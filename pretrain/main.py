#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.traverse import TraVerseMLM
from model.netclr import NetCLR
from model.scsm import SCSM
from model.swallow import SwallowBYOL
from pretrain.config import load_config, resolve
from pretrain.data import CIFDataset, DirectionDataset, MaskedPromptCollator, PromptDataset
from pretrain.distributed import cleanup, is_main, setup
from pretrain.engine import run_epoch
from pretrain.scsm_data import SCSMDataset


def components(config: dict):
    method = config["method"].lower()
    path = resolve(PROJECT_ROOT, config["dataset"]["path"])
    max_day = config["dataset"].get("max_day")
    if method == "netclr":
        return DirectionDataset(path, config["augmentation"], max_day), NetCLR(config["model"]), None
    if method == "swallow":
        return CIFDataset(path, config["augmentation"], max_day), SwallowBYOL(config["model"]), None
    if method == "scsm":
        data_config = {**config["model"], **config["augmentation"]}
        return SCSMDataset(path, data_config, max_day), SCSM(config["model"]), None
    if method == "traverse":
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_model"], trust_remote_code=bool(config["model"].get("trust_remote_code", False)))
        model = TraVerseMLM(config["model"])
        collator = MaskedPromptCollator(tokenizer, int(config["dataset"]["max_length"]), float(config["objective"]["mask_ratio"]))
        model.model.resize_token_embeddings(len(tokenizer))
        return PromptDataset(path, max_day), model, collator
    raise ValueError(f"unsupported method: {method}")


def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int, step: int, config: dict) -> None:
    raw = model.module if hasattr(model, "module") else model
    state = {"model": raw.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "global_step": step, "config": config}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified pre-training entry point")
    parser.add_argument("--config", type=Path, default="pretrain/configs/scsm_single.yaml")
    args = parser.parse_args()
    config = load_config(args.config.resolve())

    # 每个GPU都有一个进程，local_rank为当前机器的GPU编号，rank为当前任务的GPU编号
    # device根据local_rank找到
    # DistributedSampler传入world_size和rank
    # DistributedDataParallel传入local_rank
    device, rank, local_rank, world_size = setup()
    try:
        base_learning_rate = float(config["train"]["learning_rate"])
        config["train"]["base_learning_rate"] = base_learning_rate
        config["train"]["learning_rate"] = base_learning_rate * world_size

        seed = int(config["train"]["seed"]) + rank
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        dataset, model, collator = components(config)
        model.to(device)
        sampler = DistributedSampler(dataset, world_size, rank, shuffle=True, seed=int(config["train"]["seed"])) if world_size > 1 else RandomSampler(dataset)
        loader = DataLoader(dataset, batch_size=int(config["dataset"]["batch_size"]), sampler=sampler, num_workers=int(config["dataset"]["num_workers"]), pin_memory=bool(config["dataset"]["pin_memory"]), drop_last=True, persistent_workers=int(config["dataset"]["num_workers"]) > 0, collate_fn=collator)
        if config["train"].get("max_epochs") is not None:
            accumulation = int(config["train"]["accumulation_steps"])
            steps_per_epoch = math.ceil(len(loader) / accumulation)
            config["train"]["max_steps"] = int(config["train"]["max_epochs"]) * steps_per_epoch
        if world_size > 1:
            model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=True, find_unused_parameters=False)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer_name = str(config["train"].get("optimizer", "adamw")).lower()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                trainable,
                lr=float(config["train"]["learning_rate"]),
                betas=tuple(config["train"].get("betas", (0.9, 0.999))),
                weight_decay=float(config["train"].get("weight_decay", 0.0)),
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                trainable,
                lr=float(config["train"]["learning_rate"]),
                momentum=float(config["train"].get("momentum", 0.0)),
                weight_decay=float(config["train"].get("weight_decay", 0.0)),
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                trainable,
                lr=float(config["train"]["learning_rate"]),
                betas=tuple(config["train"].get("betas", (0.9, 0.999))),
                weight_decay=float(config["train"].get("weight_decay", 0.0)),
            )
        else:
            raise ValueError(f"unsupported optimizer: {optimizer_name}")
        scaler = torch.amp.GradScaler("cuda", enabled=bool(config["train"]["amp"]))
        output = resolve(PROJECT_ROOT, config["output"]["directory"])
        writer = None
        if is_main():
            output.mkdir(parents=True, exist_ok=True)
            with (output / "config.json").open("w", encoding="utf-8") as stream:
                json.dump(config, stream, ensure_ascii=False, indent=2)
            writer = SummaryWriter(output)
        epoch, global_step = 0, 0
        starting_step = global_step
        training_started = time.monotonic()

        if is_main():
            print(
                f"method={config['method']} samples={len(dataset)} world_size={world_size} "
                f"trainable_parameters={sum(p.numel() for p in trainable):,} "
                f"base_lr={base_learning_rate:.3e} effective_lr={config['train']['learning_rate']:.3e}",
                flush=True,
            )
        while epoch < int(config["train"].get("max_epochs", 2**63 - 1)) and global_step < int(config["train"]["max_steps"]):
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
            for global_step in run_epoch(
                model,
                loader,
                optimizer,
                scaler,
                device,
                epoch,
                global_step,
                config,
                writer=writer,
                training_started=training_started,
                starting_step=starting_step,
            ):
                if is_main() and global_step % int(config["output"]["save_every_steps"]) == 0:
                    save_checkpoint(output / f"checkpoint-{global_step}.pt", model, optimizer, scaler, epoch, global_step, config)
            epoch += 1
        if is_main():
            save_checkpoint(output / "checkpoint-final.pt", model, optimizer, scaler, epoch, global_step, config)
            raw = model.module if hasattr(model, "module") else model
            encoder = getattr(raw, "encoder", None)
            if encoder is not None:
                torch.save(encoder.state_dict(), output / "encoder.pt")
            if writer is not None:
                writer.close()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
