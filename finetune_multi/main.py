"""Fine-tune a supported method on one ARES multi-tab scenario."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from finetune import main as training
from finetune import models
from finetune.models import DEFAULTS
from finetune_closed.main import PRETRAINED_METHODS as SHARED_PRETRAINED_METHODS
from finetune_closed.main import build_model as build_shared_model
from finetune_multi.data import (
    MultiLabelDataset,
    MultiLabelTextCollator,
    prepare_split,
    use_preselected_train,
)
from finetune_multi.metrics import precision_and_map_at_k


BASE = ROOT / "finetune_multi"
DATASETS = {
    f"{world}-{tabs}tab": ROOT
    / f"pretrain_dataset/ARES_{world.capitalize()}_{tabs}tab_train.hdf5"
    for world in ("closed", "open")
    for tabs in range(2, 6)
}
PRETRAIN_ROOTS = {
    "gtt": ROOT / "output/pretrain/GTT_dataset",
    "swallow": ROOT / "output/pretrain/swallow_dataset",
}
PRETRAINED_METHODS = {
    "scsm-multi",
    "netclr",
    "swallow-origin",
    "swallow-multi",
    "traverse",
}
MODEL_DIRECTORIES = {
    "scsm-multi": "scsm_multi",
    "swallow-multi": "swallow-multi",
}
MODEL_IMPLEMENTATIONS = {
    "scsm-multi": "scsm-single",
    "swallow-multi": "swallow-single",
}
COUNTMAMBA_MULTI_DEFAULTS = (200, 200, 2e-3, "AdamW", 0.05, 7200)
SCSM_MULTI_DEFAULTS = (200, 64, 1e-4, "AdamW", 0.05, 7200)
COUNTMAMBA_MULTI_CONFIG = {
    "max_matrix_length": 7200,
    "maximum_cell_number": 2,
    "embedding_dim": 256,
    "depth": 3,
    "drop_path_rate": 0.2,
    "simple_mode": False,
    "kernel_size": 5,
    "maximum_load_time": 320,
    "time_interval_threshold": 0.1,
    "log_transform": True,
}
_shared_parse_args = training.parse_args


def dataset_name(value: str) -> str:
    return value.lower().replace("_", "-")


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in argv or "--help" in argv:
        print(
            "Multi-tab options:\n"
            "  --dataset-name {closed,open}-{2,3,4,5}tab  required scenario\n"
            "  --pretrain-source {none,gtt,swallow}       checkpoint family\n"
            "  --k K                                      global training count (default: 1000)\n"
        )
        _shared_parse_args(["--help"])
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument(
        "--dataset-name", type=dataset_name, choices=DATASETS, required=True
    )
    custom.add_argument(
        "--pretrain-source", choices=("none", "gtt", "swallow"), default="none"
    )
    custom.add_argument(
        "--model",
        choices=tuple(
            method
            for method in DEFAULTS
            if method not in {"scsm-single", "swallow-single"}
        )
        + ("scsm-multi", "swallow-multi"),
        default="awf",
    )
    selected, remaining = custom.parse_known_args(argv)
    implementation = MODEL_IMPLEMENTATIONS.get(selected.model, selected.model)
    remaining.extend(["--model", implementation])
    if not any(arg == "--k" or arg.startswith("--k=") for arg in remaining):
        remaining.extend(["--k", "1000"])
    if not any(
        arg == "--min-trace-length" or arg.startswith("--min-trace-length=")
        for arg in remaining
    ):
        remaining.extend(["--min-trace-length", "1"])

    args = _shared_parse_args(remaining)
    if args.weeks != 1:
        raise ValueError("ARES multi-tab datasets contain only target week 1")
    args.model = selected.model
    args.model_implementation = implementation
    is_pretrained = args.model in PRETRAINED_METHODS
    if is_pretrained and selected.pretrain_source == "none":
        raise ValueError(f"{args.model} requires --pretrain-source gtt or swallow")
    if not is_pretrained and selected.pretrain_source != "none":
        raise ValueError(f"{args.model} is not pretrained; use --pretrain-source none")

    world, tab_text = selected.dataset_name.split("-")
    args.dataset_name = selected.dataset_name
    args.pretrain_source = selected.pretrain_source
    args.world = world
    args.num_tabs = int(tab_text.removesuffix("tab"))
    args.class_count = 100 if world == "closed" else 101
    args.dataset = DATASETS[selected.dataset_name]
    args.reference_end = 49
    args.website_count = None
    args.split_dir = BASE / "splits" / selected.dataset_name
    args.output_dir = BASE / "results" / selected.dataset_name / selected.pretrain_source
    if is_pretrained:
        folder = MODEL_DIRECTORIES.get(args.model, args.model)
        args.model_dir = PRETRAIN_ROOTS[selected.pretrain_source] / folder
        config_path = args.model_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"missing {selected.pretrain_source} checkpoint config: {config_path}"
            )
    else:
        args.model_dir = None
    return args


def _pretrained_config(method: str, folder: Path, source: str) -> dict:
    config_path = folder / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing {source} checkpoint config: {config_path}")
    pretrained = json.loads(config_path.read_text())
    if pretrained.get("method") != ("scsm" if method == "scsm-multi" else "swallow"):
        raise ValueError(f"{folder}: checkpoint method does not match {method}")
    if method == "scsm-multi":
        if pretrained.get("model", {}).get("simple_mode") is not False:
            raise ValueError(f"{folder}: expected an SCSM multi checkpoint")
        expected = "GTT23_train.hdf5" if source == "gtt" else "Swallow_train.hdf5"
    else:
        expected = (
            "GTT23-Swallow-multi"
            if source == "gtt"
            else "Swallow-Swallow-multi"
        )
    actual = Path(pretrained.get("dataset", {}).get("path", "")).name
    if actual.lower() != expected.lower():
        raise ValueError(f"{folder}: expected dataset {expected}, got {actual}")
    if pretrained["dataset"].get("max_day", 0) > 49:
        raise ValueError(f"{folder}: pretraining extends beyond day 49")
    return pretrained


def build_model(method, classes, device, model_dir=None, freeze=False):
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and resolved_device.index is not None:
        torch.cuda.set_device(resolved_device)

    if method == "countmamba":
        original_defaults = DEFAULTS[method]
        try:
            DEFAULTS[method] = COUNTMAMBA_MULTI_DEFAULTS
            prototype, _, tokenizer, weights = models.build_model(
                method, classes, device, model_dir, freeze
            )
        finally:
            DEFAULTS[method] = original_defaults
        model = type(prototype)(COUNTMAMBA_MULTI_CONFIG)
        del prototype
        return model.to(device), dict(COUNTMAMBA_MULTI_CONFIG), tokenizer, weights

    if method not in {"scsm-multi", "swallow-multi"}:
        if method not in SHARED_PRETRAINED_METHODS:
            return models.build_model(method, classes, device, model_dir, freeze)
        return build_shared_model(method, classes, device, model_dir, freeze)

    from feature_similarity.evaluate import SWALLOW_MULTI_CIF_CONFIG, load_model

    folder = Path(model_dir)
    source = "swallow" if folder.parent.name == "swallow_dataset" else "gtt"
    pretrained = _pretrained_config(method, folder, source)
    encoder, checkpoint = load_model(pretrained, folder, torch.device("cpu"))
    implementation = MODEL_IMPLEMENTATIONS[method]
    dimension = (
        encoder.output_dim
        if method == "scsm-multi"
        else 2048
        if pretrained["model"]["name"] == "resnet50"
        else 512
    )
    model = models.Classifier(encoder, dimension, classes, implementation)
    if freeze:
        model.encoder.requires_grad_(False)
        model.freeze = True
    if method == "scsm-multi":
        config = {
            **pretrained["model"],
            **pretrained.get("augmentation", {}),
            "pretrained_config": pretrained,
        }
    else:
        config = {**SWALLOW_MULTI_CIF_CONFIG, "pretrained_config": pretrained}
    return model.to(device), config, None, str(checkpoint)


def output_directory(args) -> Path:
    strategy = (
        "ensemblerange5"
        if args.test_slot_strategy == "ensemble"
        else args.test_slot_strategy
    )
    test_suffix = f"_{strategy}test" if args.model == "scsm-multi" else ""
    name = (
        f"k{args.k}_seed{args.seed}_minlen{args.min_trace_length}"
        f"{'_randomslot' if args.random_slot else ''}"
        f"{test_suffix}"
        f"{'_freeze' if args.freeze else ''}"
    )
    return Path(args.output_dir) / args.model / name / "week01"


@torch.no_grad()
def evaluate(model, loader, device, classes, num_tabs):
    model.eval()
    scores = []
    targets = []
    for x, extra, y in loader:
        if isinstance(x, torch.Tensor) and x.ndim == 5:
            batch_size, views = x.shape[:2]
            logits = model(
                training.move(x.flatten(0, 1), device),
                training.move(extra.flatten(0, 1), device),
            ).reshape(batch_size, views, -1).mean(1)
        else:
            logits = model(training.move(x, device), training.move(extra, device))
        if logits.shape[-1] != classes:
            raise ValueError(f"Expected {classes} output scores, got {logits.shape[-1]}")
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits")
        scores.append(logits.cpu().numpy())
        # DataLoader worker tensors live in shared memory. Copy before retaining
        # the batch so a large test set does not keep one file descriptor per batch.
        targets.append(y.numpy().copy())
    score_array = np.concatenate(scores)
    target_array = np.concatenate(targets).astype(np.uint8, copy=False)
    result = precision_and_map_at_k(target_array, score_array, num_tabs)
    top_k = np.argsort(score_array, axis=1, kind="stable")[:, -num_tabs:][:, ::-1]
    return result, score_array, target_array, top_k


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fine-tuning but is not available")

    directory, manifest = prepare_split(args)
    with np.load(directory / "indices.npz") as data:
        tests = data["test"]
        train_records = use_preselected_train(data["pool_1"], args.k, args.seed)

    if args.model == "countmamba":
        defaults = COUNTMAMBA_MULTI_DEFAULTS
    elif args.model == "scsm-multi":
        defaults = SCSM_MULTI_DEFAULTS
    else:
        defaults = DEFAULTS[args.model_implementation]
    epochs, batch, lr, optimizer_name, weight_decay, _ = defaults
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    output = output_directory(args)
    output.mkdir(parents=True, exist_ok=False)
    model, config, tokenizer, weights = build_model(
        args.model,
        args.class_count,
        args.device,
        args.model_dir,
        args.freeze,
    )
    if weights and config["pretrained_config"]["dataset"]["max_day"] > args.reference_end:
        raise ValueError("Training/test data overlap pretraining days")
    config["random_slot"] = args.random_slot
    config["test_slot_strategy"] = args.test_slot_strategy
    config["test_slot_multiplier"] = 3.0
    config["test_slot_count"] = 5
    collate = MultiLabelTextCollator(tokenizer, config) if tokenizer else None

    def loader(records, train=False):
        return DataLoader(
            MultiLabelDataset(
                args.dataset,
                records,
                args.model_implementation,
                config,
                train=train,
            ),
            batch_size=batch,
            shuffle=train,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate,
            generator=torch.Generator().manual_seed(args.seed),
        )

    train_loader = loader(train_records, True)
    np.save(output / "train_indices.npy", train_records)
    resolved = dict(
        vars(args),
        epochs=epochs,
        batch_size=batch,
        lr=lr,
        optimizer=optimizer_name,
        weight_decay=weight_decay,
        criterion="MultiLabelSoftMarginLoss",
        week=1,
        split=str(directory),
        classes=manifest["classes"],
        features=config,
        pretrained_weights=weights,
    )
    (output / "config.json").write_text(json.dumps(resolved, indent=2, default=str))
    optimizer = getattr(torch.optim, optimizer_name)(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = None
    if args.model == "ares":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    if args.model == "countmamba":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs, eta_min=1e-6
        )
    criterion = torch.nn.MultiLabelSoftMarginLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for step, (x, extra, y) in enumerate(train_loader):
            singleton = len(y) == 1
            if singleton:
                for module in model.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
            logits = model(training.move(x, args.device), training.move(extra, args.device))
            loss = criterion(logits, y.to(args.device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            group_start = (step // args.accumulation_steps) * args.accumulation_steps
            group_size = min(args.accumulation_steps, len(train_loader) - group_start)
            (loss / group_size).backward()
            if (step + 1) % args.accumulation_steps == 0 or step + 1 == len(train_loader):
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip, error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            seen += len(y)
            total_loss += loss.item() * len(y)
            if singleton:
                model.train()
        record = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(f"{args.dataset_name} {args.model}: {json.dumps(record)}", flush=True)
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if scheduler:
            scheduler.step()

    state = model.state_dict()
    if args.model == "traverse":
        trainable = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        buffers = {name for name, _ in model.named_buffers()}
        retained = trainable | buffers
        state = {name: tensor for name, tensor in state.items() if name in retained}
    torch.save({"model": state, "epoch": epochs, "config": resolved}, output / "final.pt")

    test_strategies = (
        ("fixed", "adaptive", "ensemble")
        if args.model == "scsm-multi" and args.test_slot_strategy == "all"
        else (args.test_slot_strategy,)
    )
    results = {}
    for strategy in test_strategies:
        config["test_slot_strategy"] = strategy
        result, scores, targets, top_k = evaluate(
            model, loader(tests), args.device, args.class_count, args.num_tabs
        )
        results[strategy] = result
        suffix = f"_{strategy}" if len(test_strategies) > 1 else ""
        payload = {
            "strategy": strategy,
            "target_week": 1,
            "scenario": args.dataset_name,
            "world": args.world,
            "num_tabs": args.num_tabs,
            "overall": result,
            "by_week": {"1": result},
            "epoch": epochs,
        }
        (output / f"test_metrics{suffix}.json").write_text(
            json.dumps(payload, indent=2)
        )
        np.savez_compressed(
            output / f"predictions{suffix}.npz",
            records=tests,
            targets=targets,
            scores=scores,
            top_k=top_k,
        )
        print(f"Test ({strategy}): {result}", flush=True)
    if len(test_strategies) > 1:
        (output / "test_metrics.json").write_text(
            json.dumps(
                {
                    "target_week": 1,
                    "scenario": args.dataset_name,
                    "world": args.world,
                    "num_tabs": args.num_tabs,
                    "by_strategy": results,
                    "epoch": epochs,
                },
                indent=2,
            )
        )
    print(f"Saved to {output}", flush=True)
    del model, optimizer, state
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
