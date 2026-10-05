from __future__ import annotations

from contextlib import nullcontext
import math
import time
import torch
import torch.nn.functional as F

from .distributed import gather_with_local_gradient, is_main, reduce_mean


def format_duration(seconds: float) -> str:
    """Format a duration as HH:MM:SS, allowing runs longer than one day."""
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def cosine_lr(step: int, total: int, warmup: int, base: float, minimum: float) -> float:
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))


def nt_xent(first: torch.Tensor, second: torch.Tensor, temperature: float) -> torch.Tensor:
    first = F.normalize(gather_with_local_gradient(first), dim=-1)
    second = F.normalize(gather_with_local_gradient(second), dim=-1)
    features = torch.cat((first, second), dim=0)
    count = first.shape[0]
    logits = features @ features.T / temperature
    logits.fill_diagonal_(torch.finfo(logits.dtype).min)
    targets = (torch.arange(2 * count, device=features.device) + count) % (2 * count)
    return F.cross_entropy(logits, targets)


def cross_view_contrastive(first: torch.Tensor, second: torch.Tensor, temperature: float) -> torch.Tensor:
    first = F.normalize(gather_with_local_gradient(first), dim=-1)
    second = F.normalize(gather_with_local_gradient(second), dim=-1)
    logits = first @ second.T / temperature
    targets = torch.arange(first.shape[0], device=first.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def train_step(method: str, model, batch, device, config: dict) -> torch.Tensor:
    if method == "netclr":
        first, second = (value.to(device, non_blocking=True) for value in batch)
        features = model(torch.cat((first, second), dim=0))
        first_features, second_features = features.chunk(2, dim=0)
        return nt_xent(first_features, second_features, float(config["objective"]["temperature"]))
    if method == "swallow":
        first, second = (value.to(device, non_blocking=True) for value in batch)
        return model(first, second)
    if method == "scsm":
        first, second = batch
        first_matrix, first_index = (value.to(device, non_blocking=True) for value in first)
        second_matrix, second_index = (value.to(device, non_blocking=True) for value in second)
        features = model(
            torch.cat((first_matrix, second_matrix), dim=0),
            torch.cat((first_index, second_index), dim=0),
        )
        first_features, second_features = features.chunk(2, dim=0)
        return cross_view_contrastive(first_features, second_features, float(config["objective"]["temperature"]))
    if method == "traverse":
        return model({key: value.to(device, non_blocking=True) for key, value in batch.items()})
    raise ValueError(f"unknown method: {method}")


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch: int,
    global_step: int,
    config: dict,
    writer=None,
    training_started: float | None = None,
    starting_step: int = 0,
):
    method, train = config["method"].lower(), config["train"]
    accumulation = int(train["accumulation_steps"])
    total_steps = int(train["max_steps"])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started, loss_sum, batches = time.monotonic(), 0.0, 0
    if training_started is None:
        training_started = started
    use_amp = bool(train["amp"])
    amp_dtype = torch.bfloat16 if train.get("amp_dtype", "float16") == "bfloat16" else torch.float16
    for iteration, batch in enumerate(loader):
        if global_step >= total_steps:
            break
        sync_step = (iteration + 1) % accumulation == 0 or iteration + 1 == len(loader)
        sync_context = nullcontext() if sync_step or not hasattr(model, "no_sync") else model.no_sync()
        with sync_context:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = train_step(method, model, batch, device, config) / accumulation
            scaler.scale(loss).backward()
        loss_value = float(loss.detach()) * accumulation
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"non-finite loss at step {global_step}: {loss_value}")
        loss_sum += loss_value
        batches += 1
        if not sync_step:
            continue
        if train.get("grad_clip") is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        if train.get("lr_schedule", "cosine") == "constant":
            lr = float(train["learning_rate"])
        else:
            lr = cosine_lr(global_step, total_steps, int(train["warmup_steps"]), float(train["learning_rate"]), float(train["min_learning_rate"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        raw = model.module if hasattr(model, "module") else model
        if method == "swallow":
            raw.update_target(float(config["objective"]["target_momentum"]))
        reduced_loss = reduce_mean(loss.detach() * accumulation)
        if writer is not None:
            writer.add_scalar("train/loss", float(reduced_loss), global_step)
            writer.add_scalar("train/lr", lr, global_step)
        if global_step % int(train["log_every_steps"]) == 0:
            average = float(reduce_mean(torch.tensor(loss_sum / batches, device=device)))
            if is_main():
                now = time.monotonic()
                elapsed = now - training_started
                completed_steps = max(1, global_step - starting_step)
                remaining_steps = max(0, total_steps - global_step)
                eta = elapsed / completed_steps * remaining_steps
                print(
                    f"epoch={epoch} step={global_step}/{total_steps} loss={average:.5f} "
                    f"lr={lr:.3e} time={now-started:.1f}s "
                    f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
                    f"estimated_total={format_duration(elapsed + eta)}",
                    flush=True,
                )
            loss_sum = 0.0
            batches = 0
            started = time.monotonic()
        yield global_step
