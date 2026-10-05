"""Schedule the complete ARES multi-tab fine-tuning matrix on available GPUs."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "finetune_multi"
DATASETS = tuple(
    f"{world}-{tabs}tab" for world in ("closed", "open") for tabs in range(2, 6)
)
BASELINES = ("awf", "tmwf", "ares", "df", "tiktok", "varcnn", "rf", "countmamba")
PRETRAINED = ("scsm-multi", "netclr", "swallow-origin", "swallow-multi", "traverse")
PRETRAIN_ROOT = {
    "gtt": ROOT / "output/pretrain/GTT_dataset",
    "swallow": ROOT / "output/pretrain/swallow_dataset",
}
PRETRAIN_DIR = {"scsm-multi": "scsm_multi"}


@dataclass(frozen=True)
class Task:
    dataset: str
    model: str
    source: str
    random_slot: bool = False
    freeze: bool = False


def task_matrix(datasets, sources):
    tasks = []
    for dataset in datasets:
        tasks.extend(Task(dataset, model, "none") for model in BASELINES)
        for source in sources:
            for model in PRETRAINED:
                if model == "scsm-multi":
                    tasks.extend(
                        (Task(dataset, model, source, True), Task(dataset, model, source, False))
                    )
                elif model in {"swallow-origin", "swallow-multi"}:
                    tasks.extend(
                        (Task(dataset, model, source, freeze=True), Task(dataset, model, source))
                    )
                elif model == "traverse":
                    tasks.append(Task(dataset, model, source, freeze=True))
                else:
                    tasks.append(Task(dataset, model, source))
    return tasks


def suffixes(task, strategy):
    random_suffix = "_randomslot" if task.random_slot else ""
    freeze_suffix = "_freeze" if task.freeze else ""
    if task.model == "scsm-multi":
        test_suffix = (
            "_alltest"
            if strategy == "all"
            else "_ensemblerange5test"
            if strategy == "ensemble"
            else f"_{strategy}test"
        )
    else:
        test_suffix = ""
    return random_suffix, test_suffix, freeze_suffix


def output_path(task, k, minimum, strategy):
    random_suffix, test_suffix, freeze_suffix = suffixes(task, strategy)
    return (
        BASE
        / "results"
        / task.dataset
        / task.source
        / task.model
        / f"k{k}_seed0_minlen{minimum}{random_suffix}{test_suffix}{freeze_suffix}"
        / "week01"
    )


def is_complete(task, output, strategy):
    if not (output / "final.pt").is_file() or not (output / "test_metrics.json").is_file():
        return False
    strategies = (
        ("fixed", "adaptive", "ensemble")
        if task.model == "scsm-multi" and strategy == "all"
        else (strategy if task.model == "scsm-multi" else "fixed",)
    )
    for current in strategies:
        suffix = f"_{current}" if len(strategies) > 1 else ""
        if not (output / f"test_metrics{suffix}.json").is_file():
            return False
        if not (output / f"predictions{suffix}.npz").is_file():
            return False
    return True


def gpu_used_mib(gpu):
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.splitlines()[0].strip())


def missing_checkpoint_configs(tasks):
    missing = []
    for task in tasks:
        if task.source == "none":
            continue
        path = (
            PRETRAIN_ROOT[task.source]
            / PRETRAIN_DIR.get(task.model, task.model)
            / "config.json"
        )
        if not path.is_file() and path not in missing:
            missing.append(path)
    return missing


def command(task, args, gpu):
    result = [
        sys.executable,
        str(BASE / "main.py"),
        "--dataset-name",
        task.dataset,
        "--pretrain-source",
        task.source,
        "--model",
        task.model,
        "--k",
        str(args.k),
        "--min-trace-length",
        str(args.min_trace_length),
        "--weeks",
        "1",
        "--device",
        f"cuda:{gpu}",
        "--random-slot" if task.random_slot else "--no-random-slot",
    ]
    if task.model == "scsm-multi":
        result.extend(("--test-slot-strategy", args.test_slot_strategy))
    if task.freeze:
        result.append("--freeze")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--min-trace-length", type=int, default=1)
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=DATASETS
    )
    parser.add_argument(
        "--test-slot-strategy",
        choices=("fixed", "adaptive", "ensemble", "all"),
        default="all",
    )
    parser.add_argument(
        "--pretrain-sources",
        nargs="+",
        choices=("gtt", "swallow"),
        default=("gtt", "swallow"),
    )
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=(0, 1, 2, 3))
    parser.add_argument("--gpu-max-used-mib", type=int, default=512)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.k < 1 or args.min_trace_length < 1 or args.poll_seconds < 1:
        parser.error("k, min-trace-length, and poll-seconds must be positive")
    if args.gpu_max_used_mib < 0:
        parser.error("gpu-max-used-mib must be nonnegative")

    tasks = task_matrix(args.datasets, args.pretrain_sources)
    pending = []
    skipped = 0
    for task in tasks:
        output = output_path(task, args.k, args.min_trace_length, args.test_slot_strategy)
        if is_complete(task, output, args.test_slot_strategy):
            skipped += 1
            print(
                f"[scan] skip completed: {task.dataset} {task.source} {task.model}",
                flush=True,
            )
        elif output.exists():
            raise FileExistsError(f"Incomplete output exists; refusing to overwrite: {output}")
        else:
            pending.append((task, output))
    print(
        f"Found {len(tasks)} tasks; {skipped} completed, {len(pending)} pending",
        flush=True,
    )
    if args.dry_run:
        for task, output in pending:
            print(
                f"[dry-run] {task.dataset} {task.source} {task.model} "
                f"random-slot={task.random_slot} freeze={task.freeze} -> {output}"
            )
        return

    missing = missing_checkpoint_configs(tasks)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError("Required pretrained checkpoint configs are missing:\n" + formatted)

    for gpu in args.gpu_ids:
        gpu_used_mib(gpu)
    log_root = BASE / "logs"
    running = {}
    failures = []
    next_task = 0
    while next_task < len(pending) or running:
        for gpu, (process, stream, task, log) in list(running.items()):
            status = process.poll()
            if status is None:
                continue
            stream.close()
            del running[gpu]
            if status:
                failures.append((task, status, log))
                print(f"[cuda:{gpu}] failed ({status}): {task} (log: {log})", flush=True)
            else:
                print(
                    f"[cuda:{gpu}] done: {task.dataset} {task.source} {task.model}",
                    flush=True,
                )
        if failures:
            for process, stream, _, _ in running.values():
                process.terminate()
                stream.close()
            raise SystemExit(f"Multi-tab tasks failed: {failures}")
        for gpu in args.gpu_ids:
            if next_task >= len(pending) or gpu in running:
                continue
            if gpu_used_mib(gpu) > args.gpu_max_used_mib:
                continue
            task, _ = pending[next_task]
            next_task += 1
            random_suffix, test_suffix, freeze_suffix = suffixes(
                task, args.test_slot_strategy
            )
            log = (
                log_root
                / task.dataset
                / task.source
                / task.model
                / f"k{args.k}_minlen{args.min_trace_length}{random_suffix}{test_suffix}{freeze_suffix}"
                / "week01.log"
            )
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("w")
            process = subprocess.Popen(
                command(task, args, gpu),
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = (process, stream, task, log)
            print(
                f"[cuda:{gpu}] start: {task.dataset} {task.source} {task.model} "
                f"(log: {log})",
                flush=True,
            )
        if next_task < len(pending) or running:
            time.sleep(args.poll_seconds)
    print("All multi-tab experiments completed successfully.", flush=True)


if __name__ == "__main__":
    main()
