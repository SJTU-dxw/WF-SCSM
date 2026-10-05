"""Discover and evaluate every GTT23 Week 1 fine-tuned checkpoint on available GPUs."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(__file__).resolve().parent


def discover(results_root):
    tasks = []
    for checkpoint in sorted(results_root.glob("*/*/week01/final.pt")):
        source = checkpoint.parent
        config_path = source / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config = json.loads(config_path.read_text())
        if config.get("week") != 1:
            raise ValueError(f"Unexpected non-Week-1 checkpoint: {checkpoint}")
        relative = source.relative_to(results_root)
        tasks.append((source, BASE / "results" / relative.parent))
    if not tasks:
        raise FileNotFoundError(f"No Week 1 checkpoints under {results_root}")
    return tasks


def gpu_used_mib(gpu):
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.splitlines()[0].strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--gpu-max-used-mib", type=int, default=512)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--results-root", type=Path, default=ROOT / "finetune/results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpu_max_used_mib < 0 or args.poll_seconds < 1 or args.num_workers < 0:
        parser.error("invalid negative/zero scheduler option")
    tasks = discover(args.results_root.resolve())
    print(f"Discovered {len(tasks)} Week 1 checkpoints", flush=True)
    if args.dry_run:
        for source, output in tasks:
            print(f"{source} -> {output}")
        return

    log_root = BASE / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    pending = list(tasks)
    running = {}
    failures = []
    while pending or running:
        for gpu, (process, handle, source) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del running[gpu]
            if code:
                failures.append((source, code))
                print(f"[cuda:{gpu}] failed ({code}): {source}", flush=True)
            else:
                print(f"[cuda:{gpu}] done: {source}", flush=True)
        if failures:
            for process, handle, _ in running.values():
                process.terminate()
                handle.close()
            raise SystemExit(f"Evaluation failed: {failures}")
        for gpu in args.gpu_ids:
            if not pending or gpu in running:
                continue
            if gpu_used_mib(gpu) > args.gpu_max_used_mib:
                continue
            source, output = pending.pop(0)
            relative = source.relative_to(args.results_root.resolve())
            log = log_root / relative.parent / "week01_all_weeks.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a")
            command = [
                sys.executable, str(BASE / "evaluate.py"),
                "--source", str(source), "--output", str(output),
                "--device", f"cuda:{gpu}", "--num-workers", str(args.num_workers),
            ]
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env)
            running[gpu] = (process, handle, source)
            print(f"[cuda:{gpu}] start: {source} (log: {log})", flush=True)
        if pending or running:
            time.sleep(args.poll_seconds)
    print(f"Completed {len(tasks)} checkpoints; results: {BASE / 'results'}", flush=True)


if __name__ == "__main__":
    main()
