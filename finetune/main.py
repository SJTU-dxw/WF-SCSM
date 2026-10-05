"""Train a weekly k-shot classifier on a shared, fixed test set."""

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from finetune.data import ROOT, TextCollator, WeeklyDataset, prepare_split, sample_train
from finetune.models import DEFAULTS, build_model


def model_name(value):
    value = value.lower().replace("_", "-")
    aliases = {
        "scsm": "scsm-single",
        "swallow": "swallow-single",
        "tik-tok": "tiktok",
        "var-cnn": "varcnn",
    }
    return aliases.get(value, value)


def cuda_device(value):
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if device.type != "cuda":
        raise argparse.ArgumentTypeError("device must be CUDA")
    return str(device)


# Fixed experiment protocol; model hyperparameters live in finetune.models.DEFAULTS.
SETTINGS = dict(
    dataset=ROOT / "pretrain_dataset/GTT23_train.hdf5",
    reference_end=49,
    min_reference_samples=1000,
    min_week_samples=25,
    test_per_class=5,
    split_seed=0,
    seed=0,
    split_dir=ROOT / "finetune/splits",
    output_dir=ROOT / "finetune/results",
    num_workers=4,
    model_dir=None,
    grad_clip=1.0,
    accumulation_steps=1,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=model_name, choices=DEFAULTS, default="awf")
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument(
        "--min-trace-length",
        type=int,
        default=80,
        help="discard traces shorter than this value (default: 80)",
    )
    parser.add_argument(
        "--website-count",
        type=int,
        default=200,
        help="keep the websites with the most valid samples (default: 200)",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="freeze the pretrained encoder and train only the classifier head",
    )
    parser.add_argument(
        "--random-slot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="randomize the SCSM slot duration during fine-tuning (SCSM default: enabled)",
    )
    parser.add_argument(
        "--test-slot-strategy",
        choices=("fixed", "adaptive", "ensemble", "all"),
        default=None,
        help="SCSM test-time slot strategy (SCSM default: all three strategies)",
    )
    parser.add_argument("--weeks", type=int, default=1)
    parser.add_argument("--device", type=cuda_device, default="cuda")
    args = parser.parse_args(argv)
    if args.random_slot is None:
        args.random_slot = args.model == "scsm-single"
    if args.test_slot_strategy is None:
        args.test_slot_strategy = "all" if args.model == "scsm-single" else "fixed"
    if args.k < 1:
        parser.error("k must be positive")
    if args.min_trace_length < 1:
        parser.error("min-trace-length must be positive")
    if args.website_count is not None and args.website_count < 2:
        parser.error("website-count must be at least 2")
    if args.random_slot and args.model != "scsm-single":
        parser.error("random-slot is only supported by scsm-single")
    if args.test_slot_strategy != "fixed" and args.model != "scsm-single":
        parser.error("non-fixed test-slot-strategy is only supported by scsm-single")
    if args.weeks < 1:
        parser.error("weeks must be positive")
    for key, value in SETTINGS.items():
        setattr(args, key, value)
    return args


def move(value, device):
    if isinstance(value, dict):
        return {key: tensor.to(device) for key, tensor in value.items()}
    return value.to(device)


def metrics(confusion):
    true_positives = np.diag(confusion)
    precision = np.divide(
        true_positives,
        confusion.sum(0),
        out=np.zeros_like(true_positives, dtype=float),
        where=confusion.sum(0) > 0,
    )
    recall = np.divide(
        true_positives,
        confusion.sum(1),
        out=np.zeros_like(true_positives, dtype=float),
        where=confusion.sum(1) > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=precision + recall > 0,
    )
    return dict(
        accuracy=float(true_positives.sum() / confusion.sum()),
        precision=float(precision.mean()),
        recall=float(recall.mean()),
        macro_f1=float(f1.mean()),
        samples=int(confusion.sum()),
    )


@torch.no_grad()
def evaluate(model, loader, device, classes):
    model.eval()
    confusion = np.zeros((classes, classes), dtype=np.int64)
    predictions = []
    top3_hits = 0
    for x, extra, y in loader:
        if isinstance(x, torch.Tensor) and x.ndim == 5:
            batch_size, views = x.shape[:2]
            logits = model(
                move(x.flatten(0, 1), device),
                move(extra.flatten(0, 1), device),
            ).reshape(batch_size, views, -1).mean(1)
        else:
            logits = model(move(x, device), move(extra, device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits")
        prediction = logits.argmax(-1).cpu().numpy()
        target = y.numpy()
        top3 = logits.topk(min(3, classes), dim=-1).indices.cpu().numpy()
        top3_hits += int((top3 == target[:, None]).any(1).sum())
        np.add.at(confusion, (target, prediction), 1)
        predictions.extend(prediction.tolist())
    result = metrics(confusion)
    result['accuracy_at_3'] = top3_hits / int(confusion.sum())
    return result, np.asarray(predictions), confusion


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fine-tuning but is not available")

    directory, manifest = prepare_split(args)
    print(
        f"Split: {directory}; {len(manifest['classes'])} classes, "
        f"{len(manifest['weeks'])} weeks",
        flush=True,
    )
    week = args.weeks
    if week > len(manifest["weeks"]):
        raise ValueError(
            f"Invalid --weeks {week}; expected a value from 1 to {len(manifest['weeks'])}"
        )
    with np.load(directory / "indices.npz") as data:
        all_tests = data["test"]
        tests = all_tests[all_tests[:, 2] == week]
        train_records = sample_train(data[f"pool_{week}"], args.k, args.seed)
    epochs, batch, lr, optimizer_name, weight_decay, _ = DEFAULTS[args.model]
    label_smoothing = 0.1 if args.model in {"countmamba", "scsm-single"} else 0.0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    test_slot_suffix = ""
    if args.model == "scsm-single":
        strategy = (
            "ensemblerange5"
            if args.test_slot_strategy == "ensemble"
            else args.test_slot_strategy
        )
        test_slot_suffix = f"_{strategy}test"
    output = (
        args.output_dir
        / args.model
        / (
            f"k{args.k}_seed{args.seed}"
            f"{'_sites' + str(args.website_count) if args.website_count else ''}"
            f"_minlen{args.min_trace_length}"
            f"{'_randomslot' if args.random_slot else ''}"
            f"{test_slot_suffix}"
            f"{'_freeze' if args.freeze else ''}"
        )
        / f"week{week:02d}"
    )
    output.mkdir(parents=True, exist_ok=False)
    model, config, tokenizer, weights = build_model(
        args.model,
        len(manifest["classes"]),
        args.device,
        args.model_dir,
        args.freeze,
    )
    if weights and config["pretrained_config"]["dataset"]["max_day"] > args.reference_end:
        raise ValueError("Training/test weeks overlap pretraining days")
    config["random_slot"] = args.random_slot
    config["test_slot_strategy"] = args.test_slot_strategy
    config["test_slot_multiplier"] = 3.0
    config["test_slot_count"] = 5

    collate = TextCollator(tokenizer, config) if tokenizer else None

    def loader(records, train=False):
        return DataLoader(
            WeeklyDataset(args.dataset, records, args.model, config, train=train),
            batch_size=batch,
            shuffle=train,
            num_workers=args.num_workers,
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
        label_smoothing=label_smoothing,
        week=week,
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
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for step, (x, extra, y) in enumerate(train_loader):
            # Keep every k-shot sample, including singleton tails, without invalid BN statistics.
            singleton = len(y) == 1
            if singleton:
                for module in model.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
            logits = model(move(x, args.device), move(extra, args.device))
            loss = criterion(logits, y.to(args.device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            group_start = (step // args.accumulation_steps) * args.accumulation_steps
            group_size = min(
                args.accumulation_steps, len(train_loader) - group_start
            )
            (loss / group_size).backward()
            if (step + 1) % args.accumulation_steps == 0 or step + 1 == len(
                train_loader
            ):
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip, error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            seen += len(y)
            total_loss += loss.item() * len(y)
            if singleton:
                model.train()
        record = dict(
            epoch=epoch,
            train_loss=total_loss / seen,
            lr=optimizer.param_groups[0]["lr"],
        )
        print(f"{args.model} week {week}: {json.dumps(record)}", flush=True)
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if scheduler:
            scheduler.step()

    # Frozen Llama base weights remain in the pretrain checkpoint.
    state = model.state_dict()
    if args.model == "traverse":
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        # Buffers include BatchNorm running statistics needed to reproduce evaluation.
        buffers = {name for name, _ in model.named_buffers()}
        retained = trainable | buffers
        state = {name: tensor for name, tensor in state.items() if name in retained}
    torch.save(dict(model=state, epoch=epochs, config=resolved), output / "final.pt")
    test_strategies = (
        ("fixed", "adaptive", "ensemble")
        if args.model == "scsm-single" and args.test_slot_strategy == "all"
        else (args.test_slot_strategy,)
    )
    results = {}
    for strategy in test_strategies:
        config["test_slot_strategy"] = strategy
        result, predictions, confusion = evaluate(
            model, loader(tests), args.device, len(manifest["classes"])
        )
        results[strategy] = result
        suffix = f"_{strategy}" if len(test_strategies) > 1 else ""
        (output / f"test_metrics{suffix}.json").write_text(
            json.dumps(
                dict(
                    strategy=strategy,
                    overall=result,
                    by_week={str(week): result},
                    epoch=epochs,
                ),
                indent=2,
            )
        )
        np.savez_compressed(
            output / f"predictions{suffix}.npz",
            records=tests,
            predictions=predictions,
            confusion=confusion,
        )
        print(f"Test ({strategy}): {result}", flush=True)
    if len(test_strategies) > 1:
        (output / "test_metrics.json").write_text(
            json.dumps(dict(by_strategy=results, epoch=epochs), indent=2)
        )
    print(f"Saved to {output}", flush=True)
    del model, optimizer, state
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
