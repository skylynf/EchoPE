from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

ORIG_CWD = Path.cwd().resolve()

from analyze_full_run import run_analysis
from config import DEFAULT_DATASET_ROOT, DEFAULT_OUTPUT_DIR, DEFAULT_SEEDS, DEFAULT_TASKS
from train_full_run import default_manifest_path, run_experiment, task_run_name
from lora_run.train_lora import add_augmentation_args, build_augmentation_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full EchoPrime pooled + per-view PE fine-tuning suite.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--tasks", nargs="+", choices=list(DEFAULT_TASKS), default=list(DEFAULT_TASKS))
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing result.json for completed runs.")
    parser.add_argument("--force-run", action="store_true", help="Ignore existing run outputs and retrain.")
    parser.add_argument("--force-manifest", action="store_true", help="Rebuild split manifests for each seed.")
    parser.add_argument("--skip-analysis", action="store_true")

    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--val-frac-total", type=float, default=0.10)
    parser.add_argument("--threshold-mode", choices=("youden", "f1", "fixed"), default="youden")
    parser.add_argument("--fixed-threshold", type=float, default=0.50)

    parser.add_argument("--lora", choices=["on", "off"], default="off")
    parser.add_argument("--ft-mode", choices=["lora", "full", "partial-stage4", "frozen"], default="full")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05, dest="lora_dropout")
    parser.add_argument("--target-modules", nargs="+", default=["qkv", "proj"])

    parser.add_argument("--head", choices=["mlp", "residual"], default="mlp")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--head-dropout", type=float, default=0.3, dest="head_dropout")
    parser.add_argument("--num-blocks", type=int, default=3, dest="num_blocks")
    parser.add_argument("--head-view-mode", choices=["none", "coarse4"], default="none", dest="head_view_mode")

    parser.add_argument("--kd-mode", choices=["none", "cos", "l2", "rkd-d", "rkd-a", "combo"], default="none")
    parser.add_argument("--kd-weight", type=float, default=0.5, dest="kd_weight")
    parser.add_argument("--kd-warmup-epochs", type=int, default=2, dest="kd_warmup_epochs")
    parser.add_argument("--lam-rkd", type=float, default=1.0, dest="lam_rkd")

    parser.add_argument("--view-aux", choices=["none", "ce", "kd", "both"], default="none", dest="view_aux")
    parser.add_argument("--view-weight", type=float, default=0.3, dest="view_weight")
    parser.add_argument("--view-kd-weight", type=float, default=0.3, dest="view_kd_weight")
    parser.add_argument("--view-kd-temperature", type=float, default=2.0, dest="view_kd_temperature")

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4, dest="head_lr")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4, dest="grad_accum")
    parser.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    parser.add_argument("--warmup-epochs", type=int, default=3, dest="warmup_epochs")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4, dest="num_workers")
    parser.add_argument("--teacher-device", default="", dest="teacher_device")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    add_augmentation_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = args.output_root if Path(args.output_root).is_absolute() else (ORIG_CWD / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    aug_config = build_augmentation_config(args)

    split_manifests: dict[str, str] = {}
    all_results: list[dict[str, object]] = []

    for seed in args.seeds:
        manifest_path = default_manifest_path(output_root, seed)
        split_manifests[str(seed)] = str(manifest_path.resolve())
        for task in args.tasks:
            result_path = output_root / f"seed-{seed}" / task_run_name(task, args.head_view_mode) / "result.json"
            if result_path.is_file() and args.skip_existing and not args.force_run:
                print(f"[skip] reusing {result_path}")
                all_results.append(json.loads(result_path.read_text(encoding="utf-8")))
                continue

            run_args = copy.deepcopy(args)
            run_args.seed = int(seed)
            run_args.task = task
            run_args.manifest_path = manifest_path
            result = run_experiment(run_args)
            all_results.append(result)

    payload = {
        "config": {
            "dataset_root": str(args.dataset_root if Path(args.dataset_root).is_absolute() else (ORIG_CWD / args.dataset_root).resolve()),
            "output_root": str(output_root),
            "seeds": [int(seed) for seed in args.seeds],
            "tasks": list(args.tasks),
            "test_frac": args.test_frac,
            "val_frac_total": args.val_frac_total,
            "threshold_mode": args.threshold_mode,
            "fixed_threshold": args.fixed_threshold,
            "lora": args.lora,
            "ft_mode": args.ft_mode,
            "kd_mode": args.kd_mode,
            "view_aux": args.view_aux,
            "epochs": args.epochs,
            "lr": args.lr,
            "head_lr": args.head_lr,
            "head_view_mode": args.head_view_mode,
            "batch": args.batch,
            "grad_accum": args.grad_accum,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "patience": args.patience,
            "device": args.device,
            "smoke": args.smoke,
            "augmentation": aug_config.to_dict(),
        },
        "split_manifests": split_manifests,
        "results": all_results,
    }
    results_json = output_root / "results.json"
    results_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {results_json}")

    if not args.skip_analysis:
        out_dir = run_analysis(results_json, output_root / "analysis")
        print(f"[done] analysis written to {out_dir}")


if __name__ == "__main__":
    main()

