#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

# __echope_path_bootstrap__
import sys as _echope_sys
from pathlib import Path as _EchoPEPath
def _echope_setup_paths():
    here = _EchoPEPath(__file__).resolve().parent
    root = here
    for anc in [here, *here.parents]:
        if (anc / "echo_paths.py").exists():
            root = anc
            break
    exp = root / "experiment"
    candidates = [root, exp]
    if exp.is_dir():
        candidates += [p for p in sorted(exp.iterdir()) if p.is_dir()]
    for _p in candidates:
        _sp = str(_p)
        if _sp not in _echope_sys.path:
            _echope_sys.path.append(_sp)
_echope_setup_paths()
# __end_echope_path_bootstrap__
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from classification.config import DEFAULT_OUTPUT_DIR as FULL_RUN_OUTPUT_DIR  # noqa: E402

from analyze_interpretability import run_analysis  # noqa: E402
from extract_attributions import run_extraction  # noqa: E402
from load_interpret_model import default_interpret_output_dir, resolve_cli_path  # noqa: E402


def discover_checkpoints(full_run_output_root: Path, seeds: list[int], tasks: list[str]) -> list[Path]:
    checkpoints = []
    seed_dirs = sorted(p for p in full_run_output_root.glob("seed-*") if p.is_dir())
    for seed_dir in seed_dirs:
        try:
            seed = int(seed_dir.name.split("-")[-1])
        except ValueError:
            continue
        if seeds and seed not in seeds:
            continue
        for task_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            if tasks and task_dir.name not in tasks:
                continue
            ckpt = task_dir / "best_checkpoint.pt"
            if ckpt.is_file():
                checkpoints.append(ckpt)
    return checkpoints


def build_extract_args(args: argparse.Namespace, checkpoint: Path, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=checkpoint,
        output_dir=output_dir,
        task=None,
        manifest_path=args.manifest_path,
        dataset_root=args.dataset_root,
        splits=args.splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        detailed_samples_per_group=args.detailed_samples_per_group,
        max_detailed_cases=args.max_detailed_cases,
        occlusion_span=args.occlusion_span,
        occlusion_spans=args.occlusion_spans,
        save_figures=args.save_figures,
    )


def build_analyze_args(args: argparse.Namespace, checkpoint: Path, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=checkpoint,
        output_dir=output_dir,
        prototype_topk=args.prototype_topk,
        max_prototype_reports=args.max_prototype_reports,
    )


def suite_output_dir(args: argparse.Namespace, checkpoint: Path) -> Path:
    default_dir = default_interpret_output_dir(checkpoint)
    relative = default_dir.relative_to(HERE / "outputs")
    if args.model_tag:
        tag = str(args.model_tag)
        if not relative.parts or relative.parts[0] != tag:
            relative = Path(tag) / relative
    output_root = resolve_cli_path(args.output_root) if args.output_root else (HERE / "outputs")
    return output_root / relative


def run_suite(args: argparse.Namespace) -> dict[str, object]:
    full_root = resolve_cli_path(args.full_run_output_root)
    checkpoints = discover_checkpoints(
        full_root,
        seeds=args.seeds,
        tasks=args.tasks,
    )
    suite_results = []
    for checkpoint in checkpoints:
        output_dir = suite_output_dir(args, checkpoint)
        if args.skip_existing and (output_dir / "summary.json").is_file():
            suite_results.append(
                {
                    "checkpoint": str(checkpoint),
                    "output_dir": str(output_dir),
                    "status": "skipped_existing",
                }
            )
            continue
        extract_summary = run_extraction(build_extract_args(args, checkpoint, output_dir))
        analysis_summary = run_analysis(build_analyze_args(args, checkpoint, output_dir))
        suite_results.append(
            {
                "checkpoint": str(checkpoint),
                "output_dir": str(output_dir),
                "extract_summary": extract_summary,
                "analysis_summary": analysis_summary,
                "status": "ok",
            }
        )

    out_root = resolve_cli_path(args.output_root) if args.output_root else (HERE / "outputs")
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "full_run_output_root": str(full_root),
        "model_tag": args.model_tag,
        "n_runs": len(suite_results),
        "results": suite_results,
    }
    (out_root / "suite_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run interpretability extraction + aggregation across multiple classification checkpoints.")
    parser.add_argument("--full-run-output-root", type=Path, default=FULL_RUN_OUTPUT_DIR)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--detailed-samples-per-group", type=int, default=6)
    parser.add_argument("--max-detailed-cases", type=int, default=40)
    parser.add_argument("--occlusion-span", type=int, default=1)
    parser.add_argument("--occlusion-spans", type=int, nargs="*", default=None)
    parser.add_argument("--save-figures", action="store_true")
    parser.add_argument("--prototype-topk", type=int, default=3)
    parser.add_argument("--max-prototype-reports", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="*", default=[])
    parser.add_argument("--tasks", type=str, nargs="*", default=[])
    parser.add_argument("--model-tag", type=str, default="")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_suite(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
