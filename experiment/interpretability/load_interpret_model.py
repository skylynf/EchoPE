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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent.parent
FULL_RUN_ROOT = EXP_ROOT / "classification"
LORA_RUN_ROOT = EXP_ROOT / "finetuning"
ORIG_CWD = Path.cwd().resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))
if str(FULL_RUN_ROOT) not in sys.path:
    sys.path.insert(0, str(FULL_RUN_ROOT))
if str(LORA_RUN_ROOT) not in sys.path:
    sys.path.insert(0, str(LORA_RUN_ROOT))

from echo_paths import ECHO_ROOT, setup_echo_root_cwd  # noqa: E402

setup_echo_root_cwd()

from classification.config import DEFAULT_DATASET_ROOT  # noqa: E402
from classification.data import (  # noqa: E402
    TASKS,
    load_split_manifest,
    make_or_load_manifest,
    manifest_records,
    records_to_samples,
)
from finetuning.train_lora import count_parameters  # noqa: E402
from finetuning.train_lora_distill import (  # noqa: E402
    LoRAEchoPrimeDistillModel,
    NUM_COARSE_VIEWS,
    PEVideoDatasetWithView,
    build_classifier,
    build_encoder,
    view_collate,
)
from classification.train_full_run import default_manifest_path  # noqa: E402


def resolve_cli_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ORIG_CWD / candidate).resolve()


def path_like(value: object) -> str:
    return str(value) if isinstance(value, Path) else str(value)


def namespace_from_checkpoint_args(payload: dict[str, object]) -> argparse.Namespace:
    args_dict = dict(payload)
    for key, value in list(args_dict.items()):
        if isinstance(value, Path):
            args_dict[key] = str(value)
        elif isinstance(value, tuple):
            args_dict[key] = list(value)
    return argparse.Namespace(**args_dict)


def default_interpret_output_dir(checkpoint_path: str | Path) -> Path:
    ckpt_path = resolve_cli_path(checkpoint_path)
    parts = ckpt_path.parts
    for idx, part in enumerate(parts):
        if part == "outputs" or part.startswith("outputs_"):
            suffix = list(parts[idx + 1 : -1])
            if part.startswith("outputs_"):
                suffix = [part.removeprefix("outputs_"), *suffix]
            if suffix:
                return HERE / "outputs" / Path(*suffix)
    if "outputs" in parts:
        idx = parts.index("outputs")
        suffix = list(parts[idx + 1 : -1])
        if suffix:
            return HERE / "outputs" / Path(*suffix)
    return HERE / "outputs" / ckpt_path.stem


@dataclass
class InterpretArtifacts:
    checkpoint_path: Path
    checkpoint_payload: dict[str, object]
    checkpoint_args: argparse.Namespace
    model: LoRAEchoPrimeDistillModel
    device: torch.device
    manifest_path: Path
    manifest_payload: dict[str, object]
    records_by_split: dict[str, list[dict[str, object]]]
    loaders_by_split: dict[str, torch.utils.data.DataLoader]


def load_checkpoint_payload(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> tuple[Path, dict[str, object]]:
    resolved = resolve_cli_path(checkpoint_path)
    payload = torch.load(str(resolved), map_location=map_location, weights_only=False)
    return resolved, payload


def infer_manifest_path(
    checkpoint_args: argparse.Namespace,
    checkpoint_payload: dict[str, object],
    checkpoint_path: Path,
) -> Path:
    result_path = checkpoint_path.with_name("result.json")
    if result_path.is_file():
        try:
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            manifest_value = result_payload.get("manifest_path")
            if manifest_value:
                return resolve_cli_path(path_like(manifest_value))
        except json.JSONDecodeError:
            pass
    manifest_arg = getattr(checkpoint_args, "manifest_path", None)
    if manifest_arg:
        return resolve_cli_path(path_like(manifest_arg))
    seed = int(checkpoint_payload.get("seed", getattr(checkpoint_args, "seed", 2024)))
    output_root = getattr(checkpoint_args, "output_root", None)
    if output_root is not None:
        return resolve_cli_path(default_manifest_path(resolve_cli_path(path_like(output_root)), seed))
    return resolve_cli_path(default_manifest_path(ECHO_ROOT / "experiments" / "classification" / "outputs", seed))


def ensure_manifest(
    checkpoint_args: argparse.Namespace,
    checkpoint_payload: dict[str, object],
    checkpoint_path: Path,
    dataset_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> Path:
    resolved_manifest = (
        resolve_cli_path(manifest_path)
        if manifest_path is not None
        else infer_manifest_path(checkpoint_args, checkpoint_payload, checkpoint_path)
    )
    dataset_root_value = (
        resolve_cli_path(dataset_root)
        if dataset_root is not None
        else resolve_cli_path(path_like(getattr(checkpoint_args, "dataset_root", DEFAULT_DATASET_ROOT)))
    )
    test_frac = float(getattr(checkpoint_args, "test_frac", 0.20))
    val_frac_total = float(getattr(checkpoint_args, "val_frac_total", 0.10))
    seed = int(checkpoint_payload.get("seed", getattr(checkpoint_args, "seed", 2024)))
    return make_or_load_manifest(
        resolved_manifest,
        dataset_root=dataset_root_value,
        test_frac=test_frac,
        val_frac_total=val_frac_total,
        seed=seed,
        force_rebuild=False,
    )


def build_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
) -> tuple[Path, dict[str, object], argparse.Namespace, LoRAEchoPrimeDistillModel]:
    resolved_ckpt, checkpoint_payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    checkpoint_args = namespace_from_checkpoint_args(checkpoint_payload["args"])
    weights_path = ECHO_ROOT / "model_data" / "weights" / "echo_prime_encoder.pt"
    encoder, _injected = build_encoder(checkpoint_args, weights_path)
    head_view_mode = getattr(checkpoint_args, "head_view_mode", "none")
    classifier_in_dim = 512 + (NUM_COARSE_VIEWS if head_view_mode == "coarse4" else 0)
    classifier = build_classifier(
        checkpoint_args.head,
        checkpoint_args.hidden,
        checkpoint_args.head_dropout,
        num_blocks=getattr(checkpoint_args, "num_blocks", 3),
        in_dim=classifier_in_dim,
    )
    model = LoRAEchoPrimeDistillModel(
        encoder,
        classifier,
        view_aux=getattr(checkpoint_args, "view_aux", "none"),
        head_view_mode=head_view_mode,
        head_dropout=getattr(checkpoint_args, "head_dropout", 0.3),
    )
    model.load_state_dict(checkpoint_payload["model_state"], strict=True)
    model.eval()
    model.to(torch.device(device))
    return resolved_ckpt, checkpoint_payload, checkpoint_args, model


def build_records_and_loaders(
    manifest_payload: dict[str, object],
    task: str,
    batch_size: int,
    num_workers: int,
    splits: Iterable[str],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, torch.utils.data.DataLoader]]:
    split_set = tuple(dict.fromkeys(str(split) for split in splits))
    unknown = [split for split in split_set if split not in {"train", "val", "test"}]
    if unknown:
        raise ValueError(f"Unknown split(s): {unknown}")
    records_by_split: dict[str, list[dict[str, object]]] = {}
    use_coarse_view = str(manifest_payload.get("head_view_mode", "none")) == "coarse4"
    samples_by_split: dict[str, list[tuple[str, int, str] | tuple[str, int, str, int]]] = {}
    for split in split_set:
        records = manifest_records(manifest_payload, split, task=task)
        records_by_split[split] = records
        samples_by_split[split] = records_to_samples(records, include_coarse_view_idx=use_coarse_view)

    loaders: dict[str, DataLoader] = {}
    for split in split_set:
        loaders[split] = DataLoader(
            PEVideoDatasetWithView(samples_by_split[split]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=view_collate,
        )
    return records_by_split, loaders


def load_interpret_artifacts(
    checkpoint_path: str | Path,
    task: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    dataset_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
    splits: Iterable[str] = ("train", "val", "test"),
    device: str | torch.device | None = None,
) -> InterpretArtifacts:
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    resolved_ckpt, checkpoint_payload, checkpoint_args, model = build_model_from_checkpoint(
        checkpoint_path,
        device=resolved_device,
    )
    resolved_task = task or str(checkpoint_payload.get("task", getattr(checkpoint_args, "task", "pooled")))
    if resolved_task not in TASKS:
        raise ValueError(f"Unknown task {resolved_task!r}. Choices: {TASKS}")
    resolved_manifest = ensure_manifest(
        checkpoint_args,
        checkpoint_payload,
        resolved_ckpt,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
    )
    manifest_payload = load_split_manifest(resolved_manifest)
    manifest_payload["head_view_mode"] = getattr(checkpoint_args, "head_view_mode", "none")
    resolved_batch = int(batch_size or getattr(checkpoint_args, "batch", 4))
    resolved_workers = int(num_workers if num_workers is not None else getattr(checkpoint_args, "num_workers", 4))
    records_by_split, loaders_by_split = build_records_and_loaders(
        manifest_payload,
        task=resolved_task,
        batch_size=resolved_batch,
        num_workers=resolved_workers,
        splits=splits,
    )
    return InterpretArtifacts(
        checkpoint_path=resolved_ckpt,
        checkpoint_payload=checkpoint_payload,
        checkpoint_args=checkpoint_args,
        model=model,
        device=resolved_device,
        manifest_path=resolved_manifest,
        manifest_payload=manifest_payload,
        records_by_split=records_by_split,
        loaders_by_split=loaders_by_split,
    )


def model_summary(model: nn.Module) -> dict[str, int]:
    trainable, total = count_parameters(model)
    return {"trainable_params": int(trainable), "total_params": int(total)}

