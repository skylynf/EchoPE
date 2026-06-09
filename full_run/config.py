from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = HERE.parent.parent.parent / "dataset" / "preprocessed"
DEFAULT_OUTPUT_DIR = HERE / "outputs"

DEFAULT_SEEDS = (2024, 2025, 2026)
DEFAULT_TASKS = ("pooled", "A4", "PSL", "PSS", "Subcostal")

DEFAULT_TEST_FRAC = 0.20
DEFAULT_VAL_FRAC_TOTAL = 0.10
DEFAULT_THRESHOLD_MODE = "youden"
DEFAULT_FIXED_THRESHOLD = 0.50

BEST_MODEL_METRIC = "val_roc_auc"


@dataclass(frozen=True)
class FullRunConfig:
    dataset_root: str = str(DEFAULT_DATASET_ROOT)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    tasks: tuple[str, ...] = DEFAULT_TASKS
    test_frac: float = DEFAULT_TEST_FRAC
    val_frac_total: float = DEFAULT_VAL_FRAC_TOTAL
    threshold_mode: str = DEFAULT_THRESHOLD_MODE
    fixed_threshold: float = DEFAULT_FIXED_THRESHOLD

    # Fine-tuning defaults: start from the full-FT baseline requested in plan.
    lora: str = "off"
    ft_mode: str = "full"
    kd_mode: str = "none"
    view_aux: str = "none"
    kd_weight: float = 0.5
    view_weight: float = 0.3
    view_kd_weight: float = 0.3
    view_kd_temperature: float = 2.0
    rank: int = 8
    alpha: float = 16.0
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("qkv", "proj")

    head: str = "mlp"
    hidden: int = 256
    head_dropout: float = 0.30
    num_blocks: int = 3
    head_view_mode: str = "none"

    aug_intensity: bool = False
    aug_intensity_p: float = 0.5
    aug_brightness: float = 0.10
    aug_contrast: tuple[float, float] = (0.90, 1.10)
    aug_gamma: tuple[float, float] = (0.90, 1.10)
    aug_noise_std: float = 0.01

    aug_affine: bool = False
    aug_affine_p: float = 0.5
    aug_translate: tuple[float, float] = (0.05, 0.05)
    aug_scale: tuple[float, float] = (0.95, 1.05)
    aug_rotate: float = 8.0

    aug_temporal: bool = False
    aug_temporal_p: float = 0.3
    aug_temporal_min_frac: float = 0.80
    aug_temporal_random_start: bool = False

    epochs: int = 20
    lr: float = 1e-5
    head_lr: float = 3e-4
    batch: int = 4
    grad_accum: int = 4
    weight_decay: float = 0.01
    warmup_epochs: int = 3
    patience: int = 6
    num_workers: int = 4
    teacher_device: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

