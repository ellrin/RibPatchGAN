"""Configuration loading: configs/*.yaml -> typed dataclasses.

Every step-2 / step-3 hyperparameter is defined once in the YAML file; the
dataclasses below mirror it and hold the same defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

STEP2_DIR_NAME = "step2_inpainting_gan"
STEP3_DIR_NAME = "step3_classifier"


# --- datasets -----------------------------------------------------------------

@dataclass
class DatasetSpec:
    name: str = ""
    images_root: Path = Path()
    # operating point for test reporting: youden_valid | spec_target
    threshold: str = "youden_valid"
    target_spec: float = 0.90


@dataclass
class DatasetsConfig:
    train: DatasetSpec = field(default_factory=DatasetSpec)
    valid: DatasetSpec = field(default_factory=DatasetSpec)
    tests: List[DatasetSpec] = field(default_factory=list)


@dataclass
class PathsConfig:
    results_root: Path = Path("results")
    site_stats: Path = Path("cgmh_stats.json")


# --- shared LaMa architecture ---------------------------------------------------

@dataclass
class LamaArchitectureConfig:
    ngf: int = 64
    n_downsampling: int = 3
    n_blocks: int = 9
    ratio_g: float = 0.75
    num_classes: int = 2
    noise_dim: int = 64
    fx_label: int = 1
    mask_feather_radius: int = 4


# --- step 2 ----------------------------------------------------------------------

@dataclass
class GanTrainingConfig:
    use_nofx_training: bool = True
    force_generate_fx: bool = True
    image_size: int = 1024
    patch_size: int = 128
    mask_size: int = 84
    mask_size_jitter: int = 5
    patch_jitter: int = 6
    patches_per_image: int = 8
    max_patches_per_batch: int = 16
    batch_size: int = 8
    use_rot_aug: bool = True
    rot_aug_prob: float = 0.5
    rot_max_deg: float = 10.0
    use_clahe_aug: bool = True
    clahe_aug_prob: float = 0.5
    clahe_base_clip: float = 3.0
    clahe_clip_jitter: float = 1.5
    clahe_tile_size: List[int] = field(default_factory=lambda: [8, 8])
    epochs: int = 300
    g_lr: float = 1e-4
    d_lr: float = 1e-4
    adam_b1: float = 0.0
    adam_b2: float = 0.99
    r1_gamma: float = 10.0
    d_reg_interval: int = 16
    lambda_rec: float = 10.0
    lambda_vgg: float = 1.5
    lambda_context: float = 1.0
    lambda_seam: float = 0.5
    seam_loss_width: int = 4


# --- step 3 ----------------------------------------------------------------------

@dataclass
class ClassifierModelConfig:
    encoder: str = "convnext_tiny"
    pretrained: bool = True
    dropout: float = 0.4


@dataclass
class ClassifierTrainConfig:
    seed: int = 42
    epochs: int = 10
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-3
    amp: bool = True
    grad_clip: float = 1.0
    warmup_epochs: int = 1
    image_size: int = 1024
    patch_size: int = 224
    patch_forward_chunk_size: int = 12
    eval_patch_chunk_size: int = 24
    use_rot_aug: bool = True
    rot_aug_prob: float = 0.5
    rot_max_deg: float = 15.0
    use_rotate_scale_aug: bool = True
    use_clahe_aug: bool = True
    clahe_aug_prob: float = 0.5
    clahe_base_clip: float = 2.0
    clahe_clip_jitter: float = 0.5
    clahe_tile_size: List[int] = field(default_factory=lambda: [8, 8])
    use_intensity_aug: bool = True
    bright_aug_range: float = 0.10
    contrast_aug_range: float = 0.10
    resample_aug_scale: float = 0.15
    resample_aug_prob: float = 0.50
    resample_force_distinct: bool = False
    topk_k: int = 3
    max_train_batches: int = 0
    max_eval_batches: int = 0


@dataclass
class ClassifierLossConfig:
    w_fg: float = 1.0
    w_bg: float = 1.0
    use_dynamic_weight: bool = True
    dyn_weight_eta: float = 1.0
    dyn_weight_min: float = 0.5
    dyn_weight_max: float = 5.0
    target_sen: float = 0.70
    target_spec: float = 0.80


@dataclass
class PatchSamplingConfig:
    bag_size: int = 16
    pos_per_bag: int = 6
    neg_per_bag: int = 10
    patch_jitter: int = 24
    fx_exclusion_radius: int = 224
    pos_bbox_overlap_thresh: float = 0.05
    use_fx_balanced_batches: bool = True
    fx_bags_per_batch: int = 2


@dataclass
class GanAugmentationConfig:
    checkpoint_path: str = ""
    gan_patch_size: int = 128
    mask_size: int = 84
    mask_size_jitter: int = 5
    use_fake: bool = True
    fake_start_epoch: int = 1
    fake_fraction: float = 0.30
    fake_fraction_end: Optional[float] = 0.15
    counterfactual_fraction: float = 0.10
    counterfactual_fraction_end: Optional[float] = 0.05
    alpha: float = 0.25          # L_cls = L_real + alpha * L_fake
    ema_m: float = 0.99          # EMA smoothing coefficient m
    ema_student_for_eval: bool = True
    active_g: bool = True
    g_update_start_epoch: int = 4
    g_lr: float = 1e-6
    g_update_interval: int = 10
    g_grad_clip: float = 0.05
    lambda_m: float = 0.01       # DSR mean-matching weight
    lambda_v: float = 0.01       # DSR variance-matching weight
    lambda_f: float = 0.001      # DSR variance-floor weight
    lambda_r: float = 0.10       # L_G = L_DSR + lambda_r * L_G^recon


@dataclass
class ClassifierConfig:
    model: ClassifierModelConfig = field(default_factory=ClassifierModelConfig)
    train: ClassifierTrainConfig = field(default_factory=ClassifierTrainConfig)
    loss: ClassifierLossConfig = field(default_factory=ClassifierLossConfig)
    sampling: PatchSamplingConfig = field(default_factory=PatchSamplingConfig)
    gan_augmentation: GanAugmentationConfig = field(default_factory=GanAugmentationConfig)


# --- project root config -----------------------------------------------------------

@dataclass
class ProjectConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    datasets: DatasetsConfig = field(default_factory=DatasetsConfig)
    lama: LamaArchitectureConfig = field(default_factory=LamaArchitectureConfig)
    gan_training: GanTrainingConfig = field(default_factory=GanTrainingConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)

    def step2_output_dir(self) -> Path:
        return self.paths.results_root / STEP2_DIR_NAME

    def step3_output_dir(self) -> Path:
        return self.paths.results_root / STEP3_DIR_NAME

    def gan_checkpoint_path(self) -> Path:
        configured = str(self.classifier.gan_augmentation.checkpoint_path or "")
        if configured:
            return _resolve(Path(configured))
        return self.step2_output_dir() / "checkpoints" / "G_latest.pt"


# --- loading ---------------------------------------------------------------------

def _resolve(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _update_dataclass(obj, values: dict, context: str = ""):
    for key, value in values.items():
        if not hasattr(obj, key):
            raise KeyError(f"unknown config key: {context}{key}")
        cur = getattr(obj, key)
        if hasattr(cur, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(cur, value, context=f"{context}{key}.")
        else:
            if value is not None and isinstance(cur, Path):
                value = Path(value)
            setattr(obj, key, value)
    return obj


def load_config(path: str | Path | None = None) -> ProjectConfig:
    import yaml

    cfg = ProjectConfig()
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    test_specs = data.get("datasets", {}).pop("tests", None)
    _update_dataclass(cfg, data)
    if test_specs is not None:
        cfg.datasets.tests = [
            _update_dataclass(DatasetSpec(), spec, context="datasets.tests[].")
            for spec in test_specs
        ]

    cfg.paths.results_root = _resolve(cfg.paths.results_root)
    cfg.paths.site_stats = _resolve(cfg.paths.site_stats)
    for spec in [cfg.datasets.train, cfg.datasets.valid, *cfg.datasets.tests]:
        spec.images_root = _resolve(spec.images_root)

    if not cfg.datasets.train.name or not cfg.datasets.valid.name:
        raise ValueError("config needs datasets.train and datasets.valid")
    if not cfg.datasets.tests:
        raise ValueError("config needs at least one entry under datasets.tests")
    return cfg
