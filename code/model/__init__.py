from .ndviDropDetector import (
    NdviDropMaps,
    augment_cache_with_ndvi_drop,
    compute_ndvi_drop,
)
from .vegetationPredictor import (
    HeuristicForestMasker,
    LearnedForestMasker,
    MaskEvaluation,
    TIER_NON_FOREST,
    TIER_SOFT,
    TIER_STRONG,
    TIER_UNCERTAIN,
    evaluate_mask,
    evaluate_masker_cv,
    evaluate_masker_on_tiles,
    sweep_ndvi_threshold,
)
from .vegetationVis import plot_forest_mask_qa, plot_ndvi_drop

_TRAIN_EXPORTS = []
try:
    from .train import (
        DeforestationBaseModel,
        DANN_UNet,
        GradientReversal,
        build_model,
        random_spectral_scaling,
        weighted_bce_dice_loss,
        dann_lambda_schedule,
        train_one_epoch_baseline,
        train_one_epoch_dann,
        train_one_epoch,
        fit,
    )
    _TRAIN_EXPORTS = [
        "DeforestationBaseModel",
        "DANN_UNet",
        "GradientReversal",
        "build_model",
        "random_spectral_scaling",
        "weighted_bce_dice_loss",
        "dann_lambda_schedule",
        "train_one_epoch_baseline",
        "train_one_epoch_dann",
        "train_one_epoch",
        "fit",
    ]
except Exception:
    # Keep vegetation utilities importable even if training extras are missing.
    pass

__all__ = [
    "HeuristicForestMasker",
    "LearnedForestMasker",
    "MaskEvaluation",
    "NdviDropMaps",
    "augment_cache_with_ndvi_drop",
    "compute_ndvi_drop",
    "evaluate_mask",
    "evaluate_masker_cv",
    "evaluate_masker_on_tiles",
    "plot_forest_mask_qa",
    "plot_ndvi_drop",
    "sweep_ndvi_threshold",
    "TIER_NON_FOREST", "TIER_UNCERTAIN", "TIER_SOFT", "TIER_STRONG",
] + _TRAIN_EXPORTS
