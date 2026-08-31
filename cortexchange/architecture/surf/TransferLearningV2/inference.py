"""
Checkpoint loading, delegated to astroNNomy.

A checkpoint pickles the model class itself, so astroNNomy is imported at unpickle time whatever
we do here - there is no way to load one without it. Rather than keep a second copy of the loader
that can drift from the one the checkpoints were written by, both TransferLearning versions call
astroNNomy's, and this module only adds the friendlier errors.
"""

DINOV3_DOWNLOADS = "https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/"

ASTRONNOMY_INSTALL = (
    "Install it with `pip install "
    "git+https://github.com/LOFAR-VLBI/astroNNomy.git#egg=astroNNomy`"
)


def _astronnomy_load_checkpoint():
    """
    astroNNomy's loader, or a message explaining which part of it is missing.

    migrate_legacy_keys is the marker for a new enough astroNNomy: checkpoints store their LoRA
    tensors under names an older loader silently discarded, leaving the backbone frozen. Its
    absence cannot be detected from a version number - the package reports 0.0.0 - and an old
    ImagenetTransferLearning swallows the extra argument through **kwargs rather than failing,
    so the import itself is the check.
    """
    try:
        from astronnomy.training.utils import load_checkpoint, migrate_legacy_keys  # noqa: F401
    except ModuleNotFoundError as error:
        if "astronnomy" in str(error):
            raise ImportError(
                f"The astronnomy module is required to load these checkpoints. {ASTRONNOMY_INSTALL}"
            ) from error
        raise
    except ImportError as error:
        raise ImportError(
            "The installed astronnomy is too old: it loads checkpoints without applying their "
            f"LoRA weights, which silently leaves the backbone frozen. {ASTRONNOMY_INSTALL}"
        ) from error

    return load_checkpoint


def load_checkpoint(ckpt_path, device="cuda"):
    """
    Load a checkpoint through astroNNomy, translating a missing DINOv3 backbone into guidance.
    """
    astronnomy_load_checkpoint = _astronnomy_load_checkpoint()
    try:
        return astronnomy_load_checkpoint(ckpt_path, device=device)
    except FileNotFoundError as error:
        raise ImportError(
            "This checkpoint uses a DINOv3 backbone, whose weights are access-gated by Meta and "
            "whose code has to be a local checkout. Point $DINOV3_REPO_DIR at a clone of "
            "https://github.com/facebookresearch/dinov3 and $DINOV3_WEIGHTS at the downloaded "
            f"weights (request access at {DINOV3_DOWNLOADS}).\nOriginal error: {error}"
        ) from error
