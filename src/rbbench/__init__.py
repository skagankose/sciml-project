"""Four neural PDE surrogates on The Well `rayleigh_benard`, trained from scratch under fixed budgets."""

__version__ = "0.1.0"

# Bump when any metric definition or evaluation protocol changes; every evaluation
# file records it and aggregation refuses to mix versions.
METRIC_VERSION = "1"

MODEL_NAMES = ("cnext_unet", "tfno", "avit", "convlstm")
