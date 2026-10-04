"""PINC-MPC: physics-informed neural surrogate for MPC of a single-track vehicle."""
import os as _os

# Must be set before TensorFlow is imported anywhere: oneDNN reorders
# floating-point reductions and breaks bit-reproducibility.
_os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
