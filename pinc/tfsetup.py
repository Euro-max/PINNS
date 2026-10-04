"""Deterministic TensorFlow setup: seeds, op determinism, default float type."""
import os
import random

import numpy as np
import tensorflow as tf

_DONE = {}


def setup(seed: int = 0, dtype: str = "float64", threads: int | None = None):
    """Seed Python / NumPy / TF, enable op determinism, set Keras floatx."""
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    tf.config.experimental.enable_op_determinism()
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.keras.backend.set_floatx(dtype)
    if threads is not None:
        # only allowed before any op has run; ignore if already configured
        try:
            tf.config.threading.set_intra_op_parallelism_threads(threads)
            tf.config.threading.set_inter_op_parallelism_threads(threads)
        except RuntimeError:
            pass
    _DONE["seed"] = seed
    _DONE["dtype"] = dtype
    return tf.as_dtype(dtype)
