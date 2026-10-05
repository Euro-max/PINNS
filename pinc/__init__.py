"""PINC-MPC: physics-informed neural surrogate for MPC of a single-track vehicle."""
import os as _os

# Must be set before TensorFlow is imported anywhere: oneDNN reorders
# floating-point reductions and breaks bit-reproducibility.
_os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Allocate GPU memory on demand: the default up-front reservation of almost
# the whole device fails repeatedly under WSL2.
_os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")


def _preload_cusolver():
    """TensorFlow 2.21's pip build does not search the nvidia-cusolver wheel's lib/ folder, so
    the GPU is silently skipped ("Cannot dlopen some GPU libraries").  Loading the library
    from the installed wheel before TensorFlow lets TF's own dlopen find it."""
    import ctypes
    import glob
    import site
    for sp in site.getsitepackages():
        for lib in sorted(glob.glob(_os.path.join(sp, "nvidia", "cusolver", "lib", "libcusolver.so.*"))):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_cusolver()
