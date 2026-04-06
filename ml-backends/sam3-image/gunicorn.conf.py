# gunicorn configuration for SAM3 image backend
#
# post_fork hook — reset PyTorch CUDA state in each worker.
#
# Problem: --preload loads the Flask app in the gunicorn master process.
# Some import (transitively from label_studio_ml / SAM3 deps) initialises
# CUDA in the master.  PyTorch then sets _in_bad_fork=True in every forked
# worker, which causes RuntimeError on the first CUDA operation:
#
#   RuntimeError: Cannot re-initialize CUDA in forked subprocess.
#   To use CUDA with multiprocessing, you must use the 'spawn' start method
#
# Fix: after fork, reset PyTorch's CUDA tracking flags so the worker can
# initialise a fresh CUDA context.  This is safe because CUDA contexts are
# per-process; the master's context is not transferred to workers via fork —
# workers must initialise their own CUDA context regardless.


def post_fork(server, worker):
    """Reset PyTorch CUDA initialisation state after gunicorn forks a worker."""
    try:
        import torch.cuda as _cuda
        _cuda._initialized = False
        _cuda._in_bad_fork = False
    except Exception:
        pass
