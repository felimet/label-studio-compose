# gunicorn configuration for SAM3 video backend
#
# post_fork hook — reset PyTorch CUDA state in each worker.
# See ml-backends/sam3-image/gunicorn.conf.py for full explanation.


def post_fork(server, worker):
    """Reset PyTorch CUDA initialisation state after gunicorn forks a worker."""
    try:
        import torch.cuda as _cuda
        _cuda._initialized = False
        _cuda._in_bad_fork = False
    except Exception:
        pass
