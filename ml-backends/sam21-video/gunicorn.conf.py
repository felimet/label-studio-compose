# gunicorn configuration for SAM2.1 video backend
#
# post_fork hook — reset PyTorch CUDA state in each worker.
# Same rationale as sam21-image: CUDA must be initialised after fork.


def post_fork(server, worker):
    """Reset PyTorch CUDA state and assign one GPU per worker (multi-GPU setups)."""
    import os

    # ── 1. Reset CUDA fork state ─────────────────────────────────────────────
    try:
        import torch.cuda as _cuda
        _cuda._initialized = False
        _cuda._in_bad_fork = False
    except Exception:
        pass

    # ── 2. Per-worker GPU pinning ────────────────────────────────────────────
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible:
            gpus = [g.strip() for g in visible.split(",") if g.strip()]
            if len(gpus) > 1:
                gpu_idx = (worker.age - 1) % len(gpus)
                os.environ["CUDA_VISIBLE_DEVICES"] = gpus[gpu_idx]
    except Exception:
        pass
