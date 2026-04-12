# gunicorn configuration for SAM2.1 image backend
#
# post_fork hook — reset PyTorch CUDA state in each worker.
#
# Problem: --preload loads the Flask app in the gunicorn master process.
# Some import (transitively from label_studio_ml / SAM2 deps) initialises
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
    # worker.age: 1-indexed creation counter (first worker = 1, second = 2, …).
    # When CUDA_VISIBLE_DEVICES="0,1" and WORKERS=2:
    #   worker 1 → CUDA_VISIBLE_DEVICES="0"  → cuda:0 inside this process
    #   worker 2 → CUDA_VISIBLE_DEVICES="1"  → cuda:0 inside this process (= phys GPU 1)
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible:
            gpus = [g.strip() for g in visible.split(",") if g.strip()]
            if len(gpus) > 1:
                gpu_idx = (worker.age - 1) % len(gpus)
                os.environ["CUDA_VISIBLE_DEVICES"] = gpus[gpu_idx]
    except Exception:
        pass
