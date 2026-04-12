"""Download all SAM2.1 checkpoints at Docker build time.

Primary:  huggingface_hub.hf_hub_download (public repos, no token required)
Fallback: handled in Dockerfile via download_ckpts.sh when this script fails
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/data/models"))

MODELS: dict[str, str] = {
    "sam2.1_hiera_tiny.pt":      "facebook/sam2.1-hiera-tiny",
    "sam2.1_hiera_small.pt":     "facebook/sam2.1-hiera-small",
    "sam2.1_hiera_base_plus.pt": "facebook/sam2.1-hiera-base-plus",
    "sam2.1_hiera_large.pt":     "facebook/sam2.1-hiera-large",
}


def main() -> None:
    from huggingface_hub import hf_hub_download

    token: str | None = os.getenv("HF_TOKEN") or None
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    for filename, repo_id in MODELS.items():
        dest = MODEL_DIR / filename
        if dest.exists():
            print(f"[skip]     {filename} already exists ({dest.stat().st_size // 1_048_576} MB)")
            continue
        print(f"[download] {repo_id}/{filename} → {dest}", flush=True)
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(MODEL_DIR),
                token=token,
            )
            print(f"[ok]       {filename}")
        except Exception as exc:
            print(f"[error]    {filename}: {exc}", file=sys.stderr)
            failed.append(filename)

    if failed:
        print(f"\n[FAIL] {len(failed)} checkpoint(s) not downloaded: {failed}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[done] All SAM2.1 checkpoints in {MODEL_DIR}")


if __name__ == "__main__":
    main()
