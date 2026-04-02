"""Model download and management."""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Hakureirm/bird-id-models"
MODEL_DIR = Path(os.environ.get("BIRD_ID_MODEL_DIR", Path.home() / ".cache" / "bird-id-mcp"))

FILES = {
    "yolo.onnx": "yolo_bird_detect.onnx",
    "convnext.onnx": "convnext_bird_cls.onnx",
    "labels.txt": "labels_10753.txt",
    "labels_cn.txt": "labels_cn_10753.txt",
    "taxonomy.json": "taxonomy.json",
}


def ensure_models() -> dict[str, Path]:
    """Download models from HuggingFace if not cached. Returns paths dict."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, filename in FILES.items():
        local = MODEL_DIR / filename
        if not local.exists():
            print(f"Downloading {filename}...")
            hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=str(MODEL_DIR))
        paths[key] = local
    return paths
