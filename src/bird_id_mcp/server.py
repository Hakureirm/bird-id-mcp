"""Bird ID MCP Server — YOLO detection + ConvNeXt classification."""
from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from bird_id_mcp.models import ensure_models
from bird_id_mcp.pipeline import BirdPipeline

mcp = FastMCP(
    "Bird Species Identifier",
    description="Identify bird species from photos using YOLO detection + ConvNeXt classification. Returns Top-5 species with confidence scores and Chinese names.",
)

_pipeline: BirdPipeline | None = None


def _get_pipeline() -> BirdPipeline:
    global _pipeline
    if _pipeline is None:
        import json as _json
        paths = ensure_models()
        labels = [l.strip() for l in open(paths["labels.txt"], encoding="utf-8")]
        labels_cn = []
        if paths["labels_cn.txt"].exists():
            for line in open(paths["labels_cn.txt"], encoding="utf-8"):
                parts = line.strip().split("\t")
                labels_cn.append(parts[1] if len(parts) > 1 else parts[0])
        taxonomy = {}
        if paths["taxonomy.json"].exists():
            tax_data = _json.load(open(paths["taxonomy.json"], encoding="utf-8"))
            taxonomy = tax_data.get("species", tax_data)
        _pipeline = BirdPipeline(
            yolo_path=paths["yolo.onnx"],
            cls_path=paths["convnext.onnx"],
            labels=labels,
            labels_cn=labels_cn,
            taxonomy=taxonomy,
            threads=1,
        )
    return _pipeline


@mcp.tool()
def identify_bird(image_path: str, topk: int = 5) -> str:
    """Identify bird species from an image file.

    Args:
        image_path: Absolute path to a bird photo (jpg/png).
        topk: Number of top predictions to return (default 5).

    Returns:
        JSON string with detection info and top-k species predictions,
        each with English name, Chinese name, and confidence percentage.
    """
    pipe = _get_pipeline()
    result = pipe.identify(image_path, topk)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def identify_bird_base64(image_base64: str, topk: int = 5) -> str:
    """Identify bird species from a base64-encoded image.

    Args:
        image_base64: Base64-encoded image data (jpg/png).
        topk: Number of top predictions to return (default 5).

    Returns:
        JSON string with detection info and top-k species predictions.
    """
    pipe = _get_pipeline()
    image_bytes = base64.b64decode(image_base64)
    result = pipe.identify_bytes(image_bytes, topk)
    return json.dumps(result, ensure_ascii=False, indent=2)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
