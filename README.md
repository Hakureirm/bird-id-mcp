# bird-id-mcp

Bird species identification MCP server. YOLO detection + ConvNeXt classification, outputs Top-5 species with confidence and Chinese names.

## Install & Run

```bash
# Run directly with uvx (auto-installs)
uvx bird-id-mcp

# Or install from git
pip install git+https://github.com/Hakureirm/bird-id-mcp.git
bird-id-mcp
```

Models are automatically downloaded from HuggingFace on first run (~50MB default).

## Model Selection

| Model | Size | Speed (x86 1T) | Accuracy |
|-------|------|-----------------|----------|
| **S1v2** (default) | 37MB | ~150ms | Good |
| ConvNeXt | 144MB | ~600ms | Best |

Default is S1v2 (fast + small). To use ConvNeXt:

```bash
BIRD_ID_CLS_MODEL=convnext uvx --from git+https://github.com/Hakureirm/bird-id-mcp.git bird-id-mcp
```

## Claude Desktop / Agent Config

```json
{
  "mcpServers": {
    "bird-id": {
      "command": "uvx",
      "args": ["bird-id-mcp"]
    }
  }
}
```

## Tools

### `identify_bird`
Identify bird species from an image file path.

```
Input:  {"image_path": "/path/to/bird.jpg", "topk": 5}
Output: {
  "detections": 1,
  "detection_confidence": 0.92,
  "bbox": {"x1": 100, "y1": 50, "x2": 400, "y2": 350},
  "results": [
    {"rank": 1, "species": "Little Egret", "species_cn": "白鹭", "confidence": 78.5},
    {"rank": 2, "species": "Snowy Egret", "species_cn": "雪鹭", "confidence": 12.3},
    ...
  ]
}
```

### `identify_bird_base64`
Same as above but accepts base64-encoded image data.

## Models

- **Detection**: YOLOv8 bird detector (12MB ONNX)
- **Classification**: S1v2 (37MB, default) or ConvNeXt-Tiny (144MB), 10,753 bird species
- **Taxonomy**: eBird species info — scientific name, family, order, description
- **Inference**: ONNX Runtime CPU only, no GPU required
