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

Models are automatically downloaded from HuggingFace on first run (~150MB).

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
- **Classification**: ConvNeXt-Tiny fine-tuned on 10,753 bird species (138MB ONNX)
- **Inference**: ONNX Runtime CPU, single-thread ~300ms on x86
