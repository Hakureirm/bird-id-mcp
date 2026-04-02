"""YOLO detection + ConvNeXt classification pipeline (ONNX Runtime, CPU only)."""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path


class BirdPipeline:
    """YOLO detect → crop → ConvNeXt classify → Top-K results."""

    def __init__(self, yolo_path: str | Path, cls_path: str | Path,
                 labels: list[str], labels_cn: list[str] | None = None,
                 threads: int = 1):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.det = ort.InferenceSession(str(yolo_path), opts, providers=["CPUExecutionProvider"])
        self.cls = ort.InferenceSession(str(cls_path), opts, providers=["CPUExecutionProvider"])
        self.labels = labels
        self.labels_cn = labels_cn or [""] * len(labels)

        self._det_input = self.det.get_inputs()[0].name
        self._cls_input = self.cls.get_inputs()[0].name
        self._det_imgsz = 640
        self._cls_imgsz = 224

        # ImageNet normalization
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def _preprocess_det(self, img_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Letterbox resize to 640x640 for YOLO."""
        h, w = img_bgr.shape[:2]
        scale = min(self._det_imgsz / w, self._det_imgsz / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img_bgr, (nw, nh))

        canvas = np.full((self._det_imgsz, self._det_imgsz, 3), 114, dtype=np.uint8)
        dx, dy = (self._det_imgsz - nw) // 2, (self._det_imgsz - nh) // 2
        canvas[dy:dy+nh, dx:dx+nw] = resized

        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob[np.newaxis], scale, dx, dy

    def _detect(self, img_bgr: np.ndarray, conf_thr: float = 0.3, iou_thr: float = 0.5
                ) -> list[tuple[int, int, int, int, float]]:
        """Run YOLO detection, return list of (x1,y1,x2,y2,conf) in original coords."""
        blob, scale, dx, dy = self._preprocess_det(img_bgr)
        outputs = self.det.run(None, {self._det_input: blob})[0]

        # YOLOv8 output: [1, 5, N] → transpose to [N, 5] (x,y,w,h,conf)
        if outputs.ndim == 3:
            outputs = outputs[0].T  # [N, 5]

        results = []
        for det in outputs:
            if len(det) < 5:
                continue
            cx, cy, bw, bh = det[:4]
            conf = det[4] if len(det) == 5 else det[4:].max()
            if conf < conf_thr:
                continue
            x1 = (cx - bw / 2 - dx) / scale
            y1 = (cy - bh / 2 - dy) / scale
            x2 = (cx + bw / 2 - dx) / scale
            y2 = (cy + bh / 2 - dy) / scale
            results.append((int(x1), int(y1), int(x2), int(y2), float(conf)))

        # NMS
        if not results:
            return []
        results.sort(key=lambda x: -x[4])
        keep = []
        while results:
            best = results.pop(0)
            keep.append(best)
            results = [r for r in results if self._iou(best, r) < iou_thr]
        return keep

    @staticmethod
    def _iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def _crop_bird(self, img_bgr: np.ndarray, bbox: tuple, pad: float = 0.15) -> np.ndarray:
        """Crop detected bird with padding."""
        x1, y1, x2, y2, _ = bbox
        h, w = img_bgr.shape[:2]
        pw, ph = int((x2-x1) * pad), int((y2-y1) * pad)
        x1, y1 = max(0, x1-pw), max(0, y1-ph)
        x2, y2 = min(w, x2+pw), min(h, y2+ph)
        return img_bgr[y1:y2, x1:x2]

    def _classify(self, crop_bgr: np.ndarray, topk: int = 5) -> list[dict]:
        """Classify cropped bird image, return top-k results."""
        img = cv2.resize(crop_bgr, (self._cls_imgsz, self._cls_imgsz))
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        img = (img - self._mean) / self._std

        logits = self.cls.run(None, {self._cls_input: img[np.newaxis]})[0][0]

        # Stable softmax
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        top_idx = probs.argsort()[::-1][:topk]

        return [
            {
                "rank": i + 1,
                "species": self.labels[idx],
                "species_cn": self.labels_cn[idx] if idx < len(self.labels_cn) else "",
                "confidence": round(float(probs[idx]) * 100, 2),
            }
            for i, idx in enumerate(top_idx)
        ]

    def identify(self, image_path: str, topk: int = 5) -> dict:
        """Full pipeline: load image → detect → crop → classify → top-k."""
        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"Cannot read image: {image_path}", "detections": 0, "results": []}

        detections = self._detect(img)
        if not detections:
            return {"error": None, "detections": 0, "results": [],
                    "message": "No bird detected in image"}

        # Use highest confidence detection
        best = detections[0]
        crop = self._crop_bird(img, best)
        if crop.size == 0:
            return {"error": "Crop failed", "detections": len(detections), "results": []}

        results = self._classify(crop, topk)
        return {
            "error": None,
            "detections": len(detections),
            "bbox": {"x1": best[0], "y1": best[1], "x2": best[2], "y2": best[3]},
            "detection_confidence": round(best[4], 3),
            "results": results,
        }

    def identify_bytes(self, image_bytes: bytes, topk: int = 5) -> dict:
        """Identify from raw image bytes."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Cannot decode image bytes", "detections": 0, "results": []}

        # Same as identify but with already loaded image
        detections = self._detect(img)
        if not detections:
            return {"error": None, "detections": 0, "results": [],
                    "message": "No bird detected in image"}

        best = detections[0]
        crop = self._crop_bird(img, best)
        if crop.size == 0:
            return {"error": "Crop failed", "detections": len(detections), "results": []}

        results = self._classify(crop, topk)
        return {
            "error": None,
            "detections": len(detections),
            "bbox": {"x1": best[0], "y1": best[1], "x2": best[2], "y2": best[3]},
            "detection_confidence": round(best[4], 3),
            "results": results,
        }
