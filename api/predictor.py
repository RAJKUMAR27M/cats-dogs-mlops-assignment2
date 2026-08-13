"""Model inference predictor for Cats vs Dogs classification (PyTorch)."""

import io
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMG_SIZE = (224, 224)
CLASS_NAMES = {0: "cat", 1: "dog"}
_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


class Predictor:
    """Wrap a trained PyTorch model for single-image inference."""

    def __init__(self, model_path: str = "artifacts/model/model.pt"):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            logger.warning("Model file not found at %s - running without model", self.model_path)
            return

        try:
            import torch

            project_root = str(Path(__file__).resolve().parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from src.models.cnn_model import build_model

            self.model = build_model()
            self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            self.model.eval()
            logger.info("Model loaded from %s", self.model_path)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            self.model = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def preprocess(self, image: Image.Image):
        import torch

        img = image.convert("RGB").resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        arr = arr.transpose(2, 0, 1)
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

    def predict_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        import torch

        image = Image.open(io.BytesIO(image_bytes))
        tensor = self.preprocess(image)

        with torch.no_grad():
            raw_score = float(self.model(tensor).reshape(-1)[0])

        predicted_class = int(raw_score > 0.5)
        confidence = raw_score if predicted_class == 1 else 1.0 - raw_score

        return {
            "label": CLASS_NAMES[predicted_class],
            "confidence": confidence,
            "raw_score": raw_score,
            "class_id": predicted_class,
        }

    def predict_from_path(self, image_path: str) -> Dict[str, Any]:
        with open(image_path, "rb") as file_handle:
            return self.predict_from_bytes(file_handle.read())
