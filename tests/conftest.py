"""Shared pytest fixtures."""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


class DummyTorchModel(torch.nn.Module):
    def __init__(self, score: float = 0.75):
        super().__init__()
        self.score = score

    def forward(self, tensor):
        batch_size = tensor.shape[0]
        return torch.full((batch_size,), self.score, dtype=torch.float32)


@pytest.fixture
def sample_image_bytes():
    """224x224 RGB image serialised as JPEG bytes."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def sample_image_array():
    """Random float32 image array (224, 224, 3) in [0, 1]."""
    return np.random.rand(224, 224, 3).astype(np.float32)


@pytest.fixture
def mock_model():
    """PyTorch model stub that returns a fixed dog-score."""
    return DummyTorchModel(score=0.75)


@pytest.fixture
def mock_predictor(mock_model):
    """Predictor instance backed by a model stub without disk I/O."""
    from api.predictor import Predictor

    pred = Predictor.__new__(Predictor)
    pred.model = mock_model
    pred.model_path = "mock_model.pt"
    return pred
