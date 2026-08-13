"""Unit tests for Predictor and API health endpoint (M2 / M3)."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Predictor unit tests
# ---------------------------------------------------------------------------
class TestPredictor:
    def test_returns_label(self, mock_predictor, sample_image_bytes):
        result = mock_predictor.predict_from_bytes(sample_image_bytes)

        assert "label" in result
        assert result["label"] in ("cat", "dog")

    def test_returns_confidence_in_range(self, mock_predictor, sample_image_bytes):
        result = mock_predictor.predict_from_bytes(sample_image_bytes)

        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_returns_raw_score_in_range(self, mock_predictor, sample_image_bytes):
        result = mock_predictor.predict_from_bytes(sample_image_bytes)

        assert 0.0 <= result["raw_score"] <= 1.0

    def test_high_score_predicts_dog(self, mock_predictor, sample_image_bytes):
        mock_predictor.model.score = 0.9
        result = mock_predictor.predict_from_bytes(sample_image_bytes)

        assert result["label"] == "dog"

    def test_low_score_predicts_cat(self, mock_predictor, sample_image_bytes):
        mock_predictor.model.score = 0.1
        result = mock_predictor.predict_from_bytes(sample_image_bytes)

        assert result["label"] == "cat"

    def test_is_loaded_true(self, mock_predictor):
        assert mock_predictor.is_loaded() is True

    def test_is_loaded_false_when_no_model(self):
        from api.predictor import Predictor

        pred = Predictor.__new__(Predictor)
        pred.model = None
        pred.model_path = "nonexistent.pt"

        assert pred.is_loaded() is False

    def test_preprocess_output_shape(self, mock_predictor):
        """Preprocessed output must be NCHW for PyTorch."""
        img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
        result = mock_predictor.preprocess(img)

        assert tuple(result.shape) == (1, 3, 224, 224)

    def test_preprocess_values_normalized(self, mock_predictor):
        """Preprocessed pixel values must stay in the expected normalized range."""
        img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)
        result = mock_predictor.preprocess(img)

        assert float(result.max()) <= 1.0
        assert float(result.min()) >= -1.0

    def test_predict_raises_without_model(self, sample_image_bytes):
        from api.predictor import Predictor

        pred = Predictor.__new__(Predictor)
        pred.model = None
        pred.model_path = "nonexistent.pt"

        with pytest.raises(RuntimeError, match="not loaded"):
            pred.predict_from_bytes(sample_image_bytes)


# ---------------------------------------------------------------------------
# Health endpoint smoke test (no real model needed)
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    def _client(self):
        """Create TestClient with MODEL_PATH pointing to a non-existent file."""
        os.environ["MODEL_PATH"] = "nonexistent_for_test.pt"
        from fastapi.testclient import TestClient
        from api.main import app

        return TestClient(app)

    def test_health_returns_200(self):
        client = self._client()
        with client:
            response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self):
        client = self._client()
        with client:
            response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_status_value(self):
        client = self._client()
        with client:
            response = client.get("/health")
        assert response.json()["status"] == "healthy"
