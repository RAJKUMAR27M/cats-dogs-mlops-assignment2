"""Unit tests for data preprocessing functions (M1 – Task 1)."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocess import load_and_resize_image, normalize_image, preprocess_image


# ---------------------------------------------------------------------------
# load_and_resize_image
# ---------------------------------------------------------------------------
class TestLoadAndResizeImage:
    def test_resizes_to_224x224(self, tmp_path):
        """Image of any size must be resized to 224×224."""
        arr = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        path = str(tmp_path / "img.jpg")
        Image.fromarray(arr).save(path)

        result = load_and_resize_image(path, target_size=(224, 224))

        assert result.shape == (224, 224, 3)

    def test_output_is_rgb(self, tmp_path):
        """Output must have 3 channels (RGB)."""
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        path = str(tmp_path / "img.jpg")
        Image.fromarray(arr).save(path)

        result = load_and_resize_image(path)

        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_custom_target_size(self, tmp_path):
        """Arbitrary target size is respected."""
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        path = str(tmp_path / "img.jpg")
        Image.fromarray(arr).save(path)

        result = load_and_resize_image(path, target_size=(64, 64))

        assert result.shape == (64, 64, 3)

    def test_grayscale_converted_to_rgb(self, tmp_path):
        """Grayscale input is converted to RGB (3 channels)."""
        arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        path = str(tmp_path / "gray.jpg")
        Image.fromarray(arr, mode="L").save(path)

        result = load_and_resize_image(path, target_size=(64, 64))

        assert result.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# normalize_image
# ---------------------------------------------------------------------------
class TestNormalizeImage:
    def test_values_in_unit_range(self):
        """All pixel values must be in [0, 1] after normalization."""
        arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        result = normalize_image(arr)

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype_float32(self):
        """Output must be float32."""
        arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        result = normalize_image(arr)

        assert result.dtype == np.float32

    def test_known_values(self):
        """255 → 1.0 and 0 → 0.0."""
        arr = np.array([[[255, 0, 128]]], dtype=np.uint8)
        result = normalize_image(arr)

        np.testing.assert_almost_equal(result[0, 0, 0], 1.0, decimal=3)
        np.testing.assert_almost_equal(result[0, 0, 1], 0.0, decimal=3)

    def test_shape_preserved(self):
        """Normalization must not change the array shape."""
        arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        result = normalize_image(arr)

        assert result.shape == arr.shape


# ---------------------------------------------------------------------------
# preprocess_image  (end-to-end pipeline)
# ---------------------------------------------------------------------------
class TestPreprocessImage:
    def test_output_shape(self, tmp_path):
        """Full pipeline returns (224, 224, 3)."""
        arr = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        path = str(tmp_path / "img.jpg")
        Image.fromarray(arr).save(path)

        result = preprocess_image(path)

        assert result.shape == (224, 224, 3)

    def test_values_normalized(self, tmp_path):
        """Full pipeline returns values in [0, 1]."""
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        path = str(tmp_path / "img.jpg")
        Image.fromarray(arr).save(path)

        result = preprocess_image(path)

        assert result.min() >= 0.0
        assert result.max() <= 1.0
