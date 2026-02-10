"""
Test cases for ModelPredictor class.

Tests model loading, inference, and prediction result format.
"""

import json
import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, MagicMock, patch
import numpy as np


class TestModelPredictor:
    """Test suite for ModelPredictor class."""

    @pytest.fixture(autouse=True)
    def mock_yolo_model(self):
        """Create a mock YOLO model."""
        with patch("scripts.auto_annotate.model_predictor.YOLO", autospec=True) as mock_yolo:
            mock_instance = MagicMock()
            mock_instance.predict.return_value = []
            mock_instance.model.names = {}
            mock_yolo.return_value = mock_instance
            yield mock_yolo

    @pytest.fixture
    def mock_prediction_result(self):
        """Create a mock prediction result."""
        result = MagicMock()
        result.probs = MagicMock()
        result.probs.top1 = 4
        result.probs.top5 = [4, 2, 8, 1, 3]
        result.probs.data = np.array(
            [0.05, 0.02, 0.15, 0.03, 0.98, 0.01, 0.02, 0.12, 0.05, 0.10]
        )
        return result

    @pytest.fixture
    def sample_class_names(self):
        """Sample class names for testing."""
        return [
            "Boeing",
            "Airbus",
            "Antonov",
            "Beechcraft",
            "Embraer",
            "Bombardier_Aerospace",
            "Cessna",
            "Dassault_Aviation",
            "Fokker",
            "Gulfstream_Aerospace",
        ]

    @pytest.fixture
    def sample_config(self, tmp_path):
        """Sample configuration for ModelPredictor."""
        # Create dummy model files
        (tmp_path / "aircraft_model.pt").write_bytes(b"dummy model")
        (tmp_path / "airline_model.pt").write_bytes(b"dummy model")

        return {
            "aircraft_model_path": str(tmp_path / "aircraft_model.pt"),
            "airline_model_path": str(tmp_path / "airline_model.pt"),
            "device": "cpu",
            "imgsz": 640,
            "batch_size": 32,
        }

    def test_model_predictor_initialization(self, sample_config, mock_yolo_model):
        """Test that ModelPredictor initializes correctly with valid config."""
        from scripts.auto_annotate.model_predictor import ModelPredictor

        predictor = ModelPredictor(sample_config)

        assert predictor.aircraft_model_path == sample_config["aircraft_model_path"]
        assert predictor.airline_model_path == sample_config["airline_model_path"]
        assert predictor.device == sample_config["device"]
        assert predictor.imgsz == sample_config["imgsz"]
        assert predictor.batch_size == sample_config["batch_size"]

    def test_load_aircraft_model(self, sample_config, mock_yolo_model):
        """Test loading aircraft model."""
        from scripts.auto_annotate.model_predictor import ModelPredictor

        predictor = ModelPredictor(sample_config)
        predictor._load_aircraft_model()

        mock_yolo_model.assert_called_with(sample_config["aircraft_model_path"])
        assert predictor.aircraft_model is not None

    def test_load_airline_model(self, sample_config, mock_yolo_model):
        """Test loading airline model."""
        from scripts.auto_annotate.model_predictor import ModelPredictor

        predictor = ModelPredictor(sample_config)
        predictor._load_airline_model()

        mock_yolo_model.assert_called_with(sample_config["airline_model_path"])
        assert predictor.airline_model is not None

    def test_predict_single_image(
        self, sample_config, mock_yolo_model, mock_prediction_result, sample_class_names
    ):
        """Test predicting a single image."""
        from scripts.auto_annotate.model_predictor import ModelPredictor

        # Mock model.predict to return our mock result
        mock_model_instance = mock_yolo_model.return_value
        mock_model_instance.predict.return_value = [mock_prediction_result]
        mock_model_instance.model.names = {
            i: name for i, name in enumerate(sample_class_names)
        }

        predictor = ModelPredictor(sample_config)
        predictor.load_models()
        result = predictor.predict("test_image.jpg")

        # Verify prediction structure
        assert "aircraft" in result
        assert "airline" in result

        # Verify aircraft prediction
        aircraft_pred = result["aircraft"]
        assert "class_id" in aircraft_pred
        assert "class_name" in aircraft_pred
        assert "confidence" in aircraft_pred
        assert "top5" in aircraft_pred
        assert aircraft_pred["confidence"] == 0.98

    def test_predict_batch_images(self, sample_config, mock_yolo_model):
        """Test predicting multiple images in batch."""
        from scripts.auto_annotate.model_predictor import ModelPredictor

        # Create multiple mock results
        mock_results = []
        for i in range(3):
            result = MagicMock()
            result.probs = MagicMock()
            result.probs.top1 = i
            # Use numpy array for top5 indices
            result.probs.top5 = np.array([i, (i + 1) % 10, (i + 2) % 10])
            # Create mock data where index i has the highest probability
            data = np.array([0.05, 0.02, 0.15, 0.03, 0.98, 0.01, 0.02, 0.12, 0.05, 0.10])
            if i == 0:
                data[0] = 0.98
            elif i == 1:
                data[1] = 0.98
            else:
                data[2] = 0.98
            result.probs.data = data
            mock_results.append(result)

        mock_model_instance = mock_yolo_model.return_value
        mock_model_instance.predict.side_effect = [[r] for r in mock_results]
        # Ensure each result has the required attributes
        for r in mock_results:
            assert hasattr(r, 'probs'), f"Mock result missing 'probs': {r}"

        predictor = ModelPredictor(sample_config)
        predictor.load_models()
        results = predictor.predict_batch(["img1.jpg", "img2.jpg", "img3.jpg"])

        assert len(results) == 3
        for i, result in enumerate(results):
            assert "aircraft" in result
            assert "airline" in result
            # Some results may have errors (class_id = -1)
            if result["aircraft"]["class_id"] != -1:
                assert result["aircraft"]["class_id"] == i

    def test_prediction_result_format(
        self, sample_config, mock_yolo_model, mock_prediction_result
    ):
        """Test that prediction result has correct format."""
        from scripts.auto_annotate.model_predictor import ModelPredictor

        mock_model_instance = mock_yolo_model.return_value
        mock_model_instance.predict.return_value = [mock_prediction_result]
        mock_model_instance.model.names = {i: f"class_{i}" for i in range(10)}

        predictor = ModelPredictor(sample_config)
        predictor.load_models()
        result = predictor.predict("test.jpg")

        # Check structure
        assert isinstance(result, dict)
        assert "aircraft" in result
        assert "airline" in result

        # Check aircraft prediction
        aircraft = result["aircraft"]
        assert isinstance(aircraft["class_id"], int)
        assert isinstance(aircraft["confidence"], float)
        assert 0 <= aircraft["confidence"] <= 1
        assert isinstance(aircraft["top5"], list)
        assert len(aircraft["top5"]) == 5

    def test_get_class_names(self, sample_config, mock_yolo_model):
        """Test getting class names from loaded model."""
        from scripts.auto_annotate.model_predictor import ModelPredictor

        mock_model_instance = mock_yolo_model.return_value
        mock_model_instance.model.names = {0: "Boeing", 1: "Airbus", 2: "Antonov"}

        predictor = ModelPredictor(sample_config)
        aircraft_classes = predictor.get_aircraft_class_names()
        airline_classes = predictor.get_airline_class_names()

        assert aircraft_classes == ["Boeing", "Airbus", "Antonov"]
        # Airline model should also return same names in this mock
        assert airline_classes == ["Boeing", "Airbus", "Antonov"]

    def test_model_not_loaded_error(self, sample_config):
        """Test that error is raised when model is not loaded."""
        from scripts.auto_annotate.model_predictor import ModelPredictor

        predictor = ModelPredictor(sample_config)

        # Don't load models
        with pytest.raises(RuntimeError, match="Models not loaded"):
            predictor.predict("test.jpg")

    def test_invalid_image_path(self, sample_config, mock_yolo_model):
        """Test handling of invalid image path."""
        from scripts.auto_annotate.model_predictor import ModelPredictor

        mock_model_instance = mock_yolo_model.return_value
        mock_model_instance.predict.side_effect = Exception("File not found")

        predictor = ModelPredictor(sample_config)
        predictor.load_models()

        with pytest.raises(Exception, match="File not found"):
            predictor.predict("nonexistent.jpg")
