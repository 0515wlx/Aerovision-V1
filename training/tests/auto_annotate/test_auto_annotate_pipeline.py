"""
Test cases for AutoAnnotatePipeline class.

Tests the complete auto-annotation pipeline integration.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock, call
import json
import tempfile


class TestAutoAnnotatePipeline:
    """Test suite for AutoAnnotatePipeline class."""

    @pytest.fixture
    def sample_config(self, tmp_path):
        """Sample configuration for AutoAnnotatePipeline."""
        # Create dummy model files
        (tmp_path / "aircraft_model.pt").write_bytes(b"dummy model")
        (tmp_path / "airline_model.pt").write_bytes(b"dummy model")

        return {
            "raw_images_dir": str(tmp_path / "images"),
            "labeled_dir": str(tmp_path / "labeled"),
            "filtered_new_class_dir": str(tmp_path / "filtered_new_class"),
            "filtered_95_dir": str(tmp_path / "filtered_95"),
            "aircraft_model_path": str(tmp_path / "aircraft_model.pt"),
            "airline_model_path": str(tmp_path / "airline_model.pt"),
            "high_confidence_threshold": 0.95,
            "hdbscan": {"min_cluster_size": 5, "min_samples": 3, "metric": "euclidean"},
            "device": "cpu",
            "batch_size": 32,
        }

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories for testing."""
        raw_dir = tmp_path / "images"
        raw_dir.mkdir()
        labeled_dir = tmp_path / "labeled"
        labeled_dir.mkdir()
        new_class_dir = tmp_path / "filtered_new_class"
        new_class_dir.mkdir()
        filtered_95_dir = tmp_path / "filtered_95"
        filtered_95_dir.mkdir()

        # Create test images
        for i in range(5):
            (raw_dir / f"img_{i:03d}.jpg").write_text(f"test image {i}")

        return {
            "raw": str(raw_dir),
            "labeled": str(labeled_dir),
            "new_class": str(new_class_dir),
            "filtered_95": str(filtered_95_dir),
        }

    @pytest.fixture
    def sample_predictions(self):
        """Sample prediction results for testing."""
        return [
            {
                "filename": "img_000.jpg",
                "aircraft": {"class_id": 0, "class_name": "Boeing", "confidence": 0.98},
                "airline": {
                    "class_id": 2,
                    "class_name": "China Eastern",
                    "confidence": 0.96,
                },
                "embeddings": np.array([0.1, 0.2, 0.3, 0.4]),
            },
            {
                "filename": "img_001.jpg",
                "aircraft": {"class_id": 1, "class_name": "Airbus", "confidence": 0.97},
                "airline": {
                    "class_id": 3,
                    "class_name": "Air China",
                    "confidence": 0.96,
                },
                "embeddings": np.array([0.1, 0.2, 0.3, 0.4]),
            },
            {
                "filename": "img_002.jpg",
                "aircraft": {
                    "class_id": 2,
                    "class_name": "Antonov",
                    "confidence": 0.92,
                },
                "airline": {
                    "class_id": 1,
                    "class_name": "United Airlines",
                    "confidence": 0.88,
                },
                "embeddings": np.array([0.5, 0.6, 0.7, 0.8]),
            },
            {
                "filename": "img_003.jpg",
                "aircraft": {
                    "class_id": 5,
                    "class_name": "Unknown",
                    "confidence": 0.65,
                },
                "airline": {"class_id": 0, "class_name": "Unknown", "confidence": 0.55},
                "embeddings": np.array([0.9, 1.0, 0.1, 0.2]),
            },
            {
                "filename": "img_004.jpg",
                "aircraft": {
                    "class_id": 3,
                    "class_name": "Embraer",
                    "confidence": 0.78,
                },
                "airline": {"class_id": 4, "class_name": "Delta", "confidence": 0.72},
                "embeddings": np.array([0.9, 1.0, 0.1, 0.2]),
            },
        ]

    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings for testing."""
        return np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 0.1, 0.2],
                [0.9, 1.0, 0.1, 0.2],
            ]
        )

    def test_pipeline_initialization(self, sample_config):
        """Test that AutoAnnotatePipeline initializes correctly."""
        from scripts.auto_annotate.pipeline import AutoAnnotatePipeline

        pipeline = AutoAnnotatePipeline(sample_config)

        assert pipeline.config == sample_config
        assert pipeline.raw_images_dir == Path(sample_config["raw_images_dir"])

    def test_pipeline_load_models(self, sample_config):
        """Test loading models in pipeline."""
        from scripts.auto_annotate.pipeline import AutoAnnotatePipeline

        with patch("scripts.auto_annotate.model_predictor.YOLO") as mock_yolo:
            pipeline = AutoAnnotatePipeline(sample_config)
            pipeline.load_models()

            # Verify models were loaded
            assert pipeline.model_predictor.aircraft_model is not None
            assert pipeline.model_predictor.airline_model is not None

    def test_pipeline_collect_image_files(self, sample_config, temp_dirs):
        """Test collecting image files."""
        from scripts.auto_annotate.pipeline import AutoAnnotatePipeline

        config = sample_config.copy()
        config["raw_images_dir"] = temp_dirs["raw"]

        pipeline = AutoAnnotatePipeline(config)
        image_files = pipeline._collect_image_files()

        assert len(image_files) == 5
        assert all(file.endswith(".jpg") for file in image_files)

    def test_pipeline_predict_batch(self, sample_config, temp_dirs, sample_predictions):
        """Test batch prediction in pipeline."""
        from scripts.auto_annotate.pipeline import AutoAnnotatePipeline

        config = sample_config.copy()
        config["raw_images_dir"] = temp_dirs["raw"]

        with patch("scripts.auto_annotate.model_predictor.YOLO"):
            pipeline = AutoAnnotatePipeline(config)
            pipeline.load_models()

            # Mock predict_batch to return sample predictions
            with patch.object(
                pipeline.model_predictor,
                "predict_batch",
                return_value=sample_predictions,
            ):
                predictions = pipeline.predict_batch(
                    [f"img_{i:03d}.jpg" for i in range(5)]
                )

                assert len(predictions) == 5
                assert predictions[0]["filename"] == "img_000.jpg"

    def test_pipeline_detect_new_classes(
        self, sample_config, sample_predictions, sample_embeddings
    ):
        """Test new class detection in pipeline."""
        from scripts.auto_annotate.pipeline import AutoAnnotatePipeline

        pipeline = AutoAnnotatePipeline(sample_config)

        # Set predictions and embeddings before calling _detect_new_classes
        pipeline._predictions = sample_predictions
        pipeline._embeddings = sample_embeddings

        with patch("hdbscan.HDBSCAN") as mock_hdbscan:
            mock_clusterer = MagicMock()
            mock_clusterer.labels_ = np.array([0, 0, 0, -1, -1])
            mock_clusterer.outlier_scores_ = np.array([0.1, 0.1, 0.1, 0.9, 0.85])
            mock_hdbscan.return_value = mock_clusterer

            new_class_indices = pipeline._detect_new_classes()

            assert len(new_class_indices) == 2
            assert 3 in new_class_indices
            assert 4 in new_class_indices

    def test_pipeline_filter_by_confidence(self, sample_config, sample_predictions):
        """Test confidence filtering in pipeline."""
        from scripts.auto_annotate.pipeline import AutoAnnotatePipeline

        pipeline = AutoAnnotatePipeline(sample_config)
        filtered_result = pipeline._filter_by_confidence(sample_predictions)

        assert "high_confidence" in filtered_result
        assert "medium_confidence" in filtered_result
        assert "low_confidence" in filtered_result

        # High confidence: >= 0.95
        assert len(filtered_result["high_confidence"]) == 2
        # Medium confidence: 0.80 <= conf < 0.95
        assert len(filtered_result["medium_confidence"]) == 1
        # Low confidence: < 0.80
        assert len(filtered_result["low_confidence"]) == 2

    def test_pipeline_run_complete(
        self, sample_config, temp_dirs, sample_predictions, sample_embeddings
    ):
        """Test running complete pipeline."""
        from scripts.auto_annotate.pipeline import AutoAnnotatePipeline

        config = sample_config.copy()
        config["raw_images_dir"] = temp_dirs["raw"]
        config["labeled_dir"] = temp_dirs["labeled"]
        config["filtered_new_class_dir"] = temp_dirs["new_class"]
        config["filtered_95_dir"] = temp_dirs["filtered_95"]

        with (
            patch("scripts.auto_annotate.model_predictor.YOLO"),
            patch("hdbscan.HDBSCAN") as mock_hdbscan,
        ):
            # Setup HDBSCAN mock
            mock_clusterer = MagicMock()
            mock_clusterer.labels_ = np.array([0, 0, 0, -1, -1])
            mock_clusterer.outlier_scores_ = np.array([0.1, 0.1, 0.1, 0.9, 0.85])
            mock_hdbscan.return_value = mock_clusterer

            pipeline = AutoAnnotatePipeline(config)
            pipeline.load_models()

            # Mock predictions
            with patch.object(
                pipeline.model_predictor,
                "predict_batch",
                return_value=sample_predictions,
            ):
                result = pipeline.run()

                # Verify result structure
                assert "statistics" in result
                assert "new_class_count" in result["statistics"]
                assert "high_confidence_count" in result["statistics"]
                assert "filtered_95_count" in result["statistics"]

    def test_pipeline_saves_prediction_details(
        self, sample_config, temp_dirs, sample_predictions
    ):
        """Test that pipeline saves prediction details."""
        from scripts.auto_annotate.pipeline import AutoAnnotatePipeline

        config = sample_config.copy()
        config["raw_images_dir"] = temp_dirs["raw"]
        config["filtered_new_class_dir"] = temp_dirs["new_class"]
        config["filtered_95_dir"] = temp_dirs["filtered_95"]

        pipeline = AutoAnnotatePipeline(config)
        pipeline._save_prediction_details(sample_predictions, [], "new_class")

        # Check that details were saved
        details_file = Path(temp_dirs["new_class"]) / "prediction_details.json"
        assert details_file.exists()

        with open(details_file, "r") as f:
            data = json.load(f)
            assert "predictions" in data
            assert len(data["predictions"]) == 5

    def test_pipeline_get_statistics(self, sample_config):
        """Test getting pipeline statistics."""
        from scripts.auto_annotate.pipeline import AutoAnnotatePipeline

        pipeline = AutoAnnotatePipeline(sample_config)

        stats = pipeline._calculate_statistics(
            high_conf_count=10, medium_conf_count=5, low_conf_count=3, new_class_count=2
        )

        assert stats["total"] == 18
        assert stats["high_confidence_count"] == 10
        assert stats["medium_confidence_count"] == 5
        assert stats["low_confidence_count"] == 3
        assert stats["new_class_count"] == 2
        assert stats["filtered_95_count"] == 8  # medium + low

    def test_pipeline_integration(self, sample_config, temp_dirs):
        """Test full pipeline integration."""
        from scripts.auto_annotate.pipeline import AutoAnnotatePipeline

        config = sample_config.copy()
        config["raw_images_dir"] = temp_dirs["raw"]
        config["labeled_dir"] = temp_dirs["labeled"]
        config["filtered_new_class_dir"] = temp_dirs["new_class"]
        config["filtered_95_dir"] = temp_dirs["filtered_95"]

        with (
            patch("scripts.auto_annotate.model_predictor.YOLO"),
            patch("hdbscan.HDBSCAN"),
        ):
            pipeline = AutoAnnotatePipeline(config)
            pipeline.load_models()

            # Mock predictions
            mock_predictions = [
                {
                    "filename": f"img_{i:03d}.jpg",
                    "aircraft": {
                        "class_id": 0,
                        "class_name": "Boeing",
                        "confidence": 0.98,
                    },
                    "airline": {
                        "class_id": 2,
                        "class_name": "China Eastern",
                        "confidence": 0.96,
                    },
                    "embeddings": np.array([0.1, 0.2, 0.3, 0.4]),
                }
                for i in range(5)
            ]

            with patch.object(
                pipeline.model_predictor, "predict_batch", return_value=mock_predictions
            ):
                result = pipeline.run()

                # Check that files were organized
                labeled_path = Path(temp_dirs["labeled"])
                boeing_dir = labeled_path / "Boeing"
                assert boeing_dir.exists()

                # Check that details were saved
                details_file = (
                    Path(temp_dirs["new_class"]) / "prediction_details.json"
                )
                assert details_file.exists()
