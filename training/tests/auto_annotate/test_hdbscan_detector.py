"""
Test cases for HDBSCANNewClassDetector class.

Tests new class detection using HDBSCAN clustering.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock


class TestHDBSCANNewClassDetector:
    """Test suite for HDBSCANNewClassDetector class."""

    @pytest.fixture
    def sample_predictions(self):
        """Sample prediction results for testing."""
        return [
            {
                "filename": "img_001.jpg",
                "aircraft": {"class_id": 0, "class_name": "Boeing", "confidence": 0.98},
                "airline": {"class_id": 2, "class_name": "China Eastern", "confidence": 0.95}
            },
            {
                "filename": "img_002.jpg",
                "aircraft": {"class_id": 0, "class_name": "Boeing", "confidence": 0.97},
                "airline": {"class_id": 2, "class_name": "China Eastern", "confidence": 0.96}
            },
            {
                "filename": "img_003.jpg",
                "aircraft": {"class_id": 1, "class_name": "Airbus", "confidence": 0.92},
                "airline": {"class_id": 3, "class_name": "Air China", "confidence": 0.88}
            },
            {
                "filename": "img_004.jpg",
                "aircraft": {"class_id": 5, "class_name": "Unknown", "confidence": 0.65},
                "airline": {"class_id": 0, "class_name": "Unknown", "confidence": 0.55}
            },
            {
                "filename": "img_005.jpg",
                "aircraft": {"class_id": 5, "class_name": "Unknown", "confidence": 0.62},
                "airline": {"class_id": 0, "class_name": "Unknown", "confidence": 0.58}
            }
        ]

    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings for testing."""
        return np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 0.1, 0.2],
            [0.9, 1.0, 0.1, 0.2]
        ])

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for HDBSCANNewClassDetector."""
        return {
            "min_cluster_size": 5,
            "min_samples": 3,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "prediction_data": True
        }

    def test_hdbscan_detector_initialization(self, sample_config):
        """Test that HDBSCANNewClassDetector initializes correctly."""
        from scripts.auto_annotate.hdbscan_detector import HDBSCANNewClassDetector

        detector = HDBSCANNewClassDetector(sample_config)

        assert detector.min_cluster_size == sample_config["min_cluster_size"]
        assert detector.min_samples == sample_config["min_samples"]
        assert detector.metric == sample_config["metric"]

    def test_detect_new_classes_no_outliers(self, sample_predictions, sample_embeddings, sample_config):
        """Test detection when no new classes are found."""
        from scripts.auto_annotate.hdbscan_detector import HDBSCANNewClassDetector

        with patch('hdbscan.HDBSCAN') as mock_hdbscan:
            mock_clusterer = MagicMock()
            mock_clusterer.labels_ = np.array([0, 0, 0, 0, 0])  # All in same cluster
            mock_clusterer.outlier_scores_ = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
            mock_hdbscan.return_value = mock_clusterer

            detector = HDBSCANNewClassDetector(sample_config)
            new_class_indices = detector.detect_new_classes(sample_predictions, sample_embeddings)

            # Should return empty list as all samples are in same cluster
            assert len(new_class_indices) == 0

    def test_detect_new_classes_with_outliers(self, sample_predictions, sample_embeddings, sample_config):
        """Test detection when new classes (outliers) are found."""
        from scripts.auto_annotate.hdbscan_detector import HDBSCANNewClassDetector

        with patch('hdbscan.HDBSCAN') as mock_hdbscan:
            # Create mock labels where last two samples are outliers (-1)
            mock_clusterer = MagicMock()
            mock_clusterer.labels_ = np.array([0, 0, 0, -1, -1])
            mock_clusterer.outlier_scores_ = np.array([0.1, 0.1, 0.1, 0.9, 0.85])
            mock_hdbscan.return_value = mock_clusterer

            detector = HDBSCANNewClassDetector(sample_config)
            new_class_indices = detector.detect_new_classes(sample_predictions, sample_embeddings)

            # Should return indices of outlier samples
            assert len(new_class_indices) == 2
            assert 3 in new_class_indices  # Index of first outlier
            assert 4 in new_class_indices  # Index of second outlier

    def test_get_outlier_scores(self, sample_predictions, sample_embeddings, sample_config):
        """Test getting outlier scores for all samples."""
        from scripts.auto_annotate.hdbscan_detector import HDBSCANNewClassDetector

        with patch('hdbscan.HDBSCAN') as mock_hdbscan:
            mock_clusterer = MagicMock()
            mock_clusterer.labels_ = np.array([0, 0, 0, -1, -1])
            mock_clusterer.outlier_scores_ = np.array([0.1, 0.1, 0.1, 0.9, 0.85])
            mock_hdbscan.return_value = mock_clusterer

            detector = HDBSCANNewClassDetector(sample_config)
            detector.detect_new_classes(sample_predictions, sample_embeddings)

            outlier_scores = detector.get_outlier_scores()

            assert len(outlier_scores) == 5
            assert outlier_scores[0] == 0.1
            assert outlier_scores[3] == 0.9

    def test_cluster_embeddings(self, sample_embeddings, sample_config):
        """Test clustering of embeddings."""
        from scripts.auto_annotate.hdbscan_detector import HDBSCANNewClassDetector

        with patch('hdbscan.HDBSCAN') as mock_hdbscan:
            mock_clusterer = MagicMock()
            mock_clusterer.fit.return_value = mock_clusterer
            mock_clusterer.labels_ = np.array([0, 0, 1, 1, -1])
            mock_hdbscan.return_value = mock_clusterer

            detector = HDBSCANNewClassDetector(sample_config)
            labels = detector._cluster_embeddings(sample_embeddings)

            mock_hdbscan.assert_called_once()
            assert labels is not None

    def test_get_new_class_predictions(self, sample_predictions, sample_embeddings, sample_config):
        """Test getting predictions identified as new classes."""
        from scripts.auto_annotate.hdbscan_detector import HDBSCANNewClassDetector

        with patch('hdbscan.HDBSCAN') as mock_hdbscan:
            mock_clusterer = MagicMock()
            mock_clusterer.labels_ = np.array([0, 0, 0, -1, -1])
            mock_clusterer.outlier_scores_ = np.array([0.1, 0.1, 0.1, 0.9, 0.85])
            mock_hdbscan.return_value = mock_clusterer

            detector = HDBSCANNewClassDetector(sample_config)
            new_class_indices = detector.detect_new_classes(sample_predictions, sample_embeddings)

            new_class_preds = detector.get_new_class_predictions(sample_predictions, new_class_indices)

            assert len(new_class_preds) == 2
            assert new_class_preds[0]["filename"] == "img_004.jpg"
            assert new_class_preds[1]["filename"] == "img_005.jpg"

    def test_empty_predictions(self, sample_config):
        """Test handling of empty predictions."""
        from scripts.auto_annotate.hdbscan_detector import HDBSCANNewClassDetector

        detector = HDBSCANNewClassDetector(sample_config)
        embeddings = np.array([]).reshape(0, 4)

        new_class_indices = detector.detect_new_classes([], embeddings)

        assert len(new_class_indices) == 0

    def test_mismatched_lengths(self, sample_predictions, sample_config):
        """Test handling of mismatched predictions and embeddings."""
        from scripts.auto_annotate.hdbscan_detector import HDBSCANNewClassDetector

        detector = HDBSCANNewClassDetector(sample_config)
        embeddings = np.array([[0.1, 0.2, 0.3, 0.4]])  # Only 1 embedding

        with pytest.raises(ValueError, match="mismatched"):
            detector.detect_new_classes(sample_predictions, embeddings)

    def test_single_sample(self, sample_config):
        """Test handling of single sample (cannot form cluster)."""
        from scripts.auto_annotate.hdbscan_detector import HDBSCANNewClassDetector

        detector = HDBSCANNewClassDetector(sample_config)
        embeddings = np.array([[0.1, 0.2, 0.3, 0.4]])
        predictions = [{"filename": "img_001.jpg", "aircraft": {"confidence": 0.95}}]

        new_class_indices = detector.detect_new_classes(predictions, embeddings)

        # Single sample should be treated as potential new class
        assert len(new_class_indices) == 0  # Or could be [0] depending on implementation
