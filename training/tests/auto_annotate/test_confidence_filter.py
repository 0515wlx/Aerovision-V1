"""
Test cases for ConfidenceFilter class.

Tests confidence-based filtering and classification of predictions.
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock
import json


class TestConfidenceFilter:
    """Test suite for ConfidenceFilter class."""

    @pytest.fixture
    def sample_predictions(self):
        """Sample prediction results for testing."""
        return [
            {
                "filename": "img_001.jpg",
                "aircraft": {"class_id": 0, "class_name": "Boeing", "confidence": 0.98, "top5": [{"id": 0, "name": "Boeing", "prob": 0.98}]},
                "airline": {"class_id": 2, "class_name": "China Eastern", "confidence": 0.96, "top5": [{"id": 2, "name": "China Eastern", "prob": 0.96}]}
            },
            {
                "filename": "img_002.jpg",
                "aircraft": {"class_id": 1, "class_name": "Airbus", "confidence": 0.97, "top5": [{"id": 1, "name": "Airbus", "prob": 0.97}]},
                "airline": {"class_id": 3, "class_name": "Air China", "confidence": 0.96, "top5": [{"id": 3, "name": "Air China", "prob": 0.96}]}
            },
            {
                "filename": "img_003.jpg",
                "aircraft": {"class_id": 2, "class_name": "Antonov", "confidence": 0.92, "top5": [{"id": 2, "name": "Antonov", "prob": 0.92}]},
                "airline": {"class_id": 1, "class_name": "United Airlines", "confidence": 0.88, "top5": [{"id": 1, "name": "United Airlines", "prob": 0.88}]}
            },
            {
                "filename": "img_004.jpg",
                "aircraft": {"class_id": 5, "class_name": "Unknown", "confidence": 0.65, "top5": [{"id": 5, "name": "Unknown", "prob": 0.65}]},
                "airline": {"class_id": 0, "class_name": "Unknown", "confidence": 0.55, "top5": [{"id": 0, "name": "Unknown", "prob": 0.55}]}
            },
            {
                "filename": "img_005.jpg",
                "aircraft": {"class_id": 3, "class_name": "Embraer", "confidence": 0.78, "top5": [{"id": 3, "name": "Embraer", "prob": 0.78}]},
                "airline": {"class_id": 4, "class_name": "Delta", "confidence": 0.72, "top5": [{"id": 4, "name": "Delta", "prob": 0.72}]}
            }
        ]

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for ConfidenceFilter."""
        return {
            "high_confidence_threshold": 0.95,
            "low_confidence_threshold": 0.80
        }

    def test_confidence_filter_initialization(self, sample_config):
        """Test that ConfidenceFilter initializes correctly."""
        from scripts.auto_annotate.confidence_filter import ConfidenceFilter

        filter = ConfidenceFilter(sample_config)

        assert filter.high_threshold == sample_config["high_confidence_threshold"]
        assert filter.low_threshold == sample_config["low_confidence_threshold"]

    def test_classify_predictions_high_confidence(self, sample_predictions, sample_config):
        """Test classification of high confidence predictions."""
        from scripts.auto_annotate.confidence_filter import ConfidenceFilter

        filter = ConfidenceFilter(sample_config)
        result = filter.classify_predictions(sample_predictions)

        # High confidence predictions
        assert len(result["high_confidence"]) == 2
        assert result["high_confidence"][0]["filename"] == "img_001.jpg"
        assert result["high_confidence"][1]["filename"] == "img_002.jpg"

        # Medium confidence predictions
        assert len(result["medium_confidence"]) == 1
        assert result["medium_confidence"][0]["filename"] == "img_003.jpg"

        # Low confidence predictions
        assert len(result["low_confidence"]) == 2
        low_filenames = {p["filename"] for p in result["low_confidence"]}
        assert "img_004.jpg" in low_filenames
        assert "img_005.jpg" in low_filenames

    def test_classify_predictions_empty(self, sample_config):
        """Test classification of empty predictions."""
        from scripts.auto_annotate.confidence_filter import ConfidenceFilter

        filter = ConfidenceFilter(sample_config)
        result = filter.classify_predictions([])

        assert len(result["high_confidence"]) == 0
        assert len(result["medium_confidence"]) == 0
        assert len(result["low_confidence"]) == 0

    def test_classify_predictions_all_high(self, sample_config):
        """Test classification when all predictions are high confidence."""
        from scripts.auto_annotate.confidence_filter import ConfidenceFilter

        filter = ConfidenceFilter(sample_config)
        all_high = [
            {
                "filename": f"img_{i:03d}.jpg",
                "aircraft": {"confidence": 0.99},
                "airline": {"confidence": 0.97}
            }
            for i in range(10)
        ]

        result = filter.classify_predictions(all_high)

        assert len(result["high_confidence"]) == 10
        assert len(result["medium_confidence"]) == 0
        assert len(result["low_confidence"]) == 0

    def test_classify_predictions_all_low(self, sample_config):
        """Test classification when all predictions are low confidence."""
        from scripts.auto_annotate.confidence_filter import ConfidenceFilter

        filter = ConfidenceFilter(sample_config)
        all_low = [
            {
                "filename": f"img_{i:03d}.jpg",
                "aircraft": {"confidence": 0.5},
                "airline": {"confidence": 0.45}
            }
            for i in range(10)
        ]

        result = filter.classify_predictions(all_low)

        assert len(result["high_confidence"]) == 0
        assert len(result["medium_confidence"]) == 0
        assert len(result["low_confidence"]) == 10

    def test_get_high_confidence_predictions(self, sample_predictions, sample_config):
        """Test getting high confidence predictions only."""
        from scripts.auto_annotate.confidence_filter import ConfidenceFilter

        filter = ConfidenceFilter(sample_config)
        result = filter.classify_predictions(sample_predictions)

        high_conf = result["high_confidence"]
        assert len(high_conf) == 2
        assert all(p["aircraft"]["confidence"] >= 0.95 for p in high_conf)
        assert all(p["airline"]["confidence"] >= 0.95 for p in high_conf)

    def test_get_filtered_95_predictions(self, sample_predictions, sample_config):
        """Test getting predictions with confidence < 95% (for manual review)."""
        from scripts.auto_annotate.confidence_filter import ConfidenceFilter

        filter = ConfidenceFilter(sample_config)
        result = filter.classify_predictions(sample_predictions)

        # Filtered_95 should include medium and low confidence
        filtered_95 = result["medium_confidence"] + result["low_confidence"]
        assert len(filtered_95) == 3
        assert all(p["aircraft"]["confidence"] < 0.95 for p in filtered_95)

    def test_confidence_thresholds_boundary(self, sample_config):
        """Test boundary conditions for confidence thresholds."""
        from scripts.auto_annotate.confidence_filter import ConfidenceFilter

        filter = ConfidenceFilter(sample_config)

        # Test exactly at high threshold
        boundary_preds = [
            {"filename": "high_boundary.jpg", "aircraft": {"confidence": 0.95}, "airline": {"confidence": 0.95}}
        ]
        result = filter.classify_predictions(boundary_preds)

        # Exactly 0.95 should be in high confidence (>=)
        assert len(result["high_confidence"]) == 1
        assert result["high_confidence"][0]["filename"] == "high_boundary.jpg"

        # Test just below high threshold
        boundary_preds = [
            {"filename": "just_below_high.jpg", "aircraft": {"confidence": 0.949}, "airline": {"confidence": 0.949}}
        ]
        result = filter.classify_predictions(boundary_preds)

        assert len(result["medium_confidence"]) == 1

    def test_missing_confidence_field(self, sample_config):
        """Test handling of predictions with missing confidence field."""
        from scripts.auto_annotate.confidence_filter import ConfidenceFilter

        filter = ConfidenceFilter(sample_config)
        invalid_pred = [
            {"filename": "invalid.jpg", "aircraft": {"class_id": 0}, "airline": {"class_id": 0}}
        ]

        # Invalid predictions should be skipped, not raise KeyError
        result = filter.classify_predictions(invalid_pred)

        # All categories should be empty since the prediction was skipped
        assert len(result["high_confidence"]) == 0
        assert len(result["medium_confidence"]) == 0
        assert len(result["low_confidence"]) == 0

    def test_update_thresholds(self, sample_config):
        """Test updating confidence thresholds."""
        from scripts.auto_annotate.confidence_filter import ConfidenceFilter

        filter = ConfidenceFilter(sample_config)

        assert filter.high_threshold == 0.95
        assert filter.low_threshold == 0.80

        filter.update_thresholds(high_threshold=0.90, low_threshold=0.70)

        assert filter.high_threshold == 0.90
        assert filter.low_threshold == 0.70

    def test_get_statistics(self, sample_predictions, sample_config):
        """Test getting statistics about classification results."""
        from scripts.auto_annotate.confidence_filter import ConfidenceFilter

        filter = ConfidenceFilter(sample_config)
        result = filter.classify_predictions(sample_predictions)

        stats = filter.get_statistics(result)

        assert stats["total"] == 5
        assert stats["high_confidence_count"] == 2
        assert stats["medium_confidence_count"] == 1
        assert stats["low_confidence_count"] == 2
        assert stats["high_confidence_ratio"] == 0.4
