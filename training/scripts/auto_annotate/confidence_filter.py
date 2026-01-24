"""
ConfidenceFilter class for filtering predictions by confidence.

This module filters predictions into high, medium, and low confidence
categories based on confidence thresholds.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class ConfidenceFilter:
    """
    Filter for classification predictions based on confidence scores.

    This class categorizes predictions into:
    - High confidence: >= high_threshold (e.g., 0.95)
    - Medium confidence: low_threshold <= conf < high_threshold (e.g., 0.80-0.95)
    - Low confidence: < low_threshold (e.g., < 0.80)

    High confidence predictions are automatically labeled.
    Medium and low confidence predictions go to manual review.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize ConfidenceFilter.

        Args:
            config: Configuration dictionary containing:
                - high_confidence_threshold: Threshold for auto-labeling (default: 0.95)
                - low_confidence_threshold: Threshold for medium vs low (default: 0.80)
        """
        self.config = config
        self.high_threshold = config.get("high_confidence_threshold", 0.95)
        self.low_threshold = config.get("low_confidence_threshold", 0.80)

        logger.info(
            f"ConfidenceFilter initialized with "
            f"high_threshold={self.high_threshold}, "
            f"low_threshold={self.low_threshold}"
        )

    def classify_predictions(
        self,
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Classify predictions by confidence level.

        Uses the lower of aircraft and airline confidence for classification.

        Args:
            predictions: List of prediction results with:
                - aircraft: {confidence, ...}
                - airline: {confidence, ...}

        Returns:
            Dictionary with keys:
                - high_confidence: List of high confidence predictions
                - medium_confidence: List of medium confidence predictions
                - low_confidence: List of low confidence predictions
        """
        high_confidence = []
        medium_confidence = []
        low_confidence = []

        for pred in predictions:
            if not self._is_valid_prediction(pred):
                logger.warning(
                    f"Skipping invalid prediction: {pred.get('filename', 'unknown')}"
                )
                continue

            # Get minimum confidence (conservative approach)
            aircraft_conf = pred["aircraft"]["confidence"]
            airline_conf = pred["airline"]["confidence"]
            min_confidence = min(aircraft_conf, airline_conf)

            # Classify by confidence level
            if min_confidence >= self.high_threshold:
                high_confidence.append(pred)
            elif min_confidence >= self.low_threshold:
                medium_confidence.append(pred)
            else:
                low_confidence.append(pred)

        logger.info(
            f"Classification complete: "
            f"{len(high_confidence)} high, "
            f"{len(medium_confidence)} medium, "
            f"{len(low_confidence)} low confidence"
        )

        return {
            "high_confidence": high_confidence,
            "medium_confidence": medium_confidence,
            "low_confidence": low_confidence
        }

    def _is_valid_prediction(self, prediction: Dict[str, Any]) -> bool:
        """
        Check if a prediction is valid.

        Args:
            prediction: Prediction result

        Returns:
            True if valid, False otherwise
        """
        try:
            aircraft_conf = prediction["aircraft"]["confidence"]
            airline_conf = prediction["airline"]["confidence"]

            # Check types
            if not isinstance(aircraft_conf, (int, float)):
                return False
            if not isinstance(airline_conf, (int, float)):
                return False

            # Check ranges
            if not 0 <= aircraft_conf <= 1:
                return False
            if not 0 <= airline_conf <= 1:
                return False

            return True
        except (KeyError, TypeError):
            return False

    def get_high_confidence_predictions(
        self,
        predictions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Get high confidence predictions (>= high_threshold).

        Args:
            predictions: List of prediction results

        Returns:
            List of high confidence predictions
        """
        filtered = self.classify_predictions(predictions)
        return filtered["high_confidence"]

    def get_filtered_95_predictions(
        self,
        predictions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Get predictions with confidence < 95% (for manual review).

        This includes both medium and low confidence predictions.

        Args:
            predictions: List of prediction results

        Returns:
            List of predictions with confidence < 95%
        """
        filtered = self.classify_predictions(predictions)
        return filtered["medium_confidence"] + filtered["low_confidence"]

    def update_thresholds(
        self,
        high_threshold: Optional[float] = None,
        low_threshold: Optional[float] = None
    ) -> None:
        """
        Update confidence thresholds.

        Args:
            high_threshold: New high confidence threshold
            low_threshold: New low confidence threshold
        """
        if high_threshold is not None:
            self.high_threshold = high_threshold
            logger.info(f"High threshold updated to {high_threshold}")

        if low_threshold is not None:
            self.low_threshold = low_threshold
            logger.info(f"Low threshold updated to {low_threshold}")

    def get_statistics(
        self,
        filtered_result: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Get statistics about classification results.

        Args:
            filtered_result: Result from classify_predictions()

        Returns:
            Dictionary containing statistics
        """
        high_count = len(filtered_result["high_confidence"])
        medium_count = len(filtered_result["medium_confidence"])
        low_count = len(filtered_result["low_confidence"])
        total = high_count + medium_count + low_count

        stats = {
            "total": total,
            "high_confidence_count": high_count,
            "medium_confidence_count": medium_count,
            "low_confidence_count": low_count,
            "high_confidence_ratio": high_count / total if total > 0 else 0.0,
            "medium_confidence_ratio": medium_count / total if total > 0 else 0.0,
            "low_confidence_ratio": low_count / total if total > 0 else 0.0,
            "filtered_95_count": medium_count + low_count,
            "auto_labeled_count": high_count
        }

        return stats

    def filter_by_min_confidence(
        self,
        predictions: List[Dict[str, Any]],
        min_confidence: float
    ) -> List[Dict[str, Any]]:
        """
        Filter predictions by minimum confidence.

        Args:
            predictions: List of prediction results
            min_confidence: Minimum confidence threshold

        Returns:
            List of predictions with confidence >= min_confidence
        """
        filtered = []
        for pred in predictions:
            if not self._is_valid_prediction(pred):
                continue

            aircraft_conf = pred["aircraft"]["confidence"]
            airline_conf = pred["airline"]["confidence"]
            min_conf = min(aircraft_conf, airline_conf)

            if min_conf >= min_confidence:
                filtered.append(pred)

        logger.info(
            f"Filtered {len(filtered)}/{len(predictions)} predictions "
            f"with confidence >= {min_confidence}"
        )

        return filtered

    def get_confidence_distribution(
        self,
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get confidence distribution statistics.

        Args:
            predictions: List of prediction results

        Returns:
            Dictionary containing confidence statistics
        """
        aircraft_confs = []
        airline_confs = []
        min_confs = []

        for pred in predictions:
            if not self._is_valid_prediction(pred):
                continue

            aircraft_conf = pred["aircraft"]["confidence"]
            airline_conf = pred["airline"]["confidence"]
            min_conf = min(aircraft_conf, airline_conf)

            aircraft_confs.append(aircraft_conf)
            airline_confs.append(airline_conf)
            min_confs.append(min_conf)

        import numpy as np

        def _compute_stats(values):
            if not values:
                return {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "median": 0.0
                }
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values))
            }

        return {
            "aircraft": _compute_stats(aircraft_confs),
            "airline": _compute_stats(airline_confs),
            "min": _compute_stats(min_confs)
        }
