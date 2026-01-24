"""
HDBSCANNewClassDetector class for detecting new aircraft classes.

This module uses HDBSCAN clustering to identify images that don't
belong to any existing class cluster, indicating potential new classes.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import hdbscan, but allow graceful degradation
HDBSCAN_AVAILABLE = False
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning(
        "hdbscan package not found. Install with: pip install hdbscan\n"
        "New class detection will be disabled (all samples will be treated as existing classes)."
    )

logger = logging.getLogger(__name__)


class HDBSCANNewClassDetector:
    """
    Detector for new aircraft classes using HDBSCAN clustering.

    This class uses HDBSCAN to cluster feature embeddings and identify
    outliers that may represent new aircraft classes not seen during training.

    If hdbscan is not installed, the detector will gracefully degrade
    and treat all samples as existing classes (return empty new class list).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize HDBSCANNewClassDetector.

        Args:
            config: Configuration dictionary containing HDBSCAN parameters:
                - min_cluster_size: Minimum cluster size
                - min_samples: Minimum samples in neighborhood
                - metric: Distance metric (e.g., "euclidean", "cosine")
                - cluster_selection_method: Method for cluster selection
                - prediction_data: Whether to generate prediction data
        """
        self.config = config
        self.min_cluster_size = config.get("min_cluster_size", 5)
        self.min_samples = config.get("min_samples", 3)
        self.metric = config.get("metric", "euclidean")
        self.cluster_selection_method = config.get(
            "cluster_selection_method", "eom"
        )
        self.prediction_data = config.get("prediction_data", True)

        # Clustering results
        self._clusterer = None
        self._labels: Optional[np.ndarray] = None
        self._outlier_scores: Optional[np.ndarray] = None

        if HDBSCAN_AVAILABLE:
            logger.info(
                f"HDBSCANNewClassDetector initialized with "
                f"min_cluster_size={self.min_cluster_size}, "
                f"min_samples={self.min_samples}, "
                f"metric={self.metric}"
            )
        else:
            logger.info(
                f"HDBSCANNewClassDetector initialized (hdbscan not available, "
                f"new class detection disabled)"
            )

    def detect_new_classes(
        self,
        predictions: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> List[int]:
        """
        Detect images that may represent new aircraft classes.

        Args:
            predictions: List of prediction results
            embeddings: Feature embeddings corresponding to predictions

        Returns:
            List of indices (from predictions) that are potential new classes
        """
        if not HDBSCAN_AVAILABLE:
            logger.info(
                "hdbscan not available, returning empty new class list"
            )
            return []

        if len(predictions) == 0:
            logger.warning("No predictions provided")
            return []

        if len(predictions) != len(embeddings):
            raise ValueError(
                f"Mismatched lengths: {len(predictions)} predictions vs "
                f"{len(embeddings)} embeddings"
            )

        logger.info(
            f"Detecting new classes from {len(predictions)} predictions..."
        )

        # Cluster embeddings
        self._cluster_embeddings(embeddings)

        # Get outlier indices (label = -1)
        outlier_indices = np.where(self._labels == -1)[0]

        logger.info(
            f"Found {len(outlier_indices)} potential new class samples "
            f"({len(outlier_indices)/len(predictions)*100:.1f}%)"
        )

        return outlier_indices.tolist()

    def _cluster_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Cluster embeddings using HDBSCAN.

        Args:
            embeddings: Feature embeddings (N x D)
        """
        if not HDBSCAN_AVAILABLE:
            # Set all labels to 0 (single cluster, no outliers)
            self._labels = np.zeros(len(embeddings), dtype=int)
            self._outlier_scores = np.zeros(len(embeddings), dtype=float)
            return

        logger.info(f"Clustering {len(embeddings)} embeddings...")

        # Create HDBSCAN clusterer
        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=self.prediction_data
        )

        # Fit clusterer
        self._clusterer.fit(embeddings)

        # Store results
        self._labels = self._clusterer.labels_
        self._outlier_scores = self._clusterer.outlier_scores_

        # Log clustering results
        n_clusters = len(set(self._labels)) - (1 if -1 in self._labels else 0)
        n_noise = list(self._labels).count(-1)

        logger.info(
            f"Clustering complete: {n_clusters} clusters, "
            f"{n_noise} noise points"
        )

    def get_outlier_scores(self) -> np.ndarray:
        """
        Get outlier scores for all samples.

        Returns:
            Array of outlier scores (higher = more likely to be new class)
        """
        if self._outlier_scores is None:
            raise RuntimeError(
                "No clustering results available. "
                "Call detect_new_classes() first."
            )

        return self._outlier_scores

    def get_labels(self) -> np.ndarray:
        """
        Get cluster labels for all samples.

        Returns:
            Array of cluster labels (-1 = outlier/noise)
        """
        if self._labels is None:
            raise RuntimeError(
                "No clustering results available. "
                "Call detect_new_classes() first."
            )

        return self._labels

    def get_new_class_predictions(
        self,
        predictions: List[Dict[str, Any]],
        new_class_indices: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Get predictions identified as new classes.

        Args:
            predictions: List of prediction results
            new_class_indices: Indices of potential new class samples

        Returns:
            List of predictions for new class samples
        """
        return [predictions[i] for i in new_class_indices]

    def get_regular_class_predictions(
        self,
        predictions: List[Dict[str, Any]],
        new_class_indices: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Get predictions for regular (not new) classes.

        Args:
            predictions: List of prediction results
            new_class_indices: Indices of potential new class samples

        Returns:
            List of predictions for regular class samples
        """
        new_class_set = set(new_class_indices)
        return [
            pred for i, pred in enumerate(predictions)
            if i not in new_class_set
        ]

    def get_statistics(
        self,
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get statistics about new class detection.

        Args:
            predictions: List of prediction results

        Returns:
            Dictionary containing statistics
        """
        if not HDBSCAN_AVAILABLE:
            return {
                "total_samples": len(predictions),
                "n_clusters": 1,
                "n_noise": 0,
                "noise_ratio": 0.0,
                "mean_outlier_score": 0.0,
                "std_outlier_score": 0.0,
                "max_outlier_score": 0.0,
                "min_outlier_score": 0.0,
                "note": "hdbscan not available"
            }

        if self._labels is None:
            raise RuntimeError(
                "No clustering results available. "
                "Call detect_new_classes() first."
            )

        n_clusters = len(set(self._labels)) - (1 if -1 in self._labels else 0)
        n_noise = list(self._labels).count(-1)
        n_total = len(self._labels)

        stats = {
            "total_samples": n_total,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": n_noise / n_total if n_total > 0 else 0.0,
            "mean_outlier_score": float(np.mean(self._outlier_scores)),
            "std_outlier_score": float(np.std(self._outlier_scores)),
            "max_outlier_score": float(np.max(self._outlier_scores)),
            "min_outlier_score": float(np.min(self._outlier_scores))
        }

        return stats

    def predict_new_class(
        self,
        embedding: np.ndarray
    ) -> bool:
        """
        Predict whether a single embedding represents a new class.

        Args:
            embedding: Feature embedding

        Returns:
            True if predicted to be a new class, False otherwise
        """
        if not HDBSCAN_AVAILABLE:
            return False

        if self._clusterer is None:
            raise RuntimeError(
                "No clustering model available. "
                "Call detect_new_classes() first."
            )

        # Get membership strengths
        _, strengths = hdbscan.membership_vector(
            self._clusterer,
            [embedding]
        )

        # If all strengths are very low, it's likely a new class
        max_strength = np.max(strengths)
        return max_strength < 0.3  # Threshold can be adjusted

    @property
    def is_available(self) -> bool:
        """Check if hdbscan is available."""
        return HDBSCAN_AVAILABLE
