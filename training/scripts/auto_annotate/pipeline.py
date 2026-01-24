"""
AutoAnnotatePipeline class for complete auto-annotation workflow.

This module integrates all components to provide a complete
auto-annotation pipeline for aircraft dataset labeling.
"""

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .model_predictor import ModelPredictor
from .hdbscan_detector import HDBSCANNewClassDetector
from .confidence_filter import ConfidenceFilter
from .file_organizer import FileOrganizer

logger = logging.getLogger(__name__)


class AutoAnnotatePipeline:
    """
    Complete auto-annotation pipeline.

    Pipeline workflow:
    1. Load models (aircraft and airline classifiers)
    2. Collect all raw images
    3. Predict in batches
    4. Extract embeddings
    5. Detect new class candidates using HDBSCAN
    6. Filter by confidence
    7. Organize files:
       - High confidence (>=95%): /labeled/<class_name>/
       - New class candidates: /filtered_new_class/
       - Low confidence (<95%): /filtered_95/
    8. Save prediction details for manual review
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize AutoAnnotatePipeline.

        Args:
            config: Configuration dictionary containing:
                - raw_images_dir: Source directory for raw images
                - labeled_dir: Destination for high confidence images
                - filtered_new_class_dir: Destination for new class candidates
                - filtered_95_dir: Destination for low confidence images
                - aircraft_model_path: Path to aircraft classification model
                - airline_model_path: Path to airline classification model
                - high_confidence_threshold: Threshold for auto-labeling (default: 0.95)
                - hdbscan: Dictionary of HDBSCAN parameters
                - device: Device for inference (e.g., "cpu", "cuda:0")
                - batch_size: Batch size for inference
        """
        self.config = config
        self.raw_images_dir = Path(config["raw_images_dir"])
        self.labeled_dir = Path(config["labeled_dir"])
        self.filtered_new_class_dir = Path(config["filtered_new_class_dir"])
        self.filtered_95_dir = Path(config["filtered_95_dir"])

        # Initialize components
        self.model_predictor = ModelPredictor(config)
        self.hdbscan_detector = HDBSCANNewClassDetector(
            config.get("hdbscan", {})
        )
        self.confidence_filter = ConfidenceFilter({
            "high_confidence_threshold": config.get(
                "high_confidence_threshold", 0.95
            ),
            "low_confidence_threshold": config.get(
                "low_confidence_threshold", 0.80
            )
        })
        self.file_organizer = FileOrganizer(config)

        # Pipeline state
        self._predictions: Optional[List[Dict[str, Any]]] = None
        self._embeddings: Optional[np.ndarray] = None

        logger.info("AutoAnnotatePipeline initialized")

    def load_models(self) -> None:
        """
        Load classification models.

        This should be called before running inference.
        """
        logger.info("Loading models...")
        self.model_predictor.load_models()
        logger.info("Models loaded successfully")

    def run(self) -> Dict[str, Any]:
        """
        Run the complete auto-annotation pipeline.

        Returns:
            Dictionary containing pipeline results and statistics
        """
        logger.info("=" * 60)
        logger.info("Starting Auto-Annotation Pipeline")
        logger.info("=" * 60)

        start_time = datetime.now()

        # 1. Collect image files
        image_files = self._collect_image_files()
        logger.info(f"Found {len(image_files)} images to process")

        if len(image_files) == 0:
            logger.warning("No images found in raw directory")
            return {
                "success": False,
                "error": "No images found",
                "statistics": self._calculate_statistics(0, 0, 0, 0)
            }

        # 2. Predict in batches
        logger.info("Running predictions...")
        self._predictions = self.model_predictor.predict_batch(
            image_files,
            extract_embeddings=True
        )

        # Extract embeddings from predictions
        self._embeddings = np.array([
            pred.get("embeddings", np.zeros(10))
            for pred in self._predictions
            if "embeddings" in pred
        ])

        logger.info(f"Predictions complete for {len(self._predictions)} images")

        # 3. Detect new class candidates using HDBSCAN
        logger.info("Detecting new class candidates...")
        new_class_indices = self._detect_new_classes()
        logger.info(f"Found {len(new_class_indices)} potential new class samples")

        # 4. Filter by confidence
        logger.info("Filtering by confidence...")
        filtered_result = self._filter_by_confidence(self._predictions)

        # 5. Separate new class candidates from filtered_95
        logger.info("Separating new class candidates...")
        high_confidence = filtered_result["high_confidence"]
        low_confidence = filtered_result["medium_confidence"] + filtered_result["low_confidence"]

        # Remove new class candidates from low confidence
        new_class_set = set(new_class_indices)
        low_confidence_filtered = [
            pred for i, pred in enumerate(self._predictions)
            if i not in new_class_set
        ]

        # 6. Organize files
        logger.info("Organizing files...")

        # High confidence -> labeled directory
        self.file_organizer.organize_labeled_images(high_confidence)

        # New class candidates -> filtered_new_class directory
        new_class_preds = [
            pred for i, pred in enumerate(self._predictions)
            if i in new_class_indices
        ]
        self.file_organizer.organize_new_class_images(new_class_preds)

        # Low confidence -> filtered_95 directory
        self.file_organizer.organize_filtered_95_images(low_confidence_filtered)

        # 7. Save prediction details
        logger.info("Saving prediction details...")

        # Save for new class candidates
        self._save_prediction_details(
            new_class_preds, new_class_indices, "new_class"
        )

        # Save for filtered_95
        self._save_prediction_details(
            low_confidence_filtered, None, "filtered_95"
        )

        # 8. Calculate and return statistics
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        stats = self._calculate_statistics(
            high_conf_count=len(high_confidence),
            medium_conf_count=len(filtered_result["medium_confidence"]),
            low_conf_count=len(filtered_result["low_confidence"]),
            new_class_count=len(new_class_indices)
        )

        result = {
            "success": True,
            "statistics": stats,
            "duration_seconds": duration,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "organizer_stats": self.file_organizer.get_statistics()
        }

        logger.info("=" * 60)
        logger.info("Auto-Annotation Pipeline Complete")
        logger.info("=" * 60)
        logger.info(f"Total images: {stats['total']}")
        logger.info(f"Auto-labeled (high conf): {stats['high_confidence_count']}")
        logger.info(f"Manual review (filtered_95): {stats['filtered_95_count']}")
        logger.info(f"New class candidates: {stats['new_class_count']}")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info("=" * 60)

        return result

    def _collect_image_files(self) -> List[str]:
        """
        Collect all image files from raw directory.

        Returns:
            List of image file paths
        """
        if not self.raw_images_dir.exists():
            raise FileNotFoundError(
                f"Raw images directory not found: {self.raw_images_dir}"
            )

        # Supported image extensions
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        # Collect image files
        image_files = []
        for ext in extensions:
            image_files.extend(self.raw_images_dir.glob(f"*{ext}"))
            image_files.extend(self.raw_images_dir.glob(f"*{ext.upper()}"))

        # Convert to strings and sort
        image_files = sorted([str(f) for f in image_files])

        return image_files

    def predict_batch(
        self,
        image_files: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Run predictions on a batch of images.

        Args:
            image_files: List of image file paths

        Returns:
            List of prediction results
        """
        return self.model_predictor.predict_batch(
            image_files,
            extract_embeddings=True
        )

    def _detect_new_classes(self) -> List[int]:
        """
        Detect new class candidates using HDBSCAN.

        Returns:
            List of indices representing potential new classes
        """
        if self._predictions is None or self._embeddings is None:
            raise RuntimeError(
                "No predictions available. Run predict_batch() first."
            )

        new_class_indices = self.hdbscan_detector.detect_new_classes(
            self._predictions,
            self._embeddings
        )

        return new_class_indices

    def _filter_by_confidence(
        self,
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Filter predictions by confidence level.

        Args:
            predictions: List of prediction results

        Returns:
            Dictionary with high, medium, and low confidence predictions
        """
        return self.confidence_filter.classify_predictions(predictions)

    def _save_prediction_details(
        self,
        predictions: List[Dict[str, Any]],
        new_class_indices: Optional[List[int]],
        dest_type: str = "new_class"
    ) -> None:
        """
        Save prediction details to JSON file.

        Args:
            predictions: List of prediction results
            new_class_indices: Indices of new class samples
            dest_type: Type of destination ("new_class" or "filtered_95")
        """
        self.file_organizer.save_prediction_details(
            predictions,
            new_class_indices,
            dest_type
        )

    def _calculate_statistics(
        self,
        high_conf_count: int,
        medium_conf_count: int,
        low_conf_count: int,
        new_class_count: int
    ) -> Dict[str, Any]:
        """
        Calculate pipeline statistics.

        Args:
            high_conf_count: Number of high confidence predictions
            medium_conf_count: Number of medium confidence predictions
            low_conf_count: Number of low confidence predictions
            new_class_count: Number of new class candidates

        Returns:
            Dictionary containing statistics
        """
        total = high_conf_count + medium_conf_count + low_conf_count
        filtered_95_count = medium_conf_count + low_conf_count

        stats = {
            "total": total,
            "high_confidence_count": high_conf_count,
            "medium_confidence_count": medium_conf_count,
            "low_confidence_count": low_conf_count,
            "filtered_95_count": filtered_95_count,
            "new_class_count": new_class_count,
            "high_confidence_ratio": high_conf_count / total if total > 0 else 0.0,
            "filtered_95_ratio": filtered_95_count / total if total > 0 else 0.0,
            "new_class_ratio": new_class_count / total if total > 0 else 0.0
        }

        return stats

    def get_config(self) -> Dict[str, Any]:
        """
        Get pipeline configuration.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def verify_output(self) -> Dict[str, Any]:
        """
        Verify that files were organized correctly.

        Returns:
            Dictionary containing verification results
        """
        return self.file_organizer.verify_organization()

    def save_statistics(
        self,
        result: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> None:
        """
        Save pipeline statistics to JSON file.

        Args:
            result: Pipeline result dictionary
            output_path: Path to save statistics (default: labeled_dir/statistics.json)
        """
        if output_path is None:
            output_path = self.labeled_dir / "pipeline_statistics.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Statistics saved to: {output_path}")
