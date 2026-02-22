"""
FileOrganizer class for organizing labeled images.

This module handles copying and organizing images according to the
naming conventions and directory structure.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class FileOrganizer:
    """
    Organizer for labeled images.

    This class copies images to appropriate directories following
    the project's naming conventions:
    - High confidence: /labeled/<class_name>/<filename>
    - New class candidates: /filtered_new_class/<filename>
    - Low confidence: /filtered_95/<filename>
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize FileOrganizer.

        Args:
            config: Configuration dictionary containing:
                - raw_images_dir: Source directory for raw images
                - labeled_dir: Destination for high confidence images
                - filtered_new_class_dir: Destination for new class candidates
                - filtered_95_dir: Destination for low confidence images
        """
        self.config = config
        self.raw_images_dir = Path(config["raw_images_dir"])
        self.labeled_dir = Path(config["labeled_dir"])
        self.filtered_new_class_dir = Path(config["filtered_new_class_dir"])
        self.filtered_95_dir = Path(config["filtered_95_dir"])

        # Statistics
        self._stats = {
            "labeled_count": 0,
            "new_class_count": 0,
            "filtered_95_count": 0,
            "skipped_count": 0,
            "error_count": 0
        }

        logger.info(
            f"FileOrganizer initialized with "
            f"raw_dir={self.raw_images_dir}, "
            f"labeled_dir={self.labeled_dir}"
        )

    def organize_labeled_images(
        self,
        predictions: List[Dict[str, Any]]
    ) -> None:
        """
        Organize high confidence images into labeled directory.

        Args:
            predictions: List of prediction results with aircraft class names
        """
        logger.info(
            f"Organizing {len(predictions)} labeled images..."
        )

        self._create_output_directories()

        for pred in predictions:
            try:
                self._copy_image_to_class_dir(pred, "labeled")
            except Exception as e:
                logger.error(f"Error organizing {pred.get('filename', 'unknown')}: {e}")
                self._stats["error_count"] += 1

        logger.info(
            f"Labeled images organized: {self._stats['labeled_count']} "
            f"copied, {self._stats['skipped_count']} skipped"
        )

    def organize_new_class_images(
        self,
        predictions: List[Dict[str, Any]]
    ) -> None:
        """
        Organize potential new class images into filtered_new_class directory.

        Args:
            predictions: List of prediction results for new class candidates
        """
        logger.info(
            f"Organizing {len(predictions)} new class candidates..."
        )

        self.filtered_new_class_dir.mkdir(parents=True, exist_ok=True)

        for pred in predictions:
            try:
                self._copy_image(pred, self.filtered_new_class_dir, "new_class")
            except Exception as e:
                logger.error(f"Error organizing {pred.get('filename', 'unknown')}: {e}")
                self._stats["error_count"] += 1

        logger.info(
            f"New class candidates organized: {self._stats['new_class_count']} "
            f"copied, {self._stats['skipped_count']} skipped"
        )

    def organize_filtered_95_images(
        self,
        predictions: List[Dict[str, Any]]
    ) -> None:
        """
        Organize low confidence images into filtered_95 directory.

        Args:
            predictions: List of prediction results for manual review
        """
        logger.info(
            f"Organizing {len(predictions)} filtered_95 images..."
        )

        self.filtered_95_dir.mkdir(parents=True, exist_ok=True)

        for pred in predictions:
            try:
                self._copy_image(pred, self.filtered_95_dir, "filtered_95")
            except Exception as e:
                logger.error(f"Error organizing {pred.get('filename', 'unknown')}: {e}")
                self._stats["error_count"] += 1

        logger.info(
            f"Filtered_95 images organized: {self._stats['filtered_95_count']} "
            f"copied, {self._stats['skipped_count']} skipped"
        )

    def _copy_image_to_class_dir(
        self,
        prediction: Dict[str, Any],
        category: str = "labeled"
    ) -> None:
        """
        Copy image to its class-specific directory.

        Args:
            prediction: Prediction result with aircraft class name
            category: Category name for statistics
        """
        filename = prediction["filename"]
        src_path = self.raw_images_dir / filename

        if not src_path.exists():
            logger.warning(f"Source image not found: {src_path}")
            self._stats["skipped_count"] += 1
            return

        # Get class name and create safe directory name
        class_name = prediction["aircraft"]["class_name"]
        safe_class_name = self._create_safe_class_name(class_name)

        # Create class directory if needed
        if category == "labeled":
            dest_dir = self.labeled_dir / safe_class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            self._stats["labeled_count"] += 1
        else:
            raise ValueError(f"Unknown category: {category}")

        # Copy image
        dest_path = dest_dir / filename
        if not dest_path.exists():
            shutil.copy2(src_path, dest_path)
        else:
            logger.debug(f"Destination already exists: {dest_path}")

    def _copy_image(
        self,
        prediction: Dict[str, Any],
        dest_dir: Path,
        category: str
    ) -> None:
        """
        Copy image to destination directory.

        Args:
            prediction: Prediction result
            dest_dir: Destination directory
            category: Category name for statistics
        """
        filename = prediction["filename"]
        src_path = self.raw_images_dir / filename

        if not src_path.exists():
            logger.warning(f"Source image not found: {src_path}")
            self._stats["skipped_count"] += 1
            return

        # Copy image
        dest_path = dest_dir / filename
        if not dest_path.exists():
            shutil.copy2(src_path, dest_path)

            # Update statistics
            if category == "new_class":
                self._stats["new_class_count"] += 1
            elif category == "filtered_95":
                self._stats["filtered_95_count"] += 1
        else:
            logger.debug(f"Destination already exists: {dest_path}")

    def _create_safe_class_name(self, class_name: str) -> str:
        """
        Create a safe class name for use as directory name.

        Replaces spaces, slashes, and other problematic characters.

        Args:
            class_name: Original class name

        Returns:
            Safe class name
        """
        # Replace spaces with underscores
        safe_name = class_name.replace(" ", "_")
        # Replace forward slashes with underscores
        safe_name = safe_name.replace("/", "_")
        # Replace backslashes with underscores
        safe_name = safe_name.replace("\\", "_")
        # Replace other problematic characters if needed
        safe_name = safe_name.replace(":", "_")

        return safe_name

    def _create_class_name_directory(
        self,
        class_name: str
    ) -> Path:
        """
        Create a directory for a specific class name.

        Args:
            class_name: Aircraft class name

        Returns:
            Path to the created class directory
        """
        safe_class_name = self._create_safe_class_name(class_name)
        class_dir = self.labeled_dir / safe_class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        return class_dir

    def _create_output_directories(self) -> None:
        """Create output directories if they don't exist."""
        self.labeled_dir.mkdir(parents=True, exist_ok=True)
        self.filtered_new_class_dir.mkdir(parents=True, exist_ok=True)
        self.filtered_95_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Output directories created/verified")

    def save_prediction_details(
        self,
        predictions: List[Dict[str, Any]],
        new_class_indices: Optional[List[int]],
        dest_type: str = "new_class"
    ) -> None:
        """
        Save prediction details to JSON file.

        Args:
            predictions: List of prediction results
            new_class_indices: Indices of new class samples (if any)
            dest_type: Type of destination ("new_class" or "filtered_95")
        """
        # Determine destination directory
        if dest_type == "new_class":
            dest_dir = self.filtered_new_class_dir
        elif dest_type == "filtered_95":
            dest_dir = self.filtered_95_dir
        else:
            raise ValueError(f"Unknown dest_type: {dest_type}")

        # Prepare output data
        output_data = {
            "total_predictions": len(predictions),
            "predictions": predictions,
            "new_class_indices": new_class_indices if new_class_indices else [],
            "timestamp": self._get_timestamp()
        }

        # Save to JSON
        output_file = dest_dir / "prediction_details.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Prediction details saved to: {output_file}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about file organization.

        Returns:
            Dictionary containing statistics
        """
        return {
            "labeled_count": self._stats["labeled_count"],
            "new_class_count": self._stats["new_class_count"],
            "filtered_95_count": self._stats["filtered_95_count"],
            "skipped_count": self._stats["skipped_count"],
            "error_count": self._stats["error_count"],
            "total_processed": (
                self._stats["labeled_count"] +
                self._stats["new_class_count"] +
                self._stats["filtered_95_count"]
            )
        }

    def _get_timestamp(self) -> str:
        """
        Get current timestamp.

        Returns:
            ISO format timestamp string
        """
        from datetime import datetime
        return datetime.now().isoformat()

    def verify_organization(self) -> Dict[str, Any]:
        """
        Verify that files were organized correctly.

        Returns:
            Dictionary containing verification results
        """
        results = {
            "labeled": {},
            "new_class": {},
            "filtered_95": {}
        }

        # Verify labeled directory
        if self.labeled_dir.exists():
            class_dirs = [d for d in self.labeled_dir.iterdir() if d.is_dir()]
            results["labeled"]["num_classes"] = len(class_dirs)
            results["labeled"]["classes"] = [d.name for d in class_dirs]
            results["labeled"]["total_images"] = sum(
                len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))
                for d in class_dirs
            )

        # Verify new class directory
        if self.filtered_new_class_dir.exists():
            images = (list(self.filtered_new_class_dir.glob("*.jpg")) +
                     list(self.filtered_new_class_dir.glob("*.png")))
            results["new_class"]["num_images"] = len(images)

        # Verify filtered_95 directory
        if self.filtered_95_dir.exists():
            images = (list(self.filtered_95_dir.glob("*.jpg")) +
                     list(self.filtered_95_dir.glob("*.png")))
            results["filtered_95"]["num_images"] = len(images)

        return results
