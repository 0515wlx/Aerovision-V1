"""
ModelPredictor class for aircraft and airline classification.

This module handles loading YOLOv8 classification models and
performing inference on images.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "ultralytics package not found. Install with: pip install ultralytics"
    )

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Predictor for aircraft type and airline classification.

    This class manages loading and running inference with YOLOv8
    classification models for both aircraft type and airline identification.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the ModelPredictor.

        Args:
            config: Configuration dictionary containing:
                - aircraft_model_path: Path to aircraft classification model
                - airline_model_path: Path to airline classification model
                - device: Device to use for inference (e.g., "cpu", "cuda:0")
                - imgsz: Image size for inference
                - batch_size: Batch size for inference
        """
        self.config = config
        self.aircraft_model_path = config["aircraft_model_path"]
        self.airline_model_path = config["airline_model_path"]
        self.device = config.get("device", "cpu")
        self.imgsz = config.get("imgsz", 640)
        self.batch_size = config.get("batch_size", 32)

        # Models (loaded on demand)
        self._aircraft_model: Optional[YOLO] = None
        self._airline_model: Optional[YOLO] = None

        logger.info(f"ModelPredictor initialized with device={self.device}, imgsz={self.imgsz}")

    @property
    def aircraft_model(self) -> YOLO:
        """Get or load the aircraft model."""
        if self._aircraft_model is None:
            self._load_aircraft_model()
        return self._aircraft_model

    @property
    def airline_model(self) -> YOLO:
        """Get or load the airline model."""
        if self._airline_model is None:
            self._load_airline_model()
        return self._airline_model

    def _load_aircraft_model(self) -> None:
        """
        Load the aircraft classification model.

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        model_path = Path(self.aircraft_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Aircraft model not found: {model_path}")

        logger.info(f"Loading aircraft model from: {model_path}")
        self._aircraft_model = YOLO(str(model_path))
        logger.info("Aircraft model loaded successfully")

    def _load_airline_model(self) -> None:
        """
        Load the airline classification model.

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        model_path = Path(self.airline_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Airline model not found: {model_path}")

        logger.info(f"Loading airline model from: {model_path}")
        self._airline_model = YOLO(str(model_path))
        logger.info("Airline model loaded successfully")

    def predict(
        self,
        image_path: str,
        extract_embeddings: bool = False
    ) -> Dict[str, Any]:
        """
        Predict aircraft type and airline for a single image.

        Args:
            image_path: Path to the image file
            extract_embeddings: Whether to extract feature embeddings

        Returns:
            Dictionary containing:
                - aircraft: {class_id, class_name, confidence, top5}
                - airline: {class_id, class_name, confidence, top5}
                - embeddings: Feature embeddings (if extract_embeddings=True)
        """
        if self._aircraft_model is None or self._airline_model is None:
            raise RuntimeError(
                "Models not loaded. Call load_models() first or access the model property."
            )

        # Predict aircraft type
        aircraft_result = self._predict_single(
            self.aircraft_model, image_path, "aircraft"
        )

        # Predict airline
        airline_result = self._predict_single(
            self.airline_model, image_path, "airline"
        )

        result = {
            "aircraft": aircraft_result,
            "airline": airline_result
        }

        # Extract embeddings if requested
        if extract_embeddings:
            embeddings = self._extract_embeddings(image_path)
            result["embeddings"] = embeddings

        return result

    def predict_batch(
        self,
        image_paths: List[str],
        extract_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Predict aircraft type and airline for multiple images.

        Args:
            image_paths: List of paths to image files
            extract_embeddings: Whether to extract feature embeddings

        Returns:
            List of prediction results, one per image
        """
        if self._aircraft_model is None or self._airline_model is None:
            raise RuntimeError(
                "Models not loaded. Call load_models() first or access the model property."
            )

        logger.info(f"Predicting batch of {len(image_paths)} images")

        results = []

        # Process images in batches
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}: {len(batch_paths)} images")

            batch_results = []
            for image_path in batch_paths:
                try:
                    result = self.predict(image_path, extract_embeddings=extract_embeddings)
                    result["filename"] = Path(image_path).name
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    # Add error result
                    batch_results.append({
                        "filename": Path(image_path).name,
                        "error": str(e),
                        "aircraft": {"class_id": -1, "confidence": 0.0},
                        "airline": {"class_id": -1, "confidence": 0.0}
                    })

            results.extend(batch_results)

        return results

    def _predict_single(
        self,
        model: YOLO,
        image_path: str,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Predict with a single model.

        Args:
            model: YOLO model instance
            image_path: Path to image file
            model_name: Name of the model (for logging)

        Returns:
            Dictionary containing class_id, class_name, confidence, and top5
        """
        # Run inference
        results = model.predict(
            image_path,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )

        # Get the first result (single image)
        result = results[0]

        # Extract probabilities
        probs = result.probs

        # Get top prediction
        class_id = int(probs.top1)
        confidence = float(probs.data[class_id])

        # Get class name
        class_names = model.model.names
        class_name = class_names.get(class_id, "Unknown")

        # Get top-5 predictions
        top5_indices = probs.top5
        top5_probs = probs.data[top5_indices].tolist()
        top5 = [
            {
                "id": int(idx),
                "name": class_names.get(int(idx), "Unknown"),
                "prob": float(prob)
            }
            for idx, prob in zip(top5_indices, top5_probs)
        ]

        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "top5": top5
        }

    def _extract_embeddings(self, image_path: str) -> np.ndarray:
        """
        Extract feature embeddings from an image.

        Args:
            image_path: Path to image file

        Returns:
            Feature embedding vector
        """
        # Get features from aircraft model
        result = self.aircraft_model.predict(
            image_path,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )[0]

        # Extract features from the model's forward pass
        # This is a simplified version; in practice you might need to
        # access intermediate layers
        if hasattr(result, "features"):
            embeddings = result.features
        else:
            # Fallback: use log probabilities as features
            embeddings = result.probs.data.cpu().numpy()

        return embeddings

    def get_aircraft_class_names(self) -> List[str]:
        """
        Get list of aircraft class names.

        Returns:
            List of class names
        """
        if self._aircraft_model is None:
            self._load_aircraft_model()

        class_names = self.aircraft_model.model.names
        return [class_names.get(i, f"class_{i}") for i in range(len(class_names))]

    def get_airline_class_names(self) -> List[str]:
        """
        Get list of airline class names.

        Returns:
            List of class names
        """
        if self._airline_model is None:
            self._load_airline_model()

        class_names = self.airline_model.model.names
        return [class_names.get(i, f"class_{i}") for i in range(len(class_names))]

    def load_models(self) -> None:
        """
        Load both aircraft and airline models.

        This method is called explicitly to ensure models are loaded
        before inference.
        """
        logger.info("Loading models...")
        self._load_aircraft_model()
        self._load_airline_model()
        logger.info("All models loaded successfully")
