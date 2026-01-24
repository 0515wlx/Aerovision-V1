"""
Test cases for FileOrganizer class.

Tests file organization and naming conventions.
"""

import pytest
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile


class TestFileOrganizer:
    """Test suite for FileOrganizer class."""

    @pytest.fixture
    def sample_predictions(self):
        """Sample prediction results for testing."""
        return [
            {
                "filename": "img_001.jpg",
                "aircraft": {"class_id": 0, "class_name": "Boeing", "confidence": 0.98},
                "airline": {"class_id": 2, "class_name": "China Eastern", "confidence": 0.96}
            },
            {
                "filename": "img_002.png",
                "aircraft": {"class_id": 1, "class_name": "Airbus", "confidence": 0.97},
                "airline": {"class_id": 3, "class_name": "Air China", "confidence": 0.94}
            },
            {
                "filename": "img_003.jpeg",
                "aircraft": {"class_id": 2, "class_name": "Antonov", "confidence": 0.92},
                "airline": {"class_id": 1, "class_name": "United Airlines", "confidence": 0.88}
            }
        ]

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for FileOrganizer."""
        return {
            "raw_images_dir": "/mnt/disk/AeroVision/images",
            "labeled_dir": "/mnt/disk/AeroVision/labeled",
            "filtered_new_class_dir": "/mnt/disk/AeroVision/filtered_new_class",
            "filtered_95_dir": "/mnt/disk/AeroVision/filtered_95"
        }

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories for testing."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        labeled_dir = tmp_path / "labeled"
        labeled_dir.mkdir()
        new_class_dir = tmp_path / "filtered_new_class"
        new_class_dir.mkdir()
        filtered_95_dir = tmp_path / "filtered_95"
        filtered_95_dir.mkdir()

        # Create some test images
        test_images = raw_dir / "test_images"
        test_images.mkdir()
        for i in range(3):
            (test_images / f"img_{i:03d}.jpg").write_text(f"test image {i}")

        return {
            "raw": str(test_images),
            "labeled": str(labeled_dir),
            "new_class": str(new_class_dir),
            "filtered_95": str(filtered_95_dir)
        }

    def test_file_organizer_initialization(self, sample_config):
        """Test that FileOrganizer initializes correctly."""
        from scripts.auto_annotate.file_organizer import FileOrganizer

        organizer = FileOrganizer(sample_config)

        assert organizer.raw_images_dir == sample_config["raw_images_dir"]
        assert organizer.labeled_dir == sample_config["labeled_dir"]
        assert organizer.filtered_new_class_dir == sample_config["filtered_new_class_dir"]
        assert organizer.filtered_95_dir == sample_config["filtered_95_dir"]

    def test_create_class_name_directory(self, sample_config, tmp_path):
        """Test creating class name directory."""
        from scripts.auto_annotate.file_organizer import FileOrganizer

        # Use tmp_path for testing
        config = sample_config.copy()
        config["labeled_dir"] = str(tmp_path)

        organizer = FileOrganizer(config)
        class_dir = organizer._create_class_name_directory("Boeing")

        assert class_dir.exists()
        assert class_dir.name == "Boeing"

    def test_create_safe_class_name(self, sample_config):
        """Test creating safe class names."""
        from scripts.auto_annotate.file_organizer import FileOrganizer

        organizer = FileOrganizer(sample_config)

        # Test various class names
        assert organizer._create_safe_class_name("Boeing 737-800") == "Boeing_737-800"
        assert organizer._create_safe_class_name("Airbus A320/A321") == "Airbus_A320_A321"
        assert organizer._create_safe_class_name("China\\Eastern") == "China_Eastern"

    def test_organize_labeled_images(self, sample_predictions, sample_config, temp_dirs):
        """Test organizing images into labeled directory."""
        from scripts.auto_annotate.file_organizer import FileOrganizer

        # Use temp dirs
        config = sample_config.copy()
        config["raw_images_dir"] = temp_dirs["raw"]
        config["labeled_dir"] = temp_dirs["labeled"]

        organizer = FileOrganizer(config)
        organizer.organize_labeled_images(sample_predictions)

        # Check that images were organized
        labeled_path = Path(temp_dirs["labeled"])
        boeing_dir = labeled_path / "Boeing"
        airbus_dir = labeled_path / "Airbus"
        antonov_dir = labeled_path / "Antonov"

        assert boeing_dir.exists()
        assert airbus_dir.exists()
        assert antonov_dir.exists()

        # Check that files were copied
        assert (boeing_dir / "img_001.jpg").exists()
        assert (airbus_dir / "img_002.png").exists()
        assert (antonov_dir / "img_003.jpeg").exists()

    def test_organize_new_class_images(self, sample_predictions, sample_config, temp_dirs):
        """Test organizing images into filtered_new_class directory."""
        from scripts.auto_annotate.file_organizer import FileOrganizer

        config = sample_config.copy()
        config["raw_images_dir"] = temp_dirs["raw"]
        config["filtered_new_class_dir"] = temp_dirs["new_class"]

        organizer = FileOrganizer(config)
        organizer.organize_new_class_images(sample_predictions)

        # Check that images were copied
        new_class_path = Path(temp_dirs["new_class"])
        assert (new_class_path / "img_001.jpg").exists()
        assert (new_class_path / "img_002.png").exists()
        assert (new_class_path / "img_003.jpeg").exists()

    def test_organize_filtered_95_images(self, sample_predictions, sample_config, temp_dirs):
        """Test organizing images into filtered_95 directory."""
        from scripts.auto_annotate.file_organizer import FileOrganizer

        config = sample_config.copy()
        config["raw_images_dir"] = temp_dirs["raw"]
        config["filtered_95_dir"] = temp_dirs["filtered_95"]

        organizer = FileOrganizer(config)
        organizer.organize_filtered_95_images(sample_predictions)

        # Check that images were copied
        filtered_95_path = Path(temp_dirs["filtered_95"])
        assert (filtered_95_path / "img_001.jpg").exists()
        assert (filtered_95_path / "img_002.png").exists()
        assert (filtered_95_path / "img_003.jpeg").exists()

    def test_save_prediction_details(self, sample_predictions, sample_config, temp_dirs):
        """Test saving prediction details to JSON."""
        from scripts.auto_annotate.file_organizer import FileOrganizer

        config = sample_config.copy()
        config["filtered_new_class_dir"] = temp_dirs["new_class"]
        config["filtered_95_dir"] = temp_dirs["filtered_95"]

        organizer = FileOrganizer(config)

        # Save for new class
        organizer.save_prediction_details(sample_predictions, "new_class")

        new_class_details = Path(temp_dirs["new_class"]) / "prediction_details.json"
        assert new_class_details.exists()

        with open(new_class_details, 'r') as f:
            data = json.load(f)
            assert "predictions" in data
            assert len(data["predictions"]) == 3

        # Save for filtered_95
        organizer.save_prediction_details(sample_predictions, "filtered_95")

        filtered_95_details = Path(temp_dirs["filtered_95"]) / "prediction_details.json"
        assert filtered_95_details.exists()

    def test_copy_image_with_original_name(self, sample_config, temp_dirs):
        """Test that images are copied with original names."""
        from scripts.auto_annotate.file_organizer import FileOrganizer

        config = sample_config.copy()
        config["raw_images_dir"] = temp_dirs["raw"]
        config["labeled_dir"] = temp_dirs["labeled"]

        organizer = FileOrganizer(config)

        # Create test prediction
        prediction = {
            "filename": "img_000.jpg",
            "aircraft": {"class_name": "Boeing", "confidence": 0.98}
        }

        organizer._copy_image_to_class_dir(prediction)

        # Check file was copied with original name
        boeing_dir = Path(temp_dirs["labeled"]) / "Boeing"
        assert (boeing_dir / "img_000.jpg").exists()

    def test_missing_source_image(self, sample_config, temp_dirs):
        """Test handling of missing source image."""
        from scripts.auto_annotate.file_organizer import FileOrganizer

        config = sample_config.copy()
        config["raw_images_dir"] = temp_dirs["raw"]
        config["labeled_dir"] = temp_dirs["labeled"]

        organizer = FileOrganizer(config)

        # Create prediction for non-existent image
        prediction = {
            "filename": "nonexistent.jpg",
            "aircraft": {"class_name": "Boeing", "confidence": 0.98}
        }

        # Should not raise exception, but log warning
        organizer._copy_image_to_class_dir(prediction)

        boeing_dir = Path(temp_dirs["labeled"]) / "Boeing"
        assert not (boeing_dir / "nonexistent.jpg").exists()

    def test_create_output_directories(self, sample_config, tmp_path):
        """Test that output directories are created."""
        from scripts.auto_annotate.file_organizer import FileOrganizer

        config = sample_config.copy()
        config["labeled_dir"] = str(tmp_path / "labeled")
        config["filtered_new_class_dir"] = str(tmp_path / "new_class")
        config["filtered_95_dir"] = str(tmp_path / "filtered_95")

        organizer = FileOrganizer(config)
        organizer._create_output_directories()

        assert Path(config["labeled_dir"]).exists()
        assert Path(config["filtered_new_class_dir"]).exists()
        assert Path(config["filtered_95_dir"]).exists()

    def test_get_statistics(self, sample_predictions, sample_config, temp_dirs):
        """Test getting statistics about organization."""
        from scripts.auto_annotate.file_organizer import FileOrganizer

        config = sample_config.copy()
        config["raw_images_dir"] = temp_dirs["raw"]
        config["labeled_dir"] = temp_dirs["labeled"]

        organizer = FileOrganizer(config)
        organizer.organize_labeled_images(sample_predictions)

        stats = organizer.get_statistics()

        assert stats["labeled_count"] == 3
        assert stats["new_class_count"] == 0
        assert stats["filtered_95_count"] == 0
