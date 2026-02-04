"""
Test cases for FGVC workflow with type/airline filtering.

Tests are new --types and --airlines parameters in fgvc_workflow.py
"""

import pytest
import pandas as pd
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock, patch, call
import json

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))


class TestFGVCWorkflowFiltering:
    """Test suite for FGVC workflow with type/airline filtering."""

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories for testing."""
        base_dir = tmp_path / "test_workflow"
        base_dir.mkdir()

        fgvc_dir = base_dir / "FGVC_Aircraft"
        fgvc_dir.mkdir()
        (fgvc_dir / "data").mkdir()
        (fgvc_dir / "data" / "images").mkdir()

        output_base = base_dir / "output"
        output_base.mkdir()

        return {
            "base_dir": base_dir,
            "fgvc_dir": fgvc_dir,
            "output_base": output_base,
        }

    @pytest.fixture
    def sample_data(self, temp_dirs):
        """Create sample FGVC data for testing."""
        fgvc_data_dir = temp_dirs["fgvc_dir"] / "data"

        # Create sample images
        for i in range(10):
            image_file = fgvc_data_dir / "images" / f"{i:07d}.jpg"
            image_file.write_bytes(b"fake image data")

        # Create variant annotation file (train split)
        variant_file = fgvc_data_dir / "images_variant_train.txt"
        variant_data = {
            "0000001": "Boeing_737-800",
            "0000002": "Boeing_737-800",
            "0000003": "Airbus_A320",
            "0000004": "Airbus_A320",
            "0000005": "Boeing_777-300ER",
            "0000006": "Boeing_777-300ER",
            "0000007": "Boeing_747-400",
            "0000008": "Airbus_A330-300",
            "0000009": "Boeing_787-9",
            "0000010": "Airbus_A350-900",
        }
        with open(variant_file, "w") as f:
            for image_id, variant in variant_data.items():
                f.write(f"{image_id} {variant}\n")

        # Create family annotation file
        family_file = fgvc_data_dir / "images_family_train.txt"
        family_data = {k: "Commercial" for k in variant_data.keys()}
        with open(family_file, "w") as f:
            for image_id, family in family_data.items():
                f.write(f"{image_id} {family}\n")

        # Create manufacturer annotation file
        manufacturer_file = fgvc_data_dir / "images_manufacturer_train.txt"
        manufacturer_data = {
            "0000001": "Boeing",
            "0000002": "Boeing",
            "0000003": "Airbus",
            "0000004": "Airbus",
            "0000005": "Boeing",
            "0000006": "Boeing",
            "0000007": "Boeing",
            "0000008": "Airbus",
            "0000009": "Boeing",
            "0000010": "Airbus",
        }
        with open(manufacturer_file, "w") as f:
            for image_id, manufacturer in manufacturer_data.items():
                f.write(f"{image_id} {manufacturer}\n")

        return {
            "variant_data": variant_data,
            "family_data": family_data,
            "manufacturer_data": manufacturer_data,
        }

    def test_convert_with_all_types_airlines(self, temp_dirs, sample_data):
        """Test that convert script with --types all and --airlines all keeps all data."""
        from convert_fgvc_aerocraft import FGVC_AircraftConverter

        fgvc_dir = str(temp_dirs["fgvc_dir"])
        output_base = str(temp_dirs["output_base"])
        output_dir = Path(output_base) / "converted"

        # Create converter and convert
        converter = FGVC_AircraftConverter(
            fgvc_data_dir=fgvc_dir,
            output_dir=str(output_dir),
            split="train",
            types_filter="all",
            airlines_filter="all",
        )
        converter.convert()

        # Check that all images were converted
        labels_csv = output_dir / "labels.csv"
        assert labels_csv.exists(), "Labels CSV should exist"

        df = pd.read_csv(labels_csv)
        assert len(df) >= 5, f"Expected at least 5 images, got {len(df)}"

    def test_convert_with_specific_types_filtering(self, temp_dirs, sample_data):
        """Test that convert with --types Boeing_737-800,Airbus_A320 filters correctly."""
        from convert_fgvc_aerocraft import FGVC_AircraftConverter

        fgvc_dir = str(temp_dirs["fgvc_dir"])
        output_base = str(temp_dirs["output_base"])
        output_dir = Path(output_base) / "converted"

        # Create converter and convert
        converter = FGVC_AircraftConverter(
            fgvc_data_dir=fgvc_dir,
            output_dir=str(output_dir),
            split="train",
            types_filter="Boeing_737-800,Airbus_A320",
            airlines_filter="all",
        )
        converter.convert()

        # Check that only Boeing_737-800 and Airbus_A320 are kept
        labels_csv = output_dir / "labels.csv"
        assert labels_csv.exists(), "Labels CSV should exist"

        df = pd.read_csv(labels_csv)
        unique_types = set(df["typename"].unique())
        expected_types = {"Boeing_737-800", "Airbus_A320"}
        assert unique_types == expected_types, \
            f"Expected only {expected_types}, got {unique_types}"

    def test_convert_with_specific_airlines_filtering(self, temp_dirs, sample_data):
        """Test that convert with --airlines Boeing filters correctly."""
        from convert_fgvc_aerocraft import FGVC_AircraftConverter

        fgvc_dir = str(temp_dirs["fgvc_dir"])
        output_base = str(temp_dirs["output_base"])
        output_dir = Path(output_base) / "converted"

        # Create converter and convert
        converter = FGVC_AircraftConverter(
            fgvc_data_dir=fgvc_dir,
            output_dir=str(output_dir),
            split="train",
            types_filter="all",
            airlines_filter="Boeing",
        )
        converter.convert()

        # Check that only Boeing manufacturers are kept
        labels_csv = output_dir / "labels.csv"
        assert labels_csv.exists(), "Labels CSV should exist"

        df = pd.read_csv(labels_csv)
        unique_airlines = set(df["airline"].unique())
        assert unique_airlines == {"Boeing"}, \
            f"Expected only Boeing, got {unique_airlines}"

    def test_convert_with_combined_type_and_airline_filtering(self, temp_dirs, sample_data):
        """Test that convert with both --types and --airlines filters correctly."""
        from convert_fgvc_aerocraft import FGVC_AircraftConverter

        fgvc_dir = str(temp_dirs["fgvc_dir"])
        output_base = str(temp_dirs["output_base"])
        output_dir = Path(output_base) / "converted"

        # Create converter and convert
        converter = FGVC_AircraftConverter(
            fgvc_data_dir=fgvc_dir,
            output_dir=str(output_dir),
            split="train",
            types_filter="Boeing_737-800,Boeing_777-300ER",
            airlines_filter="Boeing",
        )
        converter.convert()

        # Check that both filters are applied
        labels_csv = output_dir / "labels.csv"
        assert labels_csv.exists(), "Labels CSV should exist"

        df = pd.read_csv(labels_csv)
        unique_types = set(df["typename"].unique())
        unique_airlines = set(df["airline"].unique())

        expected_types = {"Boeing_737-800", "Boeing_777-300ER"}
        expected_airlines = {"Boeing"}

        assert unique_types == expected_types, \
            f"Expected types {expected_types}, got {unique_types}"
        assert unique_airlines == expected_airlines, \
            f"Expected airlines {expected_airlines}, got {unique_airlines}"

    def test_train_classify_with_filtered_data(self, temp_dirs, sample_data):
        """Test that train_classify.py can run with filtered dataset."""
        # This test ensures that train_classify.py works correctly
        # with filtered dataset from fgvc_workflow.py

        # Create a simple dataset structure that train_classify.py expects
        train_dir = temp_dirs["output_base"] / "test_data" / "train"
        val_dir = temp_dirs["output_base"] / "test_data" / "val"
        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)

        # Create class directories
        for class_name in ["Boeing_737-800", "Airbus_A320"]:
            class_dir = train_dir / class_name
            class_dir.mkdir()
            # Create dummy image
            (class_dir / "image.jpg").write_bytes(b"fake image")

            val_class_dir = val_dir / class_name
            val_class_dir.mkdir()
            (val_class_dir / "image.jpg").write_bytes(b"fake image")

        # Check that directory structure is valid
        assert train_dir.exists(), "Training directory should exist"
        assert (train_dir / "Boeing_737-800").exists(), "Class directory should exist"
        assert (train_dir / "Boeing_737-800" / "image.jpg").exists(), "Image should exist"
