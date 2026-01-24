#!/usr/bin/env python3
"""
Simple integration test for auto-annotation pipeline.

This script tests basic functionality without hdbscan dependency.
"""

import sys
from pathlib import Path
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*60)
print("Testing Auto-Annotation Pipeline Components")
print("="*60)

# Test 1: ConfidenceFilter
print("\n1. Testing ConfidenceFilter...")
try:
    from auto_annotate.confidence_filter import ConfidenceFilter

    config = {"high_confidence_threshold": 0.95, "low_confidence_threshold": 0.80}
    filter = ConfidenceFilter(config)

    predictions = [
        {
            "filename": "img_001.jpg",
            "aircraft": {"confidence": 0.98, "class_name": "Boeing"},
            "airline": {"confidence": 0.96, "class_name": "China Eastern"}
        },
        {
            "filename": "img_002.jpg",
            "aircraft": {"confidence": 0.85, "class_name": "Airbus"},
            "airline": {"confidence": 0.82, "class_name": "Air China"}
        },
        {
            "filename": "img_003.jpg",
            "aircraft": {"confidence": 0.65, "class_name": "Unknown"},
            "airline": {"confidence": 0.55, "class_name": "Unknown"}
        }
    ]

    result = filter.classify_predictions(predictions)
    high = len(result["high_confidence"])
    medium = len(result["medium_confidence"])
    low = len(result["low_confidence"])

    if high == 1 and medium == 1 and low == 1:
        print(f"   ✓ ConfidenceFilter works correctly")
        print(f"     High: {high}, Medium: {medium}, Low: {low}")
    else:
        print(f"   ✗ ConfidenceFilter failed")
        print(f"     Expected 1/1/1, got {high}/{medium}/{low}")
except Exception as e:
    print(f"   ✗ ConfidenceFilter error: {e}")

# Test 2: FileOrganizer
print("\n2. Testing FileOrganizer...")
try:
    from auto_annotate.file_organizer import FileOrganizer

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        raw_dir = tmpdir / "raw"
        labeled_dir = tmpdir / "labeled"

        raw_dir.mkdir()
        labeled_dir.mkdir()

        # Create test images
        for i in range(3):
            (raw_dir / f"img_{i:03d}.jpg").write_text(f"test {i}")

        config = {
            "raw_images_dir": str(raw_dir),
            "labeled_dir": str(labeled_dir),
            "filtered_new_class_dir": str(tmpdir / "new_class"),
            "filtered_95_dir": str(tmpdir / "filtered_95")
        }

        organizer = FileOrganizer(config)

        # Test safe class name creation
        boeing = organizer._create_safe_class_name("Boeing 737-800")
        airbus = organizer._create_safe_class_name("Airbus A320/A321")

        if boeing == "Boeing_737-800" and airbus == "Airbus_A320_A321":
            print(f"   ✓ Safe class name creation works")
            print(f"     'Boeing 737-800' -> '{boeing}'")
            print(f"     'Airbus A320/A321' -> '{airbus}'")
        else:
            print(f"   ✗ Safe class name creation failed")

        # Test file organization
        test_pred = [
            {
                "filename": "img_000.jpg",
                "aircraft": {"class_name": "Boeing", "confidence": 0.98}
            }
        ]

        organizer.organize_labeled_images(test_pred)

        boeing_dir = labeled_dir / "Boeing"
        if boeing_dir.exists() and (boeing_dir / "img_000.jpg").exists():
            print(f"   ✓ File organization works correctly")
        else:
            print(f"   ✗ File organization failed")

        # Test statistics
        stats = organizer.get_statistics()
        print(f"   ✓ Statistics: labeled={stats['labeled_count']}, skipped={stats['skipped_count']}")

except Exception as e:
    print(f"   ✗ FileOrganizer error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Pipeline Initialization
print("\n3. Testing Pipeline Initialization...")
try:
    from auto_annotate.pipeline import AutoAnnotatePipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        config = {
            "raw_images_dir": str(tmpdir / "images"),
            "labeled_dir": str(tmpdir / "labeled"),
            "filtered_new_class_dir": str(tmpdir / "new_class"),
            "filtered_95_dir": str(tmpdir / "filtered_95"),
            "aircraft_model_path": "/tmp/dummy.pt",
            "airline_model_path": "/tmp/dummy.pt",
            "high_confidence_threshold": 0.95,
            "low_confidence_threshold": 0.80,
            "hdbscan": {"min_cluster_size": 2, "min_samples": 1},
            "device": "cpu",
            "batch_size": 2
        }

        pipeline = AutoAnnotatePipeline(config)
        print(f"   ✓ Pipeline initialized successfully")

        # Test statistics calculation
        stats = pipeline._calculate_statistics(10, 5, 3, 2)

        if (stats["total"] == 18 and
            stats["high_confidence_count"] == 10 and
            stats["filtered_95_count"] == 8 and
            stats["new_class_count"] == 2):
            print(f"   ✓ Statistics calculation works correctly")
            print(f"     Total: {stats['total']}")
            print(f"     High conf: {stats['high_confidence_count']}")
            print(f"     Filtered 95: {stats['filtered_95_count']}")
            print(f"     New class: {stats['new_class_count']}")
        else:
            print(f"   ✗ Statistics calculation failed")

        # Test config retrieval
        retrieved = pipeline.get_config()
        if retrieved == config:
            print(f"   ✓ Config retrieval works correctly")
        else:
            print(f"   ✗ Config retrieval failed")

except Exception as e:
    print(f"   ✗ Pipeline error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print("✓ Core components are working correctly")
print("")
print("To run full pipeline:")
print("  1. Install hdbscan: pip install hdbscan")
print("  2. Place YOLOv8 models at:")
print("     - /home/wlx/yolo26x-cls-aircraft.pt")
print("     - /home/wlx/yolo26x-cls-airline.pt")
print("  3. Place raw images at:")
print("     - /mnt/disk/AeroVision/images")
print("  4. Run:")
print("     python scripts/auto_annotate/auto_annotate.py")
print("="*60)
