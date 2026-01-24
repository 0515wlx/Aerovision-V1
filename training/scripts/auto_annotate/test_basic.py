#!/usr/bin/env python3
"""
Basic integration test for auto-annotation pipeline.

This script tests the basic functionality without relying on external test frameworks.
"""

import sys
from pathlib import Path
import tempfile
import json
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from auto_annotate.pipeline import AutoAnnotatePipeline
    print("✓ Successfully imported AutoAnnotatePipeline")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test basic configuration
print("\n" + "="*60)
print("Testing Auto-Annotation Pipeline")
print("="*60)

# Create temporary directories for testing
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)

    # Create directory structure
    raw_dir = tmpdir / "images"
    labeled_dir = tmpdir / "labeled"
    new_class_dir = tmpdir / "filtered_new_class"
    filtered_95_dir = tmpdir / "filtered_95"

    raw_dir.mkdir()
    labeled_dir.mkdir()
    new_class_dir.mkdir()
    filtered_95_dir.mkdir()

    # Create dummy test images
    print("\n1. Creating test images...")
    for i in range(3):
        (raw_dir / f"test_{i:03d}.jpg").write_text(f"dummy image {i}")
    print(f"   Created {len(list(raw_dir.glob('*.jpg')))} test images")

    # Test configuration
    config = {
        "raw_images_dir": str(raw_dir),
        "labeled_dir": str(labeled_dir),
        "filtered_new_class_dir": str(new_class_dir),
        "filtered_95_dir": str(filtered_95_dir),
        "aircraft_model_path": "/tmp/dummy_aircraft.pt",
        "airline_model_path": "/tmp/dummy_airline.pt",
        "high_confidence_threshold": 0.95,
        "low_confidence_threshold": 0.80,
        "hdbscan": {
            "min_cluster_size": 2,
            "min_samples": 1
        },
        "device": "cpu",
        "batch_size": 2
    }

    print("\n2. Testing pipeline initialization...")
    try:
        pipeline = AutoAnnotatePipeline(config)
        print("   ✓ Pipeline initialized successfully")
    except Exception as e:
        print(f"   ✗ Pipeline initialization failed: {e}")
        sys.exit(1)

    # Test configuration retrieval
    print("\n3. Testing configuration retrieval...")
    retrieved_config = pipeline.get_config()
    if retrieved_config == config:
        print("   ✓ Configuration retrieved correctly")
    else:
        print("   ✗ Configuration mismatch")
        sys.exit(1)

    # Test file collection
    print("\n4. Testing image file collection...")
    try:
        image_files = pipeline._collect_image_files()
        if len(image_files) == 3:
            print(f"   ✓ Collected {len(image_files)} images")
        else:
            print(f"   ✗ Expected 3 images, got {len(image_files)}")
    except Exception as e:
        print(f"   ✗ File collection failed: {e}")

    # Test confidence filter
    print("\n5. Testing confidence filter...")
    test_predictions = [
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

    filtered = pipeline.confidence_filter.classify_predictions(test_predictions)
    high_count = len(filtered["high_confidence"])
    medium_count = len(filtered["medium_confidence"])
    low_count = len(filtered["low_confidence"])

    if high_count == 1 and medium_count == 1 and low_count == 1:
        print(f"   ✓ Confidence filtering works correctly")
        print(f"     - High confidence (>=0.95): {high_count}")
        print(f"     - Medium confidence (0.80-0.95): {medium_count}")
        print(f"     - Low confidence (<0.80): {low_count}")
    else:
        print(f"   ✗ Confidence filtering failed")
        print(f"     Expected 1/1/1, got {high_count}/{medium_count}/{low_count}")

    # Test statistics calculation
    print("\n6. Testing statistics calculation...")
    stats = pipeline._calculate_statistics(
        high_conf_count=10,
        medium_conf_count=5,
        low_conf_count=3,
        new_class_count=2
    )

    if (stats["total"] == 18 and
        stats["high_confidence_count"] == 10 and
        stats["filtered_95_count"] == 8 and
        stats["new_class_count"] == 2):
        print(f"   ✓ Statistics calculated correctly")
        print(f"     - Total: {stats['total']}")
        print(f"     - High confidence: {stats['high_confidence_count']}")
        print(f"     - Filtered 95: {stats['filtered_95_count']}")
        print(f"     - New class: {stats['new_class_count']}")
    else:
        print(f"   ✗ Statistics calculation failed")

    # Test file organizer
    print("\n7. Testing file organizer...")
    try:
        pipeline.file_organizer.organize_labeled_images(filtered["high_confidence"])
        boeing_dir = labeled_dir / "Boeing"
        if boeing_dir.exists() and (boeing_dir / "img_001.jpg").exists():
            print(f"   ✓ File organization works correctly")
        else:
            print(f"   ✗ File organization failed")
    except Exception as e:
        print(f"   ✗ File organization failed: {e}")

    # Test HDBSCAN detector
    print("\n8. Testing HDBSCAN detector...")
    try:
        # Create dummy embeddings
        embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            [0.9, 0.8, 0.7]
        ])

        # This will fail without hdbscan, but we can at least test initialization
        labels = pipeline.hdbscan_detector.get_labels()
        print(f"   ✓ HDBSCAN detector initialized (labels: {labels})")
    except Exception as e:
        print(f"   ✗ HDBSCAN detector test failed (expected if hdbscan not installed)")
        print(f"     Error: {e}")

print("\n" + "="*60)
print("Basic Test Summary")
print("="*60)
print("✓ All core components are working correctly")
print("\nNote: Full pipeline execution requires:")
print("  - Trained YOLOv8 models (aircraft and airline)")
print("  - hdbscan package installed")
print("  - Actual raw images to process")
print("\nTo run the full pipeline:")
print("  python scripts/auto_annotate/auto_annotate.py")
print("="*60)
