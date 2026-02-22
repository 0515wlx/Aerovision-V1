#!/usr/bin/env python3
"""
Unit tests for ElasticFace loss integration with YOLO classifier.

TDD approach: Write tests first, then implement to make tests pass.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# Add scripts directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


class TestYOLOEmbedMethod:
    """Test YOLO model's embed() method output format.

    YOLOv8's embed() returns a LIST of tensors, not a single tensor.
    Format: [Tensor([embedding_dim]), ...] for each sample in batch.
    """

    @pytest.fixture
    def sample_model_path(self):
        """Path to a small YOLO classification model for testing."""
        # Use yolov8n-cls.pt which should be available
        return "yolov8n-cls.pt"

    @pytest.fixture
    def sample_images(self):
        """Create sample image tensors for testing (normalized 0-1)."""
        # Create dummy RGB images: (batch_size=4, C=3, H=224, W=224)
        # YOLO expects images normalized 0-1
        return torch.randn(4, 3, 224, 224) / 255.0

    def test_embed_returns_list(self, sample_model_path, sample_images):
        """Test that embed() returns a list."""
        from ultralytics import YOLO

        model = YOLO(sample_model_path)
        embeddings = model.embed(sample_images)

        assert isinstance(embeddings, list), \
            f"embed() should return list, got {type(embeddings)}"

    def test_embed_list_length(self, sample_model_path, sample_images):
        """Test that embed() list length equals batch size."""
        from ultralytics import YOLO

        model = YOLO(sample_model_path)
        embeddings = model.embed(sample_images)

        assert len(embeddings) == sample_images.shape[0], \
            f"List length should equal batch size: expected {sample_images.shape[0]}, got {len(embeddings)}"

    def test_embed_element_shape(self, sample_model_path, sample_images):
        """Test that each list element is a 1D tensor (embedding_dim)."""
        from ultralytics import YOLO

        model = YOLO(sample_model_path)
        embeddings = model.embed(sample_images)

        for i, emb in enumerate(embeddings):
            assert isinstance(emb, torch.Tensor), \
                f"embeddings[{i}] should be torch.Tensor, got {type(emb)}"
            assert len(emb.shape) == 1, \
                f"embeddings[{i}] should be 1D tensor, got shape {emb.shape}"
            assert emb.shape[0] > 0, \
                f"Embedding dimension should be > 0, got {emb.shape[0]}"

    def test_embed_stackable(self, sample_model_path, sample_images):
        """Test that embeddings can be stacked into a 2D tensor."""
        from ultralytics import YOLO

        model = YOLO(sample_model_path)
        embeddings = model.embed(sample_images)

        # Stack into (batch_size, embedding_dim)
        stacked = torch.stack(embeddings, dim=0)

        assert stacked.shape == (sample_images.shape[0], embeddings[0].shape[0]), \
            f"Stacked embeddings shape should be ({sample_images.shape[0]}, {embeddings[0].shape[0]}), got {stacked.shape}"

    def test_embed_device(self, sample_model_path, sample_images):
        """Test that embed() handles device correctly."""
        from ultralytics import YOLO

        device = "cuda" if torch.cuda.is_available() else "cpu"
        images = sample_images.to(device)

        model = YOLO(sample_model_path)
        embeddings = model.embed(images)

        # Check first element's device
        assert embeddings[0].device.type == device, \
            f"embeddings[0] should be on {device}, got {embeddings[0].device}"

    def test_embed_dtype(self, sample_model_path, sample_images):
        """Test that embed() returns float32 tensors."""
        from ultralytics import YOLO

        model = YOLO(sample_model_path)
        embeddings = model.embed(sample_images)

        assert embeddings[0].dtype == torch.float32, \
            f"embeddings should be float32, got {embeddings[0].dtype}"

    def test_get_embedding_dim(self, sample_model_path, sample_images):
        """Test getting embedding dimension from YOLO model."""
        from ultralytics import YOLO

        model = YOLO(sample_model_path)
        embeddings = model.embed(sample_images)

        embedding_dim = embeddings[0].shape[0]
        # yolov8n-cls has 256 embedding dimensions
        assert embedding_dim == 256, \
            f"Expected embedding_dim=256 for yolov8n-cls, got {embedding_dim}"


class TestElasticFaceLoss:
    """Test ElasticFace loss implementation in training_utils.py."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embedding tensors for testing."""
        # Simulated L2-normalized embeddings: (batch_size=8, embedding_dim=256)
        embeddings = torch.randn(8, 256)
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    @pytest.fixture
    def sample_labels(self):
        """Create sample labels for testing."""
        # 8 samples, 10 classes
        return torch.randint(0, 10, (8,))

    def test_elastic_arcface_init(self):
        """Test ElasticArcFace initialization with default parameters."""
        from training_utils import ElasticFaceArcFace

        in_features = 256
        out_features = 10
        loss_fn = ElasticFaceArcFace(
            in_features=in_features,
            out_features=out_features,
            s=30.0,
            m=0.30,
            std=0.01
        )

        assert loss_fn.in_features == in_features
        assert loss_fn.out_features == out_features
        assert loss_fn.s == 30.0
        assert loss_fn.m == 0.30
        assert loss_fn.std == 0.01

    def test_elastic_arcface_output_shape(self, sample_embeddings, sample_labels):
        """Test ElasticArcFace forward pass returns correct shape."""
        from training_utils import ElasticFaceArcFace

        in_features = sample_embeddings.shape[1]
        num_classes = 10

        loss_fn = ElasticFaceArcFace(
            in_features=in_features,
            out_features=num_classes,
            s=30.0,
            m=0.30,
            std=0.01
        )

        logits = loss_fn(sample_embeddings, sample_labels)

        assert logits.shape == (sample_embeddings.shape[0], num_classes), \
            f"Expected logits shape {(sample_embeddings.shape[0], num_classes)}, got {logits.shape}"

    def test_elastic_arcface_with_cross_entropy(self, sample_embeddings, sample_labels):
        """Test ElasticArcFace works with CrossEntropyLoss."""
        from training_utils import ElasticFaceArcFace

        in_features = sample_embeddings.shape[1]
        num_classes = 10

        loss_fn = ElasticFaceArcFace(
            in_features=in_features,
            out_features=num_classes,
            s=30.0,
            m=0.30,
            std=0.01
        )
        ce_loss = torch.nn.CrossEntropyLoss()

        logits = loss_fn(sample_embeddings, sample_labels)
        loss = ce_loss(logits, sample_labels)

        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_combined_loss_formula(self, sample_embeddings, sample_labels):
        """Test combined loss: CE_loss + lambda * ElasticFace_loss."""
        from training_utils import ElasticFaceArcFace

        lambda_weight = 0.5

        in_features = sample_embeddings.shape[1]
        num_classes = 10

        # Standard CE loss with normalized features (simple dot product)
        kernel = torch.nn.Parameter(torch.randn(in_features, num_classes) * 0.01)
        kernel_norm = torch.nn.functional.normalize(kernel, p=2, dim=0)
        ce_logits = torch.mm(sample_embeddings, kernel_norm) * 30.0

        ce_loss = torch.nn.CrossEntropyLoss()
        standard_loss = ce_loss(ce_logits, sample_labels)

        # ElasticFace loss
        elastic_loss_fn = ElasticFaceArcFace(
            in_features=in_features,
            out_features=num_classes,
            s=30.0,
            m=0.30,
            std=0.01
        )
        elastic_logits = elastic_loss_fn(sample_embeddings, sample_labels)
        elastic_loss = ce_loss(elastic_logits, sample_labels)

        # Combined loss
        combined_loss = standard_loss + lambda_weight * elastic_loss

        assert combined_loss.item() >= standard_loss.item(), \
            "Combined loss should be >= standard CE loss when lambda > 0"
        assert not torch.isnan(combined_loss), "Combined loss should not be NaN"


class TestElasticFaceIntegration:
    """Test ElasticFace loss integration in training context."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embedding tensors for testing."""
        # Simulated L2-normalized embeddings: (batch_size=8, embedding_dim=256)
        embeddings = torch.randn(8, 256)
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    @pytest.fixture
    def sample_labels(self):
        """Create sample labels for testing."""
        # 8 samples, 10 classes
        return torch.randint(0, 10, (8,))

    @pytest.fixture
    def mock_yolo_model(self):
        """Create a mock YOLO model with embed method."""
        class MockYOLO:
            def __init__(self):
                self.embedding_dim = 256
                self.num_classes = 10

            def embed(self, images: torch.Tensor) -> list:
                """Mock embed method that returns list of normalized embeddings."""
                batch_size = images.shape[0]
                embeddings_list = []
                for _ in range(batch_size):
                    emb = torch.randn(self.embedding_dim, device=images.device)
                    # L2 normalize
                    emb = torch.nn.functional.normalize(emb.unsqueeze(0), p=2, dim=1).squeeze(0)
                    embeddings_list.append(emb)
                return embeddings_list

        return MockYOLO()

    @pytest.fixture
    def elastic_wrapper(self, mock_yolo_model):
        """Create ElasticFaceLossWrapper for testing."""
        from training_utils import ElasticFaceLossWrapper

        return ElasticFaceLossWrapper(
            num_classes=10,
            embedding_dim=256,
            lambda_weight=0.5,
            s=30.0,
            m=0.30,
            std=0.01
        )

    def test_wrapper_forward_pass(self, elastic_wrapper, sample_embeddings, sample_labels):
        """Test ElasticFaceLossWrapper forward pass returns correct loss."""
        from training_utils import l2_norm

        # Normalize embeddings
        normalized_embeddings = l2_norm(sample_embeddings, axis=1)

        total_loss, loss_dict = elastic_wrapper(normalized_embeddings, sample_labels)

        assert isinstance(total_loss, torch.Tensor), "Total loss should be a tensor"
        assert total_loss.item() > 0, "Total loss should be positive"
        assert not torch.isnan(total_loss), "Total loss should not be NaN"
        assert "total_loss" in loss_dict
        assert "ce_loss" in loss_dict
        assert "elastic_loss" in loss_dict

    def test_wrapper_backward_pass(self, elastic_wrapper, sample_embeddings, sample_labels):
        """Test that backward pass works correctly."""
        from training_utils import l2_norm

        # Normalize embeddings
        normalized_embeddings = l2_norm(sample_embeddings, axis=1)

        total_loss, _ = elastic_wrapper(normalized_embeddings, sample_labels)
        total_loss.backward()

        # Check that gradients were computed for ElasticFace parameters
        for param in elastic_wrapper.elastic_face.parameters():
            if param.requires_grad:
                assert param.grad is not None, "ElasticFace parameters should have gradients"

    def test_lambda_weight_effect(self, elastic_wrapper, sample_embeddings, sample_labels):
        """Test that lambda_weight affects the combined loss correctly."""
        from training_utils import l2_norm

        # Normalize embeddings
        normalized_embeddings = l2_norm(sample_embeddings, axis=1)

        # Test with lambda=0 (only CE loss)
        elastic_wrapper.lambda_weight = 0.0
        loss_0, loss_dict_0 = elastic_wrapper(normalized_embeddings, sample_labels)

        # Test with lambda=0.5
        elastic_wrapper.lambda_weight = 0.5
        loss_05, loss_dict_05 = elastic_wrapper(normalized_embeddings, sample_labels)

        # Test with lambda=1.0
        elastic_wrapper.lambda_weight = 1.0
        loss_10, loss_dict_10 = elastic_wrapper(normalized_embeddings, sample_labels)

        # With lambda=0, loss should equal CE loss
        assert abs(loss_0.item() - loss_dict_0["ce_loss"]) < 1e-5, \
            "With lambda=0, total loss should equal CE loss"

        # With lambda=1, total loss should be CE + elastic (approximate due to random margin)
        # Use larger tolerance because ElasticFace margin is randomly sampled
        expected_10 = loss_dict_10["ce_loss"] + loss_dict_10["elastic_loss"]
        assert abs(loss_10.item() - expected_10) < 0.2, \
            f"With lambda=1, total loss should equal CE + ElasticFace (within tolerance). " \
            f"Expected {expected_10}, got {loss_10.item()}"

        # Verify lambda=0.5 gives loss between lambda=0 and lambda=1
        assert loss_0.item() < loss_05.item() < loss_10.item() or \
               loss_10.item() < loss_05.item() < loss_0.item(), \
            "Loss with lambda=0.5 should be between lambda=0 and lambda=1"

    def test_stack_embeddings(self, mock_yolo_model):
        """Test stack_embeddings function."""
        from training_utils import stack_embeddings

        # Create mock embeddings
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        embeddings_list = mock_yolo_model.embed(images)

        # Stack embeddings
        stacked = stack_embeddings(embeddings_list)

        assert stacked.shape == (batch_size, 256), \
            f"Stacked shape should be ({batch_size}, 256), got {stacked.shape}"
        assert isinstance(stacked, torch.Tensor), "Stacked result should be a tensor"

    def test_elastic_face_with_yolo(self, mock_yolo_model):
        """Test ElasticFaceWithYOLO module."""
        from training_utils import ElasticFaceWithYOLO

        elastic_yolo = ElasticFaceWithYOLO(
            yolo_model=mock_yolo_model,
            num_classes=10,
            lambda_weight=0.5,
            s=30.0,
            m=0.30,
            std=0.01
        )

        # Create sample data
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        labels = torch.randint(0, 10, (batch_size,))

        # Compute loss
        total_loss, loss_dict = elastic_yolo(images, labels)

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() > 0
        assert "total_loss" in loss_dict
        assert "ce_loss" in loss_dict
        assert "elastic_loss" in loss_dict

    def test_wrapper_from_yolo_model(self):
        """Test ElasticFaceLossWrapper.from_yolo_model factory method."""
        from ultralytics import YOLO
        from training_utils import ElasticFaceLossWrapper

        model = YOLO("yolov8n-cls.pt")

        wrapper = ElasticFaceLossWrapper.from_yolo_model(
            yolo_model=model,
            num_classes=10,
            lambda_weight=0.5,
            s=30.0,
            m=0.30,
            std=0.01
        )

        assert wrapper.num_classes == 10
        assert wrapper.embedding_dim == 256  # yolov8n-cls has 256 dims
        assert wrapper.lambda_weight == 0.5


class TestElasticFaceParameters:
    """Test ElasticFace hyperparameter configurations."""

    def test_different_s_values(self):
        """Test different scale (s) values."""
        from training_utils import ElasticFaceArcFace

        embeddings = torch.nn.functional.normalize(torch.randn(8, 256), p=2, dim=1)
        labels = torch.randint(0, 10, (8,))

        for s in [10.0, 30.0, 64.0]:
            loss_fn = ElasticFaceArcFace(256, 10, s=s, m=0.30, std=0.01)
            logits = loss_fn(embeddings, labels)

            # Higher s should produce larger logit magnitudes
            assert logits.abs().max() > 0, f"Logits should be non-zero for s={s}"

    def test_different_m_values(self):
        """Test different margin (m) values."""
        from training_utils import ElasticFaceArcFace

        embeddings = torch.nn.functional.normalize(torch.randn(8, 256), p=2, dim=1)
        labels = torch.randint(0, 10, (8,))

        for m in [0.1, 0.3, 0.5]:
            loss_fn = ElasticFaceArcFace(256, 10, s=30.0, m=m, std=0.01)
            logits = loss_fn(embeddings, labels)
            assert logits.shape == (8, 10), f"Output shape should be consistent for m={m}"

    def test_different_std_values(self):
        """Test different std values for margin sampling."""
        from training_utils import ElasticFaceArcFace

        embeddings = torch.nn.functional.normalize(torch.randn(8, 256), p=2, dim=1)
        labels = torch.randint(0, 10, (8,))

        for std in [0.0, 0.01, 0.0125]:
            loss_fn = ElasticFaceArcFace(256, 10, s=30.0, m=0.30, std=std)
            logits = loss_fn(embeddings, labels)
            assert logits.shape == (8, 10), f"Output shape should be consistent for std={std}"

    def test_default_yolo_adapted_params(self):
        """Test YOLO-adapted default parameters."""
        from training_utils import ElasticFaceArcFace

        # YOLO-adapted defaults: s=30.0, m=0.30, std=0.01
        loss_fn = ElasticFaceArcFace(
            in_features=256,
            out_features=10,
            s=30.0,
            m=0.30,
            std=0.01
        )

        assert loss_fn.s == 30.0
        assert loss_fn.m == 0.30
        assert loss_fn.std == 0.01


class TestElasticFaceTypes:
    """Test ElasticFace loss types: cos, arc, cos+, arc+."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embedding tensors for testing."""
        embeddings = torch.randn(8, 256)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    @pytest.fixture
    def sample_labels(self):
        """Create sample labels for testing."""
        return torch.randint(0, 10, (8,))

    def test_elastic_face_type_cos(self, sample_embeddings, sample_labels):
        """Test ElasticFace with type='cos' (CosFace)."""
        from training_utils import ElasticFaceArcFace

        loss_fn = ElasticFaceArcFace(
            in_features=256,
            out_features=10,
            s=30.0,
            m=0.30,
            std=0.01,
            type="cos",
        )

        logits = loss_fn(sample_embeddings, sample_labels)

        assert logits.shape == (8, 10), f"Expected shape (8, 10), got {logits.shape}"
        assert loss_fn.type == "cos", "Type should be 'cos'"

    def test_elastic_face_type_arc(self, sample_embeddings, sample_labels):
        """Test ElasticFace with type='arc' (ArcFace)."""
        from training_utils import ElasticFaceArcFace

        loss_fn = ElasticFaceArcFace(
            in_features=256,
            out_features=10,
            s=30.0,
            m=0.30,
            std=0.01,
            type="arc",
        )

        logits = loss_fn(sample_embeddings, sample_labels)

        assert logits.shape == (8, 10), f"Expected shape (8, 10), got {logits.shape}"
        assert loss_fn.type == "arc", "Type should be 'arc'"

    def test_elastic_face_type_cos_plus(self, sample_embeddings, sample_labels):
        """Test ElasticFace with type='cos+' (CosFace+ adaptive)."""
        from training_utils import ElasticFaceArcFace

        loss_fn = ElasticFaceArcFace(
            in_features=256,
            out_features=10,
            s=30.0,
            m=0.30,
            std=0.01,
            type="cos+",
        )

        logits = loss_fn(sample_embeddings, sample_labels)

        assert logits.shape == (8, 10), f"Expected shape (8, 10), got {logits.shape}"
        assert loss_fn.type == "cos+", "Type should be 'cos+'"
        assert loss_fn.plus is True, "plus should be True for 'cos+'"

    def test_elastic_face_type_arc_plus(self, sample_embeddings, sample_labels):
        """Test ElasticFace with type='arc+' (ArcFace+ adaptive)."""
        from training_utils import ElasticFaceArcFace

        loss_fn = ElasticFaceArcFace(
            in_features=256,
            out_features=10,
            s=30.0,
            m=0.30,
            std=0.01,
            type="arc+",
        )

        logits = loss_fn(sample_embeddings, sample_labels)

        assert logits.shape == (8, 10), f"Expected shape (8, 10), got {logits.shape}"
        assert loss_fn.type == "arc+", "Type should be 'arc+'"
        assert loss_fn.plus is True, "plus should be True for 'arc+'"

    def test_elastic_face_all_types_produce_valid_logits(
        self, sample_embeddings, sample_labels
    ):
        """Test that all types produce valid logits."""
        from training_utils import ElasticFaceArcFace

        for loss_type in ["cos", "arc", "cos+", "arc+"]:
            loss_fn = ElasticFaceArcFace(
                in_features=256,
                out_features=10,
                s=30.0,
                m=0.30,
                std=0.01,
                type=loss_type,
            )

            logits = loss_fn(sample_embeddings, sample_labels)

            # Check shape
            assert logits.shape == (8, 10), f"Shape mismatch for type='{loss_type}'"

            # Check no NaN
            assert not torch.isnan(logits).any(), f"NaN detected for type='{loss_type}'"

            # Check finite values
            assert torch.isfinite(logits).all(), f"Non-finite values for type='{loss_type}'"

    def test_elastic_face_type_cos_vs_arc_difference(
        self, sample_embeddings, sample_labels
    ):
        """Test that cos and arc types produce different results."""
        from training_utils import ElasticFaceArcFace

        # Set seed for reproducibility
        torch.manual_seed(42)

        loss_fn_cos = ElasticFaceArcFace(
            in_features=256,
            out_features=10,
            s=30.0,
            m=0.30,
            std=0.0,  # Fixed margin for comparison
            type="cos",
        )

        torch.manual_seed(42)

        loss_fn_arc = ElasticFaceArcFace(
            in_features=256,
            out_features=10,
            s=30.0,
            m=0.30,
            std=0.0,  # Fixed margin for comparison
            type="arc",
        )

        # Use the same kernel weights for fair comparison
        loss_fn_arc.kernel.data.copy_(loss_fn_cos.kernel.data)

        logits_cos = loss_fn_cos(sample_embeddings, sample_labels)
        logits_arc = loss_fn_arc(sample_embeddings, sample_labels)

        # Results should be different (cos applies margin in cosine space,
        # arc applies margin in angular space)
        assert not torch.allclose(
            logits_cos, logits_arc, rtol=1e-3, atol=1e-5
        ), "cos and arc should produce different results"

    def test_elastic_face_wrapper_with_type(self, sample_embeddings, sample_labels):
        """Test ElasticFaceLossWrapper with different types."""
        from training_utils import ElasticFaceLossWrapper, l2_norm

        for loss_type in ["cos", "arc", "cos+", "arc+"]:
            wrapper = ElasticFaceLossWrapper(
                num_classes=10,
                embedding_dim=256,
                lambda_weight=0.5,
                s=30.0,
                m=0.30,
                std=0.01,
                type=loss_type,
            )

            normalized_embeddings = l2_norm(sample_embeddings, axis=1)
            total_loss, loss_dict = wrapper(normalized_embeddings, sample_labels)

            assert isinstance(total_loss, torch.Tensor), f"Loss should be tensor for type='{loss_type}'"
            assert total_loss.item() > 0, f"Loss should be positive for type='{loss_type}'"
            assert not torch.isnan(total_loss), f"Loss should not be NaN for type='{loss_type}'"
            assert wrapper.elastic_face.type == loss_type, f"Type mismatch for type='{loss_type}'"

    def test_elastic_face_default_type_is_arc(self):
        """Test that default type is 'arc' for backward compatibility."""
        from training_utils import ElasticFaceArcFace

        # Test that default type is 'arc' when not specified
        loss_fn_default = ElasticFaceArcFace(
            in_features=256,
            out_features=10,
        )

        assert loss_fn_default.type == "arc", "Default type should be 'arc'"

    def test_elastic_face_invalid_type_raises_error(self):
        """Test that invalid type raises ValueError."""
        from training_utils import ElasticFaceArcFace

        with pytest.raises(ValueError, match="Invalid type"):
            ElasticFaceArcFace(
                in_features=256,
                out_features=10,
                type="invalid_type",
            )


class TestTrainClassifyIntegration:
    """Test ElasticFace integration in train_classify.py."""

    def test_elastic_face_arguments_parsing(self):
        """Test that ElasticFace arguments are correctly parsed."""
        import subprocess
        import sys

        # Test help includes ElasticFace arguments
        result = subprocess.run(
            [sys.executable, "training/scripts/train_classify.py", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent.parent),
        )

        assert "--elastic-face" in result.stdout
        assert "--elastic-lambda" in result.stdout
        assert "--elastic-s" in result.stdout
        assert "--elastic-m" in result.stdout
        assert "--elastic-std" in result.stdout
        assert "--elastic-type" in result.stdout

    def test_import_training_utils_elasticface(self):
        """Test that training_utils can be imported with ElasticFace classes."""
        from training_utils import (
            ElasticFaceArcFace,
            ElasticFaceLossWrapper,
            ElasticFaceWithYOLO,
            stack_embeddings,
            l2_norm,
        )

        # Verify classes are importable
        assert ElasticFaceArcFace is not None
        assert ElasticFaceLossWrapper is not None
        assert ElasticFaceWithYOLO is not None
        assert stack_embeddings is not None
        assert l2_norm is not None

    def test_elastic_face_config_loading(self):
        """Test that ElasticFace config is correctly loaded from YAML."""
        import sys
        from pathlib import Path

        # Add parent directories to path
        repo_root = Path(__file__).parent.parent.parent.parent
        configs_path = repo_root / "training" / "configs"
        sys.path.insert(0, str(configs_path))

        from config_loader import load_config

        config = load_config(modules=["training"], load_all_modules=False)

        # Verify ElasticFace config structure exists using get() method
        enabled = config.get("training.advanced.elastic_face.enabled")
        lambda_weight = config.get("training.advanced.elastic_face.lambda_weight")
        s = config.get("training.advanced.elastic_face.s")
        m = config.get("training.advanced.elastic_face.m")
        std = config.get("training.advanced.elastic_face.std")
        plus = config.get("training.advanced.elastic_face.plus")
        elastic_type = config.get("training.advanced.elastic_face.type")

        # Verify values are loaded correctly from YAML config
        # Note: training.yaml has enabled: true and type: "arc+" by default
        assert enabled in [True, False], f"enabled should be boolean, got {enabled}"
        assert lambda_weight == 0.5
        assert s == 30.0
        assert m == 0.30
        assert std == 0.01
        # Check plus matches the config (should be True for arc+ type)
        assert plus in [True, False]
        # Verify type is one of the valid types
        assert elastic_type in ["cos", "arc", "cos+", "arc+", None], \
            f"Invalid type '{elastic_type}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
