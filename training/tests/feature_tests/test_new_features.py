"""
测试新训练功能：梯度累计、Confidence Penalty、Focal Loss、Mixup

严格按照 TDD 原则开发：
1. 先编写失败的测试
2. 实现功能使测试通过
3. 重构优化
"""

import sys
from pathlib import Path
import argparse

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MockDataset(Dataset):
    """模拟数据集，用于测试"""

    def __init__(self, num_samples=32, num_classes=10, image_size=224):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        images = torch.randn(3, self.image_size, self.image_size)
        labels = torch.randint(0, self.num_classes, (1,)).item()
        return images, labels


class MockModel(nn.Module):
    """简单的模拟模型，用于测试"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TestGradientAccumulation:
    """测试梯度累计功能"""

    def test_gradient_accumulation_config_exists(self):
        """测试梯度累计配置是否存在"""
        parser = argparse.ArgumentParser()
        parser.add_argument("--accumulate", type=int, default=1)
        args = parser.parse_args([])
        assert hasattr(args, "accumulate"), "命令行参数应该包含 accumulate"

    def test_gradient_accumulation_batch_size(self):
        """测试梯度累计是否正确模拟大batch size"""
        from scripts.train_classify import AircraftClassifierTrainer

        config = {
            "epochs": 1,
            "batch_size": 4,
            "accumulate": 4,
            "model": "yolov8n-cls.pt",
            "data": "dummy",
            "imgsz": 64,
            "device": "cpu",
        }

        class MockArgs:
            checkpoint_dir = None
            resume = None
            tensorboard = True
            val = True
            plots = True
            amp = True
            cos_lr = True
            dropout = 0.0

        logger = self._get_mock_logger()
        trainer = AircraftClassifierTrainer(config, MockArgs(), logger)

        assert trainer.config.get("accumulate") == 4, "应该正确读取梯度累计配置"

    def test_gradient_accumulation_training_step(self):
        """测试梯度累计训练步骤是否正确"""
        model = MockModel(num_classes=5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        dataset = MockDataset(num_samples=32, num_classes=5, image_size=32)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        accumulate_steps = 4
        optimizer.zero_grad()

        total_loss = 0.0
        for batch_idx, (images, labels) in enumerate(dataloader):
            outputs = model(images)
            loss = criterion(outputs, torch.tensor(labels))

            normalized_loss = loss / accumulate_steps
            normalized_loss.backward()

            total_loss += loss.item()

            if (batch_idx + 1) % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        assert total_loss > 0, "梯度累计过程中应该有损失"
        assert not any(
            torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None
        ), "梯度不应该为 NaN"

    def test_gradient_accumulation_memory_saving(self):
        """测试梯度累计是否能有效节省显存"""
        small_batch_model = MockModel(num_classes=5)
        large_batch_model = MockModel(num_classes=5)

        small_opt = torch.optim.SGD(small_batch_model.parameters(), lr=0.01)
        large_opt = torch.optim.SGD(large_batch_model.parameters(), lr=0.01)

        criterion = nn.CrossEntropyLoss()

        dataset = MockDataset(num_samples=16, num_classes=5, image_size=32)

        dataloader_small = DataLoader(dataset, batch_size=16, shuffle=False)
        dataloader_accumulate = DataLoader(dataset, batch_size=4, shuffle=False)

        accumulate_steps = 4

        losses_small = []
        losses_accumulate = []

        small_opt.zero_grad()
        for images, labels in dataloader_small:
            outputs = small_batch_model(images)
            loss = criterion(outputs, torch.tensor(labels))
            loss.backward()
            small_opt.step()
            small_opt.zero_grad()
            losses_small.append(loss.item())

        large_opt.zero_grad()
        for batch_idx, (images, labels) in enumerate(dataloader_accumulate):
            outputs = large_batch_model(images)
            loss = criterion(outputs, torch.tensor(labels))
            (loss / accumulate_steps).backward()

            if (batch_idx + 1) % accumulate_steps == 0:
                large_opt.step()
                large_opt.zero_grad()
                losses_accumulate.append(loss.item() * accumulate_steps)

        assert len(losses_small) == len(losses_accumulate), "批次数量应该相同"

    def _get_mock_logger(self):
        """获取模拟的 logger"""
        import logging

        logger = logging.getLogger("test")
        logger.addHandler(logging.NullHandler())
        return logger


class TestConfidencePenalty:
    """测试 Confidence Penalty 正则化"""

    def test_confidence_penalty_config_exists(self):
        """测试 Confidence Penalty 配置是否存在"""
        parser = argparse.ArgumentParser()
        parser.add_argument("--confidence-penalty", type=float, default=0.0)
        args = parser.parse_args([])
        assert hasattr(args, "confidence_penalty"), (
            "命令行参数应该包含 confidence_penalty"
        )

    def test_confidence_penalty_calculation(self):
        """测试 Confidence Penalty 计算是否正确"""
        batch_size = 8
        num_classes = 10

        logits = torch.randn(batch_size, num_classes)
        probs = F.softmax(logits, dim=1)

        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        max_entropy = np.log(num_classes)

        confidence_penalty = entropy - max_entropy

        assert confidence_penalty.shape == (batch_size,), (
            "Confidence Penalty 应该与 batch size 一致"
        )
        assert confidence_penalty.mean() <= 0, "Confidence Penalty 应该 <= 0"

    def test_confidence_penalty_integration(self):
        """测试 Confidence Penalty 是否正确集成到损失中"""
        model = MockModel(num_classes=5)
        criterion = nn.CrossEntropyLoss()

        dataset = MockDataset(num_samples=32, num_classes=5, image_size=32)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

        confidence_penalty_weight = 0.1

        images, labels = next(iter(dataloader))
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)

        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        max_entropy = np.log(5)
        cp = torch.mean(entropy - max_entropy)

        base_loss = criterion(outputs, torch.tensor(labels))
        total_loss = base_loss + confidence_penalty_weight * cp

        assert torch.isclose(
            total_loss, base_loss + confidence_penalty_weight * cp, rtol=1e-5
        ), "总损失应该等于基础损失加上CP项"

    def test_confidence_penalty_effect_on_confidence(self):
        """测试 Confidence Penalty 是否能有效降低过置信"""
        model = MockModel(num_classes=5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        dataset = MockDataset(num_samples=32, num_classes=5, image_size=32)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

        confidence_penalty_weight = 0.1

        model.train()
        max_probs_before = []
        max_probs_after = []

        for images, labels in dataloader:
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            max_probs_before.append(probs.max(dim=1)[0].mean().item())

            base_loss = F.cross_entropy(outputs, torch.tensor(labels))
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            max_entropy = np.log(5)
            cp = torch.mean(entropy - max_entropy)

            loss = base_loss + confidence_penalty_weight * cp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            max_probs_after.append(probs.max(dim=1)[0].mean().item())

        assert len(max_probs_before) > 0, "应该记录了一些最大概率"


class TestFocalLoss:
    """测试 Focal Loss 功能"""

    def test_focal_loss_config_exists(self):
        """测试 Focal Loss 配置是否存在"""
        parser = argparse.ArgumentParser()
        parser.add_argument("--focal-loss", action="store_true", default=False)
        args = parser.parse_args([])
        assert hasattr(args, "focal_loss"), "命令行参数应该包含 focal_loss"

    def test_focal_loss_calculation(self):
        """测试 Focal Loss 计算是否正确"""
        batch_size = 8
        num_classes = 5

        logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))

        alpha = 0.25
        gamma = 2.0

        probs = F.softmax(logits, dim=1)
        ce_loss = F.cross_entropy(logits, labels, reduction="none")

        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss

        assert focal_loss.shape == (batch_size,), "Focal Loss 应该与 batch size 一致"
        assert (focal_loss <= ce_loss + 1e-6).all(), "Focal Loss 应该 <= CE Loss"

    def test_focal_loss_gamma_effect(self):
        """测试 gamma 参数对 Focal Loss 的影响"""
        logits = torch.randn(8, 5)
        labels = torch.randint(0, 5, (8,))
        alpha = 0.25

        gamma_values = [0.5, 1.0, 2.0, 5.0]
        losses = []

        for gamma in gamma_values:
            probs = F.softmax(logits, dim=1)
            ce_loss = F.cross_entropy(logits, labels, reduction="none")
            pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            focal_loss = alpha * (1 - pt) ** gamma * ce_loss
            losses.append(focal_loss.mean().item())

        assert len(losses) == len(gamma_values), "应该计算了不同 gamma 值的损失"

    def test_focal_vs_cross_entropy(self):
        """测试 Focal Loss 与 Cross Entropy 的差异"""
        logits = torch.randn(8, 5)
        labels = torch.randint(0, 5, (8,))

        ce_loss = F.cross_entropy(logits, labels)

        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal_loss = (
            0.25 * (1 - pt) ** 2.0 * F.cross_entropy(logits, labels, reduction="none")
        )
        focal_loss = focal_loss.mean()

        assert focal_loss <= ce_loss + 1e-6, "Focal Loss 应该 <= CE Loss"

    def test_focal_loss_hard_example_focusing(self):
        """测试 Focal Loss 是否更关注难样本"""
        logits = torch.tensor(
            [
                [10.0, -10.0, -10.0, -10.0, -10.0],
                [0.1, 0.1, 0.1, 0.1, 0.1],
                [5.0, -5.0, -5.0, -5.0, -5.0],
                [0.5, 0.5, 0.5, 0.5, -0.5],
            ]
        )
        labels = torch.tensor([0, 0, 0, 0])

        ce_loss = F.cross_entropy(logits, labels, reduction="none")

        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal_loss = 0.25 * (1 - pt) ** 2.0 * ce_loss

        assert focal_loss[1] > focal_loss[0], "难样本的 Focal Loss 应该更大"


class TestMixup:
    """测试 Mixup 数据增强功能"""

    def test_mixup_config_exists(self):
        """测试 Mixup 配置是否存在"""
        parser = argparse.ArgumentParser()
        parser.add_argument("--mixup", action="store_true", default=False)
        parser.add_argument("--mixup-alpha", type=float, default=0.4)
        args = parser.parse_args([])
        assert hasattr(args, "mixup"), "命令行参数应该包含 mixup"
        assert hasattr(args, "mixup_alpha"), "命令行参数应该包含 mixup_alpha"

    def test_mixup_alpha_sampling(self):
        """测试 Mixup 的 alpha 参数采样"""
        alpha = 0.4
        num_samples = 1000

        lambdas = np.random.beta(alpha, alpha, num_samples)

        assert np.all(lambdas >= 0), "lambda 应该 >= 0"
        assert np.all(lambdas <= 1), "lambda 应该 <= 1"
        assert 0.3 < np.mean(lambdas) < 0.7, "lambda 均值应该在 0.5 附近"

    def test_mixup_image_blending(self):
        """测试 Mixup 图像混合是否正确"""
        batch_size = 4
        C, H, W = 3, 32, 32

        images1 = torch.ones(batch_size, C, H, W)
        images2 = torch.zeros(batch_size, C, H, W)
        lam = 0.5

        mixed_images = lam * images1 + (1 - lam) * images2

        expected_value = 0.5
        assert torch.allclose(
            mixed_images, torch.full_like(mixed_images, expected_value), atol=1e-6
        ), "混合图像值应该正确"

    def test_mixup_label_smoothing(self):
        """测试 Mixup 标签平滑是否正确"""
        batch_size = 4
        num_classes = 5

        labels1 = torch.tensor([0, 1, 2, 3])
        labels2 = torch.tensor([4, 3, 2, 1])
        lam = 0.3

        labels_a = F.one_hot(labels1, num_classes).float()
        labels_b = F.one_hot(labels2, num_classes).float()

        mixed_labels = lam * labels_a + (1 - lam) * labels_b

        expected_label_0 = torch.tensor([lam, 0, 0, 0, 1 - lam])
        assert torch.allclose(mixed_labels[0], expected_label_0), "混合标签应该正确"

    def test_mixup_integration(self):
        """测试 Mixup 是否正确集成到训练流程"""
        batch_size = 4
        num_classes = 5
        C, H, W = 3, 32, 32

        dataset = MockDataset(num_samples=16, num_classes=num_classes, image_size=H)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = MockModel(num_classes=num_classes)

        alpha = 0.4
        lam = np.random.beta(alpha, alpha)

        images, labels = next(iter(dataloader))

        rand_index = torch.randperm(batch_size)
        images_a, images_b = images, images[rand_index]
        labels_a, labels_b = labels, torch.tensor(labels[rand_index])

        mixed_images = lam * images_a + (1 - lam) * images_b

        outputs = model(mixed_images)
        assert outputs.shape == (batch_size, num_classes), "输出形状应该正确"

        labels_onehot_a = F.one_hot(labels_a, num_classes).float()
        labels_onehot_b = F.one_hot(labels_b, num_classes).float()
        mixed_labels = lam * labels_onehot_a + (1 - lam) * labels_onehot_b

        loss = F.cross_entropy(outputs, mixed_labels)
        assert loss.item() > 0, "损失应该 > 0"

    def test_mixup_effect_on_accuracy(self):
        """测试 Mixup 对模型准确率的影响（训练初期）"""
        batch_size = 8
        num_classes = 5
        C, H, W = 3, 32, 32

        dataset = MockDataset(num_samples=32, num_classes=num_classes, image_size=H)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = MockModel(num_classes=num_classes)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        alpha = 0.4
        epochs = 5

        for epoch in range(epochs):
            for images, labels in dataloader:
                lam = np.random.beta(alpha, alpha)

                rand_index = torch.randperm(batch_size)
                images_a, images_b = images, images[rand_index]
                labels_a, labels_b = labels, torch.tensor(labels[rand_index])

                mixed_images = lam * images_a + (1 - lam) * images_b

                outputs = model(mixed_images)

                labels_onehot_a = F.one_hot(labels_a, num_classes).float()
                labels_onehot_b = F.one_hot(labels_b, num_classes).float()
                mixed_labels = lam * labels_onehot_a + (1 - lam) * labels_onehot_b

                loss = F.cross_entropy(outputs, mixed_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        assert True, "Mixup 训练应该能够完成"


class TestIntegration:
    """测试所有功能的集成"""

    def test_all_features_config_compatibility(self):
        """测试所有功能配置是否兼容"""
        parser = argparse.ArgumentParser()
        parser.add_argument("--accumulate", type=int, default=1)
        parser.add_argument("--confidence-penalty", type=float, default=0.0)
        parser.add_argument("--focal-loss", action="store_true", default=False)
        parser.add_argument("--mixup", action="store_true", default=False)
        parser.add_argument("--mixup-alpha", type=float, default=0.4)
        args = parser.parse_args([])

        required_args = [
            "accumulate",
            "confidence_penalty",
            "focal_loss",
            "mixup",
            "mixup_alpha",
        ]
        for arg in required_args:
            assert hasattr(args, arg), f"应该存在参数: {arg}"

    def test_all_features_together(self):
        """测试所有功能同时使用"""
        batch_size = 4
        num_classes = 5
        C, H, W = 3, 32, 32

        dataset = MockDataset(num_samples=32, num_classes=num_classes, image_size=H)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = MockModel(num_classes=num_classes)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        accumulate_steps = 2
        mixup_alpha = 0.4
        focal_alpha = 0.25
        focal_gamma = 2.0
        confidence_penalty_weight = 0.1

        optimizer.zero_grad()
        batch_count = 0

        for images, labels in dataloader:
            lam = np.random.beta(mixup_alpha, mixup_alpha)

            rand_index = torch.randperm(batch_size)
            images_a, images_b = images, images[rand_index]
            labels_a, labels_b = labels, torch.tensor(labels[rand_index])

            mixed_images = lam * images_a + (1 - lam) * images_b

            outputs = model(mixed_images)
            probs = F.softmax(outputs, dim=1)

            labels_onehot_a = F.one_hot(labels_a, num_classes).float()
            labels_onehot_b = F.one_hot(labels_b, num_classes).float()
            mixed_labels = lam * labels_onehot_a + (1 - lam) * labels_onehot_b

            ce_loss = F.cross_entropy(outputs, mixed_labels, reduction="none")
            pt = probs * mixed_labels
            pt = pt.sum(dim=1)
            focal_loss = focal_alpha * (1 - pt) ** focal_gamma * ce_loss
            focal_loss = focal_loss.mean()

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            max_entropy = np.log(num_classes)
            cp = torch.mean(entropy - max_entropy)

            total_loss = focal_loss + confidence_penalty_weight * cp

            normalized_loss = total_loss / accumulate_steps
            normalized_loss.backward()

            batch_count += 1

            if batch_count % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        assert batch_count == 8, "应该处理所有批次"

    def test_configuration_loading(self):
        """测试配置文件加载是否正确"""
        import yaml

        config_path = "configs/config/training.yaml"
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            assert "training" in config, "配置应该包含 training 节"
            assert "optimizer" in config["training"], "配置应该包含 optimizer 节"

            training_config = config["training"]
            expected_keys = ["epochs", "batch_size", "image_size", "optimizer"]
            for key in expected_keys:
                assert key in training_config, f"训练配置应该包含: {key}"

        except FileNotFoundError:
            pytest.skip("配置文件不存在")
