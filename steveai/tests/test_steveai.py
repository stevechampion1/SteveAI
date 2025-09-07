# SteveAI - Unit Tests

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
import torch.nn as nn

from ..core.data_utils import (
    StudentDataset,
    TeacherLogitsLoader,
    prepare_student_dataset,
)

# Import our custom modules
from ..core.distillation_loss import (
    AdvancedDistillationLoss,
    DistillationLoss,
    FocalDistillationLoss,
)
from ..utils.config_manager import ConfigManager
from ..utils.model_utils import count_trainable_parameters, get_model_summary
from ..utils.utils import check_disk_space, print_memory_usage, setup_logging


class TestDistillationLoss(unittest.TestCase):
    """Test distillation loss functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.seq_len = 10
        self.vocab_size = 1000

        # Create dummy data
        self.student_logits = torch.randn(
            self.batch_size, self.seq_len, self.vocab_size
        )
        self.teacher_logits = torch.randn(
            self.batch_size, self.seq_len, self.vocab_size
        )
        self.labels = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

    def test_distillation_loss_forward(self):
        """Test basic distillation loss forward pass."""
        loss_fn = DistillationLoss()

        dist_loss, hard_loss = loss_fn(
            self.student_logits, self.teacher_logits, self.labels
        )

        self.assertIsInstance(dist_loss, torch.Tensor)
        self.assertIsInstance(hard_loss, torch.Tensor)
        self.assertGreater(dist_loss.item(), 0)
        self.assertGreater(hard_loss.item(), 0)

    def test_distillation_loss_parameters(self):
        """Test distillation loss with different parameters."""
        loss_fn = DistillationLoss()

        # Test with different temperature
        dist_loss1, hard_loss1 = loss_fn(
            self.student_logits, self.teacher_logits, self.labels, temperature=2.0
        )

        dist_loss2, hard_loss2 = loss_fn(
            self.student_logits, self.teacher_logits, self.labels, temperature=4.0
        )

        # Different temperatures should produce different losses
        self.assertNotEqual(dist_loss1.item(), dist_loss2.item())

    def test_advanced_distillation_loss(self):
        """Test advanced distillation loss."""
        loss_fn = AdvancedDistillationLoss()

        # Create dummy hidden states and attention
        hidden_size = 512
        num_heads = 8
        student_hidden = torch.randn(self.batch_size, self.seq_len, hidden_size)
        teacher_hidden = torch.randn(self.batch_size, self.seq_len, hidden_size)
        student_attention = torch.randn(
            self.batch_size, num_heads, self.seq_len, self.seq_len
        )
        teacher_attention = torch.randn(
            self.batch_size, num_heads, self.seq_len, self.seq_len
        )

        dist_loss, hard_loss, hidden_loss, attention_loss = loss_fn(
            self.student_logits,
            self.teacher_logits,
            self.labels,
            student_hidden,
            teacher_hidden,
            student_attention,
            teacher_attention,
        )

        self.assertIsInstance(dist_loss, torch.Tensor)
        self.assertIsInstance(hard_loss, torch.Tensor)
        self.assertIsInstance(hidden_loss, torch.Tensor)
        self.assertIsInstance(attention_loss, torch.Tensor)

    def test_focal_distillation_loss(self):
        """Test focal distillation loss."""
        loss_fn = FocalDistillationLoss()

        dist_loss, hard_loss = loss_fn(
            self.student_logits, self.teacher_logits, self.labels
        )

        self.assertIsInstance(dist_loss, torch.Tensor)
        self.assertIsInstance(hard_loss, torch.Tensor)
        self.assertGreater(dist_loss.item(), 0)
        self.assertGreater(hard_loss.item(), 0)


class TestDataUtils(unittest.TestCase):
    """Test data utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logits_dir = os.path.join(self.temp_dir, "teacher_logits")
        os.makedirs(self.logits_dir, exist_ok=True)

        # Create dummy logits files
        for i in range(3):
            logits = torch.randn(2, 10, 1000)
            torch.save(
                {"teacher_logits": logits},
                os.path.join(self.logits_dir, f"teacher_logits_batch_{i}.pt"),
            )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_teacher_logits_loader(self):
        """Test TeacherLogitsLoader."""
        loader = TeacherLogitsLoader(self.logits_dir)

        # Test loading single batch
        logits = loader.load_logits(0)
        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (2, 10, 1000))

        # Test loading all logits
        all_logits = loader.load_all_logits()
        self.assertEqual(len(all_logits), 3)

        # Test getting logits info
        info = loader.get_logits_info()
        self.assertIn("num_batches", info)
        self.assertIn("logits_shape", info)

    def test_student_dataset(self):
        """Test StudentDataset."""
        batch_size, seq_len, vocab_size = 4, 10, 1000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = input_ids.clone()
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

        dataset = StudentDataset(input_ids, attention_mask, labels, teacher_logits)

        self.assertEqual(len(dataset), batch_size)

        # Test getting item
        item = dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("labels", item)
        self.assertIn("teacher_logits", item)

    def test_student_dataset_validation(self):
        """Test StudentDataset validation."""
        batch_size, seq_len, vocab_size = 4, 10, 1000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = input_ids.clone()
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

        dataset = StudentDataset(input_ids, attention_mask, labels, teacher_logits)

        # Test validation
        from ..core.data_utils import validate_dataset

        is_valid = validate_dataset(dataset)
        self.assertTrue(is_valid)


class TestConfigManager(unittest.TestCase):
    """Test configuration manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.yaml")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_config_manager_default(self):
        """Test ConfigManager with default configuration."""
        config = ConfigManager()

        # Test getting values
        teacher_model = config.get("model.teacher_model_id")
        self.assertIsNotNone(teacher_model)

        # Test setting values
        config.set("model.teacher_model_id", "test-model")
        updated_model = config.get("model.teacher_model_id")
        self.assertEqual(updated_model, "test-model")

    def test_config_manager_validation(self):
        """Test configuration validation."""
        config = ConfigManager()

        # Test valid configuration
        is_valid = config.validate()
        self.assertTrue(is_valid)

        # Test invalid configuration
        config.set("training.learning_rate", -1.0)
        is_valid = config.validate()
        self.assertFalse(is_valid)

    def test_config_manager_save_load(self):
        """Test saving and loading configuration."""
        config = ConfigManager()
        config.set("model.teacher_model_id", "test-model")

        # Save configuration
        config.save(self.config_file)
        self.assertTrue(os.path.exists(self.config_file))

        # Load configuration
        loaded_config = ConfigManager(self.config_file)
        loaded_model = loaded_config.get("model.teacher_model_id")
        self.assertEqual(loaded_model, "test-model")

    def test_config_manager_output_paths(self):
        """Test output paths generation."""
        config = ConfigManager()
        paths = config.get_output_paths()

        self.assertIn("base_dir", paths)
        self.assertIn("logits_dir", paths)
        self.assertIn("checkpoint_dir", paths)
        self.assertIn("final_model_dir", paths)


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_setup_logging(self):
        """Test logging setup."""
        # This should not raise an exception
        setup_logging(level="INFO")

    def test_format_time(self):
        """Test time formatting."""
        from ..utils.utils import format_time

        self.assertEqual(format_time(30), "30.00s")
        self.assertEqual(format_time(90), "1.50m")
        self.assertEqual(format_time(7200), "2.00h")

    def test_format_size(self):
        """Test size formatting."""
        from ..utils.utils import format_size

        self.assertEqual(format_size(1024), "1.00 KB")
        self.assertEqual(format_size(1024**2), "1.00 MB")
        self.assertEqual(format_size(1024**3), "1.00 GB")

    def test_safe_divide(self):
        """Test safe division."""
        from ..utils.utils import safe_divide

        self.assertEqual(safe_divide(10, 2), 5.0)
        self.assertEqual(safe_divide(10, 0, default=999), 999)

    def test_clamp(self):
        """Test value clamping."""
        from ..utils.utils import clamp

        self.assertEqual(clamp(5, 0, 10), 5)
        self.assertEqual(clamp(-5, 0, 10), 0)
        self.assertEqual(clamp(15, 0, 10), 10)

    def test_calculate_accuracy(self):
        """Test accuracy calculation."""
        from ..utils.utils import calculate_accuracy

        predictions = torch.tensor([1, 2, 3, 4])
        targets = torch.tensor([1, 2, 3, 4])

        accuracy = calculate_accuracy(predictions, targets)
        self.assertEqual(accuracy, 1.0)

        # Test with ignore index
        targets_with_ignore = torch.tensor([1, 2, -100, 4])
        accuracy = calculate_accuracy(predictions, targets_with_ignore)
        self.assertEqual(accuracy, 1.0)

    def test_calculate_perplexity(self):
        """Test perplexity calculation."""
        from ..utils.utils import calculate_perplexity

        loss = 1.0
        perplexity = calculate_perplexity(loss)
        expected = np.exp(1.0)
        self.assertAlmostEqual(perplexity, expected, places=5)


class TestModelUtils(unittest.TestCase):
    """Test model utility functions."""

    def setUp(self):
        """Set up test fixtures."""

        # Create a simple test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 5)
                self.linear2 = nn.Linear(5, 1)

            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = self.linear2(x)
                return x

        self.model = TestModel()

    def test_get_model_summary(self):
        """Test model summary generation."""
        summary = get_model_summary(self.model)

        self.assertIn("total_parameters", summary)
        self.assertIn("trainable_parameters", summary)
        self.assertIn("model_size_mb", summary)
        self.assertGreater(summary["total_parameters"], 0)

    def test_count_trainable_parameters(self):
        """Test trainable parameter counting."""
        count = count_trainable_parameters(self.model)
        self.assertGreater(count, 0)

        # Test after freezing some parameters
        from ..utils.model_utils import freeze_model_parameters

        freeze_model_parameters(self.model, ["linear1"])

        count_after_freeze = count_trainable_parameters(self.model)
        self.assertLess(count_after_freeze, count)

    def test_model_parameter_operations(self):
        """Test model parameter operations."""
        from ..utils.model_utils import (
            freeze_model_parameters,
            unfreeze_model_parameters,
        )

        # Test freezing
        freeze_model_parameters(self.model, ["linear1"])
        self.assertFalse(self.model.linear1.weight.requires_grad)
        self.assertTrue(self.model.linear2.weight.requires_grad)

        # Test unfreezing
        unfreeze_model_parameters(self.model, ["linear1"])
        self.assertTrue(self.model.linear1.weight.requires_grad)
        self.assertTrue(self.model.linear2.weight.requires_grad)


class TestIntegration(unittest.TestCase):
    """Test integration between components."""

    def test_distillation_pipeline(self):
        """Test complete distillation pipeline."""
        # Create dummy data
        batch_size, seq_len, vocab_size = 2, 5, 100
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Test distillation loss
        loss_fn = DistillationLoss()
        dist_loss, hard_loss = loss_fn(student_logits, teacher_logits, labels)

        self.assertIsInstance(dist_loss, torch.Tensor)
        self.assertIsInstance(hard_loss, torch.Tensor)

        # Test total loss
        total_loss = dist_loss + hard_loss
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertGreater(total_loss.item(), 0)

    def test_data_loading_pipeline(self):
        """Test data loading pipeline."""
        # Create temporary directory and files
        temp_dir = tempfile.mkdtemp()
        logits_dir = os.path.join(temp_dir, "teacher_logits")
        os.makedirs(logits_dir, exist_ok=True)

        try:
            # Create dummy logits files
            for i in range(2):
                logits = torch.randn(1, 5, 100)
                torch.save(
                    {"teacher_logits": logits},
                    os.path.join(logits_dir, f"teacher_logits_batch_{i}.pt"),
                )

            # Test loading
            loader = TeacherLogitsLoader(logits_dir)
            all_logits = loader.load_all_logits()

            self.assertEqual(len(all_logits), 2)
            self.assertIsInstance(all_logits[0], torch.Tensor)

        finally:
            import shutil

            shutil.rmtree(temp_dir)


class TestErrorHandling(unittest.TestCase):
    """Test error handling."""

    def test_distillation_loss_errors(self):
        """Test distillation loss error handling."""
        loss_fn = DistillationLoss()

        # Test with mismatched shapes
        student_logits = torch.randn(2, 5, 100)
        teacher_logits = torch.randn(2, 5, 200)  # Different vocab size
        labels = torch.randint(0, 100, (2, 5))

        with self.assertRaises(ValueError):
            loss_fn(student_logits, teacher_logits, labels)

    def test_data_utils_errors(self):
        """Test data utils error handling."""
        # Test with non-existent directory
        with self.assertRaises(FileNotFoundError):
            TeacherLogitsLoader("/non/existent/path")

        # Test with invalid batch index
        temp_dir = tempfile.mkdtemp()
        logits_dir = os.path.join(temp_dir, "teacher_logits")
        os.makedirs(logits_dir, exist_ok=True)

        try:
            loader = TeacherLogitsLoader(logits_dir)
            with self.assertRaises(IndexError):
                loader.load_logits(0)  # No files exist
        finally:
            import shutil

            shutil.rmtree(temp_dir)

    def test_config_manager_errors(self):
        """Test config manager error handling."""
        config = ConfigManager()

        # Test with invalid key
        value = config.get("non.existent.key", default="default")
        self.assertEqual(value, "default")

        # Test validation with invalid values
        config.set("training.learning_rate", -1.0)
        is_valid = config.validate()
        self.assertFalse(is_valid)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestDistillationLoss,
        TestDataUtils,
        TestConfigManager,
        TestUtils,
        TestModelUtils,
        TestIntegration,
        TestErrorHandling,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    # Setup logging for tests
    setup_logging(level="WARNING")

    # Run tests
    success = run_tests()

    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)
