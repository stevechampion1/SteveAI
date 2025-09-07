# SteveAI - Performance Tests

import gc
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import psutil
import torch
import torch.nn as nn

from ..core.data_utils import StudentDataset, TeacherLogitsLoader, create_dataloader

# Import our custom modules
from ..core.distillation_loss import (
    AdvancedDistillationLoss,
    DistillationLoss,
    FocalDistillationLoss,
)
from ..evaluation.benchmark import PerformanceBenchmark
from ..optimization.optimize_model import ModelOptimizer
from ..utils.config_manager import ConfigManager
from ..utils.model_utils import count_trainable_parameters, get_model_summary
from ..utils.utils import format_size, format_time, get_system_info, print_memory_usage


class TestMemoryPerformance(unittest.TestCase):
    """Test memory performance and usage."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logits_dir = os.path.join(self.temp_dir, "teacher_logits")
        os.makedirs(self.logits_dir, exist_ok=True)

        # Create dummy teacher logits
        for i in range(5):
            logits = torch.randn(2, 10, 1000)
            torch.save(
                {"teacher_logits": logits},
                os.path.join(self.logits_dir, f"teacher_logits_batch_{i}.pt"),
            )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)
        gc.collect()

    def test_memory_usage_during_logits_loading(self):
        """Test memory usage during logits loading."""
        # Get initial memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Load logits
        loader = TeacherLogitsLoader(self.logits_dir)
        all_logits = loader.load_all_logits()

        # Get memory usage after loading
        after_loading_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Calculate memory increase
        memory_increase = after_loading_memory - initial_memory

        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"After loading memory: {after_loading_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")

        # Memory increase should be reasonable (less than 100MB for this test)
        self.assertLess(memory_increase, 100)

        # Clean up
        del all_logits
        gc.collect()

    def test_memory_usage_during_training(self):
        """Test memory usage during training simulation."""

        # Create a simple model for testing
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(1000, 512)
                self.linear2 = nn.Linear(512, 1000)

            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = self.linear2(x)
                return x

        model = TestModel()
        loss_fn = DistillationLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Get initial memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Simulate training steps
        batch_size, seq_len, vocab_size = 4, 10, 1000
        for step in range(10):
            # Create dummy data
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            student_logits = torch.randn(batch_size, seq_len, vocab_size)
            teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
            labels = input_ids.clone()

            # Forward pass
            optimizer.zero_grad()
            dist_loss, hard_loss = loss_fn(student_logits, teacher_logits, labels)
            total_loss = dist_loss + hard_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Clean up
            del input_ids, student_logits, teacher_logits, labels
            del dist_loss, hard_loss, total_loss

            if step % 3 == 0:
                gc.collect()

        # Get memory usage after training
        after_training_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Calculate memory increase
        memory_increase = after_training_memory - initial_memory

        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"After training memory: {after_training_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")

        # Memory increase should be reasonable
        self.assertLess(memory_increase, 200)

        # Clean up
        del model, loss_fn, optimizer
        gc.collect()

    def test_memory_cleanup_effectiveness(self):
        """Test memory cleanup effectiveness."""
        # Get initial memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Create and use large tensors
        large_tensors = []
        for i in range(10):
            tensor = torch.randn(100, 100, 100)
            large_tensors.append(tensor)

        # Get memory usage with large tensors
        with_tensors_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Clean up tensors
        del large_tensors
        gc.collect()

        # Get memory usage after cleanup
        after_cleanup_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"With tensors memory: {with_tensors_memory:.2f} MB")
        print(f"After cleanup memory: {after_cleanup_memory:.2f} MB")

        # Memory should be cleaned up effectively
        memory_recovered = with_tensors_memory - after_cleanup_memory
        self.assertGreater(memory_recovered, 0)


class TestSpeedPerformance(unittest.TestCase):
    """Test speed performance."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logits_dir = os.path.join(self.temp_dir, "teacher_logits")
        os.makedirs(self.logits_dir, exist_ok=True)

        # Create dummy teacher logits
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

    def test_logits_loading_speed(self):
        """Test logits loading speed."""
        loader = TeacherLogitsLoader(self.logits_dir)

        # Test loading all logits
        start_time = time.time()
        all_logits = loader.load_all_logits()
        end_time = time.time()

        loading_time = end_time - start_time
        print(f"Loading time: {loading_time:.4f} seconds")

        # Loading should be fast (less than 1 second for this test)
        self.assertLess(loading_time, 1.0)

        # Test loading single batch
        start_time = time.time()
        single_logits = loader.load_logits(0)
        end_time = time.time()

        single_loading_time = end_time - start_time
        print(f"Single batch loading time: {single_loading_time:.4f} seconds")

        # Single batch loading should be very fast
        self.assertLess(single_loading_time, 0.1)

    def test_distillation_loss_computation_speed(self):
        """Test distillation loss computation speed."""
        loss_fn = DistillationLoss()

        # Create dummy data
        batch_size, seq_len, vocab_size = 8, 20, 1000
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Warm up
        for _ in range(5):
            _ = loss_fn(student_logits, teacher_logits, labels)

        # Test computation speed
        start_time = time.time()
        for _ in range(100):
            dist_loss, hard_loss = loss_fn(student_logits, teacher_logits, labels)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / 100

        print(f"Total time for 100 computations: {total_time:.4f} seconds")
        print(f"Average time per computation: {avg_time:.6f} seconds")

        # Each computation should be fast (less than 0.01 seconds)
        self.assertLess(avg_time, 0.01)

    def test_dataset_creation_speed(self):
        """Test dataset creation speed."""
        batch_size, seq_len, vocab_size = 100, 20, 1000

        # Test dataset creation speed
        start_time = time.time()

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = input_ids.clone()
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

        dataset = StudentDataset(input_ids, attention_mask, labels, teacher_logits)

        end_time = time.time()

        creation_time = end_time - start_time
        print(f"Dataset creation time: {creation_time:.4f} seconds")

        # Dataset creation should be fast
        self.assertLess(creation_time, 0.1)

    def test_dataloader_iteration_speed(self):
        """Test dataloader iteration speed."""
        batch_size, seq_len, vocab_size = 100, 20, 1000

        # Create dataset
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = input_ids.clone()
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

        dataset = StudentDataset(input_ids, attention_mask, labels, teacher_logits)
        dataloader = create_dataloader(dataset, batch_size=10, shuffle=False)

        # Test iteration speed
        start_time = time.time()

        for batch in dataloader:
            # Simulate some processing
            _ = batch["input_ids"].sum()

        end_time = time.time()

        iteration_time = end_time - start_time
        print(f"Dataloader iteration time: {iteration_time:.4f} seconds")

        # Iteration should be fast
        self.assertLess(iteration_time, 0.5)


class TestScalabilityPerformance(unittest.TestCase):
    """Test scalability performance."""

    def test_batch_size_scalability(self):
        """Test performance with different batch sizes."""
        loss_fn = DistillationLoss()
        seq_len, vocab_size = 20, 1000

        batch_sizes = [1, 4, 8, 16, 32]
        times = []

        for batch_size in batch_sizes:
            # Create dummy data
            student_logits = torch.randn(batch_size, seq_len, vocab_size)
            teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len))

            # Measure computation time
            start_time = time.time()
            for _ in range(10):
                dist_loss, hard_loss = loss_fn(student_logits, teacher_logits, labels)
            end_time = time.time()

            avg_time = (end_time - start_time) / 10
            times.append(avg_time)

            print(f"Batch size {batch_size}: {avg_time:.6f} seconds")

        # Performance should scale reasonably with batch size
        # Larger batches should be more efficient per sample
        for i in range(1, len(batch_sizes)):
            prev_efficiency = batch_sizes[i - 1] / times[i - 1]
            curr_efficiency = batch_sizes[i] / times[i]

            # Efficiency should not degrade too much
            efficiency_ratio = curr_efficiency / prev_efficiency
            self.assertGreater(efficiency_ratio, 0.5)

    def test_sequence_length_scalability(self):
        """Test performance with different sequence lengths."""
        loss_fn = DistillationLoss()
        batch_size, vocab_size = 8, 1000

        seq_lengths = [10, 20, 50, 100, 200]
        times = []

        for seq_len in seq_lengths:
            # Create dummy data
            student_logits = torch.randn(batch_size, seq_len, vocab_size)
            teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len))

            # Measure computation time
            start_time = time.time()
            for _ in range(10):
                dist_loss, hard_loss = loss_fn(student_logits, teacher_logits, labels)
            end_time = time.time()

            avg_time = (end_time - start_time) / 10
            times.append(avg_time)

            print(f"Sequence length {seq_len}: {avg_time:.6f} seconds")

        # Performance should scale reasonably with sequence length
        # Longer sequences should take more time, but not exponentially
        for i in range(1, len(seq_lengths)):
            prev_efficiency = seq_lengths[i - 1] / times[i - 1]
            curr_efficiency = seq_lengths[i] / times[i]

            # Efficiency should not degrade too much
            efficiency_ratio = curr_efficiency / prev_efficiency
            self.assertGreater(efficiency_ratio, 0.3)

    def test_vocab_size_scalability(self):
        """Test performance with different vocabulary sizes."""
        loss_fn = DistillationLoss()
        batch_size, seq_len = 8, 20

        vocab_sizes = [100, 500, 1000, 2000, 5000]
        times = []

        for vocab_size in vocab_sizes:
            # Create dummy data
            student_logits = torch.randn(batch_size, seq_len, vocab_size)
            teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len))

            # Measure computation time
            start_time = time.time()
            for _ in range(10):
                dist_loss, hard_loss = loss_fn(student_logits, teacher_logits, labels)
            end_time = time.time()

            avg_time = (end_time - start_time) / 10
            times.append(avg_time)

            print(f"Vocabulary size {vocab_size}: {avg_time:.6f} seconds")

        # Performance should scale reasonably with vocabulary size
        # Larger vocabularies should take more time, but not exponentially
        for i in range(1, len(vocab_sizes)):
            prev_efficiency = vocab_sizes[i - 1] / times[i - 1]
            curr_efficiency = vocab_sizes[i] / times[i]

            # Efficiency should not degrade too much
            efficiency_ratio = curr_efficiency / prev_efficiency
            self.assertGreater(efficiency_ratio, 0.2)


class TestOptimizationPerformance(unittest.TestCase):
    """Test optimization performance."""

    def setUp(self):
        """Set up test fixtures."""

        # Create a test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(1000, 512)
                self.linear2 = nn.Linear(512, 1000)

            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = self.linear2(x)
                return x

        self.model = TestModel()
        self.optimizer = ModelOptimizer(self.model)

    def tearDown(self):
        """Clean up test fixtures."""
        gc.collect()

    def test_quantization_performance(self):
        """Test quantization performance."""
        # Test dynamic quantization
        start_time = time.time()
        quantized_model = self.optimizer.quantize_model(quantization_type="dynamic")
        end_time = time.time()

        quantization_time = end_time - start_time
        print(f"Dynamic quantization time: {quantization_time:.4f} seconds")

        # Quantization should be fast
        self.assertLess(quantization_time, 1.0)

        # Test model size reduction
        original_size = get_model_summary(self.model)["model_size_mb"]
        quantized_size = get_model_summary(quantized_model)["model_size_mb"]

        print(f"Original model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quantized_size:.2f} MB")

        # Quantized model should be smaller
        self.assertLess(quantized_size, original_size)

    def test_pruning_performance(self):
        """Test pruning performance."""
        # Test magnitude pruning
        start_time = time.time()
        pruned_model = self.optimizer.prune_model(
            pruning_type="magnitude", pruning_ratio=0.1
        )
        end_time = time.time()

        pruning_time = end_time - start_time
        print(f"Magnitude pruning time: {pruning_time:.4f} seconds")

        # Pruning should be fast
        self.assertLess(pruning_time, 1.0)

        # Test parameter reduction
        original_params = count_trainable_parameters(self.model)
        pruned_params = count_trainable_parameters(pruned_model)

        print(f"Original parameters: {original_params}")
        print(f"Pruned parameters: {pruned_params}")

        # Pruned model should have fewer parameters
        self.assertLess(pruned_params, original_params)

    def test_inference_optimization_performance(self):
        """Test inference optimization performance."""
        # Test inference optimization
        start_time = time.time()
        optimized_model = self.optimizer.optimize_for_inference()
        end_time = time.time()

        optimization_time = end_time - start_time
        print(f"Inference optimization time: {optimization_time:.4f} seconds")

        # Optimization should be fast
        self.assertLess(optimization_time, 1.0)

        # Test inference speed
        input_tensor = torch.randn(1, 1000)

        # Warm up
        for _ in range(5):
            _ = optimized_model(input_tensor)

        # Measure inference time
        start_time = time.time()
        for _ in range(100):
            _ = optimized_model(input_tensor)
        end_time = time.time()

        avg_inference_time = (end_time - start_time) / 100
        print(f"Average inference time: {avg_inference_time:.6f} seconds")

        # Inference should be fast
        self.assertLess(avg_inference_time, 0.01)


class TestSystemPerformance(unittest.TestCase):
    """Test system performance."""

    def test_system_info_collection(self):
        """Test system info collection performance."""
        start_time = time.time()
        system_info = get_system_info()
        end_time = time.time()

        collection_time = end_time - start_time
        print(f"System info collection time: {collection_time:.4f} seconds")

        # Collection should be fast
        self.assertLess(collection_time, 1.0)

        # Should contain expected information
        self.assertIn("cpu_count", system_info)
        self.assertIn("memory_total", system_info)
        self.assertIn("memory_available", system_info)
        self.assertIn("disk_total", system_info)
        self.assertIn("disk_available", system_info)

    def test_memory_monitoring_performance(self):
        """Test memory monitoring performance."""
        start_time = time.time()
        print_memory_usage("Test step")
        end_time = time.time()

        monitoring_time = end_time - start_time
        print(f"Memory monitoring time: {monitoring_time:.4f} seconds")

        # Monitoring should be fast
        self.assertLess(monitoring_time, 0.1)

    def test_config_management_performance(self):
        """Test config management performance."""
        config = ConfigManager()

        # Test getting values
        start_time = time.time()
        for _ in range(1000):
            _ = config.get("model.teacher_model_id")
        end_time = time.time()

        get_time = (end_time - start_time) / 1000
        print(f"Average config get time: {get_time:.6f} seconds")

        # Getting values should be fast
        self.assertLess(get_time, 0.001)

        # Test setting values
        start_time = time.time()
        for i in range(1000):
            config.set(f"test.key_{i}", f"value_{i}")
        end_time = time.time()

        set_time = (end_time - start_time) / 1000
        print(f"Average config set time: {set_time:.6f} seconds")

        # Setting values should be fast
        self.assertLess(set_time, 0.001)


def run_performance_tests():
    """Run all performance tests."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestMemoryPerformance,
        TestSpeedPerformance,
        TestScalabilityPerformance,
        TestOptimizationPerformance,
        TestSystemPerformance,
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
    from utils import setup_logging

    setup_logging(level="WARNING")

    # Run tests
    success = run_performance_tests()

    if success:
        print("\n✅ All performance tests passed!")
    else:
        print("\n❌ Some performance tests failed!")
        exit(1)
