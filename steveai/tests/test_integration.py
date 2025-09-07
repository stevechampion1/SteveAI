# SteveAI - Integration Tests

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import torch
import torch.nn as nn

from ..core.data_utils import (
    StudentDataset,
    TeacherLogitsLoader,
    create_dataloader,
    prepare_student_dataset,
)
from ..core.distillation_loss import DistillationLoss

# Import our custom modules
from ..core.student_training import (
    StudentTrainingConfig,
    evaluate_model,
    load_student_model_and_tokenizer,
    train_epoch,
)
from ..deployment.deploy import FlaskModelServer, ModelServer
from ..evaluation.benchmark import PerformanceBenchmark
from ..evaluation.evaluate import ModelEvaluator
from ..monitoring.monitor_training import TrainingMonitor
from ..optimization.optimize_model import ModelOptimizer
from ..utils.config_manager import ConfigManager


class TestStudentTrainingIntegration(unittest.TestCase):
    """Test student training integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = StudentTrainingConfig()

        # Create dummy teacher logits
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
        shutil.rmtree(self.temp_dir)

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_student_model_loading(self, mock_tokenizer, mock_model):
        """Test student model and tokenizer loading."""
        # Mock the model and tokenizer
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()

        model, tokenizer = load_student_model_and_tokenizer(self.config)

        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)
        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()

    def test_training_config_validation(self):
        """Test training configuration validation."""
        from student_training import validate_training_config

        # Test valid config
        is_valid = validate_training_config(self.config)
        self.assertTrue(is_valid)

        # Test invalid config
        self.config.LEARNING_RATE = -1.0
        is_valid = validate_training_config(self.config)
        self.assertFalse(is_valid)

    def test_data_preparation_pipeline(self):
        """Test data preparation pipeline."""
        # Create dummy data
        batch_size, seq_len, vocab_size = 4, 10, 1000
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = input_ids.clone()
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

        # Create dataset
        dataset = StudentDataset(input_ids, attention_mask, labels, teacher_logits)

        # Create dataloader
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=True)

        self.assertEqual(len(dataloader), 2)  # 4 samples / 2 batch_size = 2 batches

        # Test dataloader iteration
        for batch in dataloader:
            self.assertIn("input_ids", batch)
            self.assertIn("attention_mask", batch)
            self.assertIn("labels", batch)
            self.assertIn("teacher_logits", batch)
            break


class TestDistillationPipeline(unittest.TestCase):
    """Test complete distillation pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logits_dir = os.path.join(self.temp_dir, "teacher_logits")
        os.makedirs(self.logits_dir, exist_ok=True)

        # Create dummy teacher logits
        for i in range(2):
            logits = torch.randn(1, 5, 100)
            torch.save(
                {"teacher_logits": logits},
                os.path.join(self.logits_dir, f"teacher_logits_batch_{i}.pt"),
            )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_teacher_logits_loading(self):
        """Test teacher logits loading."""
        loader = TeacherLogitsLoader(self.logits_dir)

        # Test loading all logits
        all_logits = loader.load_all_logits()
        self.assertEqual(len(all_logits), 2)

        # Test loading single batch
        logits = loader.load_logits(0)
        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (1, 5, 100))

    def test_distillation_loss_computation(self):
        """Test distillation loss computation."""
        loss_fn = DistillationLoss()

        # Create dummy data
        batch_size, seq_len, vocab_size = 2, 5, 100
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Compute loss
        dist_loss, hard_loss = loss_fn(student_logits, teacher_logits, labels)

        self.assertIsInstance(dist_loss, torch.Tensor)
        self.assertIsInstance(hard_loss, torch.Tensor)
        self.assertGreater(dist_loss.item(), 0)
        self.assertGreater(hard_loss.item(), 0)

        # Test total loss
        total_loss = dist_loss + hard_loss
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertGreater(total_loss.item(), 0)


class TestEvaluationIntegration(unittest.TestCase):
    """Test evaluation integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.temp_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)

        # Create dummy model files
        dummy_model = Mock()
        dummy_tokenizer = Mock()

        # Save dummy model
        torch.save(
            dummy_model.state_dict(), os.path.join(self.model_dir, "pytorch_model.bin")
        )

        # Create dummy tokenizer config
        tokenizer_config = {"vocab_size": 1000}
        with open(os.path.join(self.model_dir, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_model_evaluator_initialization(self, mock_tokenizer, mock_model):
        """Test model evaluator initialization."""
        # Mock the model and tokenizer
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()

        evaluator = ModelEvaluator(self.model_dir, self.model_dir)

        self.assertIsNotNone(evaluator.model)
        self.assertIsNotNone(evaluator.tokenizer)
        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()

    def test_evaluation_metrics(self):
        """Test evaluation metrics computation."""
        # Create dummy dataloader
        batch_size, seq_len, vocab_size = 2, 5, 100
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = input_ids.clone()

        dataset = StudentDataset(
            input_ids,
            attention_mask,
            labels,
            torch.randn(batch_size, seq_len, vocab_size),
        )
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False)

        # Mock evaluator
        evaluator = Mock()
        evaluator.model = Mock()
        evaluator.tokenizer = Mock()
        evaluator.device = torch.device("cpu")

        # Mock model forward pass
        evaluator.model.return_value = Mock()
        evaluator.model.return_value.logits = torch.randn(
            batch_size, seq_len, vocab_size
        )

        # Test perplexity evaluation
        from evaluate import ModelEvaluator

        with patch.object(ModelEvaluator, "__init__", return_value=None):
            evaluator = ModelEvaluator.__new__(ModelEvaluator)
            evaluator.model = Mock()
            evaluator.tokenizer = Mock()
            evaluator.device = torch.device("cpu")

            evaluator.model.return_value = Mock()
            evaluator.model.return_value.logits = torch.randn(
                batch_size, seq_len, vocab_size
            )

            # Test perplexity computation
            perplexity = evaluator.evaluate_perplexity(dataloader, max_samples=2)
            self.assertIn("perplexity", perplexity)
            self.assertGreater(perplexity["perplexity"], 0)


class TestBenchmarkIntegration(unittest.TestCase):
    """Test benchmark integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.temp_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)

        # Create dummy model files
        dummy_model = Mock()
        torch.save(
            dummy_model.state_dict(), os.path.join(self.model_dir, "pytorch_model.bin")
        )

        # Create dummy tokenizer config
        tokenizer_config = {"vocab_size": 1000}
        with open(os.path.join(self.model_dir, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_performance_benchmark_initialization(self, mock_tokenizer, mock_model):
        """Test performance benchmark initialization."""
        # Mock the model and tokenizer
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()

        benchmark = PerformanceBenchmark(self.model_dir, self.model_dir)

        self.assertIsNotNone(benchmark.model)
        self.assertIsNotNone(benchmark.tokenizer)
        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()

    def test_benchmark_metrics(self):
        """Test benchmark metrics computation."""
        # Mock benchmark
        benchmark = Mock()
        benchmark.model = Mock()
        benchmark.tokenizer = Mock()
        benchmark.device = torch.device("cpu")

        # Mock model forward pass
        benchmark.model.return_value = Mock()
        benchmark.model.return_value.logits = torch.randn(2, 5, 100)

        # Test inference speed benchmark
        from benchmark import PerformanceBenchmark

        with patch.object(PerformanceBenchmark, "__init__", return_value=None):
            benchmark = PerformanceBenchmark.__new__(PerformanceBenchmark)
            benchmark.model = Mock()
            benchmark.tokenizer = Mock()
            benchmark.device = torch.device("cpu")

            benchmark.model.return_value = Mock()
            benchmark.model.return_value.logits = torch.randn(2, 5, 100)

            # Test inference speed computation
            speed_results = benchmark.benchmark_inference_speed(
                num_samples=2, sequence_length=5
            )
            self.assertIn("avg_inference_time", speed_results)
            self.assertIn("throughput", speed_results)
            self.assertGreater(speed_results["avg_inference_time"], 0)


class TestOptimizationIntegration(unittest.TestCase):
    """Test optimization integration."""

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
        self.optimizer = ModelOptimizer(self.model)

    def test_model_optimization_pipeline(self):
        """Test model optimization pipeline."""
        # Test quantization
        quantized_model = self.optimizer.quantize_model(quantization_type="dynamic")
        self.assertIsInstance(quantized_model, nn.Module)

        # Test pruning
        pruned_model = self.optimizer.prune_model(
            pruning_type="magnitude", pruning_ratio=0.1
        )
        self.assertIsInstance(pruned_model, nn.Module)

        # Test inference optimization
        optimized_model = self.optimizer.optimize_for_inference()
        self.assertIsInstance(optimized_model, nn.Module)

    def test_model_export(self):
        """Test model export functionality."""
        # Test ONNX export
        onnx_path = os.path.join(tempfile.mkdtemp(), "model.onnx")
        try:
            self.optimizer.export_to_onnx(onnx_path, input_shape=(1, 10))
            self.assertTrue(os.path.exists(onnx_path))
        finally:
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

        # Test TorchScript export
        torchscript_path = os.path.join(tempfile.mkdtemp(), "model.pt")
        try:
            self.optimizer.export_to_torchscript(torchscript_path, input_shape=(1, 10))
            self.assertTrue(os.path.exists(torchscript_path))
        finally:
            if os.path.exists(torchscript_path):
                os.remove(torchscript_path)


class TestDeploymentIntegration(unittest.TestCase):
    """Test deployment integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.temp_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)

        # Create dummy model files
        dummy_model = Mock()
        torch.save(
            dummy_model.state_dict(), os.path.join(self.model_dir, "pytorch_model.bin")
        )

        # Create dummy tokenizer config
        tokenizer_config = {"vocab_size": 1000}
        with open(os.path.join(self.model_dir, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_model_server_initialization(self, mock_tokenizer, mock_model):
        """Test model server initialization."""
        # Mock the model and tokenizer
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()

        server = ModelServer(self.model_dir, self.model_dir)

        self.assertIsNotNone(server.model)
        self.assertIsNotNone(server.tokenizer)
        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()

    def test_flask_server_creation(self):
        """Test Flask server creation."""
        # Mock model server
        mock_model_server = Mock()
        mock_model_server.generate_text.return_value = "Generated text"
        mock_model_server.get_model_info.return_value = {"model_id": "test-model"}

        # Create Flask server
        flask_server = FlaskModelServer(mock_model_server)

        self.assertIsNotNone(flask_server.app)
        self.assertIsNotNone(flask_server.model_server)

    def test_text_generation_endpoint(self):
        """Test text generation endpoint."""
        # Mock model server
        mock_model_server = Mock()
        mock_model_server.generate_text.return_value = "Generated text"
        mock_model_server.get_model_info.return_value = {"model_id": "test-model"}

        # Create Flask server
        flask_server = FlaskModelServer(mock_model_server)

        # Test generation endpoint
        with flask_server.app.test_client() as client:
            response = client.post("/generate", json={"prompt": "Test prompt"})
            self.assertEqual(response.status_code, 200)

            data = response.get_json()
            self.assertIn("generated_text", data)
            self.assertEqual(data["generated_text"], "Generated text")


class TestMonitoringIntegration(unittest.TestCase):
    """Test monitoring integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_training_monitor_initialization(self):
        """Test training monitor initialization."""
        monitor = TrainingMonitor(self.log_dir)

        self.assertIsNotNone(monitor.log_dir)
        self.assertIsNotNone(monitor.metrics)
        self.assertIsNotNone(monitor.system_metrics)

    def test_metrics_logging(self):
        """Test metrics logging."""
        monitor = TrainingMonitor(self.log_dir)

        # Test epoch metrics logging
        monitor.log_epoch_metrics(
            epoch=1, train_loss=0.5, val_loss=0.6, learning_rate=0.001
        )

        self.assertIn("epoch_metrics", monitor.metrics)
        self.assertEqual(len(monitor.metrics["epoch_metrics"]), 1)

        # Test system metrics logging
        monitor.log_system_metrics()

        self.assertIn("system_metrics", monitor.metrics)
        self.assertGreater(len(monitor.metrics["system_metrics"]), 0)

    def test_metrics_saving(self):
        """Test metrics saving."""
        monitor = TrainingMonitor(self.log_dir)

        # Log some metrics
        monitor.log_epoch_metrics(
            epoch=1, train_loss=0.5, val_loss=0.6, learning_rate=0.001
        )
        monitor.log_system_metrics()

        # Save metrics
        metrics_file = os.path.join(self.log_dir, "training_metrics.json")
        monitor.save_metrics(metrics_file)

        self.assertTrue(os.path.exists(metrics_file))

        # Load and verify metrics
        with open(metrics_file, "r") as f:
            saved_metrics = json.load(f)

        self.assertIn("epoch_metrics", saved_metrics)
        self.assertIn("system_metrics", saved_metrics)


class TestEndToEndPipeline(unittest.TestCase):
    """Test end-to-end pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logits_dir = os.path.join(self.temp_dir, "teacher_logits")
        os.makedirs(self.logits_dir, exist_ok=True)

        # Create dummy teacher logits
        for i in range(2):
            logits = torch.randn(1, 5, 100)
            torch.save(
                {"teacher_logits": logits},
                os.path.join(self.logits_dir, f"teacher_logits_batch_{i}.pt"),
            )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_complete_distillation_workflow(self):
        """Test complete distillation workflow."""
        # 1. Load teacher logits
        loader = TeacherLogitsLoader(self.logits_dir)
        all_logits = loader.load_all_logits()
        self.assertEqual(len(all_logits), 2)

        # 2. Prepare student dataset
        batch_size, seq_len, vocab_size = 2, 5, 100
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = input_ids.clone()
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

        dataset = StudentDataset(input_ids, attention_mask, labels, teacher_logits)
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False)

        # 3. Test distillation loss
        loss_fn = DistillationLoss()

        for batch in dataloader:
            student_logits = torch.randn(
                batch["input_ids"].shape[0], batch["input_ids"].shape[1], vocab_size
            )
            teacher_logits = batch["teacher_logits"]
            labels = batch["labels"]

            dist_loss, hard_loss = loss_fn(student_logits, teacher_logits, labels)

            self.assertIsInstance(dist_loss, torch.Tensor)
            self.assertIsInstance(hard_loss, torch.Tensor)
            self.assertGreater(dist_loss.item(), 0)
            self.assertGreater(hard_loss.item(), 0)
            break

        # 4. Test monitoring
        log_dir = os.path.join(self.temp_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        monitor = TrainingMonitor(log_dir)
        monitor.log_epoch_metrics(
            epoch=1, train_loss=0.5, val_loss=0.6, learning_rate=0.001
        )
        monitor.log_system_metrics()

        self.assertIn("epoch_metrics", monitor.metrics)
        self.assertIn("system_metrics", monitor.metrics)


def run_integration_tests():
    """Run all integration tests."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestStudentTrainingIntegration,
        TestDistillationPipeline,
        TestEvaluationIntegration,
        TestBenchmarkIntegration,
        TestOptimizationIntegration,
        TestDeploymentIntegration,
        TestMonitoringIntegration,
        TestEndToEndPipeline,
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
    success = run_integration_tests()

    if success:
        print("\n✅ All integration tests passed!")
    else:
        print("\n❌ Some integration tests failed!")
        exit(1)
