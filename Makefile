# SteveAI - Makefile

.PHONY: help install install-dev test test-unit test-integration test-performance lint format clean build docs serve-docs

# Default target
help:
	@echo "SteveAI - Available commands:"
	@echo ""
	@echo "Installation:"
	@echo "  install          Install SteveAI in production mode"
	@echo "  install-dev      Install SteveAI in development mode"
	@echo "  install-docs     Install documentation dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-performance Run performance tests only"
	@echo "  test-quick       Run quick tests (unit tests only)"
	@echo "  test-ci          Run CI tests (unit + integration)"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run type checking with mypy"
	@echo ""
	@echo "Documentation:"
	@echo "  docs             Build documentation"
	@echo "  serve-docs       Serve documentation locally"
	@echo ""
	@echo "Development:"
	@echo "  clean            Clean build artifacts"
	@echo "  build            Build package"
	@echo "  pre-commit       Run pre-commit hooks"
	@echo ""
	@echo "Examples:"
	@echo "  example-basic    Run basic usage example"
	@echo "  example-advanced Run advanced usage example"
	@echo ""
	@echo "Quick Start:"
	@echo "  quickstart       Run complete quickstart pipeline"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

install-docs:
	pip install -e ".[docs]"

# Testing
test:
	python run_tests.py --type all

test-unit:
	python run_tests.py --type unit

test-integration:
	python run_tests.py --type integration

test-performance:
	python run_tests.py --type performance

test-quick:
	python run_tests.py --type quick

test-ci:
	python run_tests.py --type ci

# Code Quality
lint:
	flake8 .
	mypy .
	black --check .
	isort --check-only .

format:
	black .
	isort .

type-check:
	mypy .

# Documentation
docs:
	cd docs && make html

serve-docs:
	cd docs/_build/html && python -m http.server 8000

# Development
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:
	python setup.py sdist bdist_wheel

pre-commit:
	pre-commit run --all-files

# Examples
example-basic:
	python examples/basic_usage.py

example-advanced:
	python examples/advanced_usage.py

# Quick Start Pipeline
quickstart: install-dev
	@echo "=== SteveAI Quick Start Pipeline ==="
	@echo "1. Running teacher inference..."
	python teacher_inference_gpu.py
	@echo "2. Training student model..."
	python student_training.py --teacher_logits_path ./output/SteveAI_Teacher_Inference/teacher_logits_float16 --output_dir ./output/student_model --num_epochs 1 --batch_size 4
	@echo "3. Evaluating model..."
	python evaluate.py --model_path ./output/student_model/final_model --tokenizer_path ./output/student_model/final_model --teacher_logits_path ./output/SteveAI_Teacher_Inference/teacher_logits_float16 --output_dir ./output/evaluation_results
	@echo "4. Running benchmark..."
	python benchmark.py --model_path ./output/student_model/final_model --tokenizer_path ./output/student_model/final_model --teacher_logits_path ./output/SteveAI_Teacher_Inference/teacher_logits_float16 --output_dir ./output/benchmark_results
	@echo "=== Quick Start Pipeline Completed ==="

# Docker commands
docker-build:
	docker build -t steveai:latest .

docker-run:
	docker run --gpus all -it steveai:latest

# Environment setup
setup-env:
	python -m venv steveai_env
	@echo "Virtual environment created. Activate with:"
	@echo "  source steveai_env/bin/activate  # Linux/Mac"
	@echo "  steveai_env\\Scripts\\activate  # Windows"

# Configuration
config-template:
	python -c "from config_manager import create_config_template; create_config_template('config_template.yaml')"
	@echo "Configuration template created: config_template.yaml"

# Monitoring
monitor-training:
	python monitor_training.py --log_dir ./logs

# Visualization
visualize:
	python visualize_results.py --results_dir ./output --output_dir ./visualizations

# Deployment
deploy-local:
	python deploy.py --model_path ./output/student_model/final_model --tokenizer_path ./output/student_model/final_model --server_type fastapi --port 8000

# Performance profiling
profile:
	python -m cProfile -o profile_output.prof teacher_inference_gpu.py
	python -c "import pstats; pstats.Stats('profile_output.prof').sort_stats('cumulative').print_stats(20)"

# Memory profiling
profile-memory:
	python -m memory_profiler teacher_inference_gpu.py

# Security check
security-check:
	bandit -r . -f json -o security_report.json
	safety check

# Dependencies
update-deps:
	pip-compile requirements.in
	pip-compile requirements-dev.in

# Version management
version:
	@python -c "import steveai; print(steveai.__version__)"

bump-version:
	@echo "Current version: $$(python -c "import steveai; print(steveai.__version__)")"
	@read -p "Enter new version: " version; \
	sed -i "s/version = \".*\"/version = \"$$version\"/" setup.py; \
	sed -i "s/version = \".*\"/version = \"$$version\"/" pyproject.toml

# Git hooks
setup-git-hooks:
	pre-commit install
	pre-commit install --hook-type commit-msg

# CI/CD helpers
ci-install:
	pip install -e ".[dev]"

ci-test:
	python run_tests.py --type ci

ci-lint:
	flake8 .
	black --check .
	isort --check-only .

# Development workflow
dev-setup: install-dev setup-git-hooks
	@echo "Development environment setup complete!"

dev-test: format lint test
	@echo "All development checks passed!"

# Release helpers
release-check: clean test lint
	@echo "Release checks passed!"

release-build: release-check build
	@echo "Release build complete!"

# Help for specific targets
help-test:
	@echo "Testing commands:"
	@echo "  test             - Run all tests"
	@echo "  test-unit        - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-performance - Run performance tests only"
	@echo "  test-quick       - Run quick tests (unit tests only)"
	@echo "  test-ci          - Run CI tests (unit + integration)"

help-dev:
	@echo "Development commands:"
	@echo "  dev-setup        - Setup development environment"
	@echo "  dev-test         - Run all development checks"
	@echo "  format           - Format code"
	@echo "  lint             - Run linting"
	@echo "  type-check       - Run type checking"
	@echo "  pre-commit       - Run pre-commit hooks"
