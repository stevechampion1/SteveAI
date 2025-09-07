# SteveAI - Setup Script

import os

from setuptools import find_packages, setup


# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]


setup(
    name="steveai",
    version="1.0.0",
    author="SteveAI Team",
    author_email="steveai@example.com",
    description="A comprehensive knowledge distillation framework for large language models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/SteveAI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "deploy": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "flask>=2.0.0",
            "gunicorn>=20.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "steveai-teacher=scripts.teacher_inference:main",
            "steveai-student=steveai.core.student_training:main",
            "steveai-evaluate=steveai.evaluation.evaluate:main",
            "steveai-benchmark=steveai.evaluation.benchmark:main",
            "steveai-optimize=steveai.optimization.optimize_model:main",
            "steveai-deploy=steveai.deployment.deploy:main",
            "steveai-monitor=steveai.monitoring.monitor_training:main",
            "steveai-visualize=steveai.visualization.visualize_results:main",
            "steveai-test=steveai.tests.run_tests:main",
        ],
    },
    include_package_data=True,
    package_data={
        "steveai": [
            "config.yaml",
            "*.md",
            "examples/*.py",
        ],
    },
    keywords=[
        "machine learning",
        "deep learning",
        "knowledge distillation",
        "language models",
        "transformer",
        "nlp",
        "ai",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/SteveAI/issues",
        "Source": "https://github.com/your-username/SteveAI",
        "Documentation": "https://steveai.readthedocs.io/",
    },
)
