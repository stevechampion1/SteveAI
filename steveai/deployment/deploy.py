# SteveAI - Model Deployment Script

import argparse
import asyncio
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from flask import Flask, jsonify, request
from pydantic import BaseModel

from ..utils.config_manager import ConfigManager

# Import our custom modules
from ..utils.model_utils import get_model_summary, load_model_artifacts
from ..utils.utils import check_disk_space, print_memory_usage, setup_logging

logger = logging.getLogger(__name__)


class ModelServer:
    """Model serving server for SteveAI models."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: Optional[str] = None,
        max_batch_size: int = 8,
        max_sequence_length: int = 256,
    ):
        """
        Initialize the model server.

        Args:
            model_path: Path to the model
            tokenizer_path: Path to the tokenizer
            device: Device to use for inference
            max_batch_size: Maximum batch size for inference
            max_sequence_length: Maximum sequence length
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length

        self.model = None
        self.tokenizer = None
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()

        logger.info(f"Model server initialized on device: {self.device}")

    def load_model(self):
        """Load model and tokenizer."""
        logger.info("Loading model and tokenizer...")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

        # Log model info
        summary = get_model_summary(self.model)
        logger.info(
            f"Model loaded: {summary['total_parameters']:,} parameters, "
            f"{summary['model_size_mb']:.2f} MB"
        )

    def generate_text(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            num_return_sequences: Number of sequences to generate

        Returns:
            List of generated texts
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_sequence_length,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode generated text
        generated_texts = []
        for output in outputs:
            generated_text = self.tokenizer.decode(
                output[inputs["input_ids"].size(1) :], skip_special_tokens=True
            )
            generated_texts.append(generated_text)

        return generated_texts

    def batch_generate(
        self,
        prompts: List[str],
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[List[str]]:
        """
        Generate text for multiple prompts in batch.

        Args:
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            List of generated texts for each prompt
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        # Process in batches
        results = []
        for i in range(0, len(prompts), self.max_batch_size):
            batch_prompts = prompts[i : i + self.max_batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_sequence_length,
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode generated text
            batch_results = []
            for j, output in enumerate(outputs):
                input_length = inputs["input_ids"][j].size(0)
                generated_text = self.tokenizer.decode(
                    output[input_length:], skip_special_tokens=True
                )
                batch_results.append([generated_text])

            results.extend(batch_results)

        return results


class FlaskModelServer:
    """Flask-based model server."""

    def __init__(self, model_server: ModelServer):
        """
        Initialize Flask model server.

        Args:
            model_server: ModelServer instance
        """
        self.model_server = model_server
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/health", methods=["GET"])
        def health_check():
            """Health check endpoint."""
            return jsonify(
                {
                    "status": "healthy",
                    "model_loaded": self.model_server.model is not None,
                    "device": self.model_server.device,
                }
            )

        @self.app.route("/generate", methods=["POST"])
        def generate():
            """Text generation endpoint."""
            try:
                data = request.get_json()

                prompt = data.get("prompt", "")
                max_length = data.get("max_length", 256)
                temperature = data.get("temperature", 0.7)
                top_p = data.get("top_p", 0.9)
                num_return_sequences = data.get("num_return_sequences", 1)

                if not prompt:
                    return jsonify({"error": "Prompt is required"}), 400

                generated_texts = self.model_server.generate_text(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                )

                return jsonify(
                    {
                        "generated_texts": generated_texts,
                        "prompt": prompt,
                        "parameters": {
                            "max_length": max_length,
                            "temperature": temperature,
                            "top_p": top_p,
                            "num_return_sequences": num_return_sequences,
                        },
                    }
                )

            except Exception as e:
                logger.error(f"Error in generate endpoint: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/batch_generate", methods=["POST"])
        def batch_generate():
            """Batch text generation endpoint."""
            try:
                data = request.get_json()

                prompts = data.get("prompts", [])
                max_length = data.get("max_length", 256)
                temperature = data.get("temperature", 0.7)
                top_p = data.get("top_p", 0.9)

                if not prompts:
                    return jsonify({"error": "Prompts are required"}), 400

                if len(prompts) > self.model_server.max_batch_size:
                    return (
                        jsonify(
                            {
                                "error": f"Too many prompts. Maximum batch size: {self.model_server.max_batch_size}"
                            }
                        ),
                        400,
                    )

                generated_texts = self.model_server.batch_generate(
                    prompts=prompts,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                )

                return jsonify(
                    {
                        "generated_texts": generated_texts,
                        "prompts": prompts,
                        "parameters": {
                            "max_length": max_length,
                            "temperature": temperature,
                            "top_p": top_p,
                        },
                    }
                )

            except Exception as e:
                logger.error(f"Error in batch_generate endpoint: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/model_info", methods=["GET"])
        def model_info():
            """Model information endpoint."""
            if self.model_server.model is None:
                return jsonify({"error": "Model not loaded"}), 500

            summary = get_model_summary(self.model_server.model)

            return jsonify(
                {
                    "model_path": self.model_server.model_path,
                    "device": self.model_server.device,
                    "max_batch_size": self.model_server.max_batch_size,
                    "max_sequence_length": self.model_server.max_sequence_length,
                    "model_summary": summary,
                }
            )

    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
        """
        Run Flask server.

        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Enable debug mode
        """
        logger.info(f"Starting Flask server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


class FastAPIModelServer:
    """FastAPI-based model server."""

    def __init__(self, model_server: ModelServer):
        """
        Initialize FastAPI model server.

        Args:
            model_server: ModelServer instance
        """
        self.model_server = model_server
        self.app = FastAPI(title="SteveAI Model Server", version="1.0.0")
        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "model_loaded": self.model_server.model is not None,
                "device": self.model_server.device,
            }

        class GenerateRequest(BaseModel):
            prompt: str
            max_length: int = 256
            temperature: float = 0.7
            top_p: float = 0.9
            num_return_sequences: int = 1

        @self.app.post("/generate")
        async def generate(request: GenerateRequest):
            """Text generation endpoint."""
            try:
                generated_texts = self.model_server.generate_text(
                    prompt=request.prompt,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    num_return_sequences=request.num_return_sequences,
                )

                return {
                    "generated_texts": generated_texts,
                    "prompt": request.prompt,
                    "parameters": request.dict(),
                }

            except Exception as e:
                logger.error(f"Error in generate endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        class BatchGenerateRequest(BaseModel):
            prompts: List[str]
            max_length: int = 256
            temperature: float = 0.7
            top_p: float = 0.9

        @self.app.post("/batch_generate")
        async def batch_generate(request: BatchGenerateRequest):
            """Batch text generation endpoint."""
            try:
                if len(request.prompts) > self.model_server.max_batch_size:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Too many prompts. Maximum batch size: {self.model_server.max_batch_size}",
                    )

                generated_texts = self.model_server.batch_generate(
                    prompts=request.prompts,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                )

                return {
                    "generated_texts": generated_texts,
                    "prompts": request.prompts,
                    "parameters": request.dict(),
                }

            except Exception as e:
                logger.error(f"Error in batch_generate endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/model_info")
        async def model_info():
            """Model information endpoint."""
            if self.model_server.model is None:
                raise HTTPException(status_code=500, detail="Model not loaded")

            summary = get_model_summary(self.model_server.model)

            return {
                "model_path": self.model_server.model_path,
                "device": self.model_server.device,
                "max_batch_size": self.model_server.max_batch_size,
                "max_sequence_length": self.model_server.max_sequence_length,
                "model_summary": summary,
            }

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Run FastAPI server.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        logger.info(f"Starting FastAPI server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


def create_deployment_config(
    model_path: str,
    tokenizer_path: str,
    server_type: str = "fastapi",
    host: str = "0.0.0.0",
    port: int = 8000,
    max_batch_size: int = 8,
    max_sequence_length: int = 256,
) -> Dict[str, Any]:
    """
    Create deployment configuration.

    Args:
        model_path: Path to the model
        tokenizer_path: Path to the tokenizer
        server_type: Type of server ('flask' or 'fastapi')
        host: Host to bind to
        port: Port to bind to
        max_batch_size: Maximum batch size
        max_sequence_length: Maximum sequence length

    Returns:
        Deployment configuration
    """
    config = {
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "server_type": server_type,
        "host": host,
        "port": port,
        "max_batch_size": max_batch_size,
        "max_sequence_length": max_sequence_length,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "created_at": time.time(),
    }

    return config


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy SteveAI models")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model to deploy"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to the tokenizer"
    )
    parser.add_argument(
        "--server_type",
        type=str,
        default="fastapi",
        choices=["flask", "fastapi"],
        help="Type of server to use",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--max_batch_size", type=int, default=8, help="Maximum batch size"
    )
    parser.add_argument(
        "--max_sequence_length", type=int, default=256, help="Maximum sequence length"
    )
    parser.add_argument(
        "--config_file", type=str, default=None, help="Deployment configuration file"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("--- Starting SteveAI Model Deployment ---")

        # Load configuration if provided
        if args.config_file and os.path.exists(args.config_file):
            with open(args.config_file, "r") as f:
                config = json.load(f)

            # Override with command line arguments
            for key, value in vars(args).items():
                if value is not None and key in config:
                    config[key] = value
        else:
            config = create_deployment_config(
                model_path=args.model_path,
                tokenizer_path=args.tokenizer_path,
                server_type=args.server_type,
                host=args.host,
                port=args.port,
                max_batch_size=args.max_batch_size,
                max_sequence_length=args.max_sequence_length,
            )

        # Initialize model server
        model_server = ModelServer(
            model_path=config["model_path"],
            tokenizer_path=config["tokenizer_path"],
            device=config["device"],
            max_batch_size=config["max_batch_size"],
            max_sequence_length=config["max_sequence_length"],
        )

        # Load model
        model_server.load_model()

        # Create and run server
        if config["server_type"] == "flask":
            flask_server = FlaskModelServer(model_server)
            flask_server.run(host=config["host"], port=config["port"], debug=False)
        elif config["server_type"] == "fastapi":
            fastapi_server = FastAPIModelServer(model_server)
            fastapi_server.run(host=config["host"], port=config["port"])

    except Exception as e:
        logger.error(f"Deployment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
