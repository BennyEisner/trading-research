#!/usr/bin/env python3

"""
Model Manager for Lazy Loading Keras Models
Handles .keras model discovery, loading, and caching
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import tensorflow as tf


class ModelManager:
    def __init__(self, models_directory: str = "./models"):
        self.models_directory = Path(models_directory)
        self.models: Dict[str, Any] = {}  # Cache for loaded models
        self.models_registry: Dict[str, Any] = {}  # Available models metadata
        self.logger = logging.getLogger(__name__)

    async def load_models(self):
        """Loads all available models on startup"""
        self.models_registry.clear()

        if not self.models_directory.exists():
            self.logger.warning(f"Models directory not found: {self.models_directory}")
            return

        # Scan for .keras files recursively
        keras_files = list(self.models_directory.rglob("*.keras"))

        for model_file in keras_files:
            try:
                model_info = self._parse_model_file(model_file)
                if model_info:
                    self.models_registry[model_info["name"]] = model_info
                    self.logger.info(f"Discovered model: {model_info['name']}")
            except Exception as e:
                self.logger.error(f"Error processing {model_file}: {e}")

        self.logger.error(f"Discovered {len(self.models_registry)} models")

    def get_model(self, name):
        """Get a loaded model by name"""
        # TODO Implement actual model retrieval
        return None

    def list_models(self) -> List[str]:
        """List all available models"""
        # TODO Implement actual model listing
        return []

    def _parse_model_file(self, model_file: Path) -> Optional[Dict]:
        """Extract metadata from .keras files and JSON files"""

        try:
            filename = model_file.stem  # Removes .keras
            parts = filename.split("_")

            if len(parts) < 3:
                self.logger.warning(
                    f"Unexpected file name format: {filename}"
                )  # Since all files in the form .keras.model.___
                return None

            # Extract components
            model_type = "_".join(parts[:2])  # Everything except datetine
            date = parts[-2]
            time = parts[-1]

            metadata_file = model_file.parent / f"metadata_{date}_{time}.json"
            metadata = {}

            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

            model_info = {
                "name": filename,
                "file_path": str(model_file.absolute()),
                "model_type": model_type,
                "training_date": date,
                "training_time": time,
                "file_size_md": model_file.stat().st_size / (1024 * 1024),  # Readability
                "created_timestamp": datetime.fromtimestamp(model_file.stat().st_ctime),
                "modified_timestamp": datetime.fromtimestamp(model_file.stat().st_mtime),
            }

            if metadata:
                model_info["metadata"] = metadata
                model_info["tickers"] = metadata.get("config", {}).get("tickers", [])
                model_info["parameters"] = metadata.get("model_parameters", 0)
                model_info["training_epochs"] = metadata.get("training_epochs", 0)
                model_info["final_loss"] = metadata.get("final_loss", 0)
                model_info["final_val_loss"] = metadata.get("final_val_loss")
                model_info["cofig"] = metadata.get("config", {})

            return model_info

        except Exception as e:
            self.logger.error(f"Error parsing model file: {e}")
            return None
