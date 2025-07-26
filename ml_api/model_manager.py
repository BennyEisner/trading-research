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

import keras
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

        self.logger.info(f"Discovered {len(self.models_registry)} models")

    def get_model(self, name):
        """Get a loaded model by name"""
        try:
            if name in self.models:
                self.logger.info(f"Returning cached model: {name}")
                return self.models[name]

            if name not in self.models_registry:
                self.logger.warning(f"Model not found in registry: {name}")
                return None

            model_info = self.models_registry[name]
            model_path = model_info["file_path"]

            self.logger.info(f"Loading model from: {model_path}")

            # Enable unsafe deserialization for Lambda layers (needed for custom layers)
            import keras
            keras.config.enable_unsafe_deserialization()
            
            # Define custom functions for model loading
            def directional_mse_loss(y_true, y_pred):
                """Custom directional MSE loss function"""
                alpha = 0.4  # Default from training
                mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
                y_true_sign = tf.sign(y_true)
                y_pred_sign = tf.sign(y_pred)
                directional_error = 0.5 * (1 - y_true_sign * y_pred_sign)
                directional_loss = tf.reduce_mean(directional_error)
                return (1 - alpha) * mse_loss + alpha * directional_loss
            
            def _directional_accuracy(y_true, y_pred):
                """Custom metric: Directional accuracy"""
                y_true_sign = tf.sign(y_true)
                y_pred_sign = tf.sign(y_pred)
                correct_directions = tf.equal(y_true_sign, y_pred_sign)
                return tf.reduce_mean(tf.cast(correct_directions, tf.float32))
            
            def _weighted_directional_accuracy(y_true, y_pred):
                """Custom metric: Weighted directional accuracy"""
                y_true_sign = tf.sign(y_true)
                y_pred_sign = tf.sign(y_pred)
                correct_directions = tf.cast(tf.equal(y_true_sign, y_pred_sign), tf.float32)
                weights = tf.abs(y_true)
                weights = weights / (tf.reduce_mean(weights) + 1e-8)
                weighted_accuracy = tf.reduce_sum(correct_directions * weights) / tf.reduce_sum(weights)
                return weighted_accuracy
            
            def _correlation_metric(y_true, y_pred):
                """Custom metric: Pearson correlation coefficient"""
                x = y_true - tf.reduce_mean(y_true)
                y = y_pred - tf.reduce_mean(y_pred)
                numerator = tf.reduce_sum(x * y)
                denominator = tf.sqrt(tf.reduce_sum(x**2) * tf.reduce_sum(y**2))
                correlation = numerator / (denominator + 1e-8)
                return correlation
            
            def _up_down_accuracy(y_true, y_pred):
                """Custom metric: Balanced accuracy for up vs down moves"""
                y_true_sign = tf.sign(y_true)
                y_pred_sign = tf.sign(y_pred)
                
                # Up moves accuracy
                up_mask = y_true > 0
                up_correct = tf.logical_and(up_mask, tf.equal(y_true_sign, y_pred_sign))
                up_total = tf.reduce_sum(tf.cast(up_mask, tf.float32))
                up_accuracy = tf.reduce_sum(tf.cast(up_correct, tf.float32)) / (up_total + 1e-8)
                
                # Down moves accuracy  
                down_mask = y_true < 0
                down_correct = tf.logical_and(down_mask, tf.equal(y_true_sign, y_pred_sign))
                down_total = tf.reduce_sum(tf.cast(down_mask, tf.float32))
                down_accuracy = tf.reduce_sum(tf.cast(down_correct, tf.float32)) / (down_total + 1e-8)
                
                # Return balanced accuracy
                return (up_accuracy + down_accuracy) / 2.0
            
            # Load model with custom objects
            custom_objects = {
                'directional_mse_loss': directional_mse_loss,
                '_directional_accuracy': _directional_accuracy,
                '_weighted_directional_accuracy': _weighted_directional_accuracy,
                '_correlation_metric': _correlation_metric,
                '_up_down_accuracy': _up_down_accuracy
            }
            
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

            self.models[name] = model

            self.logger.info(f"Successfully loaded and cached model: {name}")
            return model

        except Exception as e:
            self.logger.error(f"Failed to load model {name}: {e}")
            return None

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with metadata"""
        return list(self.models_registry.values())
    
    def list_model_names(self) -> List[str]:
        """List just the model names"""
        return list(self.models_registry.keys())
    
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get model metadata without loading the model"""
        return self.models_registry.get(name)
    
    def is_model_loaded(self, name: str) -> bool:
        """Check if model is currently loaded in cache"""
        return name in self.models
    
    def unload_model(self, name: str) -> bool:
        """Remove model from cache to free memory"""
        if name in self.models:
            del self.models[name]
            self.logger.info(f"Unloaded model from cache: {name}")
            return True
        return False
    
    def clear_cache(self):
        """Clear all loaded models from cache"""
        count = len(self.models)
        self.models.clear()
        self.logger.info(f"Cleared {count} models from cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "total_models_available": len(self.models_registry),
            "models_loaded_in_cache": len(self.models),
            "cached_model_names": list(self.models.keys()),
            "available_model_names": list(self.models_registry.keys())
        }

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
                model_info["config"] = metadata.get("config", {})

            return model_info

        except Exception as e:
            self.logger.error(f"Error parsing model file: {e}")
            return None
