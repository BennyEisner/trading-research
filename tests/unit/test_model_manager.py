#!/usr/bin/env python3
"""
Unit tests for Model Manager
"""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tensorflow as tf

from ml_api.model_manager import ModelManager


class TestModelManager:
    """Test cases for ModelManager class"""

    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary models directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir) / "models"
            models_dir.mkdir()
            trained_dir = models_dir / "trained"
            trained_dir.mkdir()
            yield models_dir

    @pytest.fixture
    def mock_metadata(self):
        """Sample model metadata for testing"""
        return {
            "timestamp": "20250724_123456",
            "config": {
                "tickers": ["TEST", "DEMO"],
                "lookback_window": 30,
                "epochs": 5
            },
            "model_parameters": 50000,
            "training_epochs": 5,
            "final_loss": 0.5,
            "final_val_loss": 0.6
        }

    @pytest.fixture
    def model_manager(self, temp_models_dir):
        """Create ModelManager instance with temporary directory"""
        return ModelManager(str(temp_models_dir))

    def test_init(self, model_manager, temp_models_dir):
        """Test ModelManager initialization"""
        assert model_manager.models_directory == temp_models_dir
        assert model_manager.models == {}
        assert model_manager.models_registry == {}
        assert model_manager.logger is not None

    @pytest.mark.asyncio
    async def test_load_models_empty_directory(self, model_manager):
        """Test loading models from empty directory"""
        await model_manager.load_models()
        assert len(model_manager.models_registry) == 0

    @pytest.mark.asyncio
    async def test_load_models_nonexistent_directory(self):
        """Test loading models from nonexistent directory"""
        manager = ModelManager("/nonexistent/path")
        await manager.load_models()
        assert len(manager.models_registry) == 0

    def test_parse_model_file_valid_filename(self, model_manager, temp_models_dir, mock_metadata):
        """Test parsing valid model filename with metadata"""
        # Create mock model file
        model_file = temp_models_dir / "trained" / "test_model_20250724_123456.keras"
        model_file.touch()
        
        # Create corresponding metadata file
        metadata_file = temp_models_dir / "trained" / "metadata_20250724_123456.json"
        with open(metadata_file, 'w') as f:
            json.dump(mock_metadata, f)
        
        # Test parsing
        result = model_manager._parse_model_file(model_file)
        
        assert result is not None
        assert result["name"] == "test_model_20250724_123456"
        assert result["model_type"] == "test_model"
        assert result["training_date"] == "20250724"
        assert result["training_time"] == "123456"
        assert result["tickers"] == ["TEST", "DEMO"]
        assert result["parameters"] == 50000

    def test_parse_model_file_invalid_filename(self, model_manager, temp_models_dir):
        """Test parsing invalid model filename"""
        # Create file with invalid name
        model_file = temp_models_dir / "trained" / "invalid.keras"
        model_file.touch()
        
        result = model_manager._parse_model_file(model_file)
        assert result is None

    def test_parse_model_file_no_metadata(self, model_manager, temp_models_dir):
        """Test parsing model file without metadata"""
        # Create mock model file without metadata
        model_file = temp_models_dir / "trained" / "solo_model_20250724_123456.keras"
        model_file.touch()
        
        result = model_manager._parse_model_file(model_file)
        
        assert result is not None
        assert result["name"] == "solo_model_20250724_123456"
        assert result["model_type"] == "solo_model"
        assert "metadata" not in result

    @pytest.mark.asyncio
    async def test_load_models_with_valid_files(self, model_manager, temp_models_dir, mock_metadata):
        """Test loading models with valid .keras files"""
        # Create mock model files
        model_file1 = temp_models_dir / "trained" / "model1_20250724_123456.keras"
        model_file2 = temp_models_dir / "trained" / "model2_20250725_654321.keras"
        model_file1.touch()
        model_file2.touch()
        
        # Create metadata for first model
        metadata_file1 = temp_models_dir / "trained" / "metadata_20250724_123456.json"
        with open(metadata_file1, 'w') as f:
            json.dump(mock_metadata, f)
        
        await model_manager.load_models()
        
        assert len(model_manager.models_registry) == 2
        assert "model1_20250724_123456" in model_manager.models_registry
        assert "model2_20250725_654321" in model_manager.models_registry

    def test_list_model_names(self, model_manager):
        """Test listing model names"""
        # Add mock models to registry
        model_manager.models_registry = {
            "model1": {"name": "model1"},
            "model2": {"name": "model2"}
        }
        
        names = model_manager.list_model_names()
        assert names == ["model1", "model2"]

    def test_list_models(self, model_manager):
        """Test listing all models with metadata"""
        # Add mock models to registry
        mock_model1 = {"name": "model1", "parameters": 1000}
        mock_model2 = {"name": "model2", "parameters": 2000}
        model_manager.models_registry = {
            "model1": mock_model1,
            "model2": mock_model2
        }
        
        models = model_manager.list_models()
        assert len(models) == 2
        assert mock_model1 in models
        assert mock_model2 in models

    def test_get_model_info(self, model_manager):
        """Test getting model info without loading"""
        mock_info = {"name": "test_model", "parameters": 1000}
        model_manager.models_registry = {"test_model": mock_info}
        
        info = model_manager.get_model_info("test_model")
        assert info == mock_info
        
        # Test nonexistent model
        info = model_manager.get_model_info("nonexistent")
        assert info is None

    def test_is_model_loaded(self, model_manager):
        """Test checking if model is loaded"""
        mock_model = Mock()
        model_manager.models = {"loaded_model": mock_model}
        
        assert model_manager.is_model_loaded("loaded_model") is True
        assert model_manager.is_model_loaded("not_loaded") is False

    def test_unload_model(self, model_manager):
        """Test unloading model from cache"""
        mock_model = Mock()
        model_manager.models = {"test_model": mock_model}
        
        # Test successful unload
        result = model_manager.unload_model("test_model")
        assert result is True
        assert "test_model" not in model_manager.models
        
        # Test unloading nonexistent model
        result = model_manager.unload_model("nonexistent")
        assert result is False

    def test_clear_cache(self, model_manager):
        """Test clearing all models from cache"""
        model_manager.models = {
            "model1": Mock(),
            "model2": Mock(),
            "model3": Mock()
        }
        
        model_manager.clear_cache()
        assert len(model_manager.models) == 0

    def test_get_cache_stats(self, model_manager):
        """Test getting cache statistics"""
        # Setup registry and cache
        model_manager.models_registry = {
            "model1": {"name": "model1"},
            "model2": {"name": "model2"},
            "model3": {"name": "model3"}
        }
        model_manager.models = {
            "model1": Mock(),
            "model2": Mock()
        }
        
        stats = model_manager.get_cache_stats()
        
        assert stats["total_models_available"] == 3
        assert stats["models_loaded_in_cache"] == 2
        assert set(stats["cached_model_names"]) == {"model1", "model2"}
        assert set(stats["available_model_names"]) == {"model1", "model2", "model3"}

    def test_get_model_not_in_registry(self, model_manager):
        """Test getting model that doesn't exist in registry"""
        result = model_manager.get_model("nonexistent_model")
        assert result is None

    @patch('tensorflow.keras.models.load_model')
    def test_get_model_cached(self, mock_load_model, model_manager):
        """Test getting already cached model"""
        mock_model = Mock()
        model_manager.models = {"cached_model": mock_model}
        model_manager.models_registry = {"cached_model": {"file_path": "/fake/path"}}
        
        result = model_manager.get_model("cached_model")
        
        assert result == mock_model
        mock_load_model.assert_not_called()  # Should not load from file

    @patch('keras.config.enable_unsafe_deserialization')
    @patch('tensorflow.keras.models.load_model')
    def test_get_model_load_from_file(self, mock_load_model, mock_enable_unsafe, model_manager):
        """Test loading model from file"""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        model_manager.models_registry = {
            "test_model": {"file_path": "/fake/path/test_model.keras"}
        }
        
        result = model_manager.get_model("test_model")
        
        assert result == mock_model
        assert model_manager.models["test_model"] == mock_model
        mock_enable_unsafe.assert_called_once()
        mock_load_model.assert_called_once()

    @patch('tensorflow.keras.models.load_model')
    def test_get_model_load_failure(self, mock_load_model, model_manager):
        """Test handling model loading failure"""
        mock_load_model.side_effect = Exception("Loading failed")
        
        model_manager.models_registry = {
            "failing_model": {"file_path": "/fake/path/failing_model.keras"}
        }
        
        result = model_manager.get_model("failing_model")
        
        assert result is None
        assert "failing_model" not in model_manager.models


@pytest.mark.integration
class TestModelManagerIntegration:
    """Integration tests using real files"""

    @pytest.fixture
    def real_model_manager(self):
        """Create ModelManager pointing to actual models directory"""
        return ModelManager("./models")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_load_real_models(self, real_model_manager):
        """Test loading actual trained models"""
        await real_model_manager.load_models()
        
        # Should find at least one model if they exist
        model_names = real_model_manager.list_model_names()
        if model_names:
            # Test that we can get info about real models
            for name in model_names:
                info = real_model_manager.get_model_info(name)
                assert info is not None
                assert "file_path" in info
                assert "model_type" in info