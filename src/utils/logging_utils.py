#!/usr/bin/env python3

"""
Production-quality logging utilities
"""

import json
import time
from datetime import datetime


class ProductionLogger:
    """Production-level logger with timestamps and progress tracking"""

    def __init__(self, log_file="training_log.json"):
        self.log_file = log_file
        self.start_time = time.time()
        self.phase_times = {}
        self.results = {}

        # Initialize log file
        self._initialize_log_file()

    def _initialize_log_file(self):
        """Initialize log file with session start"""
        try:
            with open(self.log_file, "w") as f:
                f.write("")  # Clear existing content
        except Exception:
            pass  # Don't fail on logging errors

    def log(self, message, phase=None):
        """Log message with timestamp and elapsed time"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - self.start_time

        log_entry = {
            "timestamp": timestamp,
            "elapsed_seconds": round(elapsed, 2),
            "phase": phase,
            "message": message,
        }

        # Console output
        print(f"[{timestamp}] [{elapsed:.1f}s] {message}")

        # File logging
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass  # Silent failure for logging

    def start_phase(self, phase_name):
        """Start timing a phase"""
        self.phase_times[phase_name] = time.time()
        self.log(f"Starting {phase_name}", phase_name)

    def end_phase(self, phase_name):
        """End timing a phase and log duration"""
        if phase_name in self.phase_times:
            duration = time.time() - self.phase_times[phase_name]
            self.log(f"Completed {phase_name} in {duration:.1f}s", phase_name)
            return duration
        return 0

    def save_results(self, results):
        """Save results to timestamped file"""
        self.results = results
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_results_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=str)
            self.log(f"Results saved to {filename}")
        except Exception as e:
            self.log(f"Failed to save results: {e}")


class ProgressCallback:
    """Custom callback for detailed training progress with visual progress bar"""

    def __init__(self, logger, phase_name, total_epochs):
        self.logger = logger
        self.phase_name = phase_name
        self.total_epochs = total_epochs
        self.epoch_start_time = None
        self.training_start_time = None

    def on_train_begin(self, logs=None):
        """Called at beginning of training"""
        self.training_start_time = time.time()
        self.logger.log(f"Starting {self.phase_name} - {self.total_epochs} epochs")

    def on_epoch_begin(self, epoch, logs=None):
        """Called at beginning of each epoch"""
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """Called at end of each epoch with detailed progress reporting"""
        epoch_time = time.time() - self.epoch_start_time
        elapsed_total = time.time() - self.training_start_time

        # Calculate ETA
        epochs_remaining = self.total_epochs - (epoch + 1)
        avg_epoch_time = elapsed_total / (epoch + 1)
        eta_seconds = epochs_remaining * avg_epoch_time
        eta_minutes = eta_seconds / 60

        # Extract metrics with defaults
        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)
        dir_acc = logs.get("directional_accuracy", 0)
        val_dir_acc = logs.get("val_directional_accuracy", 0)
        mae = logs.get("mae", 0)
        val_mae = logs.get("val_mae", 0)
        lr = logs.get("lr", logs.get("learning_rate", 0.002))

        # Create visual progress bar
        progress_pct = ((epoch + 1) / self.total_epochs) * 100
        bar_length = 30
        filled_length = int(bar_length * (epoch + 1) // self.total_epochs)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)

        # Build validation metrics string
        val_metrics = ""
        if val_loss > 0:
            val_metrics = f" | Val Loss: {val_loss:.4f} | Val Dir: {val_dir_acc:.1%} | Val MAE: {val_mae:.4f}"

        # Construct progress message
        progress_msg = (
            f"[{bar}] {progress_pct:.1f}% "
            f"Epoch {epoch + 1}/{self.total_epochs} ({epoch_time:.1f}s) | "
            f"Loss: {loss:.4f} | Dir: {dir_acc:.1%} | MAE: {mae:.4f}"
            f"{val_metrics} | LR: {lr:.2e} | ETA: {eta_minutes:.1f}min"
        )

        self.logger.log(progress_msg, self.phase_name)

    def on_train_end(self, logs=None):
        """Called at end of training"""
        total_time = time.time() - self.training_start_time
        self.logger.log(f"COMPLETED {self.phase_name} in {total_time:.1f}s", self.phase_name)


class MetricsTracker:
    """Track and log training metrics over time"""

    def __init__(self, logger):
        self.logger = logger
        self.metrics_history = {}

    def update_metrics(self, epoch, metrics_dict):
        """Update metrics for given epoch"""
        for metric_name, value in metrics_dict.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append(value)

    def log_best_metrics(self):
        """Log best achieved metrics"""
        if not self.metrics_history:
            return

        self.logger.log("=== BEST METRICS ACHIEVED ===")

        # Find best directional accuracy
        if "val_directional_accuracy" in self.metrics_history:
            best_dir_acc = max(self.metrics_history["val_directional_accuracy"])
            best_epoch = self.metrics_history["val_directional_accuracy"].index(best_dir_acc) + 1
            self.logger.log(f"Best Val Directional Accuracy: {best_dir_acc:.1%} (Epoch {best_epoch})")

        # Find best validation loss
        if "val_loss" in self.metrics_history:
            best_val_loss = min(self.metrics_history["val_loss"])
            best_epoch = self.metrics_history["val_loss"].index(best_val_loss) + 1
            self.logger.log(f"Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})")

    def get_metrics_summary(self):
        """Get summary statistics of all metrics"""
        summary = {}
        for metric_name, values in self.metrics_history.items():
            summary[metric_name] = {
                "best": max(values) if "accuracy" in metric_name else min(values),
                "final": values[-1] if values else 0,
                "mean": sum(values) / len(values) if values else 0,
                "total_epochs": len(values),
            }
        return summary


class FileLogger:
    """Simple file-based logger for debugging"""

    def __init__(self, filename="debug.log"):
        self.filename = filename

    def log(self, message, level="INFO"):
        """Log message to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.filename, "a") as f:
                f.write(f"[{timestamp}] [{level}] {message}\n")
        except Exception:
            pass  # Silent failure


# Convenience functions
def setup_production_logger(config):
    """Setup logger with config parameters"""
    log_file = config.get("log_file", "training_log.json")
    return ProductionLogger(log_file)


def log_system_info(logger):
    """Log system information for debugging"""
    import platform
    import sys

    logger.log("=== SYSTEM INFORMATION ===")
    logger.log(f"Python Version: {sys.version}")
    logger.log(f"Platform: {platform.platform()}")
    logger.log(f"Processor: {platform.processor()}")

    try:
        import tensorflow as tf

        logger.log(f"TensorFlow Version: {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        logger.log(f"Available GPUs: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            logger.log(f"  GPU {i}: {gpu.name}")
    except ImportError:
        logger.log("TensorFlow not available")

    try:
        import psutil

        memory = psutil.virtual_memory()
        logger.log(f"Total Memory: {memory.total / (1024**3):.1f} GB")
        logger.log(f"Available Memory: {memory.available / (1024**3):.1f} GB")
    except ImportError:
        logger.log("psutil not available for memory info")
