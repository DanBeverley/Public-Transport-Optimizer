"""
Utility functions for the model training pipeline.

Includes logging setup, object serialization, and reproducibility helpers.
"""

import logging
import random
import os
import pickle
import joblib
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[Union[str, Path]] = None,
    log_to_console: bool = True
    ) -> logging.Logger:
    """
    Configures logging for the training process.

    Args:
        log_level: The logging level (e.g., 'DEBUG', 'INFO').
        log_file: Optional path to a file to save logs.
        log_to_console: Whether to also log messages to the console.

    Returns:
        The configured root logger instance.
    """
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        # If already configured (e.g., in interactive session), just return
        # Or clear existing handlers if re-configuration is desired:
        # for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
        return root_logger

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = []
    if log_to_console:
        handlers.append(logging.StreamHandler())

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='a'))

    logging.basicConfig(level=numeric_level, format=log_format, datefmt=date_format, handlers=handlers, force=True)
    logger = logging.getLogger() 
    logger.info(f"Logging setup complete. Level: {log_level}, File: {log_file}, Console: {log_to_console}")
    return logger

def save_object(obj: Any, path: Union[str, Path]):
    """
    Saves a Python object to a file using joblib (preferred for scikit-learn)
    or pickle as a fallback.

    Args:
        obj: The Python object to save.
        path: The file path where the object will be saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    try:
        joblib.dump(obj, path)
        logger.info(f"Object saved successfully to {path} using joblib.")
    except (TypeError, ValueError, pickle.PicklingError) as e_joblib:
        logger.warning(f"Could not save object using joblib ({e_joblib}). Falling back to pickle.")
        try:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
            logger.info(f"Object saved successfully to {path} using pickle.")
        except Exception as e_pickle:
            logger.error(f"Failed to save object to {path} using both joblib and pickle: {e_pickle}", exc_info=True)
            raise

def load_object(path: Union[str, Path]) -> Any:
    """
    Loads a Python object from a file using joblib or pickle.

    Args:
        path: The file path from where the object will be loaded.

    Returns:
        The loaded Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If loading fails with both joblib and pickle.
    """
    path = Path(path)
    logger = logging.getLogger(__name__)
    if not path.exists():
        logger.error(f"File not found at {path}")
        raise FileNotFoundError(f"No file found at the specified path: {path}")

    try:
        obj = joblib.load(path)
        logger.info(f"Object loaded successfully from {path} using joblib.")
        return obj
    except (TypeError, ValueError, pickle.UnpicklingError, EOFError, AttributeError) as e_joblib: # Added AttributeError common in joblib load issues
        logger.warning(f"Could not load object using joblib ({e_joblib}). Trying pickle.")
        try:
            with open(path, 'rb') as f:
                import sys
                original_recursion_limit = sys.getrecursionlimit()
                sys.setrecursionlimit(max(original_recursion_limit, 2000))
                obj = pickle.load(f)
                sys.setrecursionlimit(original_recursion_limit)
            logger.info(f"Object loaded successfully from {path} using pickle.")
            return obj
        except Exception as e_pickle:
            logger.error(f"Failed to load object from {path} using both joblib and pickle: {e_pickle}", exc_info=True)
            raise

def set_seed(seed_value: int):
    """
    Sets random seeds for Python, NumPy, PyTorch, and TensorFlow (if available)
    for reproducibility.

    Args:
        seed_value: The integer seed value.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    logger = logging.getLogger(__name__)
    logger.info(f"Set random seed to {seed_value} for random, numpy, os.environ.")

    if _TORCH_AVAILABLE:
        try:
            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed_value)
                torch.cuda.manual_seed_all(seed_value) # for multi-GPU
                # Potentially makes things slower, but more reproducible
                # torch.backends.cudnn.deterministic = True
                # torch.backends.cudnn.benchmark = False
            logger.info(f"Set random seed for PyTorch.")
        except Exception as e:
            logger.warning(f"Could not set PyTorch seeds: {e}")

    if _TF_AVAILABLE:
        try:
            tf.random.set_seed(seed_value)
            # Optional: Configure for determinism (might impact performance)
            # os.environ['TF_DETERMINISTIC_OPS'] = '1'
            # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
            logger.info(f"Set random seed for TensorFlow.")
        except Exception as e:
            logger.warning(f"Could not set TensorFlow seeds: {e}")

def check_gpu_availability() -> tuple[bool, bool]:
    """
    Checks and logs the availability of GPUs for PyTorch and TensorFlow.

    Returns:
        A tuple (torch_gpu_available, tf_gpu_available).
    """
    logger = logging.getLogger(__name__)
    torch_gpu_available = False
    if _TORCH_AVAILABLE:
        try:
            torch_gpu_available = torch.cuda.is_available()
            if torch_gpu_available:
                gpu_count = torch.cuda.device_count()
                logger.info(f"PyTorch: GPU available. Found {gpu_count} CUDA device(s).")
            else:
                logger.info("PyTorch: No CUDA GPU available.")
        except Exception as e:
            logger.warning(f"Error checking PyTorch GPU availability: {e}")
    else:
        logger.info("PyTorch: Not installed, GPU check skipped.")

    tf_gpu_available = False
    if _TF_AVAILABLE:
        try:
            gpus = tf.config.list_physical_devices('GPU')
            tf_gpu_available = len(gpus) > 0
            if tf_gpu_available:
                logger.info(f"TensorFlow: GPU available. Found {len(gpus)} GPU device(s).")
                # Log detail for each GPU
                for gpu in gpus:
                   logger.debug(f"  - {gpu}")
            else:
                logger.info("TensorFlow: No GPU available.")
        except Exception as e:
            logger.warning(f"Error checking TensorFlow GPU availability: {e}")
    else:
        logger.info("TensorFlow: Not installed, GPU check skipped.")

    return torch_gpu_available, tf_gpu_available

def format_metrics(metrics: dict[str, Any], precision: int = 4) -> str:
    """
    Formats a dictionary of metrics into a readable string for logging.

    Args:
        metrics: Dictionary containing metric names and values.
        precision: The number of decimal places for float values.

    Returns:
        A formatted string representation of the metrics.
    """
    formatted_parts = []
    for key, value in sorted(metrics.items()): # Sort for consistent order
        if isinstance(value, float):
            formatted_parts.append(f"{key}={value:.{precision}f}")
        else:
            formatted_parts.append(f"{key}={value}")
    return ", ".join(formatted_parts)