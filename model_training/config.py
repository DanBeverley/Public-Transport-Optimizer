"""
Configuration settings for the model training pipeline.

Centralizes hyperparameters, paths, and other parameters for reproducibility.
"""
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PIPELINE_OUTPUT_DIR = PROJECT_ROOT / "data_pipeline" / "output" 
PREDICTION_SERVICE_DIR = PROJECT_ROOT / "prediction_service"
MODEL_SAVE_DIR = PREDICTION_SERVICE_DIR / "models"
LOG_DIR = PROJECT_ROOT / "logs" / "training"

MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Ensure directory exist
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Data loading and splitting
DATA_SOURCE_TYPE = "parquet"
HISTORICAL_DATA_PATH_PATTERN = DATA_PIPELINE_OUTPUT_DIR/"processed_history"/"history_{date}.parquet"

FEATURE_COLUMNS = [
    'timestamp', 'trip_id', 'route_id', 'stop_sequence', 'vehicle_id',
    'current_delay_seconds', 'current_speed_mps', 'latitude', 'longitude',
    'day_of_week', 'time_of_day_seconds', 'month', 'is_weekend',
    'weather_temp_c', 'weather_precip_mm', 'traffic_level_index',
]
TARGET_VARIABLE = 'actual_delay_at_stop_seconds' 

# Chronological splitting
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15

# Preprocessing
MISSING_VALUE_STRATEGY = 'median' # 'mean', 'median', 'zero', 'ffill', 'bfill'
SCALER_TYPE = 'standard' # 'standard', 'minmax', None
SCALER_SAVE_PATH = MODEL_SAVE_DIR / "scaler.joblib"

# Feature Engineering
LAG_FEATURES = { # Target variable based lags
    TARGET_VARIABLE: [1, 2, 3] # Lag by 1, 2, 3 time steps (relative to trip/vehicle)
}
ROLLING_FEATURES = { 
    'current_delay_seconds': {'window': [3, 5], 'agg': ['mean', 'std']} # Rolling mean/std over 3/5 steps
}
CYCLICAL_FEATURES = ['day_of_week', 'time_of_day_seconds', 'month'] # Features to encode cyclically (sin/cos)

MODEL_TYPE = 'LSTM'

INPUT_DIM = None 
OUTPUT_DIM = 1 
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT_RATE = 0.2

SEQ_LENGTH = 10 

GRAPH_TYPE = 'static_route_stop' 
GNN_HIDDEN_DIM = 64
GNN_LAYERS = 2
GRAPH_FEATURE_DIM = None

SEED = 42
DEVICE = 'cuda' # 'cuda', 'cpu', 'mps' (for Apple Silicon)
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
OPTIMIZER = 'adam' # 'adam', 'sgd', 'rmsprop'
LOSS_FUNCTION = 'mse' # 'mse' (Mean Squared Error), 'mae' (Mean Absolute Error)
EARLY_STOPPING_PATIENCE = 5 
GRADIENT_CLIP_VALUE = 1.0 

# Evaluation
EVALUATION_METRICS = ['mae', 'rmse', 'r2'] # Mean Absolute Error, Root Mean Squared Error, R-squared
BEST_MODEL_SAVE_PATH = MODEL_SAVE_DIR / f"{MODEL_TYPE}_best_model.pt" # Or .h5, .joblib depending on model type

# Logging
LOG_LEVEL = 'INFO' # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_FILE = LOG_DIR / f"{MODEL_TYPE}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log" # Timestamped log file