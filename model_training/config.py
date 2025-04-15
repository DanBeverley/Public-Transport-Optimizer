# --- START OF FILE model_training/config.py ---
"""
Configuration settings for the model training pipeline.

Centralizes hyperparameters, paths, and other parameters for reproducibility.
Supports configurations for multiple datasets (MTA, HSL, CTA).
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union, Literal, Any

# --- Project Structure ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTION_SERVICE_DIR = PROJECT_ROOT / "prediction_service"
MODEL_SAVE_DIR = PREDICTION_SERVICE_DIR / "models"
LOG_DIR = PROJECT_ROOT / "logs" / "training"
# Directory where raw/interim data pipeline outputs might be stored
DATA_PIPELINE_RAW_OUTPUT_DIR = PROJECT_ROOT / "data" / "historical" # Example

# --- Ensure Directories Exist ---
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- Training Mode ---
# 'base': Train a new model from scratch.
# 'finetune': Load a pre-trained model and fine-tune on a new dataset.
TRAINING_MODE: Literal['base', 'finetune'] = 'base'

# --- Active Dataset Selection ---
# Specify which dataset configuration to use for this run (MTA, HSL, CTA)
ACTIVE_DATASET_NAME: Literal['MTA', 'HSL', 'CTA'] = 'MTA'

# --- Dataset Specific Configurations ---
# Define parameters for each dataset. Paths and columns might need adjustment
# based on how your data_pipeline stores/processes data for each agency.
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "MTA": {
        "name": "MTA",
        "description": "Metropolitan Transportation Authority (New York City)",
        "data_path_pattern": DATA_PIPELINE_RAW_OUTPUT_DIR / "mta" / "processed" / "history_{date}.parquet", # Example path
        "file_type": "parquet",
        "start_date": "2023-01-01", # Training range for MTA
        "end_date": "2023-12-31",
        "timestamp_col": "timestamp",
        # List columns *as they appear* in this dataset's files BEFORE standardization
        "feature_columns": [
            'timestamp', 'trip_id', 'route_id', 'stop_sequence', 'vehicle_id',
            'gtfs_realtime_delay', # Example MTA-specific name
            'speed', 'latitude', 'longitude',
            'day_of_week', 'time_of_day_secs', 'month_of_year', 'is_weekend_flag',
            'weather_temp_celsius', 'weather_precip_mm', 'traffic_congestion_level',
        ],
        "target_variable": 'actual_stop_delay_seconds', # Target column name for MTA
        "scaler_save_path": MODEL_SAVE_DIR / "mta_scaler.joblib",
    },
    "HSL": {
        "name": "HSL",
        "description": "Helsinki Region Transport (Finland)",
        "data_path_pattern": DATA_PIPELINE_RAW_OUTPUT_DIR / "hsl" / "processed" / "history_{date}.csv", # Example path
        "file_type": "csv",
        "start_date": "2024-01-01", # Fine-tuning range for HSL
        "end_date": "2024-03-31",
        "timestamp_col": "observation_time",
        "feature_columns": [
            'observation_time', 'journey_id', 'line_id', 'stop_point_sequence', 'vehicle_ref',
            'observed_delay', 
            'velocity_kmh', 'lat', 'lon',
            'weekday', 'secs_since_midnight', 'month', 'weekend_indicator',
            'air_temp', 'precipitation_intensity', 'road_traffic_index',
        ],
        "target_variable": 'actual_delay_stop_secs', # Target column name for HSL
        "scaler_save_path": MODEL_SAVE_DIR / "hsl_finetune_scaler.joblib", # Potentially different scaler for finetuning
    },
    "CTA": {
        "name": "CTA",
        "description": "Chicago Transit Authority (USA)",
        "data_path_pattern": DATA_PIPELINE_RAW_OUTPUT_DIR / "cta" / "processed" / "{date}_cta_data.parquet", # Example path
        "file_type": "parquet",
        "start_date": "2024-04-01", # Fine-tuning range for CTA
        "end_date": "2024-06-30",
        "timestamp_col": "ts", # Example different column name
        "feature_columns": [
            'ts', 'tripid', 'rt', 'stpid', 'seq', 'vid',
            'delay_rt', # Example CTA-specific name
            'spd_mph', 'lat', 'lon',
            'daynum', 'timesec', 'mon', 'is_wknd',
            'temp_f', 'precip_in', 'traffic_score',
        ],
        "target_variable": 'actual_delay', # Target column name for CTA
        "scaler_save_path": MODEL_SAVE_DIR / "cta_finetune_scaler.joblib",
    },
}

def get_active_dataset_config() -> Dict[str, Any]:
    if ACTIVE_DATASET_NAME not in DATASET_CONFIGS:
        raise ValueError(f"Configuration for dataset '{ACTIVE_DATASET_NAME}' not found in DATASET_CONFIGS.")
    return DATASET_CONFIGS[ACTIVE_DATASET_NAME]

_active_config = get_active_dataset_config()

DATA_PATH_PATTERN = _active_config["data_path_pattern"]
DATA_FILE_TYPE = _active_config["file_type"]
DATA_START_DATE = _active_config["start_date"]
DATA_END_DATE = _active_config["end_date"]
RAW_FEATURE_COLUMNS = _active_config["feature_columns"] 
RAW_TARGET_VARIABLE = _active_config["target_variable"]
TIMESTAMP_COL = _active_config["timestamp_col"]

# Define standardized column names that preprocessing/feature engineering will map to
STANDARDIZED_FEATURE_COLUMNS = [
    'timestamp', 'trip_id', 'route_id', 'stop_sequence', 'vehicle_id',
    'current_delay_seconds', 'current_speed_mps', 'latitude', 'longitude',
    'day_of_week', 'time_of_day_seconds', 'month', 'is_weekend',
    'weather_temp_c', 'weather_precip_mm', 'traffic_level_index',
    # Add standardized names for lag/rolling features etc.
]
STANDARDIZED_TARGET_VARIABLE = 'target_delay_seconds'

# Chronological splitting (ratios remain general)
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15

# Preprocessing
MISSING_VALUE_STRATEGY = 'median' # 'mean', 'median', 'zero', 'ffill', 'bfill'
SCALER_TYPE = 'standard' # 'standard', 'minmax', None
SCALER_SAVE_PATH = Path(_active_config["scaler_save_path"])
BASE_SCALER_LOAD_PATH = MODEL_SAVE_DIR / "mta_scaler.joblib" if TRAINING_MODE == 'finetune' else None

# Feature Engineering
LAG_FEATURES = { STANDARDIZED_TARGET_VARIABLE: [1, 2, 3] } # Apply lags to standardized target
ROLLING_FEATURES = { 'current_delay_seconds': {'window': [3, 5], 'agg': ['mean', 'std']} }
CYCLICAL_FEATURES = ['day_of_week', 'time_of_day_seconds', 'month'] # Apply to standardized columns

# Model Parameters
MODEL_TYPE = 'LSTM'
INPUT_DIM = None 
OUTPUT_DIM = 1
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT_RATE = 0.2
SEQ_LENGTH = 10 

# --- Training ---
SEED = 42
DEVICE = 'cuda' # 'cuda', 'cpu', 'mps'
BATCH_SIZE = 64
GRADIENT_CLIP_VALUE = 1.0

# Training parameters potentially adjusted for base vs fine-tuning
if TRAINING_MODE == 'base':
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    OPTIMIZER = 'adam'
    LOSS_FUNCTION = 'mse'
    EARLY_STOPPING_PATIENCE = 5
    BASE_MODEL_LOAD_PATH = None 
elif TRAINING_MODE == 'finetune':
    LEARNING_RATE = 5e-5 # Lower LR for fine-tuning
    NUM_EPOCHS = 20 # Fewer epochs for fine-tuning
    OPTIMIZER = 'adam'
    LOSS_FUNCTION = 'mse'
    EARLY_STOPPING_PATIENCE = 3
    BASE_MODEL_LOAD_PATH = MODEL_SAVE_DIR / "MTA_LSTM_base_model.pt" 
else:
    raise ValueError(f"Invalid TRAINING_MODE: {TRAINING_MODE}")

# Evaluation
EVALUATION_METRICS = ['mae', 'rmse', 'r2']
MODEL_SUFFIX = "base" if TRAINING_MODE == 'base' else f"finetuned_{ACTIVE_DATASET_NAME}"
BEST_MODEL_SAVE_PATH = MODEL_SAVE_DIR / f"{ACTIVE_DATASET_NAME}_{MODEL_TYPE}_{MODEL_SUFFIX}_best.pt"

# For logging
LOG_LEVEL = 'INFO'
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = LOG_DIR / f"{ACTIVE_DATASET_NAME}_{MODEL_TYPE}_{MODEL_SUFFIX}_{RUN_TIMESTAMP}.log"