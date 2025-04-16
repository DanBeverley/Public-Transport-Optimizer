"""
Handles preprocessing of loaded historical data for model training.

Key steps include:
1. Renaming and standardizing columns from source-specific names.
2. Performing necessary unit conversions.
3. Handling missing values.
4. Fitting and applying feature scaling.
"""
import logging
from typing import Tuple, Dict, Any, Optional, Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from . import config
from . import utils

logger = logging.getLogger(__name__)


# Column renaming and Unit conversion mapping
# Define explicit mappings from raw dataset columns to standardized ones.
# This is crucial for handling different dataset schemas.
# Add entries for ALL expected raw columns from MTA, HSL, CTA configs.

STANDARDIZATION_MAP:Dict[str, Dict[str, str]] = {
    "MTA":{
        # Raw MTA Name -> Standardized name
        config.DATASET_CONFIGS["MTA"]["timestamp_col"]:"timestamp",
        "trip_id":"trip_id",
        "route_id":"route_id",
        "stop_sequence":"stop_sequence",
        "vehicle_id":"vehicle_id",
        "gtfs_realtime_delay":"current_delay_seconds",
        "speed":"current_speed_mps",
        "latitude": "latitude",
        "longitude": "longitude",
        "day_of_week": "day_of_week",
        "time_of_day_secs": "time_of_day_seconds",
        "month_of_year": "month",
        "is_weekend_flag": "is_weekend",
        "weather_temp_celsius": "weather_temp_c",
        "weather_precip_mm": "weather_precip_mm",
        "traffic_congestion_level": "traffic_level_index",
        config.DATASET_CONFIGS["MTA"]["target_variable"]: config.STANDARDIZED_TARGET_VARIABLE,
    },
    "HSL":{
        config.DATASET_CONFIGS["HSL"]["timestamp_col"]: "timestamp",
        "journey_id": "trip_id", 
        "line_id": "route_id",
        "stop_point_sequence": "stop_sequence",
        "vehicle_ref": "vehicle_id",
        "observed_delay": "current_delay_seconds",
        "velocity_kmh": "current_speed_kmh",
        "lat": "latitude",
        "lon": "longitude",
        "weekday": "day_of_week", 
        "secs_since_midnight": "time_of_day_seconds",
        "month": "month",
        "weekend_indicator": "is_weekend",
        "air_temp": "weather_temp_c",
        "precipitation_intensity": "weather_precip_mm",
        "road_traffic_index": "traffic_level_index",
        config.DATASET_CONFIGS["HSL"]["target_variable"]: config.STANDARDIZED_TARGET_VARIABLE,
    },
    "CTA":{
        config.DATASET_CONFIGS["CTA"]["timestamp_col"]: "timestamp",
        "tripid": "trip_id",
        "rt": "route_id",
        "seq": "stop_sequence",
        "vid": "vehicle_id",
        "delay_rt": "current_delay_seconds", 
        "spd_mph": "current_speed_mph", 
        "lat": "latitude",
        "lon": "longitude",
        "daynum": "day_of_week", 
        "timesec": "time_of_day_seconds",
        "mon": "month", 
        "is_wknd": "is_weekend",
        "temp_f": "weather_temp_f",
        "precip_in": "weather_precip_in",
        "traffic_score": "traffic_level_index",
        config.DATASET_CONFIGS["CTA"]["target_variable"]: config.STANDARDIZED_TARGET_VARIABLE,
    }
}

def _perform_unit_conversions(df:pd.DataFrame) -> pd.DataFrame:
    """Applies necessary unit conversions after initial renaming"""
    logger.debug("Performing unit conversions...")
    if "current_speed_kmh" in df.columns:
        logger.debug("Converting speed from km/h to m/s")
        df['current_speed_mps'] = df['current_speed_kmh'] * 1000 / 3600
        df.drop(columns = ["current_speed_kmh"], inplace = True)
    if "current_speed_mph" in df.columns:
        logger.debug("Converting speed from mph to m/s")
        df["current_speed_mps"] = df["current_speed_mph"] * 1609.34 / 3600
        df.drop(columns = ["current_speed_mph"], inplace = True)
    if "weather_temp_f" in df.columns:
        logger.debug("Converting temperature from Fahrenheit to Celsius")
        df["weather_temp_c"] = (df["weather_temp_f"] - 32) * 5/9
        df.drop(columns = ["weather_temp_f"], inplace = True)
    if "weather_precip_in" in df.columns:
        logger.debug("Converting precipitation from inches to mm")
        df["weather_precip_mm"] = df["weather_precip_in"] * 25.4
        df.drop(columns = ["weather_precip_in"], inplace = True)
    # TODO: Add other conversions as identified (e.g., HSL precipitation intensity?)
    # Verify day-of-week mappings (0=Monday standard)

    return df

def standardize_data(df:pd.DataFrame, dataset_name:str) -> Optional[pd.DataFrame]:
    """
    Renames columns to standardized names and performs unit conversions.

    Args:
        df: Input DataFrame with raw column names for the dataset.
        dataset_name: The name of the dataset ('MTA', 'HSL', 'CTA').

    Returns:
        DataFrame with standardized column names and units, or None on error.
    """
    logger.info(f"Standardizing columns and units for dataset: {dataset_name}")
    if dataset_name not in STANDARDIZATION_MAP:
        logger.error(f"Standardization mapping not found for dataset {dataset_name}")
        return None
    mapping = STANDARDIZATION_MAP[dataset_name]
    raw_columns_in_df = set(df.columns)
    required_raw_columns = set(mapping.keys())
    # Check if all expected raw columns are present
    missing_raw_cols = required_raw_columns - raw_columns_in_df
    if missing_raw_cols:
        logger.warning(f"Dataset '{dataset_name}' is missing expected raw columns needed for standardization: {missing_raw_cols}.")
    # Select only columns that are in the mapping
    cols_to_rename = {raw:std for raw, std in mapping.items() if raw in raw_columns_in_df}
    try:
        standardized_df = df.rename(columns = cols_to_rename)
        logger.info(f"Renamed {len(cols_to_rename)} columns")

        standardized_columns_present = list(cols_to_rename.values())
        standardized_df = standardized_df[standardized_columns_present]
        # Perform unit conversions
        standardized_df = _perform_unit_conversions(standardized_df)
        # Verify standardized columns are present
        expected_std_cols = set(config.STANDARDIZED_FEATURE_COLUMNS + [config.STANDARDIZED_TARGET_VARIABLE])
        current_std_cols = set(standardized_df.columns)
        missing_std_cols = expected_std_cols - current_std_cols
        if missing_std_cols:
            logger.warning(f"After standardization, expected columns are missing: {missing_std_cols}"
                           "Might cause issues in feature engineering or modeling")
            # Optionally add missing columns filled with NaN
            for col in missing_std_cols:standardized_df[col] = np.nan
        logger.info(f"Standardization complete. Resulting columns: {list(standardized_df.columns)}")
        return standardized_df
    except Exception as e:
        logger.error(f"Error during data standardization for {dataset_name}: {e}, exc_info = True")
        return None

def handle_missing_values(df:pd.DataFrame, strategy:str = config.MISSING_VALUE_STRATEGY) -> pd.DataFrame:
    """
    Handles missing values in the DataFrame using the specified strategy.

    Args:
        df: Input DataFrame (assumed to have standardized column names).
        strategy: 'mean', 'median', 'zero', 'ffill', 'bfill'.

    Returns:
        DataFrame with missing values handled.
    """
    logger.info(f"Handling missing values using strategy: '{strategy}'")
    numeric_cols = df.select_dtypes(include=np.number).columns
    initial_na_counts = df[numeric_cols].isnull().sum()
    initial_na_counts = initial_na_counts[initial_na_counts > 0]
    if initial_na_counts.empty:
        logger.info("No missing numeric values found")
        #TODO: Handle non-numeric
        return df
    logger.info(f"Missing values before handling: \n{initial_na_counts}")
    if strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == "zero":
        df[numeric_cols] = df[numeric_cols].fillna(0)
    elif strategy == "ffill":
         df[numeric_cols] = df.groupby("trip_id")[numeric_cols].ffill() # Grouped fill
         # df[numeric_cols] = df[numeric_cols].ffill() # Non-grouped fill
    elif strategy == "bfill":
        df[numeric_cols] = df.groupby("trip_id")[numeric_cols].bfill()
        # df[numeric_cols] = df[numeric_cols].bfill()
    else:
        logger.warning(f"Unknown missing value strategy: '{strategy}'. No imputation performed")
        return df
    
    final_na_counts = df[numeric_cols].isnull().sum()
    final_na_counts = final_na_counts[final_na_counts > 0]
    if not final_na_counts.empty:
        logger.warning(f"Missing values remaining after handling ({strategy}):\n{final_na_counts}")
    else:
        logger.info("Missing value handling complete")
    return df

def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = config.STANDARDIZED_TARGET_VARIABLE,
    scaler_type: str = config.SCALER_TYPE,
    save_path: Optional[Path] = config.SCALER_SAVE_PATH,
    load_path: Optional[Path] = config.BASE_SCALER_LOAD_PATH if config.TRAINING_MODE == 'finetune' else None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Any]]:
    """
    Applies feature scaling (StandardScaler or MinMaxScaler) to the datasets.
    Fits the scaler ONLY on the training data, or loads a pre-fitted scaler.

    Args:
        train_df: Training DataFrame with standardized columns.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        target_col: Name of the target variable column (excluded from scaling).
        scaler_type: 'standard', 'minmax', or None.
        save_path: Path to save the fitted scaler object. Required if not loading.
        load_path: Path to load a pre-fitted scaler object (e.g., for fine-tuning or inference).

    Returns:
        Tuple: (scaled_train_df, scaled_val_df, scaled_test_df, fitted_scaler_object)
    """
    logger.info(f"Applying feature scaling (type: {scaler_type})")
    if scaler_type not in ["standard", "minmax"]:
        logger.info("No scaling applied")
        return train_df.copy(), val_df.copy(), test_df.copy(), None
    
    # Identify numeric feature columns
    numeric_cols = train_df.select_dtypes(include = np.number).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if not numeric_cols:
        logger.warning("No numeric feature columns found to scale (excluding target). Skipping scaling.")
        return train_df.copy(), val_df.copy(), test_df.copy(), None
    
    logger.debug(f"Columns to be scale: {numeric_cols}")

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()
    scaler = None
    if load_path and Path(load_path).exists():
        try:
            scaler = utils.load_object(load_path)
            logger.info(f"Loaded pre-fitted scaler from: {load_path}")
            if scaler_type == "standard" and not isinstance(scaler, StandardScaler):
                logger.warning(f"Loaded scaler type ({type(scaler)}) does not match config 'standard'.")
            if scaler_type == 'minmax' and not isinstance(scaler, MinMaxScaler):
                logger.warning(f"Loaded scaler type ({type(scaler)}) does not match config 'minmax'.")
        except Exception as e:
            logger.error(f"Failed to load scaler from {load_path}: {e}. Will fit a new scaler.", exc_info=True)
            scaler = None # Force refitting
    if scaler is None:
        # Fit new scaler on the training data
        logger.info("Fitting new scaler on training data...")
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        # Ensure training data for fitting does not contain NaNs in numeric cols
        if train_scaled[numeric_cols].isnull().any().any():
            logger.warning(f"NaNs found in training data numeric columns before scaling fit: {train_scaled[numeric_cols].isnull().sum()[lambda x:x>0].to_dict()}")
            #TODO: Improve handle missing values
            train_data_for_fit = train_scaled[numeric_cols].fillna(0)
        else:
            train_data_for_fit = train_scaled[numeric_cols]
        try:
            scaler.fit(train_data_for_fit)
            logger.info("Scaler fitted successfully.")
            if save_path:
                utils.save_object(scaler, save_path)
            else:
                logger.warning("Scaler was fitted but 'save_path' is not provided in config. Scaler will not be saved.")
        except Exception as e:
            logger.error(f"Error fitting scaler: {e}", exc_info=True)
            # Cannot proceed without a scaler
            raise ValueError("Scaler fitting failed.") from e

    # Transform all datasets using the fitted (or loaded) scaler
    try:
        logger.info("Transforming datasets with the scaler...")
        # Handle potential NaNs before transform by imputing with 0 (or mean/median if scaler stores them)
        train_scaled[numeric_cols] = scaler.transform(train_scaled[numeric_cols].fillna(0))
        if not val_scaled.empty:
            val_scaled[numeric_cols] = scaler.transform(val_scaled[numeric_cols].fillna(0))
        if not test_scaled.empty:
            test_scaled[numeric_cols] = scaler.transform(test_scaled[numeric_cols].fillna(0))
        logger.info("Scaling transformation complete.")
    except Exception as e:
        logger.error(f"Error applying scaler transform: {e}", exc_info=True)
        raise ValueError("Scaler transformation failed.") from e

    return train_scaled, val_scaled, test_scaled, scaler

def preprocess_data(
    train_df_raw: pd.DataFrame,
    val_df_raw: pd.DataFrame,
    test_df_raw: pd.DataFrame,
    dataset_name: str = config.ACTIVE_DATASET_NAME
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Any]]:
    """
    Applies the full preprocessing pipeline: standardization, missing values, scaling.

    Args:
        train_df_raw: Raw training DataFrame from data_loader.
        val_df_raw: Raw validation DataFrame.
        test_df_raw: Raw test DataFrame.
        dataset_name: Name of the dataset being processed.

    Returns:
        Tuple: (processed_train_df, processed_val_df, processed_test_df, fitted_scaler)
               Returns original dataframes if steps fail where appropriate.
    """
    logger.info(f"--- Starting Preprocessing Pipeline for Dataset: {dataset_name} ---")

    # 1. Standardize Columns and Units
    train_std = standardize_data(train_df_raw, dataset_name)
    if train_std is None: return train_df_raw, val_df_raw, test_df_raw, None # Return raws if standardization fails
    val_std = standardize_data(val_df_raw, dataset_name) if not val_df_raw.empty else pd.DataFrame(columns=train_std.columns)
    test_std = standardize_data(test_df_raw, dataset_name) if not test_df_raw.empty else pd.DataFrame(columns=train_std.columns)
    # Ensure val/test got standardized if not None
    if not val_df_raw.empty and val_std is None: return train_df_raw, val_df_raw, test_df_raw, None
    if not test_df_raw.empty and test_std is None: return train_df_raw, val_df_raw, test_df_raw, None


    # 2. Handle Missing Values
    train_clean = handle_missing_values(train_std)
    val_clean = handle_missing_values(val_std) if not val_std.empty else val_std
    test_clean = handle_missing_values(test_std) if not test_std.empty else test_std

    # 3. Scale Features
    try:
        train_processed, val_processed, test_processed, scaler = scale_features(
            train_clean, val_clean, test_clean
        )
    except ValueError as e: # Handle potential failure in scaling
         logger.error(f"Scaling failed: {e}. Returning unscaled data.")
         return train_clean, val_clean, test_clean, None


    logger.info("--- Preprocessing Pipeline Complete ---")
    return train_processed, val_processed, test_processed, scaler
