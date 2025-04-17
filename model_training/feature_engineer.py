"""
Generates time series features from preprocessed data for model training.

Assumes input DataFrame has standardized column names and scaled numeric features.
Creates lag features, rolling window features, and cyclical encodings.
"""
import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional

from . import utils
from . import config

logger = logging.getLogger(__name__)

def generate_cyclical_features(df:pd.DataFrame, cols:List[str]) -> pd.DataFrame:
    """
    Encodes specified time-related columns into sin/cos cyclical features.

    Args:
        df: Input DataFrame (preprocessed).
        cols: List of standardized column names to encode (e.g., 'day_of_week', 'time_of_day_seconds').

    Returns:
        DataFrame with original columns dropped and new sin/cos columns added.
    """
    logger.info(f"Generating cyclical features for columns: {cols}")
    df_out = df.copy()
    for col in cols:
        if col not in df_out.columns:
            logger.warning(f"Column '{col}' not found for cyclical encoding. Skipping...")
            continue
        logger.debug(f"Encoding column:{col}")
        values = df_out[col]
        if col == "day_of_week": # For 0 - 6 (Mon - Sun)
            max_val = 6
            period = 7
        elif col == "time_of_day_seconds":
            max_val = 24 * 60 * 60 - 1
            period = 24 * 60 * 60
        elif col == "month":
            max_val = 12
            period = 12
            # Test and adjust 1-12 range to 0-11 for easer sin/cos calculation if needed
            values = values - 1
        else:
            logger.warning(f"No predefined period found for cyclical column '{col}'. Attempting auto-detection")
            max_val = values.max()
            period = max_val + 1
            if max_val <= 0:
                logger.warning(f"Cannot encode non-positive max value column '{col}'. Skipping")
                continue
        # Normalize to [0, 2*pi] in a more simple term , a full circle
        norm_values = 2* np.pi * values / period
        df_out[f"{col}_sin"] = np.sin(norm_values)
        df_out[f"{col}_cos"] = np.cos(norm_values)
        df_out.drop(columns=[col], inplace = True)
    
    return df_out

def generate_lag_features(df:pd.DataFrame, target_col:str, lags:List[int],
                          group_col:str="trip_id") -> pd.DataFrame:
    """
    Generates lag features for the target variable, grouped by trip.

    Args:
        df: Input DataFrame (preprocessed, time-sorted).
        target_col: The standardized name of the target variable.
        lags: List of integers representing the time steps to lag by.
        group_col: Column to group by before lagging (e.g., 'trip_id', 'vehicle_id').

    Returns:
        DataFrame with new lag feature columns added.
    """
    logger.info(f"Generating lag features for '{target_col}' (lags:{lags}),
                groupsed by '{group_col}'")
    df_out = df.copy()
    if target_col not in df_out.columns:
        logger.error(f"Target column '{target_col}' not found for lag features")
        return df_out
    if group_col not in df_out.columns:
        logger.warning(f"Group column '{group_col}' not found. Performing global lag (maybe incorrect)")
        for lag in lags:
            df_out[f"{target_col}_lag_{lag}"] = df_out[target_col].shift(lag)
    else:
        logger.debug(f"Applying lags within groups defined by {group_col}")
        for lag in lags:
            df_out[f"{target_col}_lag_{lag}"] = df_out.groupby(group_col)[target_col].shift(lag)
    # Lags introduce NaNs at the beginning of each group/series
    num_nans = df_out[[f"{target_col}_lag_{l}" for l in lags]].isnull().sum().sum()
    logger.info(f"Lag features generated. Introduced {num_nans} NaN values")
    # TODO: implement missing value handling for lags after feature generation
    return df_out

def generate_rolling_features(df:pd.DataFrame, features_config:Dict[str, Dict],
                              group_col:str="trip_id") -> pd.DataFrame:
    """
    Generates rolling window features (mean, std, etc.) grouped by trip.

    Args:
        df: Input DataFrame (preprocessed, time-sorted).
        features_config: Dictionary defining rolling features.
                         Example: {'col_name': {'window': [3, 5], 'agg': ['mean', 'std']}}
        group_col: Column to group by before calculating rolling features.

    Returns:
        DataFrame with new rolling feature columns added.
    """
    logger.info(f"Generating rolling features (grouped by '{group_col}')")
    df_out = df.copy()
    for col, params in features_config.items():
        if col not in df_out.columns:
            logger.warning(f"Column '{col}' not found for rolling features. Skipping.")
            continue
        windows = params.get('window', [])
        aggs = params.get('agg', [])
        for window in windows:
            logger.debug(f"Calculating rolling window={window} for column '{col}'")
            # closed='left' ensures window uses past data only up to (but not including) current step
            # min_periods=1 allows calculation even if window isn't full at start
            rolling_obj = df_out.groupby(group_col)[col].rolling(window = window,
                                                                 min_periods = 1,
                                                                 closed="left")
            for agg_func in aggs:
                try:
                    agg_series = rolling_obj.agg(agg_func)
                    # Result needs index alignment back to original df after groupby/rolling
                    # Remove the group index level, keep the original index level
                    agg_series = agg_series.reset_index(level=group_col, drop = True)
                    new_col_name = f"{col}_roll_{window}_{agg_func}"
                    df_out[new_col_name] = agg_series
                    logger.debug(f"Generated rolling feature: {new_col_name}")
                except Exception as e:
                    logger.error(f"Failed to calculate rolling aggregate '{agg_func}' for window {window} on column '{col}': {e}", exc_info = True)
    return df_out

def select_final_features(df:pd.DataFrame, target_col:str = config.STANDARDIZED_TARGET_VARIABLE) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Selects the final feature columns for the model and separates the target variable.
    Drops rows with NaNs introduced by feature engineering (lags/rolling).

    Args:
        df: DataFrame after all feature engineering steps.
        target_col: Name of the standardized target variable.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (features_df, target_series)
    """
    logger.info("Selecting final features and handling NaNs from feature engineering.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in final DataFrame.")

    # Define final feature columns (excluding target and identifiers like timestamp, trip_id etc.)
    exclude_cols = [target_col, 'timestamp', 'trip_id', 'route_id', 'vehicle_id', 'stop_sequence'] # Add others?
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

    logger.info(f"Final selected features ({len(feature_cols)}): {feature_cols}")

    # Handle NaNs introduced by lag/rolling features
    initial_rows = len(df)
    df_final = df.dropna(subset=feature_cols + [target_col])
    rows_dropped = initial_rows - len(df_final)
    logger.info(f"Dropped {rows_dropped} rows due to NaNs after feature engineering (expected from lags/rolling).")

    if df_final.empty:
         raise ValueError("DataFrame is empty after dropping NaNs from feature engineering. Check lag/rolling parameters or data size.")

    X = df_final[feature_cols]
    y = df_final[target_col]

    return X, y

def engineer_features_for_training(train_df:pd.DataFrame,
                                    val_df:pd.DataFrame,
                                    test_df:pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies the full feature engineering pipeline to preprocessed train, val, test sets.

    Args:
        train_df: Preprocessed training DataFrame (standardized columns, scaled).
        val_df: Preprocessed validation DataFrame.
        test_df: Preprocessed test DataFrame.

    Returns:
        Tuple: (train_X, train_y, val_X, val_y, test_X, test_y) as NumPy arrays.
    """
    logger.info("Starting Feature Engineering Pipeline")
    # Apply feature generation steps consistently to all sets
    datasets = {"train":train_df, "val":val_df, "test":test_df}
    processed_datasets = {}
    for name, df in datasets.items():
        logger.info(f"Processing dataset split: {name} (shape:{df.shape})")
        if df.empty:
            logger.warning(f"Dataset split '{name}' is emptyu. Skipping feature engineering")
            processed_datasets[name] = pd.DataFrame()
            continue
        if not all(col in df.columns for col in config.CYCLICAL_FEATURES + list(config.LAG_FEATURES.keys()) + list(config.ROLLING_FEATURES.keys())):
            logger.warning(f"Dataset split '{name}' missing some columns required for feature engineering. Results may be incomplete.")
        # 1. Cyclical Features
        df_processed = generate_cyclical_features(df, config.CYCLICAL_FEATURES)

        # 2. Lag Features
        df_processed = generate_lag_features(df_processed, config.STANDARDIZED_TARGET_VARIABLE, config.LAG_FEATURES[config.STANDARDIZED_TARGET_VARIABLE])

        # 3. Rolling Features
        df_processed = generate_rolling_features(df_processed, config.ROLLING_FEATURES)

        processed_datasets[name] = df_processed


    # 4. Select Final Features and Handle NaNs
    # Important: Handle NaNs *after* generating all features for all splits
    logger.info("Selecting final features and handling NaNs across all splits...")
    final_data = {}
    try:
        X_train, y_train = select_final_features(processed_datasets['train'])
        final_data['train'] = (X_train, y_train)

        if not processed_datasets['val'].empty:
            X_val, y_val = select_final_features(processed_datasets['val'])
            final_data['val'] = (X_val, y_val)
        else: final_data['val'] = (pd.DataFrame(), pd.Series()) # Empty

        if not processed_datasets['test'].empty:
            X_test, y_test = select_final_features(processed_datasets['test'])
            final_data['test'] = (X_test, y_test)
        else: final_data['test'] = (pd.DataFrame(), pd.Series()) # Empty

        # Check feature alignment (columns should match between train, val, test X)
        if not final_data['val'][0].empty and list(final_data['train'][0].columns) != list(final_data['val'][0].columns):
             logger.warning("Feature columns mismatch between train and validation sets after NaN handling.")
        if not final_data['test'][0].empty and list(final_data['train'][0].columns) != list(final_data['test'][0].columns):
             logger.warning("Feature columns mismatch between train and test sets after NaN handling.")

        # Convert to NumPy arrays for model input
        train_X, train_y = final_data['train'][0].values, final_data['train'][1].values
        val_X = final_data['val'][0].values if not final_data['val'][0].empty else np.array([])
        val_y = final_data['val'][1].values if not final_data['val'][1].empty else np.array([])
        test_X = final_data['test'][0].values if not final_data['test'][0].empty else np.array([])
        test_y = final_data['test'][1].values if not final_data['test'][1].empty else np.array([])

        logger.info("--- Feature Engineering Pipeline Complete ---")
        logger.info(f"Final shapes: Train X={train_X.shape}, y={train_y.shape}; "
                    f"Val X={val_X.shape}, y={val_y.shape}; Test X={test_X.shape}, y={test_y.shape}")

        return train_X, train_y, val_X, val_y, test_X, test_y

    except ValueError as e:
         logger.error(f"Error during final feature selection/NaN handling: {e}", exc_info=True)
         return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])


