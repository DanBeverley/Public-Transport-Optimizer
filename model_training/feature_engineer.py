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

def select_final_features_and_target(
    df: pd.DataFrame,
    target_col: str = config.STANDARDIZED_TARGET_VARIABLE,
    identifier_cols: List[str] = ['timestamp', 'trip_id', 'route_id', 'vehicle_id', 'stop_id', 'stop_sequence'] # Standardized identifiers
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    """
    Selects final numeric feature columns, separates target and identifiers,
    and drops rows with NaNs introduced by feature engineering.

    Args:
        df: DataFrame after all feature engineering steps.
        target_col: Name of the standardized target variable.
        identifier_cols: List of standardized identifier columns to keep separately.

    Returns:
        Tuple: (
            features_df: DataFrame containing only the final numeric features.
            target_series: Series containing the target variable.
            identifiers_df: DataFrame containing the specified identifier columns.
            final_feature_names: List of column names in features_df.
        )
        Returns empty structures if processing fails or results in no data.
    """
    logger.info("Selecting final features, target, identifiers and handling NaNs from feature engineering.")
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in final DataFrame.")
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), []

    # Identify numeric feature columns (excluding target and specified identifiers)
    exclude_cols = set([target_col] + identifier_cols)
    potential_feature_cols = [col for col in df.columns if col not in exclude_cols]
    # Select only numeric types from potential features
    numeric_feature_cols = df[potential_feature_cols].select_dtypes(include=np.number).columns.tolist()

    if not numeric_feature_cols:
         logger.error("No numeric feature columns found after exclusions.")
         return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), []

    logger.info(f"Identified {len(numeric_feature_cols)} numeric features: {numeric_feature_cols[:5]}...")

    # Columns to keep temporarily for NaN dropping and final selection
    cols_to_keep = numeric_feature_cols + [target_col] + [id_col for id_col in identifier_cols if id_col in df.columns]

    # Drop rows with NaNs ONLY in feature columns or target column
    initial_rows = len(df)
    # Use subset based on numeric features + target to decide which rows to drop
    dropna_subset = numeric_feature_cols + [target_col]
    df_final = df[cols_to_keep].dropna(subset=dropna_subset)
    rows_dropped = initial_rows - len(df_final)
    logger.info(f"Dropped {rows_dropped} rows due to NaNs in features/target (expected from lags/rolling).")

    if df_final.empty:
         logger.error("DataFrame is empty after dropping NaNs from feature engineering.")
         return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), []

    # Separate features, target, and identifiers
    X = df_final[numeric_feature_cols]
    y = df_final[target_col]
    identifiers = df_final[[id_col for id_col in identifier_cols if id_col in df_final.columns]]
    final_feature_names = numeric_feature_cols # Store the final list of names

    return X, y, identifiers, final_feature_names

def engineer_features_for_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, # Train X, y, ids
               pd.DataFrame, pd.Series, pd.DataFrame, # Val X, y, ids
               pd.DataFrame, pd.Series, pd.DataFrame, # Test X, y, ids
               List[str]]:                           # Final feature names
    """
    Applies the full feature engineering pipeline to preprocessed train, val, test sets.
    Returns DataFrames for features, targets, and identifiers, plus the final feature names list.

    Args:
        train_df: Preprocessed training DataFrame (standardized columns, scaled).
        val_df: Preprocessed validation DataFrame.
        test_df: Preprocessed test DataFrame.

    Returns:
        Tuple containing features, target, identifiers for train, val, test,
        and the list of final feature names.
    """
    logger.info("--- Starting Feature Engineering Pipeline ---")
    final_data = {}
    final_feature_names = []

    target_col = config.STANDARDIZED_TARGET_VARIABLE

    for name, df_in in [('train', train_df), ('val', val_df), ('test', test_df)]:
        logger.info(f"Processing dataset split: {name} (shape: {df_in.shape})")
        if df_in.empty:
            logger.warning(f"Dataset split '{name}' is empty. Skipping.")
            # Store empty structures
            final_data[name] = (pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame())
            continue

        df_processed = generate_cyclical_features(df_in, config.CYCLICAL_FEATURES)
        df_processed = generate_lag_features(df_processed, target_col, config.LAG_FEATURES[target_col])
        df_processed = generate_rolling_features(df_processed, config.ROLLING_FEATURES)

        X_df, y_series, ids_df, current_feature_names = select_final_features_and_target(df_processed, target_col)
        final_data[name] = (X_df, y_series, ids_df)

        if name == 'train':
            final_feature_names = current_feature_names
            if not final_feature_names:
                 logger.error("No features selected from training data. Aborting.")
                 # Return empty tuples to signal failure
                 empty_res = (pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame())
                 return empty_res, empty_res, empty_res, []

    if not final_data['val'][0].empty and list(final_data['val'][0].columns) != final_feature_names:
        logger.warning("Validation set feature names mismatch training set features after engineering!")
        try:
            final_data['val'] = (final_data['val'][0].reindex(columns=final_feature_names, fill_value=0), final_data['val'][1], final_data['val'][2])
            logger.info("Reindexed validation features to match training features.")
        except Exception as e: logger.error(f"Failed to reindex validation features: {e}")

    if not final_data['test'][0].empty and list(final_data['test'][0].columns) != final_feature_names:
        logger.warning("Test set feature names mismatch training set features after engineering!")
        try:
            final_data['test'] = (final_data['test'][0].reindex(columns=final_feature_names, fill_value=0), final_data['test'][1], final_data['test'][2])
            logger.info("Reindexed test features to match training features.")
        except Exception as e: logger.error(f"Failed to reindex test features: {e}")


    logger.info("--- Feature Engineering Pipeline Complete ---")
    train_res = final_data['train']
    val_res = final_data['val']
    test_res = final_data['test']
    logger.info(f"Final shapes: Train X={train_res[0].shape}, y={train_res[1].shape}, ids={train_res[2].shape}; "
                f"Val X={val_res[0].shape}, y={val_res[1].shape}, ids={val_res[2].shape}; "
                f"Test X={test_res[0].shape}, y={test_res[1].shape}, ids={test_res[2].shape}")

    return train_res, val_res, test_res, final_feature_names


