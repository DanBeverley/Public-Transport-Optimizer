"""
Handles loading, splitting, and batching of historical transit data for model training.
Loads data based on the configuration specified in config.py for the ACTIVE_DATASET_NAME.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List, Union, Dict, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Used only for type hint example

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    class Dataset: pass
    class DataLoader: pass

from . import config
from . import utils

logger = logging.getLogger(__name__)

def load_historical_data(
    dataset_name: str = config.ACTIVE_DATASET_NAME
    ) -> Optional[pd.DataFrame]:
    """
    Loads historical transit data for the specified dataset name using its configuration.

    Args:
        dataset_name: The name of the dataset to load (e.g., 'MTA', 'HSL', 'CTA').
                      Must have a corresponding entry in config.DATASET_CONFIGS.

    Returns:
        A pandas DataFrame containing the loaded historical data, sorted by timestamp,
        or None if loading fails.
    """
    logger.info(f"Attempting to load historical data for dataset: '{dataset_name}'")

    try:
        ds_config = config.DATASET_CONFIGS[dataset_name]
        data_path_pattern = ds_config["data_path_pattern"]
        start_date_str = ds_config["start_date"]
        end_date_str = ds_config["end_date"]
        file_type = ds_config["file_type"]
        # Load only the necessary raw columns specified for this dataset
        columns_to_load = ds_config["feature_columns"] + [ds_config["target_variable"]]
        timestamp_col = ds_config["timestamp_col"]
        # date_format in pattern (e.g., %Y-%m-%d or %Y%m%d) - determine from pattern or add to config
        # Heuristic: Assume common formats if not explicitly defined
        date_format = "%Y-%m-%d" 
        if "{date:%Y%m%d}" in str(data_path_pattern): date_format = "%Y%m%d"
        elif "{date}" in str(data_path_pattern): pass # Use default %Y-%m-%d assumption
        #TODO: Implement more robust datetime handling

    except KeyError:
        logger.error(f"Configuration for dataset '{dataset_name}' not found or incomplete in config.py.")
        return None

    logger.info(f"Using configuration for '{ds_config.get('description', dataset_name)}'")
    logger.info(f"Path pattern: {data_path_pattern}, Date range: {start_date_str} to {end_date_str}")

    all_data = []
    try:
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        # Generate daily dates (e.g., 'W' for weekly files)
        date_range = pd.date_range(start_date, end_date, freq='D')

        loaded_files = 0
        required_cols_set = set(columns_to_load)

        for dt in date_range:
            date_str_for_path = dt.strftime(date_format) 
            try:
                 file_path = Path(str(data_path_pattern).format(date=date_str_for_path))
            except KeyError: 
                 file_path = Path(data_path_pattern) 
                 if dt != start_date and loaded_files > 0: 
                     continue
                 if dt > start_date and loaded_files == 0: 
                     logger.error(f"Data path pattern '{data_path_pattern}' seems fixed, but date range is multiple days.")
                     return None


            if file_path.exists():
                logger.debug(f"Loading data from: {file_path}")
                try:
                    if file_type == 'parquet':
                        df = pd.read_parquet(file_path) 
                    elif file_type == 'csv':
                        df = pd.read_csv(file_path)
                    else:
                        logger.error(f"Unsupported file type: {file_type}")
                        return None

                    missing_cols = required_cols_set - set(df.columns)
                    if missing_cols:
                        logger.warning(f"File {file_path} missing required columns: {missing_cols}. Skipping file.")
                        continue

                    df = df[columns_to_load]

                    all_data.append(df)
                    loaded_files += 1
                except FileNotFoundError:
                    logger.debug(f"File vanished before loading: {file_path}")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load or process file {file_path}: {e}")
                    continue # Skip problematic files
            else:
                 logger.debug(f"File does not exist, skipping: {file_path}")


        if not all_data:
            logger.error(f"No '{dataset_name}' data files found or loaded for the specified date range and pattern.")
            return None

        logger.info(f"Loaded data from {loaded_files} files for dataset '{dataset_name}'.")
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined raw data shape: {combined_df.shape}")

        # Data Cleaning & Validation
        # 1. Timestamp Handling
        if timestamp_col not in combined_df.columns:
             logger.error(f"Loaded data for '{dataset_name}' missing configured timestamp column: '{timestamp_col}'")
             return None

        logger.info(f"Converting timestamp column '{timestamp_col}' to datetime...")
        # Handle potential errors during conversion
        original_timestamp_dtype = combined_df[timestamp_col].dtype
        try:
            # Attempt direct conversion first
             combined_df[timestamp_col] = pd.to_datetime(combined_df[timestamp_col], errors='coerce')
        except Exception as e1:
            logger.warning(f"Direct pd.to_datetime failed for column '{timestamp_col}' (dtype: {original_timestamp_dtype}): {e1}. Trying unit='s' if numeric.")
            # If it's numeric, it might be a Unix timestamp
            if pd.api.types.is_numeric_dtype(original_timestamp_dtype):
                 try:
                      combined_df[timestamp_col] = pd.to_datetime(combined_df[timestamp_col], unit='s', errors='coerce')
                 except Exception as e2:
                      logger.error(f"Failed converting numeric timestamp column '{timestamp_col}' with unit='s': {e2}")
                      return None
            else:
                 logger.error(f"Cannot convert non-numeric timestamp column '{timestamp_col}' after initial failure.")
                 return None

        # Check for conversion errors (NaT values)
        if combined_df[timestamp_col].isnull().any():
            num_null_ts = combined_df[timestamp_col].isnull().sum()
            logger.warning(f"Found {num_null_ts} rows with invalid timestamps after conversion in column '{timestamp_col}'. Dropping these rows.")
            combined_df.dropna(subset=[timestamp_col], inplace=True)
            if combined_df.empty:
                 logger.error("DataFrame is empty after dropping rows with invalid timestamps.")
                 return None

        # 2. Sort by timestamp
        logger.info("Sorting data by timestamp...")
        combined_df.sort_values(by=timestamp_col, inplace=True)

        # 3. Filter final dataset by exact date range
        # Ensure start/end_date are timezone-naive if the timestamp column is, or localize otherwise
        ts_col_tz = combined_df[timestamp_col].dt.tz
        if ts_col_tz is not None:
             logger.info(f"Timestamp column '{timestamp_col}' has timezone {ts_col_tz}. Localizing date range.")
             start_date = start_date.tz_localize(ts_col_tz)
             end_date = end_date.tz_localize(ts_col_tz)
        # Ensure end_date includes the whole day for filtering
        end_date_inclusive = end_date + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

        combined_df = combined_df[
            (combined_df[timestamp_col] >= start_date) &
            (combined_df[timestamp_col] <= end_date_inclusive)
        ]
        logger.info(f"Data shape after final date filtering: {combined_df.shape}")

        if combined_df.empty:
            logger.error(f"No data remaining for '{dataset_name}' after filtering for date range {start_date_str} to {end_date_str}.")
            return None

        # Rename columns to standardized names 
        # TODO: Implement the actual renaming logic in preprocess.py or feature_engineer.py
        # For now, just log the intention. The DataFrame returned here still has RAW column names.
        logger.info(f"Column renaming to standardized names (e.g., '{timestamp_col}' -> 'timestamp', "
                    f"'{ds_config['target_variable']}' -> '{config.STANDARDIZED_TARGET_VARIABLE}') "
                    f"should be performed in the preprocessing step.")

        # Add memory usage info
        mem_usage = combined_df.memory_usage(index=True, deep=True).sum()
        logger.info(f"Loaded data memory usage for '{dataset_name}': {mem_usage / 1024**2:.2f} MB")

        return combined_df

    except ValueError as e:
        logger.error(f"Error parsing dates or creating date range for '{dataset_name}': {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading for '{dataset_name}': {e}", exc_info=True)
        return None

def split_data(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VALIDATION_RATIO,
    timestamp_col: str = config.TIMESTAMP_COL 
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame chronologically into training, validation, and test sets.
    Args:
        df: The input DataFrame, assumed to be sorted by timestamp_col.
        train_ratio: Proportion of data for the training set.
        val_ratio: Proportion of data for the validation set.
        timestamp_col: The name of the timestamp column to sort/split by.
    Returns:
        A tuple containing (train_df, validation_df, test_df).
    """
    logger.info(f"Splitting data chronologically using column '{timestamp_col}': Train={train_ratio:.2f}, Val={val_ratio:.2f}")

    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame for splitting.")

    if not df[timestamp_col].is_monotonic_increasing:
         logger.warning(f"Data is not sorted by '{timestamp_col}'. Sorting now.")
         df = df.sort_values(by=timestamp_col)

    n = len(df)
    if n == 0:
        raise ValueError("Input DataFrame is empty, cannot split.")

    train_end_idx = int(n * train_ratio)
    val_end_idx = train_end_idx + int(n * val_ratio)

    # Ensure indices are valid
    train_end_idx = max(0, min(n, train_end_idx))
    val_end_idx = max(train_end_idx, min(n, val_end_idx))

    train_df = df.iloc[:train_end_idx]
    val_df = df.iloc[train_end_idx:val_end_idx]
    test_df = df.iloc[val_end_idx:]

    logger.info(f"Split complete:")
    if not train_df.empty:
        logger.info(f"  Train shape: {train_df.shape}, Time range: {train_df[timestamp_col].min()} -> {train_df[timestamp_col].max()}")
    else: logger.warning("  Train split is empty.")
    if not val_df.empty:
        logger.info(f"  Val shape:   {val_df.shape}, Time range: {val_df[timestamp_col].min()} -> {val_df[timestamp_col].max()}")
    else: logger.warning("  Validation split is empty.")
    if not test_df.empty:
        logger.info(f"  Test shape:  {test_df.shape}, Time range: {test_df[timestamp_col].min()} -> {test_df[timestamp_col].max()}")
    else: logger.warning("  Test split is empty.")


    if train_df.empty or val_df.empty: 
        logger.error("Training or Validation split resulted in an empty DataFrame. Check ratios and data size.")
        

    return train_df, val_df, test_df

if _TORCH_AVAILABLE:
    logger.info("PyTorch found. Defining TransitDataset and DataLoaders.")

    class TransitDataset(Dataset):
        """
        PyTorch Dataset for transit time series data.
        Handles sequence creation if applicable (e.g., for RNNs).
        """
        def __init__(self, features: np.ndarray, targets: np.ndarray, seq_length: Optional[int] = None):
            """
            Args:
                features (np.ndarray): Feature data (num_samples, num_features).
                targets (np.ndarray): Target data (num_samples,).
                seq_length (Optional[int]): Length of sequences for RNNs/Transformers.
                                            If None, treats each row as an independent sample.
            """
            if features.shape[0] != targets.shape[0]:
                raise ValueError("Features and targets must have the same number of samples.")
            self.features = features
            self.targets = targets
            self.seq_length = seq_length if seq_length is not None else 1
            logger.debug(f"TransitDataset created. Features shape: {self.features.shape}, "
                        f"Targets shape: {self.targets.shape}, Seq length: {self.seq_length}")
        def __len__(self) -> int:
            return max(0, self.features.shape[0] - self.seq_length + 1)
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Returns a single sequence (or sample) and its corresponding target.

            Args:
                idx: The starting index of the sequence (or the sample index if seq_length=1).

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: (feature_sequence, target_value)
                 - feature_sequence: shape (seq_length, num_features)
                 - target_value: shape (1,) - typically the target at the end of the sequence
            """
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of bounds for dataset length {len(self)}")
            feature_seq = self.features[idx : idx + self.seq_length]
            target_idx = idx + self.seq_length - 1
            target_val = self.targets[target_idx]
            feature_tensor = torch.tensor(feature_seq, dtype=torch.float32)
            target_tensor = torch.tensor([target_val], dtype=torch.float32)
            return feature_tensor, target_tensor

    def create_dataloaders(
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray],
        batch_size: int = config.BATCH_SIZE,
        seq_length: Optional[int] = config.SEQ_LENGTH if config.MODEL_TYPE in ['LSTM', 'GRU', 'Transformer', 'GNN_LSTM'] else None,
        num_workers: int = 0 
        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Creates PyTorch DataLoaders for training, validation, and test sets.

        Args:
            train_data: Tuple (train_features_array, train_targets_array).
            val_data: Tuple (val_features_array, val_targets_array).
            test_data: Tuple (test_features_array, test_targets_array).
            batch_size: Number of samples per batch.
            seq_length: Sequence length for time series models.
            num_workers: Number of subprocesses for data loading.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for create_dataloaders.")

        logger.info(f"Creating DataLoaders with Batch Size: {batch_size}, Sequence Length: {seq_length}")
        train_dataset = TransitDataset(train_data[0], train_data[1], seq_length=seq_length)
        val_dataset = TransitDataset(val_data[0], val_data[1], seq_length=seq_length)
        test_dataset = TransitDataset(test_data[0], test_data[1], seq_length=seq_length)
        logger.info(f"Dataset sizes (num sequences): Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        if len(train_dataset) == 0 or len(val_dataset) == 0:
             logger.warning("Training or Validation dataset has zero length after sequence creation. "
                           "Check data size and sequence length.")

        pin_memory = True if config.DEVICE == 'cuda' else False
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        logger.info("DataLoaders created successfully.")
        return train_loader, val_loader, test_loader

else:
    logger.warning("PyTorch not found. TransitDataset and create_dataloaders will not be available.")
    class TransitDataset(Dataset): pass
    def create_dataloaders(*args, **kwargs):
        raise NotImplementedError("PyTorch is not installed, cannot create DataLoaders.")