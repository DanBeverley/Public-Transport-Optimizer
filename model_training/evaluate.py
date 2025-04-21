"""
Evaluates the trained transit delay prediction model on the test set.
Loads the best model, scaler, and feature names list saved during training.
"""
import logging
import time
from pathlib import Path
import datetime
import sys
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional, Any, Tuple, Union, List, Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from . import config, utils, data_loader, preprocess, feature_engineer, graph_constructor, model_def

logger = logging.getLogger(__name__)

@torch.no_grad()
def make_predictions(
    model: nn.Module,
    dataloader: data_loader.DataLoader,
    device: torch.device,
    graph_data: Optional[Any] = None 
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Makes predictions on the data provided by the dataloader."""
    model.eval() 
    all_predictions = []
    all_targets = []

    for batch in dataloader:
        features, targets = batch
        node_indices = None
        if isinstance(features, (list, tuple)):
             if len(features) == 2:
                 node_indices = features[1].to(device)
                 features = features[0].to(device)
             else: raise ValueError(f"Unexpected feature batch format: {type(features)}")
        else: features = features.to(device)
        if config.MODEL_TYPE == 'GNN_LSTM':
            if graph_data is None or node_indices is None:
                raise ValueError("GNN_LSTM requires graph_data and node_indices for forward pass.")
            outputs = model(features, graph_data.to(device), node_indices)
        else:
            outputs = model(features)

        # Collect predictions and targets (move outputs to CPU, detach from graph)
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(targets.numpy()) 

    predictions_np = np.concatenate(all_predictions, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)

    return predictions_np, targets_np


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric_list: List[str]) -> Dict[str, float]:
    """Calculates specified regression metrics."""
    results = {}
    logger.info("Calculating evaluation metrics...")
    for metric in metric_list:
        try:
            if metric == 'mae':
                score = mean_absolute_error(y_true, y_pred)
            elif metric == 'mse':
                 score = mean_squared_error(y_true, y_pred)
            elif metric == 'rmse':
                score = np.sqrt(mean_squared_error(y_true, y_pred))
            elif metric == 'r2':
                score = r2_score(y_true, y_pred)
            else:
                logger.warning(f"Unsupported metric: {metric}. Skipping.")
                continue
            results[metric] = score
            logger.info(f"  {metric.upper()}: {score:.4f}")
        except Exception as e:
            logger.error(f"Error calculating metric '{metric}': {e}", exc_info=True)
    return results

def generate_evaluation_report(metrics: Dict[str, float], plots_path: Optional[Path] = None):
    """Logs metrics and potentially saves plots."""
    logger.info("--- Evaluation Report ---")
    for name, value in metrics.items():
        logger.info(f"  Test {name.upper()}: {value:.4f}")
    logger.info("-------------------------")

    # TODO: Add plotting functionality (e.g., using matplotlib)
    # - Scatter plot of True vs Predicted
    # - Residual plot
    # Save plots if plots_path is provided
    if plots_path:
         logger.warning("Plot generation not yet implemented.")
         pass


def evaluate_model():
    """Main function to evaluate the trained model."""
    eval_run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') # Separate timestamp for eval log
    log_eval_file = config.LOG_DIR / f"evaluation_{config.ACTIVE_DATASET_NAME}_{config.MODEL_TYPE}_{config.MODEL_SUFFIX}_{eval_run_timestamp}.log"
    logger = utils.setup_logging(config.LOG_LEVEL, log_eval_file) # Setup fresh logger for eval
    logger.info("=" * 50)
    logger.info(f"Starting Evaluation Run: Dataset='{config.ACTIVE_DATASET_NAME}', Mode='{config.TRAINING_MODE}', Model='{config.MODEL_TYPE}'")
    logger.info(f"Evaluating model: {config.BEST_MODEL_SAVE_PATH}")
    logger.info(f"Using scaler: {config.SCALER_SAVE_PATH}")
    logger.info("=" * 50)
    start_time = time.time()
    utils.set_seed(config.SEED)
    if config.DEVICE == 'cuda' and not torch.cuda.is_available(): device = torch.device('cpu'); logger.warning("CUDA not available.")
    elif config.DEVICE =='mps' and not torch.backends.mps.is_available(): device = torch.device('cpu'); logger.warning("MPS not available.")
    else: device = torch.device(config.DEVICE)
    logger.info(f"Using device: {device}")

    # 1. Load Scaler and Feature Names
    scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
    final_feature_names: Optional[List[str]] = None
    feature_names_path = config.MODEL_SAVE_DIR / f"{config.ACTIVE_DATASET_NAME}_{config.MODEL_TYPE}_{config.MODEL_SUFFIX}_feature_names.json"

    logger.info("Loading scaler and feature names list...")
    try:
        if config.SCALER_TYPE is not None:
            scaler = utils.load_object(config.SCALER_SAVE_PATH)
            if scaler is None: raise FileNotFoundError("Scaler object loaded as None.")
            logger.info("Scaler loaded successfully.")
        else: logger.info("Scaling was not used during training.")

        if not feature_names_path.exists(): raise FileNotFoundError("Feature names list not found.")
        with open(feature_names_path, 'r') as f:
            final_feature_names = json.load(f)
        if not final_feature_names: raise ValueError("Loaded feature names list is empty.")
        logger.info(f"Loaded {len(final_feature_names)} feature names successfully.")
        input_dim = len(final_feature_names)

    except FileNotFoundError as e: logger.error(f"Missing required file: {e}. Cannot proceed."); return
    except Exception as e: logger.error(f"Error loading scaler/features list: {e}", exc_info=True); return

    # 2. Load and Prepare Test Data
    logger.info("Loading and preparing test data...")
    test_X_scaled_df: Optional[pd.DataFrame] = None
    test_y_s: Optional[pd.Series] = None
    test_ids_df: Optional[pd.DataFrame] = None
    try:
        df_raw = data_loader.load_historical_data(config.ACTIVE_DATASET_NAME)
        if df_raw is None: raise ValueError("Failed raw data load.")
        _, _, test_df_raw = data_loader.split_data(df_raw, config.TIMESTAMP_COL)
        del df_raw
        if test_df_raw.empty: raise ValueError("Test split empty.")

        # Preprocess (Std+Impute)
        test_std = preprocess.standardize_data(test_df_raw, config.ACTIVE_DATASET_NAME)
        if test_std is None: raise ValueError("Standardization failed.")
        test_clean = preprocess.handle_missing_values(test_std)

        logger.info("Engineering features for test set...")
        test_feat_eng = feature_engineer.generate_cyclical_features(test_clean, config.CYCLICAL_FEATURES)
        test_feat_eng = feature_engineer.generate_lag_features(test_feat_eng, config.STANDARDIZED_TARGET_VARIABLE, config.LAG_FEATURES[config.STANDARDIZED_TARGET_VARIABLE])
        test_feat_eng = feature_engineer.generate_rolling_features(test_feat_eng, config.ROLLING_FEATURES)

        identifier_cols = ['timestamp', 'trip_id', 'route_id', 'vehicle_id', 'stop_id', 'stop_sequence'] # Standardized identifiers
        cols_to_keep = final_feature_names + [config.STANDARDIZED_TARGET_VARIABLE] + [id_col for id_col in identifier_cols if id_col in test_feat_eng.columns]
        dropna_subset = final_feature_names + [config.STANDARDIZED_TARGET_VARIABLE]
        test_final_rows = test_feat_eng[cols_to_keep].dropna(subset=dropna_subset)

        if test_final_rows.empty: raise ValueError("No rows remaining in test set after feature engineering and NaN drop.")

        test_X_unscaled_df = test_final_rows[final_feature_names]
        test_y_s = test_final_rows[config.STANDARDIZED_TARGET_VARIABLE]
        test_ids_df = test_final_rows[[id_col for id_col in identifier_cols if id_col in test_final_rows.columns]]

        if scaler is not None:
            logger.info("Applying loaded scaler to test features...")
            if hasattr(scaler, 'feature_names_in_') and list(scaler.feature_names_in_) != final_feature_names:
                 logger.warning("Feature names mismatch between loaded scaler and current features! Evaluation might be incorrect.")
                 try:
                     test_X_reordered = test_X_unscaled_df[list(scaler.feature_names_in_)]
                     test_X_scaled_np = scaler.transform(test_X_reordered)
                     test_X_scaled_df = pd.DataFrame(test_X_scaled_np, columns=scaler.feature_names_in_, index=test_X_reordered.index)
                     logger.info("Reordered test features to match scaler and applied transform.")
                 except Exception as e:
                     logger.error(f"Failed to reorder/scale features using loaded scaler's names: {e}. Using unscaled data.")
                     test_X_scaled_df = test_X_unscaled_df 
            else:
                 test_X_scaled_np = scaler.transform(test_X_unscaled_df)
                 test_X_scaled_df = pd.DataFrame(test_X_scaled_np, columns=final_feature_names, index=test_X_unscaled_df.index)
                 logger.info("Test features scaled successfully.")
        else: # No scaler used
             test_X_scaled_df = test_X_unscaled_df

        logger.info(f"Test data prepared. Features shape: {test_X_scaled_df.shape}, Target shape: {test_y_s.shape}")

    except Exception as e: logger.error(f"Error preparing test data: {e}.", exc_info=True); return

    # 3. Graph Construction
    graph_data = None
    stop_id_to_idx_map = None
    if 'GNN' in config.MODEL_TYPE:
        logger.info("Constructing/Loading graph for evaluation...")
        graph_data = graph_constructor.get_transit_graph(config.ACTIVE_DATASET_NAME)
        if graph_data is None: logger.error("Failed to get graph data. Exiting."); return
        if not hasattr(graph_data, 'stop_id_to_idx'): logger.error("Graph missing 'stop_id_to_idx'. Exiting."); return
        stop_id_to_idx_map = graph_data.stop_id_to_idx

    # 4. Create Test DataLoader
    logger.info("Creating test dataloader...")
    try:
        # Pass DataFrames to dataloader creator
        test_loader = data_loader.create_dataloaders(
             (pd.DataFrame(), pd.Series(), pd.DataFrame()),
             (pd.DataFrame(), pd.Series(), pd.DataFrame()),
             (test_X_scaled_df, test_y_s, test_ids_df), # Test data
             stop_id_to_idx_map=stop_id_to_idx_map,
             batch_size=config.BATCH_SIZE * 2, # Larger batch for eval
             seq_length=config.SEQ_LENGTH if config.MODEL_TYPE != 'NonSequential' else None
        )[2] 
        logger.info(f"Test DataLoader created. Num batches: {len(test_loader)}")
    except Exception as e: logger.error(f"Error creating test dataloader: {e}.", exc_info=True); return

    # 5. Load Trained Model
    logger.info("Loading trained model...")
    try:
        model = model_def.build_model(input_dim=input_dim, graph_data=graph_data)
        model_path = config.BEST_MODEL_SAVE_PATH
        if not model_path.exists(): logger.error(f"Model file not found: {model_path}"); return
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        logger.info(f"Trained model loaded successfully from {model_path}")
    except Exception as e: logger.error(f"Error loading trained model: {e}.", exc_info=True); return

    # 6. Make Predictions
    logger.info("Making predictions on test set...")
    try:
        predictions_scaled_np, targets_scaled_np = make_predictions(model, test_loader, device, graph_data)
        logger.info(f"Predictions made. Shape: {predictions_scaled_np.shape}")
    except Exception as e: logger.error(f"Error during prediction loop: {e}.", exc_info=True); return

    # 7. Inverse Transform Predictions and Targets 
    predictions_inv, targets_inv = None, None
    if scaler is not None:
        logger.info("Inverse transforming predictions and targets...")
        try:
            target_col_name = config.STANDARDIZED_TARGET_VARIABLE
            if hasattr(scaler, 'feature_names_in_'): fitted_feature_names = list(scaler.feature_names_in_)
            else: fitted_feature_names = final_feature_names; logger.warning("Scaler missing 'feature_names_in_'. Assuming test feature names match.")

            if target_col_name not in fitted_feature_names:
                 logger.error(f"Target '{target_col_name}' not in scaler features. Cannot inverse transform.")
            else:
                 target_col_index = fitted_feature_names.index(target_col_name)
                 logger.debug(f"Target column index for inverse transform: {target_col_index}")
                 num_samples = predictions_scaled_np.shape[0]
                 num_features_fitted = len(fitted_feature_names)
                 dummy_pred_array = np.zeros((num_samples, num_features_fitted))
                 dummy_targ_array = np.zeros((num_samples, num_features_fitted))
                 dummy_pred_array[:, target_col_index] = predictions_scaled_np.flatten()
                 dummy_targ_array[:, target_col_index] = targets_scaled_np.flatten()

                 predictions_inv = scaler.inverse_transform(dummy_pred_array)[:, target_col_index]
                 targets_inv = scaler.inverse_transform(dummy_targ_array)[:, target_col_index]
                 logger.info("Inverse transform successful.")

        except Exception as e:
             logger.error(f"Error during inverse transform: {e}. Metrics based on scaled values.", exc_info=True)
    else:
        logger.info("No scaler used. Metrics based on original/unscaled target.")

    # Use inverse transformed if available, else use scaled
    y_pred_final = predictions_inv if predictions_inv is not None else predictions_scaled_np.flatten()
    y_true_final = targets_inv if targets_inv is not None else targets_scaled_np.flatten()

    # 8. Calculate Metrics 
    metrics = calculate_metrics(y_true_final, y_pred_final, config.EVALUATION_METRICS)

    # 9. Generate Report 
    generate_evaluation_report(metrics)

    total_eval_time = time.time() - start_time
    logger.info(f"Total evaluation script run time: {total_eval_time:.2f}s"); logger.info("="*50)