"""
Main script for training the transit delay prediction model.

Orchestrates data loading, preprocessing, feature engineering, model building,
training loop, validation, and saving the best model and scaler.
"""
import json
import logging
import time
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from typing import Optional, Any
from . import config
from . import utils
from . import data_loader
from . import preprocess
from . import feature_engineer
from . import graph_constructor
from . import model_def

logger = utils.setup_logging(config.LOG_LEVEL, config.LOG_FILE)

def run_training_epoch(model:nn.Module, dataloader:data_loader.DataLoader,
                       optimizer:optim.Optimizer, criterion:nn.Module,
                       device:torch.device, clip_value:Optional[float] = config.GRADIENT_CLIP_VALUE,
                       graph_data:Optional[Any] = None) -> float:
    """Run a single training epoch"""
    model.train() # Set model to training mode
    running_loss = 0.0
    num_batches = len(dataloader)
    for i, batch in enumerate(dataloader):
        batch_features, targets = batch
        targets = targets.to(device)
        node_indices = None
        if isinstance(batch_features,(list, tuple)):
            node_indices = batch_features[1].to(device)
            features = batch_features[0].to(device)
        else: # Non-GNN case
            features = batch_features.to(device)
        optimizer.zero_grad()
        if config.MODEL_TYPE == "GNN_LSTM":
            outputs = model(features, graph_data.to(device), node_indices)
        else:
            outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        if clip_value is not None and clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters, clip_value)
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 100 == 0 or (i + 1) == num_batches:
                logger.debug(f"  Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}")
                return running_loss/num_batches
            
def run_validation_epoch(model:nn.Module, dataloader:data_loader.DataLoader,
                         criterion:nn.Module, device:torch.device,
                         graph_data:Optional[Any] = None) -> float:
    model.eval() 
    running_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad(): 
        for batch in dataloader:
            features, targets = batch
            node_indices = None
            if isinstance(features, (list, tuple)):
                 if len(features) == 2:
                     node_indices = features[1].to(device)
                     features = features[0].to(device)
                 else: raise ValueError(f"Unexpected feature batch format: {type(features)}")
            else: features = features.to(device)
            targets = targets.to(device)

            # Forward pass
            if config.MODEL_TYPE == 'GNN_LSTM':
                if graph_data is None or node_indices is None:
                     raise ValueError("GNN_LSTM requires graph_data and node_indices for forward pass.")
                outputs = model(features, graph_data.to(device), node_indices)
            else:
                outputs = model(features)

            loss = criterion(outputs, targets)
            running_loss += loss.item()

    avg_loss = running_loss / num_batches
    return avg_loss

def train_model():
    """Main function to orchestrate the training process."""
    logger.info("=" * 50)
    logger.info(f"Starting Training Run: Dataset='{config.ACTIVE_DATASET_NAME}', Mode='{config.TRAINING_MODE}', Model='{config.MODEL_TYPE}'")
    logger.info(f"Log file: {config.LOG_FILE}")
    logger.info("=" * 50)
    start_time = time.time()
    utils.set_seed(config.SEED)
    if config.DEVICE == 'cuda' and not torch.cuda.is_available(): device = torch.device('cpu'); logger.warning("CUDA not available, using CPU.")
    elif config.DEVICE =='mps' and not torch.backends.mps.is_available(): device = torch.device('cpu'); logger.warning("MPS not available, using CPU.")
    else: device = torch.device(config.DEVICE)
    logger.info(f"Using device: {device}")

    logger.info("Loading raw data...")
    df_raw = data_loader.load_historical_data(config.ACTIVE_DATASET_NAME)
    if df_raw is None or df_raw.empty: logger.error("Data loading failed."); return
    logger.info("Splitting data...")
    train_df_raw, val_df_raw, test_df_raw = data_loader.split_data(df_raw, config.TIMESTAMP_COL)
    del df_raw
    if train_df_raw.empty or val_df_raw.empty: logger.error("Empty train/val split."); return

    logger.info("Preprocessing data...")
    try:
        train_df_proc, val_df_proc, test_df_proc, scaler = preprocess.preprocess_data(
            train_df_raw, val_df_raw, test_df_raw, config.ACTIVE_DATASET_NAME
        )
        del train_df_raw, val_df_raw, test_df_raw
    except Exception as e: logger.error(f"Preprocessing error: {e}", exc_info=True); return

    logger.info("Engineering features...")
    try:
        (train_X_df, train_y_s, train_ids_df,
         val_X_df, val_y_s, val_ids_df,
         test_X_df, test_y_s, test_ids_df,
         final_feature_names) = feature_engineer.engineer_features_for_training(
            train_df_proc, val_df_proc, test_df_proc
         )
        del train_df_proc, val_df_proc, test_df_proc
        if train_X_df.empty: logger.error("Empty training features after engineering."); return
        feature_names_path = config.MODEL_SAVE_DIR / f"{config.ACTIVE_DATASET_NAME}_{config.MODEL_TYPE}_{config.MODEL_SUFFIX}_feature_names.json"
        with open(feature_names_path, 'w') as f:
            json.dump(final_feature_names, f)
        logger.info(f"Saved final feature names list to {feature_names_path}")
    except Exception as e: logger.error(f"Feature engineering error: {e}", exc_info=True); return

    graph_data = None
    stop_id_to_idx_map = None 
    if 'GNN' in config.MODEL_TYPE:
        logger.info("Constructing graph...")
        try:
            graph_data = graph_constructor.get_transit_graph(config.ACTIVE_DATASET_NAME)
            if graph_data is None: raise ValueError("Failed to construct graph.")
            if not hasattr(graph_data, 'stop_id_to_idx'): raise ValueError("Graph data missing 'stop_id_to_idx' map.")
            stop_id_to_idx_map = graph_data.stop_id_to_idx # Get map for DataLoader
            logger.info("Graph constructed successfully.")
        except Exception as e: logger.error(f"Graph construction error: {e}", exc_info=True); return

    logger.info("Creating dataloaders...")
    try:
        train_loader, val_loader, test_loader = data_loader.create_dataloaders(
            (train_X_df, train_y_s, train_ids_df), 
            (val_X_df, val_y_s, val_ids_df),
            (test_X_df, test_y_s, test_ids_df),
            stop_id_to_idx_map=stop_id_to_idx_map, 
            batch_size=config.BATCH_SIZE,
        )
        input_dim = len(final_feature_names)
        logger.info(f"DataLoaders created. Input dimension set to: {input_dim}")
    except Exception as e: logger.error(f"Dataloader creation error: {e}", exc_info=True); return

    logger.info("Building model...")
    try:
        model = model_def.build_model(input_dim=input_dim, graph_data=graph_data)
        model.to(device)
    except Exception as e: logger.error(f"Model building error: {e}", exc_info=True); return

    if config.TRAINING_MODE == 'finetune':
        if config.BASE_MODEL_LOAD_PATH and Path(config.BASE_MODEL_LOAD_PATH).exists():
             logger.info(f"Loading base model weights from: {config.BASE_MODEL_LOAD_PATH}")
             try: model.load_state_dict(torch.load(config.BASE_MODEL_LOAD_PATH, map_location=device), strict=False) # Use strict=False for potential layer mismatches in finetuning
             except Exception as e: logger.error(f"Error loading base model weights: {e}. Starting from scratch.", exc_info=True)
        else: logger.warning(f"Fine-tuning: Base model path not found: {config.BASE_MODEL_LOAD_PATH}. Training from scratch.")

    if config.OPTIMIZER.lower() == 'adam': optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    elif config.OPTIMIZER.lower() == 'adamw': optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    else: logger.warning("Using Adam optimizer as default."); optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    logger.info(f"Using optimizer: {config.OPTIMIZER}, LR: {config.LEARNING_RATE}")

    if config.LOSS_FUNCTION.lower() == 'mse': criterion = nn.MSELoss()
    elif config.LOSS_FUNCTION.lower() == 'mae': criterion = nn.L1Loss()
    else: logger.warning("Using MSE loss as default."); criterion = nn.MSELoss()
    logger.info(f"Using loss function: {config.LOSS_FUNCTION}")

    logger.info("--- Starting Training Loop ---")
    best_val_loss = float('inf'); epochs_no_improve = 0; training_start_time = time.time()
    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = time.time()
        train_loss = run_training_epoch(model, train_loader, optimizer, criterion, device, config.GRADIENT_CLIP_VALUE, graph_data)
        val_loss = run_validation_epoch(model, val_loader, criterion, device, graph_data)
        epoch_duration = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Duration: {epoch_duration:.2f}s")
        if val_loss < best_val_loss:
            best_val_loss = val_loss; epochs_no_improve = 0
            try: torch.save(model.state_dict(), config.BEST_MODEL_SAVE_PATH); logger.info(f"Val loss improved. Saved best model to {config.BEST_MODEL_SAVE_PATH}")
            except Exception as e: logger.error(f"Error saving model checkpoint: {e}", exc_info=True)
        else:
            epochs_no_improve += 1; logger.info(f"Val loss did not improve. ({epochs_no_improve}/{config.EARLY_STOPPING_PATIENCE})")
            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE: logger.info(f"Early stopping triggered after {epoch+1} epochs."); break
    training_duration = time.time() - training_start_time
    logger.info(f"--- Training Loop Finished --- Total Time: {training_duration:.2f}s, Best Val Loss: {best_val_loss:.4f}")

    total_run_time = time.time() - start_time
    logger.info(f"Total script run time: {total_run_time:.2f}s"); logger.info("="*50)      