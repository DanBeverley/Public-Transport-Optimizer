"""
Predicts transit delays or travel times for specific trip segments using ML models.
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Any, Tuple

import joblib 
import pandas as pd

from . import feature_engineer

try:
    from data_pipeline.utils import time_it_async
except ImportError:
    def time_it_async(func): return func


logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DELAY_MODEL_PATH = os.path.join(MODEL_DIR, "delay_model.joblib")

_delay_model:Optional[Any] = None

def load_delay_model(model_path:str = DELAY_MODEL_PATH) -> Any:
    """Load the pre-trained delay prediction model"""
    global _delay_model
    if _delay_model is not None:
        return _delay_model
    if not os.path.exist(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    logger.info(f"Loading delay prediction model from {model_path}")
    try:
        _delay_model = joblib.load(model_path)
        logger.info("Delay prediction model loaded")
        logger.info(f"Model type: {type(_delay_model)}")
        return _delay_model
    except Exception as e:
        logger.error(f"Failed to load delay model from {model_path}: {e}", exc_info = True)
        raise

try:
    load_delay_model()
except Exception:
    logger.error("Delay model could not be loaded on module import. Predictions will fail")

@time_it_async
async def predict_arrival_delay(trip_id:str, target_stop_sequence:int, query_time:datetime,
                                route_id:Optional[str] = None, current_stop_sequence:Optional[int] = None,
                                static_data_accessor:Optional[Any] = None ) -> Optional[timedelta]:
    """
    Predicts the arrival delay at a specific stop sequence for a given trip.

    Args:
        trip_id: The ID of the trip.
        target_stop_sequence: The sequence number of the stop for prediction.
        query_time: The time at which the prediction is being made.
        route_id: The route ID (optional context).
        current_stop_sequence: Current known stop sequence (optional).
        static_data_accessor: Object for accessing static GTFS data (optional).

    Returns:
        Predicted delay as a timedelta object, or None if prediction fails.
        The delay is relative to the scheduled arrival time. Positive means late.
    """
    global _delay_model
    if _delay_model is None:
        logger.warning("Delay prediction model is not loaded. Cannot predict")
        return None 
    logger.debug(f"Requesting delay prediction for trip {trip_id}, target sequence {target_stop_sequence}")

    # 1. Create Features
    features = await feature_engineer.create_delay_features(
        trip_id = trip_id,
        route_id = route_id,
        current_stop_sequence=current_stop_sequence,
        target_stop_sequence=target_stop_sequence,
        query_time=query_time,
        static_data_accessor=static_data_accessor
    )
    if features is None:
        logger.warning(f"Could not generate features for delay prediction (Trip:{trip_id})")
        return None
    
    # 2. Predict with the loaded model
    try:
        # Models often expect a 2D array (e.g., DataFrame or NumPy array)
        # NOTE:Ensure features are in the correct format/order
        features_df = pd.DataFrame([features])
        predicted_delay_seconds = _delay_model.predict(features_df)[0]
        logger.debug(f"Raw prediction for {trip_id}:{predicted_delay_seconds} seconds")
        # Basic sanity check on prediction
        # E.g., prevent extremely large negative delays if nonsensical
        predicted_delay_seconds = max(-60*15, predicted_delay_seconds)
        predicted_delay_seconds = min(60*60*2, predicted_delay_seconds)
        return timedelta(seconds=float(predicted_delay_seconds))
    except Exception as e:
        logger.error(f"Error during delay prediction for trip {trip_id}:{e}", exc_info = True)
        return None
