"""
Analyzes the probability of successfully making a connection between two transit legs.

Uses predicted arrival/departure times and potentially ML models or heuristics.
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Any, Tuple

import joblib
import pandas as pd
import numpy as np

from . import feature_engineer
from . import delay_predictor
from data_pipeline.utils import time_it_async

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
CONNECTION_MODEL_PATH = os.path.join(MODEL_DIR, "connection_model.joblib")

_connection_model:Optional[Any] = None
USE_CONNECTION_MODEL = False

def load_connection_model(model_path:str = CONNECTION_MODEL_PATH) -> Any:
    """Loads the pre-trained connection success probability model."""
    global _connection_model
    if not USE_CONNECTION_MODEL:
        logger.info("Connection ML model usage is disabled")
        return None
    if _connection_model is not None:
        return _connection_model
    if not os.path.exists(model_path):
        logger.error(f"Connection prediction model not found at: {model_path}")
        raise FileNotFoundError(f"Model file not found : {model_path}")
    logger.info(f"Loading connection prediction model from: {model_path}")
    try:
        _connection_model = joblib.load(model_path)
        logger.info("Connection prediction model loaded successfully")
        return _connection_model
    except Exception as e:
        logger.error(f"Failed to load connection model from {model_path}:{e}", exc_info=True)
        raise

# Heuristic / Rules
DEFAULT_MIN_BUFFER_SECONDS = 45
TRANSITION_WIDTH_SECONDS = 60 # Width of the transition window (e.g., 60 seconds)

def heuristic_connection_probability(predicted_arrival_time:datetime,
                                     scheduled_departure_time:datetime,
                                     predicted_departure_delay:timedelta,
                                     walk_time:timedelta,
                                     min_buffer:timedelta = timedelta(seconds=DEFAULT_MIN_BUFFER_SECONDS)) -> float:
    """
    Calculates connection probability based on buffer time with a smooth transition.

    Returns:
        Probability (0.0 to 1.0).
    """
    predicted_departure_time = scheduled_departure_time + predicted_departure_delay
    effective_arrival_at_departure = predicted_arrival_time + walk_time
    available_buffer = predicted_departure_time - effective_arrival_at_departure

    transition_width = timedelta(seconds=TRANSITION_WIDTH_SECONDS)
    lower_bound_buffer = min_buffer - transition_width / 2
    upper_bound_buffer = min_buffer + transition_width / 2

    if available_buffer >= upper_bound_buffer:
        logger.debug(f"Heuristic: Connection highly likely (Buffer:{available_buffer} >= {upper_bound_buffer})")
        return 1.0 # Buffer is comfortably above the minimum + transition
    elif available_buffer <= lower_bound_buffer:
        logger.debug(f"Heuristic: Connection unlikely (Buffer: {available_buffer} <= {lower_bound_buffer})")
        return 0.0 # Buffer is below the minimum - transition
    else:
        # Linear interpolation within the transition window
        try:
            probability = (available_buffer.total_seconds() - lower_bound_buffer.total_seconds()) / transition_width.total_seconds()
            # Clamp probability to [0, 1] to handle potential floating point inaccuracies
            probability = max(0.0, min(1.0, probability))
            logger.debug(f"Heuristic: Connection possible (Buffer: {available_buffer}, Min: {min_buffer}, Prob: {probability:.2f})")
            return probability
        except ZeroDivisionError:
            logger.warning("Transition width is zero, returning binary probability based on min_buffer.")
            # Fallback to original binary logic if transition width is zero
            return 1.0 if available_buffer >= min_buffer else 0.0

# Prediction function
@time_it_async
async def predict_connection_success_probability(
    route_id:str, 
    arrival_trip_id: str,
    arrival_stop_id: str,
    scheduled_arrival_time: datetime, 
    predicted_arrival_delay: timedelta, 
    current_stop_sequence:int,

    departure_trip_id: str,
    departure_stop_id: str,
    departure_stop_sequence: int, 
    scheduled_departure_time: datetime,

    query_time: datetime,
    static_data_accessor: Optional[Any] = None # For walk times etc.
    ) -> float:
    """
    Predicts the probability (0.0 to 1.0) of making a connection.

    Args:
        arrival_trip_id: ID of the arriving trip.
        arrival_stop_id: Stop ID where arrival occurs.
        scheduled_arrival_time: Scheduled arrival time of the first leg.
        predicted_arrival_delay: Predicted delay for the arriving trip at arrival_stop_id.
        departure_trip_id: ID of the departing trip.
        departure_stop_id: Stop ID where departure occurs.
        departure_stop_sequence: Stop sequence for the departure stop on the departing trip.
        scheduled_departure_time: Scheduled departure time of the second leg.
        query_time: Current time.
        static_data_accessor: Object for static data lookups.

    Returns:
        Connection success probability (float between 0.0 and 1.0).
    """
    global _connection_model, USE_CONNECTION_MODEL
    logger.debug(f"Analyzing connection: {arrival_trip_id}@{arrival_stop_id} -> {departure_trip_id}@{departure_stop_id}")
    
    predicted_arrival_time = scheduled_arrival_time + predicted_arrival_delay

    # Predict the delay of the "departing" service at its departure stop
    predicted_departure_delay = await delay_predictor.predict_arrival_delay(trip_id = departure_trip_id,
                                                                          target_stop_sequence=departure_stop_sequence,
                                                                          query_time=query_time,
                                                                          static_data_accessor=static_data_accessor,
                                                                          route_id=route_id,
                                                                          current_stop_sequence=current_stop_sequence)
    if predicted_departure_delay is None:
        logger.warning(f"Could not predict departure delay for {departure_trip_id}, assuming 0 delay for connection analysis")
        predicted_departure_delay = timedelta(0)
    if USE_CONNECTION_MODEL and _connection_model:
        logger.debug("Using connection prediction model")
        features = await feature_engineer.create_connection_features(
            arrival_trip_id=arrival_trip_id,
            departure_trip_id=departure_trip_id,
            arrival_stop_id=arrival_stop_id,
            departure_stop_id=departure_stop_id,
            predicted_arrival_time=predicted_arrival_time,
            scheduled_departure_time=scheduled_departure_time,
            predicted_departure_delay_seconds = int(predicted_departure_delay.total_seconds()),
            query_time=query_time,
            static_data_accessor=static_data_accessor
        )
        if features is None:
            logger.warning(f"Could not generate features for connection model. Falling back to heuristic")
            # TODO: Need walk time for heuristic
            walk_time = timedelta(seconds = 60) # PLaceholder
            return heuristic_connection_probability(predicted_arrival_time, scheduled_departure_time,
                                                    predicted_departure_delay, walk_time)
        try:
            features_df = pd.DataFrame([features])
            probability = _connection_model.predict_proba(features_df)[0,1]
            probability = float(np.clip(probability, 0.0, 1.0)) 
            logger.debug(f"Model prediction probability: {probability:.3f}")
            return probability
        except Exception as e:
            logger.error(f"Error during connection model prediction: {e}. Falling back to heuristic.", exc_info=True)
            # Fallback to heuristic on model error
            # TODO: Need walk time for heuristic
            walk_time = timedelta(seconds=60) # Placeholder
            return heuristic_connection_probability(predicted_arrival_time, scheduled_departure_time,
                                                    predicted_departure_delay, walk_time)

    else:
        # Use heuristic rules if model is disabled or not loaded
        logger.debug("Using heuristic connection probability rules.")
        # TODO: Need walk time for heuristic
        walk_time = timedelta(seconds=60) # Placeholder: Get walk time from static_data_accessor
        if static_data_accessor and hasattr(static_data_accessor, 'get_transfer_time'):
            fetched_walk_time = await static_data_accessor.get_transfer_time(arrival_stop_id, departure_stop_id)
            if fetched_walk_time: walk_time = fetched_walk_time

        return heuristic_connection_probability(predicted_arrival_time, scheduled_departure_time,
                                                predicted_departure_delay, walk_time)



    
