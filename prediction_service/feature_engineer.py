"""
Feature Engineering for Delay and Connection Prediction Models.

Gathers real-time and static data to create input features for ML models.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import numpy as np

try:
    from data_pipeline import data_storage
    from data_pipeline import api_clients
    from data_pipeline.utils import get_current_utc_datetime, parse_flexible_timestamp, safe_float, safe_int
    from routing_engine.models import Trip, Route, Stop
except ImportError:
    logging.error("Failed to import necessary modules from data_pipeline or routine engine")

logger = logging.getLogger(__name__)

# CONSTANT
DELAY_MODEL_FEATURES = [
    'time_of_day_sin', 'time_of_day_cos', 'day_of_week', 'month_of_year',
    'route_type', 'current_reported_delay', 'current_speed',
    'segment_distance_km', 'stops_remaining', 'recent_avg_delay_route', # Example advanced features
    'weather_temp', 'weather_precip_mm', 'traffic_level' # Example contextual features
]

CONNECTION_MODEL_FEATURES = [
    'arrival_delay_predicted', 'departure_delay_scheduled', 'buffer_time_seconds',
    'walk_time_seconds', 'arr_route_type', 'dep_route_type', 'time_of_day_sin',
    'time_of_day_cos', 'is_peak_hour'
]

# Helper functions
def _encode_time_features(dt:datetime) -> Tuple[float, float, int, int]:
    """Encodes datetime into cyclical and categorical features"""
    # Time of day (cyclical)
    seconds_in_day = 24*60*60
    time_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
    time_of_day_sin = np.sin(2 * np.pi * time_seconds / seconds_in_day)
    time_of_day_cos = np.cos(2 * np.pi * time_seconds / seconds_in_day)

    # Day of week (0 = monday, 6 = Sunday)
    day_of_week = dt.weekday()
    # Month (1-12)
    month_of_year = dt.month
    return time_of_day_sin, time_of_day_cos, day_of_week, month_of_year

async def _get_cached_trip_update(trip_id:str) -> Optional[Dict[str, Any]]:
    """Fetches cached trip update data from Redis"""
    cache_key = f"trip_update: {trip_id}"
    return await data_storage.get_cached_data(cache_key)

async def _get_cached_vehicle_position(vehicle_id:Optional[str], trip_id:Optional[str]) -> Optional[Dict[str, Any]]:
    """Fetches cached vehicle position data from Redis, trying vehicle_id first"""
    if vehicle_id:
        cache_key = f"vehicle:{vehicle_id}"
        data = await data_storage.get_cached_data(cache_key)
        if data: return data
    if trip_id:
        # Fallback if vehicle_id is missing or position not found by vehicle_id
        cache_key = f"vehicle:trip:{trip_id}"
        data = await data_storage.get_cached_data(cache_key)
        if data:return data
    return None

async def _get_contextual_data(location:Tuple[float, lat])-> Dict[str, Optional[Any]]:
    """Fetches weather and traffic data"""
    # TODO: Implement bounding boxes for traffic, caching strategies and error handling
    context = {"Weather":None, "traffic":None}
    lat, lon = location
    
    weather_cache_key = f"weather:{lat:.3f}:{lon:.3f}"
    cached_weather = await data_storage.get_cached_data(weather_cache_key)
    if cached_weather:
        context["weather"] = cached_weather
    else:
        session = api_clients.httpx.AsyncClient()
        try:
            weather_data = await api_clients.get_weather_forecast(api_clients.WEATHER_API_KEY,
                                                                  location, session)
            if weather_data:
                context["weather"] = weather_data
                # Cache result
                await data_storage.cache_data(weather_cache_key, weather_data, ttl_seconds=60*15)
        finally:
            await session.aclose()
    # TODO: Implemnt Traffic (needs mapping route segment to roads)
    # For now, placeholder - assume traffic data is fetched elsewhere and put in cache maybe
    # traffic_cache_key = f"traffic:area:{lat:.2f}:{lon:.2f}"
    # context['traffic'] = await data_storage.get_cached_data(traffic_cache_key)
    return context

async def create_delay_features(trip_id:str, route_id:Optional[str],
                                current_stop_sequence:Optional[int],
                                target_stop_sequence:int,
                                query_time:datetime,
                                trip_context:Optional[Trip] = None,
                                route_context:Optional[Route] = None,
                                static_data_accessor:Optional[Any] = None,
                                ) -> Optional[pd.Series]:
    """
    Creates a feature vector (Pandas Series) for predicting delay/travel time
    for a trip segment.

    Args:
        trip_id: The ID of the trip.
        route_id: The ID of the route.
        current_stop_sequence: The sequence number of the stop the vehicle is
                               approaching or has just left.
        target_stop_sequence: The sequence number of the stop for which delay is predicted.
        query_time: The time at which the prediction is being made.
        trip_context: Pre-fetched Trip object (optional).
        route_context: Pre-fetched Route object (optional).
        static_data_accessor: Object to get static data like stop distances (optional).

    Returns:
        A Pandas Series containing features, or None if essential data is missing.
        Feature names should align with model expectations (e.g., DELAY_MODEL_FEATURES).
    """
    logger.debug(f"Creating delay features for trip {trip_id}, target sequence {target_stop_sequence}")
    features = {}
    # 1.Time features
    time_sin, time_cos, dow, month = _encode_time_features(query_time)
    features['time_of_day_sin'] = time_sin
    features['time_of_day_cos'] = time_cos
    features['day_of_week'] = dow
    features['month_of_year'] = month
    features['is_peak_hour'] = 1 if (7 <= query_time.hour <= 9 or 16 <= query_time.hour <= 18) else 0 

    # 2. Trip/Route Context (Static)
     # TODO: Fetch trip_context, route_context if not provided (e.g., from DB cache)
    if route_context:
        features['route_type'] = safe_int(route_context.route_type, default=-1) # Use -1 for unknown
    else:
        features['route_type'] = -1 # Or fetch from DB/cache based on route_id
    
    # 3. Real-time Trip Status
    trip_update = await _get_cached_trip_update(trip_id)
    vehicle_id = trip_update.get("vehicle_id") if trip_update else None
    vehicle_pos = await _get_cached_vehicle_position(vehicle_id, trip_id)

    current_delay_reported = None
    if trip_update and "stop_time_updates" in trip_update:
        # Find the most relevant reported delay from the trip update
        relevant_stu = None
        for stu in reversed(trip_update['stop_time_updates']): # Check recent updates first
            seq = stu.get('stop_sequence')
            if seq is not None and current_stop_sequence is not None and seq <= current_stop_sequence:
                relevant_stu = stu
                break
            if not relevant_stu and trip_update.get("delay") is not None:
                current_delay_reported = safe_int(trip_update["delay"])
        if relevant_stu:
            # Prefer departure delay of available, else arrival delay
            dep_delay = relevant_stu.get("departure_delay")
            arr_delay = relevant_stu.get("arrival_delay")
            if dep_delay is not None:
                current_delay_reported = safe_int(dep_delay)
            elif arr_delay is not None:
                current_delay_reported = safe_int(arr_delay)
    features["current_reported_delay"] = current_delay_reported if current_delay_reported is not None else 0

    # 4. Real time vehicle status
    features["current_speed"] = safe_float(vehicle_pos.get("speed"), default=0.0) if vehicle_pos else 0.0

    # 5. Segment / Positional features
    # TODO: Implement logic using static_data_accessor
    features["segment_distance_km"] = 1.0 # Just a placeholder calculate distance from current pos/seq to target seq
    features["stops_remaining"] = max(0, target_stop_sequence - (current_stop_sequence or 0))

    # 6. Contextual Features (Weather, Traffic) - need more work on
    current_loc = (safe_float(vehicle_pos['latitude']), safe_float(vehicle_pos['longitude'])) if vehicle_pos else None # Needs lat/lon
    context_data = {}
    if current_loc:
        context_data = await _get_contextual_data(current_loc)

    weather_info = context_data.get('weather')
    traffic_info = context_data.get('traffic') # Placeholder

    features['weather_temp'] = safe_float(weather_info['main']['temp']) if weather_info and 'main' in weather_info else 15.0 # Default temp?
    features['weather_precip_mm'] = safe_float(weather_info.get('rain', {}).get('1h')) if weather_info else 0.0 # Example rain

    #NOTE: Traffic is complex. Requires workaround for mapping route segment to roads and getting traffic levels.
    features['traffic_level'] = 0.5 # Placeholder (e.g., 0=low, 1=high)

    # 7. Advanced / Historical Features (Placeholders)
    # TODO: Implement logic to fetch/calculate these
    features['recent_avg_delay_route'] = 10.0 # Placeholder: Avg delay for this route/time recently (from DB)

    feature_series = pd.Series(features)
    try:
        # Select and reindex to ensure correct order and columns, fill missing with 0 or mean/median
        feature_series = feature_series.reindex(DELAY_MODEL_FEATURES).fillna(0)
        # Ensure correct types if model is strict (e.g., convert all to float)
        # feature_series = feature_series.astype(float)
    except Exception as e:
        logger.error(f"Error finalizing feature vector for trip {trip_id}: {e}", exc_info=True)
        return None

    logger.debug(f"Generated features for {trip_id}: {feature_series.to_dict()}")
    return feature_series

async def create_connection_features(arrival_trip_id:str, departure_trip_id:str,
                                     arrival_stop_id:str, departure_stop_id:str,
                                     predicted_arrival_time:datetime,
                                     scheduled_departure_time:datetime, predicted_departure_delay_seconds:int,
                                     query_time:datetime, static_data_accessor:Optional[Any] = None)->Optional[pd.Series]:
    """
    Creates a feature vector for predicting connection success probability.

    Args:
        arrival_trip_id: Trip ID of the arriving vehicle.
        departure_trip_id: Trip ID of the departing vehicle.
        arrival_stop_id: Stop ID where the arrival occurs.
        departure_stop_id: Stop ID where the departure occurs.
        predicted_arrival_time: The *predicted* arrival time of the first leg.
        scheduled_departure_time: The *scheduled* departure time of the second leg.
        predicted_departure_delay_seconds: The *predicted* delay for the departing trip at its departure stop.
        query_time: The time the prediction is being made.
        static_data_accessor: Object to get static data like walk times.

    Returns:
        A Pandas Series containing features, or None if essential data is missing.
    """
    features = {}
    logger.debug(f"Creating connection features for {arrival_trip_id}->{departure_trip_id} at {arrival_stop_id}/{departure_stop_id}")

    #1. Time features
    time_sin, time_cos, dow, month = _encode_time_features(query_time)
    features['time_of_day_sin'] = time_sin
    features['time_of_day_cos'] = time_cos
    features['day_of_week'] = dow # Maybe less relevant?
    features['is_peak_hour'] = 1 if (7 <= query_time.hour <= 9 or 16 <= query_time.hour <= 18) else 0

    # 2. Timing / Buffer Calculation
    # Predicted departure = scheduled + predicted delay
    predicted_departure_time = scheduled_departure_time + timedelta(seconds=predicted_departure_delay_seconds)

    # Walking time (needs lookup from static data)
    walk_time = timedelta(seconds=60) # Placeholder: Fetch from static_data_accessor or RoutingData
    if static_data_accessor and hasattr(static_data_accessor, 'get_transfer_time'):
         walk_time = await static_data_accessor.get_transfer_time(arrival_stop_id, departure_stop_id) or walk_time

    features['walk_time_seconds'] = walk_time.total_seconds()

    # Effective arrival time at departure platform/stop
    effective_arrival_time = predicted_arrival_time + walk_time

    # Buffer time = Predicted Departure Time - Effective Arrival Time
    buffer_time = predicted_departure_time - effective_arrival_time
    features['buffer_time_seconds'] = buffer_time.total_seconds()

    # 3. Trip Delays
    # Arrival delay: predicted arrival vs scheduled arrival (need scheduled arrival time)
    # TODO: Need access to scheduled arrival time of the *first* trip.
    # features['arrival_delay_predicted'] = (predicted_arrival_time - scheduled_arrival_time).total_seconds()
    features['arrival_delay_predicted'] = 0 # Placeholder

    features['departure_delay_scheduled'] = predicted_departure_delay_seconds # Already predicted

    # 4. Route Types (Context)
    # TODO: Fetch route types for arrival_trip_id and departure_trip_id
    features['arr_route_type'] = 3 # Placeholder (Bus)
    features['dep_route_type'] = 3 # Placeholder (Bus)

    feature_series = pd.Series(features)
    try:
        feature_series = feature_series.reindex(CONNECTION_MODEL_FEATURES).fillna(0)
    except Exception as e:
        logger.error(f"Error finalizing connection features for {arrival_trip_id} -> {departure_trip_id}: {e}", exc_info=True)
        return None

    logger.debug(f"Generated connection features: {feature_series.to_dict()}")
    return feature_series
