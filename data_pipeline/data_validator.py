"""
Validates data fetched from GTFS feeds and external APIs.

Checks for schema compliance, data consistency, and timeliness.
Leverages gtfs-kit for static GTFS validation and performs custom checks
for GTFS-RT and API responses.
"""
import logging
from time import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import gtfs_kit as gk
from google.transit import gtfs_realtime_pb2

from .utils import (is_timestamp_recent, get_current_utc_datetime,
                    parse_flexible_timestamp, DEFAULT_GTFS_RT_STALENESS_SECONDS,
                    MIN_LATITUDE, MAX_LATITUDE, MAX_LONGITUDE, MIN_LONGITUDE)

logger = logging.getLogger(__name__)

def validate_static_gtfs(feed:gk.Feed) -> Tuple[List[str], List[str]]:
    """
    Validates a static GTFS feed using gtfs-kit and custom checks.

    Args:
        feed: A gtfs_kit.Feed object.

    Returns:
        A tuple containing two lists: (errors, warnings).
    """
    errors = []
    warnings = []
    logger.info(f"Starting validation for static GTFS feed: {feed.agency.iloc[0]['agency_name'] if not feed.agency.empty else 'Unknown Agency'}")
    # 1. Use gtfs-kit's built-in validation
    # Note: gtfs-kit's validate() returns a DataFrame
    validation_results = feed.validate()
    for _, row in validation_results.iterrows():
        message = f"[{row['type']}/{row['message']}]
        Table: {row['table']}, Column: {row['column']}, Rows: {row['rows']}"
        if row["level"] == "error":
            errors.append(message)
        elif row["level"] == "warnings":
            warnings.append(message)
    # 2. Custom sanity checks
    # Check stop coordinates validity
    if 'stops' in feed.files and not feed.stops.empty:
        invalid_lat = feed.stops[(feed.stops['stop_lat'] < MIN_LATITUDE | (feed.stops['stop_lat'] > MAX_LATITUDE))]
        if not invalid_lat.empty:
            warnings.append(f"Found {len(invalid_lat)} stops with invalid latitudes (outside {MIN_LATITUDE} to {MAX_LATITUDE}). IDS: {invalid_lat['stop_id'].tolist()[:5]}...")
        invalid_lon = feed.stops[(feed.stops["stop_lon"] < MIN_LONGITUDE) | (feed.stops['stop_lon'] > MAX_LONGITUDE)]
        if not invalid_lon.empty:
            warnings.append(f"Found {len(invalid_lon)} stops with invalid longitudes (outside {MIN_LONGITUDE} to {MAX_LONGITUDE}). IDS: {invalid_lon["stop_id"].tolist()[:5]}...")
    # Check for resonable service dates (e.g., not too far in the past/ future)
    if feed.calendar is not None and not feed.calendar.empty:
        try:
            min_date = feed.calendar["start_date"].min()
            max_date = feed.calendar["end_date"].max()
            today_date = get_current_utc_datetime().strftime("%Y%m%d")
            if max_date < today_date:
                warnings.append(f"Lastest service end date ({max_date}) is in the past")
            #TODO: Add check for start_date being too far in future
        except KeyError:
            warnings.append("Could not perform date checks on calendar.txt (missing columns?)")
        except Exception as e:
            warnings.append(f"Error during calendar data check: {e}")
            # TODO : More custom checks (e.g, duplicate entries, logical time)
    if errors:
        logger.error(f"Static GTFS validation found {len(errors)} errors")
    if warnings:
        logger.warning(f"Static GTFS validation found {len(warnings)} warnings")
    return errors, warnings

def validate_realtime_feed(feed_message:Optional[gtfs_realtime_pb2.FeedMessage], feed_url:str,
                            max_staleness_seconds:int = DEFAULT_GTFS_RT_STALENESS_SECONDS) -> Tuple[List[str], List[str]]:
    """
    Validates a GTFS-Realtime FeedMessage.

    Checks for presence, header timestamp validity, and basic entity sanity.

    Args:
        feed_message: The parsed FeedMessage object (or None if parsing failed).
        feed_url: The source URL for context in logs/errors.
        max_staleness_seconds: Maximum allowed age of the feed header timestamp.

    Returns:
        A tuple containing two lists: (errors, warnings).
    """
    errors = []
    warnings = []
    if feed_message is None:
        errors.append(f"Received null FeedMessage object for {feed_url}. Cannot validate")
        return errors, warnings
    #1. Check header Timestamp
    if not feed_message.header.HasField("timestamp"):
        warnings.append(f"GTFS-RT feed header from {feed_url} is missing timestamp.")
    elif not is_timestamp_recent(feed_message.header.timestamp, max_staleness_seconds):
        ts_dt = datetime.fromtimestamp(feed_message.header.timestamp, tz=timezone.utc)
        warnings.append(
            f"GTFS-RT feed from {feed_url} is stale. Header timestamp: {ts_dt} "
            f"(> {max_staleness_seconds}s old)."
        )

    # 2. Check GTFS Version
    if feed_message.header.gtfs_realtime_version != "2.0": # Current standard
         warnings.append(
             f"GTFS-RT feed from {feed_url} uses version "
             f"{feed_message.header.gtfs_realtime_version} (expected 2.0)."
         )

    # 3. Basic Entity Validation (can be expanded)
    entity_count = len(feed_message.entity)
    if entity_count == 0:
        warnings.append(f"GTFS-RT feed from {feed_url} contains 0 entities.")
    else:
        # Example: Check a sample of entities for basic required fields
        vehicle_updates = 0
        trip_updates = 0
        alerts = 0
        for entity in feed_message.entity[:10]: # Check first few
            if entity.HasField("vehicle"):
                vehicle_updates += 1
                if not entity.vehicle.HasField("trip") and not entity.vehicle.HasField("vehicle"):
                     warnings.append(f"VehiclePosition entity (ID: {entity.id}) missing trip and vehicle descriptor.")
                if not entity.vehicle.HasField("position"):
                     warnings.append(f"VehiclePosition entity (ID: {entity.id}) missing position data.")
            elif entity.HasField("trip_update"):
                trip_updates += 1
                if not entity.trip_update.HasField("trip"):
                    warnings.append(f"TripUpdate entity (ID: {entity.id}) missing trip descriptor.")
                if not entity.trip_update.stop_time_update:
                    warnings.append(f"TripUpdate entity (ID: {entity.id}) has no stop_time_updates.")
            elif entity.HasField("alert"):
                alerts += 1
                if not entity.alert.informed_entity:
                    warnings.append(f"Alert entity (ID: {entity.id}) has no informed_entity.")

        logger.debug(f"Validated {entity_count} entities in feed from {feed_url} "
                     f"(Vehicles: {vehicle_updates}, Trips: {trip_updates}, Alerts: {alerts} in sample).")


    if errors:
        logger.error(f"GTFS-RT validation for {feed_url} found {len(errors)} errors.")
    if warnings:
        logger.warning(f"GTFS-RT validation for {feed_url} found {len(warnings)} warnings.")

    return errors, warnings


# TODO: Implement Pydantic for robust schema validation of API responses
# Example using dictionary checks for now:

def validate_traffic_response(data: Optional[Dict[str, Any]], source_desc: str) -> Tuple[List[str], List[str]]:
    """Placeholder for validating traffic API response structure."""
    errors = []
    warnings = []
    if data is None:
        errors.append(f"Received null traffic data from {source_desc}.")
        return errors, warnings
    # TODO: Add specific checks based on the chosen Traffic API structure
    # e.g., check for required keys like 'flow_segments', 'speed', 'jam_factor'
    if not isinstance(data, dict):
         errors.append(f"Traffic data from {source_desc} is not a dictionary.")
    elif not data: # Check if empty
        warnings.append(f"Received empty traffic data dictionary from {source_desc}.")
    # Example check:
    # if "incidents" not in data:
    #     warnings.append(f"Traffic data from {source_desc} missing 'incidents' key.")

    return errors, warnings

def validate_weather_response(data: Optional[Dict[str, Any]], source_desc: str) -> Tuple[List[str], List[str]]:
    """Placeholder for validating weather API response structure."""
    errors = []
    warnings = []
    if data is None:
        errors.append(f"Received null weather data from {source_desc}.")
        return errors, warnings
    # Example for OpenWeatherMap 'weather' endpoint
    if not isinstance(data, dict):
         errors.append(f"Weather data from {source_desc} is not a dictionary.")
         return errors, warnings

    required_keys = ["coord", "weather", "main", "wind", "dt", "sys", "id", "name"]
    for key in required_keys:
        if key not in data:
            errors.append(f"Weather data from {source_desc} missing required key: '{key}'.")
    if "weather" in data and not isinstance(data["weather"], list):
         errors.append(f"Weather data key 'weather' is not a list in response from {source_desc}.")
    elif "weather" in data and data["weather"]:
        if "main" not in data["weather"][0] or "description" not in data["weather"][0]:
             warnings.append(f"Weather data 'weather' list item missing 'main' or 'description' in {source_desc}.")

    if "main" in data and "temp" not in data["main"]:
        errors.append(f"Weather data missing 'temp' in 'main' object from {source_desc}.")

    # Check timestamp is recent
    if "dt" in data:
        if not is_timestamp_recent(data["dt"], max_staleness_seconds=3600): # Allow 1 hour old weather
            ts = parse_flexible_timestamp(data["dt"])
            warnings.append(f"Weather data from {source_desc} timestamp 'dt' is older than 1 hour: {ts}")

    return errors, warnings

def validate_events_response(data: Optional[List[Dict[str, Any]]], source_desc: str) -> Tuple[List[str], List[str]]:
    """Placeholder for validating events API response structure."""
    errors = []
    warnings = []
    if data is None:
        # Note: API client returns None on failure, [] on empty success.
        # An empty list is valid here, meaning no events found.
        warnings.append(f"Received null/failed events data fetch from {source_desc}.")
        return errors, warnings # Treat null as error, but [] as just a warning/info

    if not isinstance(data, list):
        errors.append(f"Events data from {source_desc} is not a list.")
        return errors, warnings

    if not data:
        logger.info(f"No events found from {source_desc}. Valid empty response.")
        return errors, warnings

    # Check structure of the first event item as an example
    first_event = data[0]
    if not isinstance(first_event, dict):
        errors.append(f"First event item from {source_desc} is not a dictionary.")
    else:
        # TODO: Adapt to multiple source
        required_keys = ["name", "type", "id", "dates", "_embedded"]
        for key in required_keys:
            if key not in first_event:
                warnings.append(f"First event from {source_desc} missing suggested key: '{key}'.")
        if "dates" in first_event and "start" not in first_event["dates"]:
            warnings.append(f"First event from {source_desc} missing 'start' in 'dates'.")
        if "_embedded" in first_event and "venues" not in first_event["_embedded"]:
             warnings.append(f"First event from {source_desc} missing 'venues' in '_embedded'.")

    return errors, warnings
