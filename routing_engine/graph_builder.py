"""
Builds the necessary data structures for efficient time-dependent routing
from a static GTFS feed.

Pre-calculates connections (trip segments) and transfers (walking between stops).
"""
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Set

import gtfs_kit as gk
import pandas as pd

from .models import Route, Trip, Connection, Transfer, Location, RoutingData
from data_pipeline.utils import time_it

logger = logging.getLogger(__name__)

# Constants
DEFAULT_WALKING_SPEED_KHM = 4.5
# Maximum distance for considering a walking transfer (basing on city density)
DEFAULT_MAX_TRANSFER_DISTANCE_KM = 0.5
# Maximum duration for a transfer (to avoid unreasonbale walks)
DEFAULT_MAX_TRANSFER_DURATION_MINUTES = 15

# Helper to convert GTFS time strings (HH:MM:SS, potentially > 24:00:00) to timedelta
def _parse_gtfs_time(time_str:Optional[str]) -> Optional[timedelta]:
    if pd.isna(time_str) or not isinstance(time_str, str):
        return None
    try:
        h, m, s = map(int, time_str.split(":"))
        return timedelta(hours = h, minutes=m, seconds=s)
    except ValueError:
        logger.warning(f"Could not parse GTFS time string: {time_str}", exc_info = True)
        return None

@time_it # For performance mesurement
def _prepare_stops(feed:gk.Feed) -> Dict[str, Stop]:
    """Extract stop information from GTFS feed"""
    logger.info("Preparing stops data...")
    stops_by_id = {}
    if feed.stops is None or feed.stops.empty:
        logger.warning("GTFS feed is missing stops data")
        return stops_by_id
    
    for _, row in feed.stops.iterrows():
        stop_id = row["stop_id"]
        loc = Location(latitude=row["stop_lat"], longitude = row["stop_lon"])
        stops_by_id[stop_id] = Stop(id = stop_id,
                                    name = row.get("stop_name"),
                                    location = loc)
    logger.info(f"Prepared {len(stops_by_id)} stops")
    return stops_by_id

@time_it
def _prepare_routes_and_trips(feed:gk.Feed) -> Tuple[Dict[str, Route], Dict[str, Trip]]:
    """Extracts route and trip information from the GTFS feed"""
    logger.info("Preparing routes and trips data...")
    routes_by_id = {}
    trips_by_id = {}
    if feed.routes is not None and not feed.routes.empty:
        for _, row in feed.routes.iterrows():
            route_id = row["route_id"]
            routes_by_id[route_id] = Route(id = route_id,
                                           short_name=row.get("route_short_name"),
                                           long_name=row.get("route_long_name"),
                                           route_type=row.get("route_type"),
                                           agency_id=row.get("agency_id"))
            logger.info(f"Prepared {len(routes_by_id)} routes")
        else:
            logger.warning("GTFS feed is missing routes data")
        if feed.trips is not None and not feed.trips.empty:
            required_trip_cols = ["trip_id", "route_id", "service_id"]
            if not all(col in feed.trips.columns for col in required_trip_cols):
                logger.error(f"Trips table missing required columns: {required_trip_cols}")
                return routes_by_id, trips_by_id
        for _, row in feed.trips.iterrows():
            trip_id = row["trip_id"]
            trips_by_id[trip_id] = Trip(id = trip_id,
                                        route_id = row["route_id"],
                                        service_id = row["service_id"],
                                        direction_id = row.get("direction_id"),
                                        shape_id = row.get("shape_id"))
        logger.info(f"Prepared {len(trips_by_id)} trips")
    else:
        logger.warning("GTFS feed is missing trips data")
    return routes_by_id, trips_by_id

#@timeit
def _prepare_connections(feed:gk.Feed, query_date:date) -> List[Connection]:
    """
    Generates a list of all individual trip segments (connections)
    from the GTFS feed, sorted by departure time. Only includes connections
    active on the given query_date.

    Args:
        feed: The gtfs_kit Feed object.
        query_date: The specific date for which to find active services.

    Returns:
        A list of Connection objects sorted by departure time.
    """
    logger.info(f"Preparing connections active on {query_date}...")
    connections = []
    if feed.stop_times is None or feed.stop_times.empty:
        logger.warning("GTFS feed is missing stop_times data. Cannot create connections")
        return connections
    if feed.trips is None or feed.trips.empty:
        logger.warning("GTFS feed is missing trips data. Cannot link connections to trips")
        return connections
    # 1. Filter trips active on the query date
    try:
        active_trip_ids = set(feed.get_trips_for_date(query_date)["trip_id"])
        if not active_trip_ids:
            logger.warning(f"No active trips found for date {query_date}")
            return connections
        logger.info(f"Found {len(active_trip_ids)} active trips for {query_date}")
    except Exception as e:
        logger.error(f"Failed to determine active trips for date {query_date}:{e}", exc_info = True)
        # This might happen if calendar/calendar_dates are missing or malformed
        return connections # Cannot proceed without knowing active trips
    
    # 2.Merge stop_times with trips to get service_id and filter
    stop_times_df = feed.stop_times.copy()
    trips_df = feed.trips[["trip_id", "service_id"]].copy()
    # Ensure trip_id is the same type if merging
    stop_times_df["trip_id"] = stop_times_df["trip_id"].astype(str)
    trips_df["trip_id"] = trips_df["trip_id"].astype(str)
    st_merged = pd.merge(stop_times_df, trips_df, on="trip_id", how="inner")

    # Filter stop_times to only include those from active trips
    st_active = st_merged[st_merged["trip_id"].isin(active_trip_ids)].copy()
    if st_active.empty:
        logger.warning(f"No stop_times found for active trips on {query_date}")
        return connections
    
    # 3.Parse time and sort
    logger.info("Parsing departure/arrival times for active stop_times...")
    st_active["depature_timedelta"] = st_active["depature_time"].apply(_parse_gtfs_time)
    st_active["arrival_timedelta"] = st_active["arrival_time"].apply(_parse_gtfs_time)

    # Drop rows where time parsing failed
    st_active.dropna(subset=["departure_timedelta", "arrival_timedelta"], inplace = True)
    if st_active.empty:
        logger.warning("No valid departure/arrival times found after parsing")
        return connections
    
    # Critical: Sort by trip and sequence to easily link consecutive steps
    st_active.sort_values(by=["trip_id", "stop_sequence"], inplace = True)
    logger.info(f"Processing {len(st_active)} sorted active stop_time entries...")

    # 4. Iterate and create Connection objects
    # Use shift(-1) to get the 'next stop in the sequence for each trip
    st_active["next_trip_id"] = st_active["trip_id"].shift(-1)
    st_active["next_stop_id"] = st_active["stop_id"].shift(-1)
    st_active["next_arrival_timedelta"] = st_active["arrival_timedelta"].shift(-1)
    # Filter out the last stop of each trip (no departure from there to a *next* stop)
    # and rows where time parsing failed
    valid_connections = st_active[(st_active["trip_id"] == st_active["next_trip_id"]) &
                                  st_active["departure_timedelta"].notna() &
                                  st_active["next_arrival_timedelta"].nowna()]
    logger.info(f"Generating {len(valid_connections)} connection objects...")
    for _, row in valid_connections.iterrows():
        dep_time = row["departure_timedelta"]
        arr_time = row["next_arrival_timedelta"]
        # Sanity check: arrival should not be before departure on the same segment
        if arr_time < dep_time:
            logger.debug(f"Skipping connection for trip {row["trip_id"]} from stop"
                         f"{row["stop_id"]} to {row["next_stop_id"]} due to arrival time"
                         f"({arr_time}) being before departure time ({dep_time})")
            continue
        connections.append(Connection(departure_time=dep_time,
                                      arrival_time=arr_time,
                                      depature_stop_id=row["stop_id"],
                                      arrival_stop_id=row["next_stop_id"],
                                      trip_id = row["trip_id"]))
    
    # 5.Final sort - sort by departure time primarily
    logger.info("Sorting connections by departure time...")
    connections.sort() # Relies on Connection dataclass = True
    logger.info(f"Prepared {len(connections)} connections active on {query_date}")
    return connections

