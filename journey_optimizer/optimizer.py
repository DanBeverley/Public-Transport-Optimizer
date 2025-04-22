"""
Core journey optimization logic.

Combines static routing with real-time predictions (delay, connection probability)
to find optimal journeys based on user criteria. Includes basic disruption handling.
"""

import logging
import asyncio
from datetime import datetime, timedelta, date, time
from typing import List, Optional, Tuple, Dict, Any
import math


from .models import OptimizationCriteria, ScoredJourney, DisruptionInfo
from routing_engine.models import Location, Journey, JourneyLeg, Stop, Trip, Route
from routing_engine import router as static_router
from routing_engine import graph_builder 
from prediction_service import delay_predictor, connection_analyzer
from data_pipeline import data_storage

logger = logging.getLogger(__name__)

# Constants
MAX_CANDIDATE_JOURNEYS = 5 
MAX_RESULTS = 3            
DEFAULT_WALK_SPEED_MPS = 1.3 
MAX_INITIAL_WALK_KM = 1.0 

# Helper Functions
def _haversine_distance(loc1:Location, loc2:Location) -> float:
    """Calculate distance between two lat/lon points in kilometers(approx)"""
    R = 6371 # Earth's radius in km
    lat1, lon1 = math.radians(loc1.latitude), math.radians(loc1.longitude)
    lat2, lon2 = math.radians(loc2.latitude), math.radians(loc2.longitude)
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = math.sin(d_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c # Distance in km
    return distance

async def _find_neaby_stops(location:Location, max_distance_km:float,
                            stops_by_id:Dict[str, Stop]) -> List[Tuple[Stop, float]]:
    """Finds stop within a certain distance, returning (Stop, distance_km) tuples"""
    nearby = []
    if not stops_by_id:
        logger.warning("No stops data provided to _find_nearby_stops")
        return []
    for stop in stops_by_id.values():
        distance = _haversine_distance(location, stop.location)
        if distance <= max_distance_km:
            nearby.append((stop, distance))
    # Sort by distance
    nearby.sort(key=lambda x:x[1])
    return nearby

async def _get_scheduled_times(leg:JourneyLeg, static_data_accessor:Any) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Retrieve the scheduled (not real-time) arrival and departure times for a journey leg.
    For transit legs, attempts to look up the times from the static GTFS data using the accessor.
    For walk legs, returns the leg's start and end times (as walking is not scheduled in GTFS).

    Args:
        leg: The JourneyLeg to get scheduled times for.
        static_data_accessor: An object providing access to GTFS static data, expected to have a method like
            get_scheduled_times(trip_id, stop_id) -> datetime or (arrival, departure).

    Returns:
        (scheduled_departure_time, scheduled_arrival_time) as datetimes, or (None, None) if unavailable.
    """
    if leg.leg_type != "transit":
        # For walking, scheduled times are just the times in the leg
        return leg.start_time, leg.end_time

    trip = leg.trip
    dep_stop = leg.departure_stop
    arr_stop = leg.arrival_stop
    if not (trip and dep_stop and arr_stop):
        logger.warning("Transit leg missing trip or stop information. Returning leg times as fallback.")
        return leg.start_time, leg.end_time

    trip_id = trip.id
    dep_stop_id = dep_stop.id
    arr_stop_id = arr_stop.id

    # Try to use static_data_accessor if available
    if static_data_accessor is not None:
        # Try to use a method for getting scheduled times (by convention)
        get_sched = getattr(static_data_accessor, "get_scheduled_times", None)
        if callable(get_sched):
            try:
                # Should return (dep_time, arr_time) as datetime objects
                result = await get_sched(trip_id, dep_stop_id, arr_stop_id)
                if result and isinstance(result, (tuple, list)) and len(result) == 2:
                    sched_dep, sched_arr = result
                    return sched_dep, sched_arr
                else:
                    logger.warning(f"static_data_accessor.get_scheduled_times returned unexpected result for trip {trip_id}")
            except Exception as e:
                logger.error(f"Error fetching scheduled times from static_data_accessor: {e}", exc_info=True)
        else:
            # Try to use lower-level methods if available (e.g., get_scheduled_departure/arrival)
            get_dep = getattr(static_data_accessor, "get_scheduled_departure", None)
            get_arr = getattr(static_data_accessor, "get_scheduled_arrival", None)
            try:
                sched_dep = await get_dep(trip_id, dep_stop_id) if callable(get_dep) else None
                sched_arr = await get_arr(trip_id, arr_stop_id) if callable(get_arr) else None
                if sched_dep or sched_arr:
                    return sched_dep or leg.start_time, sched_arr or leg.end_time
            except Exception as e:
                logger.error(f"Error fetching scheduled departure/arrival: {e}", exc_info=True)

    # Default fallback: use leg's times
    logger.info(f"Falling back to leg times for scheduled times of trip {trip_id}")
    return leg.start_time, leg.end_time

async def _get_stop_sequence(trip_id: str, stop_id: str, static_data_accessor: Any) -> Optional[int]:
    """
    Retrieve the stop_sequence for a stop within a trip using static GTFS data.
    Attempts to use the static_data_accessor if provided, falling back to None if unavailable.

    Args:
        trip_id: The GTFS trip_id.
        stop_id: The GTFS stop_id.
        static_data_accessor: An object providing access to GTFS static data, expected to have a method like
            get_stop_sequence(trip_id, stop_id) -> int.

    Returns:
        The stop_sequence (int) if found, otherwise None.
    """
    if static_data_accessor is not None:
        # Try to use a method for getting stop_sequence directly
        get_seq = getattr(static_data_accessor, "get_stop_sequence", None)
        if callable(get_seq):
            try:
                seq = await get_seq(trip_id, stop_id)
                if seq is not None:
                    return seq
                else:
                    logger.warning(f"static_data_accessor.get_stop_sequence returned None for trip {trip_id} stop {stop_id}")
            except Exception as e:
                logger.error(f"Error fetching stop_sequence from static_data_accessor: {e}", exc_info=True)
        else:
            # Try to use a lower-level method (e.g., get_stop_times_for_trip)
            get_stop_times = getattr(static_data_accessor, "get_stop_times_for_trip", None)
            if callable(get_stop_times):
                try:
                    stop_times = await get_stop_times(trip_id)
                    # stop_times should be a list/dict of stops for the trip, each with stop_id and stop_sequence
                    if stop_times:
                        for st in stop_times:
                            # Accept dict or object
                            st_id = st.get("stop_id") if isinstance(st, dict) else getattr(st, "stop_id", None)
                            st_seq = st.get("stop_sequence") if isinstance(st, dict) else getattr(st, "stop_sequence", None)
                            if st_id == stop_id:
                                return int(st_seq) if st_seq is not None else None
                    logger.warning(f"Stop_id {stop_id} not found in stop_times for trip {trip_id}")
                except Exception as e:
                    logger.error(f"Error fetching stop_times for trip {trip_id}: {e}", exc_info=True)
    # Not found
    logger.info(f"Could not determine stop_sequence for trip {trip_id}, stop {stop_id}")
    return None

async def _score_journey(candidate_journey: Journey, criteria: OptimizationCriteria,
                        query_time: datetime, static_data_accessor: Optional[Any]) -> Optional[ScoredJourney]:
    """
    Evaluates a single candidate journey using real-time predictions and static data.
    Returns a ScoredJourney with predicted times and scores, or None if scoring fails.
    """
    logger.debug(f"Scoring candidate journey starting around: {candidate_journey.start_time}")
    if not candidate_journey or not candidate_journey.legs:
        logger.warning("Cannot score empty candidate journey")
        return None

    num_legs = len(candidate_journey.legs)
    predicted_leg_times = []  # List of (pred_start, pred_end) for each leg
    current_predicted_time = candidate_journey.start_time
    cumulative_reliability = 1.0
    prediction_failed = False

    # Precompute scheduled times and stop sequences for all legs
    scheduled_times = []
    stop_sequences = []
    for leg in candidate_journey.legs:
        sched_start, sched_end = await _get_scheduled_times(leg, static_data_accessor)
        scheduled_times.append((sched_start, sched_end))
        if leg.leg_type == "transit" and leg.trip and leg.arrival_stop:
            arr_seq = getattr(leg, 'arrival_stop_sequence', None)
            if arr_seq is None:
                arr_seq = await _get_stop_sequence(leg.trip.id, leg.arrival_stop.id, static_data_accessor)
            stop_sequences.append(arr_seq)
        else:
            stop_sequences.append(None)

    for i, leg in enumerate(candidate_journey.legs):
        pred_start = current_predicted_time
        sched_start, sched_end = scheduled_times[i]
        pred_end = None
        predicted_arrival_delay = timedelta(0)

        if leg.leg_type == "transit":
            if not leg.trip or not leg.arrival_stop:
                logger.warning(f"Transit leg {i} missing trip/stop info. Skipping scoring for this journey")
                return None
            target_stop_sequence = stop_sequences[i]
            if target_stop_sequence is None:
                logger.warning(f"Could not determine stop sequence for leg {i}. Using 0 delay.")
                predicted_arrival_delay = timedelta(0)
                prediction_failed = True
            else:
                try:
                    predicted_arrival_delay = await delay_predictor.predict_arrival_delay(
                        trip_id=leg.trip.id,
                        target_stop_sequence=target_stop_sequence,
                        query_time=query_time,
                        route_id=leg.trip.route_id if hasattr(leg.trip, 'route_id') else None,
                        static_data_accessor=static_data_accessor
                    )
                    if predicted_arrival_delay is None:
                        predicted_arrival_delay = timedelta(0)
                        prediction_failed = True
                except Exception as e:
                    logger.error(f"Delay prediction failed for leg {i}: {e}", exc_info=True)
                    predicted_arrival_delay = timedelta(0)
                    prediction_failed = True
            pred_end = sched_end + predicted_arrival_delay
        elif leg.leg_type == "walk":
            pred_end = pred_start + leg.duration
        else:
            logger.warning(f"Unknown leg type: {leg.leg_type}. Skipping journey.")
            return None
        predicted_leg_times.append((pred_start, pred_end))
        # --- Predict Connection Probability for Transfers ---
        if i + 1 < num_legs:
            next_leg = candidate_journey.legs[i + 1]
            # Only consider transfer if current is transit and next is transit (with or without walk in between)
            if leg.leg_type == "transit" and next_leg.leg_type == "transit":
                try:
                    dep_stop_id = next_leg.departure_stop.id if next_leg.departure_stop else None
                    dep_stop_sequence = getattr(next_leg, 'departure_stop_sequence', None)
                    if dep_stop_sequence is None and next_leg.trip and dep_stop_id:
                        dep_stop_sequence = await _get_stop_sequence(next_leg.trip.id, dep_stop_id, static_data_accessor)
                    sched_dep_start, _ = scheduled_times[i + 1]
                    conn_prob = await connection_analyzer.predict_connection_success_probability(
                        arrival_trip_id=leg.trip.id,
                        arrival_stop_id=leg.arrival_stop.id,
                        scheduled_arrival_time=sched_end,
                        predicted_arrival_delay=predicted_arrival_delay,
                        departure_trip_id=next_leg.trip.id if next_leg.trip else None,
                        departure_stop_id=dep_stop_id,
                        departure_stop_sequence=dep_stop_sequence,
                        scheduled_departure_time=sched_dep_start,
                        query_time=query_time,
                        static_data_accessor=static_data_accessor
                    ) if (dep_stop_id and dep_stop_sequence and sched_dep_start) else 1.0
                    if conn_prob is None:
                        conn_prob = 1.0
                    conn_prob = max(0.0, min(1.0, conn_prob))
                    cumulative_reliability *= conn_prob
                except Exception as e:
                    logger.error(f"Connection probability prediction failed between leg {i} and {i+1}: {e}", exc_info=True)
                    cumulative_reliability *= 1.0
                    prediction_failed = True
        current_predicted_time = pred_end

    # --- Calculate Final Scores ---
    final_predicted_arrival = predicted_leg_times[-1][1]
    final_predicted_duration = final_predicted_arrival - predicted_leg_times[0][0]
    duration_score = final_predicted_duration.total_seconds() / 60.0
    reliability_penalty = (1.0 - cumulative_reliability) * 100
    transfer_penalty = candidate_journey.num_transfers * 20
    optimization_score = (criteria.weight_duration * duration_score +
                          criteria.weight_reliability * reliability_penalty +
                          criteria.weight_transfers * transfer_penalty)
    if prediction_failed:
        optimization_score *= 1.5
        logger.warning(f"Applying penalty to score for journey {getattr(candidate_journey, 'id', '')} due to prediction failures.")
    return ScoredJourney(
        journey=candidate_journey,
        predicted_arrival_time=final_predicted_arrival,
        predicted_duration=final_predicted_duration,
        reliability_score=cumulative_reliability,
        optimization_score=optimization_score
    )

def _create_walking_leg(start_loc: Location, end_loc: Location, start_time: datetime, walk_speed_mps: float = DEFAULT_WALK_SPEED_MPS) -> JourneyLeg:
    distance_km = _haversine_distance(start_loc, end_loc)
    distance_m = distance_km * 1000
    walk_duration = timedelta(seconds=distance_m / walk_speed_mps)
    return JourneyLeg(
        leg_type="walk",
        start_location=start_loc,
        end_location=end_loc,
        start_time=start_time,
        end_time=start_time + walk_duration,
        trip=None,
        route=None,
        departure_stop=None,
        arrival_stop=None,
        geometry=None
    )

async def find_optimal_journeys(
    start_location: Location,
    end_location: Location,
    departure_time: datetime,
    criteria: OptimizationCriteria = OptimizationCriteria(),
    static_routing_data: Optional[Any] = None, # e.g., routing_engine.RoutingData object
    max_results: int = MAX_RESULTS
    ) -> List[ScoredJourney]:
    """
    Finds the top N optimal journeys based on real-time predictions and criteria.
    Returns a list of ScoredJourney objects, sorted by optimization score (best first).
    """
    logger.info(f"Finding optimal journeys from {start_location} to {end_location} around {departure_time}")

    if static_routing_data is None or not hasattr(static_routing_data, 'stops_by_id'):
         logger.error("Static routing data (with stops_by_id) is required for find_optimal_journeys.")
         return []

    # 1. Find nearest stops to origin and destination
    try:
        start_stops_dist = await _find_nearby_stops(start_location, MAX_INITIAL_WALK_KM, static_routing_data.stops_by_id)
        end_stops_dist = await _find_nearby_stops(end_location, MAX_INITIAL_WALK_KM * 1.5, static_routing_data.stops_by_id)
    except Exception as e:
        logger.error(f"Error finding nearby stops: {e}", exc_info=True)
        return []

    if not start_stops_dist or not end_stops_dist:
        logger.warning("Could not find nearby start or end stops within radius.")
        return []

    start_stops = [s for s, d in start_stops_dist[:3]]
    end_stops = [s for s, d in end_stops_dist[:3]]
    logger.info(f"Found potential start stops: {[s.id for s in start_stops]}, end stops: {[s.id for s in end_stops]}")

    # 2. Get candidate journeys from static router
    candidate_journeys_tasks = []
    try:
        static_csa_router = static_router.ConnectionScanAlgorithm(static_routing_data)
    except Exception as e:
        logger.error(f"Error initializing static router: {e}", exc_info=True)
        return []

    loop = asyncio.get_running_loop()
    for start_stop in start_stops:
        for end_stop in end_stops:
            logger.debug(f"Querying static router for {start_stop.id} -> {end_stop.id}")
            candidate_journeys_tasks.append(
                loop.run_in_executor(
                    None,
                    static_csa_router.find_earliest_arrival_journey,
                    start_stop.id, end_stop.id, departure_time
                )
            )

    try:
        scheduled_journeys_results: List[Optional[Journey]] = await asyncio.gather(*candidate_journeys_tasks)
    except Exception as e:
        logger.error(f"Error retrieving candidate journeys: {e}", exc_info=True)
        return []
    candidate_journeys: List[Journey] = [j for j in scheduled_journeys_results if j is not None]

    if not candidate_journeys:
        logger.warning("Static router found no candidate journeys.")
        return []

    logger.info(f"Found {len(candidate_journeys)} candidate journeys from static router.")

    # Add initial/final walking legs to candidate journeys if needed
    enriched_journeys = []
    for journey in candidate_journeys:
        legs = list(journey.legs)
        # Prepend walk to first stop if needed
        first_leg = legs[0] if legs else None
        walk_legs = []
        if first_leg and _haversine_distance(start_location, first_leg.start_location) > 0.05:  # >50m
            walk_leg = _create_walking_leg(start_location, first_leg.start_location, departure_time)
            walk_legs.append(walk_leg)
        # Append walk from last stop if needed
        last_leg = legs[-1] if legs else None
        if last_leg and _haversine_distance(last_leg.end_location, end_location) > 0.05:
            # Start walk after last_leg.end_time
            walk_leg = _create_walking_leg(last_leg.end_location, end_location, last_leg.end_time)
            legs.append(walk_leg)
        enriched_journeys.append(Journey(legs=walk_legs + legs))
    candidate_journeys = enriched_journeys

    # Optional: De-duplicate journeys based on sequence of stops
    unique_journeys = {}
    for journey in candidate_journeys:
        key = tuple((leg.leg_type, getattr(leg, 'trip', None) and getattr(leg.trip, 'id', None), getattr(leg, 'departure_stop', None) and getattr(leg.departure_stop, 'id', None), getattr(leg, 'arrival_stop', None) and getattr(leg.arrival_stop, 'id', None)) for leg in journey.legs)
        if key not in unique_journeys:
            unique_journeys[key] = journey
    candidate_journeys = list(unique_journeys.values())

    # 3. Score candidate journeys concurrently
    logger.info("Scoring candidate journeys with real-time predictions...")
    try:
        import utils
        query_time = utils.get_current_utc_datetime() # Use consistent time for all predictions in this run
    except Exception:
        query_time = datetime.utcnow()
    scoring_tasks = []
    for journey in candidate_journeys[:MAX_CANDIDATE_JOURNEYS]:
        scoring_tasks.append(
            _score_journey(journey, criteria, query_time, static_data_accessor=static_routing_data)
        )
    try:
        scored_journey_results: List[Optional[ScoredJourney]] = await asyncio.gather(*scoring_tasks)
    except Exception as e:
        logger.error(f"Error scoring candidate journeys: {e}", exc_info=True)
        return []
    valid_scored_journeys: List[ScoredJourney] = [sj for sj in scored_journey_results if sj is not None]

    if not valid_scored_journeys:
        logger.warning("Failed to score any candidate journeys.")
        return []

    # 4. Rank journeys
    valid_scored_journeys.sort()
    logger.info(f"Successfully scored and ranked {len(valid_scored_journeys)} journeys.")
    return valid_scored_journeys[:max_results]


def _estimate_last_completed_leg(current_journey: Journey, current_location: Location, threshold_m: float = 100.0) -> int:
    """Estimate the index of the last completed leg based on proximity to leg end locations."""
    for i, leg in enumerate(current_journey.legs):
        if _haversine_distance(current_location, leg.end_location) < (threshold_m / 1000.0):
            return i
    return -1

async def handle_disruption(
    current_journey: Journey,
    current_location: Location,
    disruption_info: DisruptionInfo,
    static_routing_data: Optional[Any] = None,
    criteria: OptimizationCriteria = OptimizationCriteria()
    ) -> List[ScoredJourney]:
    """
    Checks if a disruption affects the current journey and suggests alternatives.
    Returns a list of alternative ScoredJourney options, or an empty list if
    the journey is not affected or no alternatives are found.
    """
    logger.info(f"Handling disruption: {disruption_info.disruption_type} - {disruption_info.message}")
    # Determine user's progress along the journey based on current_location and leg end locations
    last_completed_leg_index = _estimate_last_completed_leg(current_journey, current_location)
    affected = False

    for i, leg in enumerate(current_journey.legs):
        if i <= last_completed_leg_index:
            continue
        if leg.leg_type == "transit" and leg.trip:
            if leg.trip.id in disruption_info.affected_trips:
                affected = True
                break
            if leg.route and leg.route.id in disruption_info.affected_routes:
                affected = True
                break
            if leg.departure_stop and leg.departure_stop.id in disruption_info.affected_stops:
                affected = True
                break
            if leg.arrival_stop and leg.arrival_stop.id in disruption_info.affected_stops:
                affected = True
                break
        if leg.leg_type == "walk":
            # Walking leg affected if start or end is at a closed stop
            if (getattr(leg, 'departure_stop', None) and leg.departure_stop.id in disruption_info.affected_stops) or \
               (getattr(leg, 'arrival_stop', None) and leg.arrival_stop.id in disruption_info.affected_stops):
                affected = True
                break
    if not affected:
        logger.info("Current journey does not seem directly affected by this disruption.")
        return []
    logger.warning(f"Disruption affects current journey at/after leg {last_completed_leg_index + 1}. Finding alternatives...")
    reroute_start_location = current_location
    reroute_departure_time = datetime.utcnow() + timedelta(minutes=1)
    original_destination_location = current_journey.legs[-1].end_location
    try:
        alternative_journeys = await find_optimal_journeys(
            start_location=reroute_start_location,
            end_location=original_destination_location,
            departure_time=reroute_departure_time,
            criteria=criteria,
            static_routing_data=static_routing_data
        )
        logger.info(f"Found {len(alternative_journeys)} alternative journeys.")
        return alternative_journeys
    except Exception as e:
        logger.error(f"Error finding alternative journeys during disruption handling: {e}", exc_info=True)
        return []
