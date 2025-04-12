"""
Implements time-dependent public transport routing algorithms.

Currently features the Connection Scan Algorithm (CSA) for finding the
earliest arrival journey based on a static schedule.
"""
import logging
from datetime import datetime, timedelta, date, time
from typing import Dict, Optional, Tuple, Union, List, Set
import heapq # For potential Dijkstra-like extensions, though CSA is simpler

from .models import Stop, Route, Trip, Connection, Transfer, Journey, JourneyLeg, RoutingData, Location

logger = logging.getLogger(__name__)

# Constant infinity for time calculations
INFINITY_TIME = timedelta.max

class ConnectionScanAlgorithm:
    """
    Finds the earliest arrival journey using the Connection Scan Algorithm.

    Processes pre-sorted connections and applies transfers to find the
    fastest path according to the timetable.
    """
    def __init__(self, routing_data:RoutingData):
        """
        Initializes the router with pre-processed routing data.

        Args:
            routing_data: A RoutingData object containing sorted connections,
                          transfers, stop info, etc.
        """
        if not routing_data or not routing_data.connections:
            raise ValueError("RoutingData is invalid or missing connections. Cannot initialize router")
        self.routing_data = routing_data
        self.stops = routing_data.stops_by_id
        self.transfers = routing_data.transfers_by_departure_stop
        self.connections = routing_data.connections 

        self.trips = routing_data.trips_by_id
        self.routes = routing_data.routes_by_id
        logger.info(f"CSA Router initialized with {len(self.connections)} connections and {len(self.stops)} stops")
    
    def find_earliest_arrival_journey(self, origin_stop_id:str,
                                      destination_stop_id:str,
                                      departure_datetime:datetime,
                                      max_transfers:Optional[int] = None) -> Optional[Journey]:
        """
        Finds the journey with the earliest arrival time.

        Args:
            origin_stop_id: The ID of the starting stop.
            destination_stop_id: The ID of the destination stop.
            departure_datetime: The desired departure time and date.
            max_transfers: (Currently informational for basic CSA) Maximum number
                           of transfers allowed. Algorithms like RAPTOR handle this better.

        Returns:
            A Journey object representing the best path found, or None if no
            path exists or input is invalid.
        """
        logger.info(f"Starting CSA search from {origin_stop_id} to {destination_stop_id} departing around {departure_datetime}")
        if origin_stop_id not in self.stops or destination_stop_id not in self.stops:
            logger.error("Origin or destination stop ID not found in routing data.")
            return None
        
        # Earliest known arrival time at each stop (relative to departure_datetime's date midnight)
        earliest_arrival_time: Dict[str, timedelta] = {stop_id: INFINITY_TIME for stop_id in self.stops}
        # How we reached each stop (stores the previous stop/leg that led to the earliest arrival)
        # Format: {stop_id: (previous_stop_id, leg_object)} where leg_object is Connection or Transfer
        journey_details: Dict[str, Tuple[str, Union[Connection, Transfer]]] = {}
        # Time relative to midnight of the departure date
        departure_time_delta = timedelta(hours = departure_datetime.hour,
                                         minutes = departure_datetime.minute,
                                         seconds = departure_datetime.second)
        earliest_arrival_time[origin_stop_id] = departure_time_delta
        journey_details[origin_stop_id] = ("origin", None) # mark origin

        if origin_stop_id in self.transfers:
            for transfer in self.transfers[origin_stop_id]:
                arrival_at_transfer_stop = departure_time_delta + transfer.duration
                if arrival_at_transfer_stop < earliest_arrival_time[transfer.to_stop_id]:
                    earliest_arrival_time[transfer.to_stop_id] = arrival_at_transfer_stop
                    journey_details[transfer.to_stop_id] = (origin_stop_id, transfer)
        # Main CSA loop
        logger.debug("Scanning connections...")
        connections_scanned = 0
        updates_made = 0
        for connection in self.connections:
            connections_scanned += 1
            dep_stop = connection.departure_stop_id
            arr_stop = connection.arrival_stop_id
            dep_time = connection.departure_time
            arr_time = connection.arrival_time
            min_time_at_dep_stop = earliest_arrival_time[dep_stop]

            if min_time_at_dep_stop <= dep_time:
                # Yes, we can potentially board this connection.
                # Does this connection offer a faster path to the arrival stop?
                if arr_time < earliest_arrival_time[arr_stop]:
                    # Found a better path! Update arrival time and predecessor.
                    earliest_arrival_time[arr_stop] = arr_time
                    journey_details[arr_stop] = (dep_stop, connection)
                    updates_made += 1
                    if arr_stop in self.transfers:
                        for transfer in self.transfers[arr_stop]:
                            transfer_dest_stop = transfer.to_stop_id
                            arrival_at_transfer_dest = arr_time + transfer.duration
                            if arrival_at_transfer_dest < earliest_arrival_time[transfer_dest_stop]:
                                # This transfer provides a faster way to reach the transfer destination
                                earliest_arrival_time[transfer_dest_stop] = arrival_at_transfer_dest
                                journey_details[transfer_dest_stop] = (arr_stop, transfer)
                                updates_made += 1

        logger.debug(f"Scan complete. Scanned {connections_scanned} connections, made {updates_made} updates.")

        # Path reconstruction
        if earliest_arrival_time[destination_stop_id] == INFINITY_TIME:
            logger.warning(f"No path found from {origin_stop_id} to {destination_stop_id} on {departure_datetime.date()}")
            return None

        logger.info(f"Path found! Reconstructing journey to {destination_stop_id}...")
        journey = self._reconstruct_journey(origin_stop_id, destination_stop_id, journey_details, departure_datetime.date())

        if journey:
            journey.calculate_metrics()
            logger.info(f"Journey reconstructed: Start={journey.start_time}, End={journey.end_time}, Duration={journey.total_duration}, Transfers={journey.num_transfers}")
        else:
             logger.error("Failed to reconstruct journey despite finding a path.")

        return journey
    
    def reconstruct_journey(self, origin_stop_id:str, destination_stop_id:str,
                            journey_details:Dict[str, Tuple[str, Union[Connection, Transfer]]],
                            query_date:date) -> Optional[Journey]:
        """
        Reconstructs the journey path backwards from the destination using
        the predecessor information stored during the scan.
        """
        legs:List[JourneyLeg] = []
        current_stop_id = destination_stop_id
        while current_stop_id != origin_stop_id:
            if current_stop_id not in journey_details:
                logger.error(f"Error reconstructing path: Stop {current_stop_id} has no predecessor info.")
                return None
            prev_stop_id, leg_info = journey_details[current_stop_id]
            if leg_info is None and prev_stop_id == "origin":
                if current_stop_id == origin_stop_id: break
                else:
                    logger.error("Error reconstructing path: Reached origin marker unexpectedly")
                    return None
            if isinstance(leg_info, Connection):
                # This leg was a transit connection
                conn = leg_info
                dep_stop = self.stops.get(conn.departure_stop_id)
                arr_stop = self.stops.get(conn.arrival_stop_id)
                trip = self.trips.get(conn.trip_id)
                route = self.routes.get(trip.route_id) if trip else None
                
                # Calculate absolute datetime objects
                start_dt = datetime.combine(query_date, time()) + conn.departure_time
                end_dt = datetime.combine(query_date, time()) + conn.arrival_time
                # Handle overnight times (timedelta > 24h) - datetime automatically handles this
                duration = end_dt - start_dt
                if not dep_stop or not arr_stop or not trip or not route:
                    logger.warning(f"Missing data during reconstruction for connection {conn}. Skipping leg")
                    # TODO: Add robust handling here
                    current_stop_id = prev_stop_id
                    continue
                legs.append(JourneyLeg(leg_type = "transit", start_time = start_dt,
                                       end_time = end_dt, duration = duration,
                                       start_location = dep_stop.location,
                                       end_location = arr_stop.location,
                                       route = route,  trip = trip,
                                       departure_stop=dep_stop,
                                       arrival_stop = arr_stop))
                current_stop_id = conn.departure_stop_id # Move to the start of the connection
            
            elif isinstance(leg_info, Transfer):
                # This leg was a walking transfer
                transfer = leg_info
                from_stop = self.stops.get(transfer.stop_from_id)
                to_stop = self.stops.get(transfer.to_stop_id) # == current_stop_id
                if not from_stop or not to_stop:
                    logger.warning(f"Missing stop data during reconstruction for transfer {transfer}. Skipping leg")
                    current_stop_id = prev_stop_id
                    continue
                end_timedelta = self._get_arrival_timedelta(current_stop_id, journey_details) # TODO: Need helper
                start_timedelta = end_timedelta - transfer.duration
                if end_timedelta == INFINITY_TIME:
                    logger.error(f"Could not determine arrival time for transfer end stop {current_stop_id}")
                    return None
                start_dt = datetime.combine(query_date, time()) + start_timedelta
                end_dt = datetime.combine(query_date, time()) + end_timedelta
                legs.append(JourneyLeg(leg_type="walk", start_time = start_dt, end_time=end_dt,
                                        duration = transfer.duration, start_location=from_stop.location,
                                        end_location = to_stop.location))
                current_stop_id = transfer.from_stop_id # Move to the start of the walk
            else:
                if current_stop_id == origin_stop_id and prev_stop_id == "origin":
                    break # reached the start
                else:
                    logger.error(f"Invalid leg_info type '{type(leg_info)}' during reconstruction")
                    return None
        if current_stop_id != origin_stop_id:
            logger.error("Journey reconstruction did not reach the origin stop")
            return None
        # Legs are currently backwards, reverse them
        journey = Journey(legs = list(reversed(legs)))
        return journey
    
    def _get_arrival_timedelta(self, stop_id:str,
                               journey_details:Dict[str, Tuple[str, Union[Connection, Transfer]]]) -> timedelta:
        """Helper to trace back and find the arrival time at a specific stop during reconstruction."""
        # NOTE: This is inefficient and complex. The main search should ideally store the arrival times.
        # A better way: the main loop's `earliest_arrival_time` dict should be available here.
        # Let's simulate having access to it (or refactor to pass it).
        # --- THIS IS A PLACEHOLDER - Needs refactoring for robust time reconstruction ---
        # When implement, pass the `earliest_arrival_time` dict computed in the main
        # function to the reconstructor, or store it alongside journey_details.

        # Placeholder logic 
        if stop_id not in journey_details: return INFINITY_TIME
        _, leg_info = journey_details[stop_id]
        if isinstance(leg_info, Connection):
            return leg_info.arrival_time # Direct arrival time from connection
        elif isinstance(leg_info, Transfer):
             # Need time at previous stop + transfer duration
             prev_stop_id = leg_info.from_stop_id
             prev_arrival_time = self._get_arrival_timedelta(prev_stop_id, journey_details)
             if prev_arrival_time != INFINITY_TIME:
                 return prev_arrival_time + leg_info.duration
             else:
                 return INFINITY_TIME
        else: # Origin
            # Need the actual departure time from the initial query
            # This highlights the need to pass more context or the earliest_arrival_time dict.
            return timedelta(0) # Incorrect placeholder
