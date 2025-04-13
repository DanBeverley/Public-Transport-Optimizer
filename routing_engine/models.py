"""
Data models for representing the transit network graph components,
routing requests, and resulting journeys.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union
from datetime import datetime, timedelta, time

@dataclass(frozen = True)
class Location:
    """Represents a geographic coordinate"""
    latitude:float
    longitude:float

@dataclass(frozen=True)
class Stop:
    """Represent public transport stop or station"""
    id:str # Corresponds to GTFS stop_id
    name:Optional[str] = None
    location:Optional[Location] = None

@dataclass(frozen = True)
class Route:
    """Represents a public transport route"""
    id:str
    short_name:Optional[str] = None
    long_name:Optional[str] = None
    route_type:Optional[int] = None # GTFS route_type (e.g., 3 for Bus)
    agency_id:Optional[str] = None

@dataclass(frozen=True)
class Trip:
    """Represents a specific instance of a vehicle traveling along a route"""
    id:str # Corresponds to GTFS trip_id
    route_id:str
    service_id:str # Links to calendar/calendar_notes
    direction_id:Optional[int] = None
    shape_id:Optional[str] = None 

# Structure for routing algorithms (especially CSA)

@dataclass(frozen=True, order=True)
class Connection:
    """
    Represents a direct connection between two stops via a specific trip,
    sorted primarily by departure time. Essential for Connection Scan Algorithm.
    """
    # Sort key field must come first for implicit ordering
    departure_time:timedelta # Time past midnight on the service day
    arrival_time:timedelta
    depature_stop_id:str
    arrival_stop_id:str
    trip_id:str

@dataclass(frozen=True)
class Transfer:
    """A potential transfer between two stops"""
    from_stop_id:str
    to_stop_id:str
    duration:timedelta # Walking time

@dataclass(frozen=True)
class JourneyLeg:
    """A single leg of a journey (either transit or walking)"""
    leg_type:str # 'transit' | 'walk'
    start_time:datetime
    end_time:datetime
    duration:timedelta
    start_location:Location
    end_location:Location
    route:Optional[Route] = None
    trip:Optional[Trip] = None
    departure_stop:Optional[Stop] = None
    arrival_stop:Optional[Stop] = None
    geometry:Optional[List[Location]] = None

@dataclass
class Journey:
    """A complete journey from an origin to a destination"""
    legs:List[JourneyLeg] = field(default_factory=list)
    total_duration:timedelta = timedelta(0)
    start_time:Optional[datetime] = None
    end_time:Optional[datetime] = None
    num_transfers:int = 0 # Number of transit vehicle changes

    def calculate_metrics(self):
        """Calculates overall metrics after legs are added"""
        if not self.legs:
            self.total_duration = timedelta(0)
            self.start_time = None
            self.end_time = None
            self.num_transfers = 0
            return
        self.start_time = self.legs[0].start_time
        self.end_time = self.legs[-1].end_time
        self.total_duration = self.end_time - self.start_time

        # Count transfers between distinct transit legs
        self.num_transfers = 0
        last_trip_id = None
        for leg in self.legs:
            if leg.leg_type == "transit":
                current_trip_id = leg.trip.id if leg.trip else None
                if last_trip_id is not None and current_trip_id != last_trip_id:
                    self.num_transfers += 1
                last_trip_id = current_trip_id

# Holding prepared routing data

@dataclass
class RoutingData:
    """Container for data structures needed by the routing algorithms"""
    connections:List[Connection] = field(default_factory=list) # Must be sorted by departure time
    transfers_by_departure_stop:Dict[str, List[Transfer]] = field(default_factory=dict)
    stops_by_id:Dict[str, Stop] = field(default_factory=dict)
    trips_by_id:Dict[str, Trip] = field(default_factory=dict)
    routes_by_id:Dict[str, Route] = field(default_factory=dict)
    active_service_ids:Set[str] = field(default_factory = set) # Service ids on specific date

