"""
Data models specific to the journey optimization process.

Includes structures for optimization criteria and scored journey results.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from enum import Enum
from routing_engine.models import Journey, Location, Stop 


class DisruptionType(Enum):
    """Represents the type of service disruption."""
    CANCELLED = "CANCELLED"
    DELAYED = "DELAYED"
    DETOUR = "DETOUR"
    STOP_CLOSURE = "STOP_CLOSURE"
    UNKNOWN = "UNKNOWN"

class SeverityLevel(Enum):
    """Represents the severity level of a service disruption."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"

@dataclass(frozen=True)
class OptimizationCriteria:
    """Defines user preferences for journey optimization."""
    # Weights should ideally sum to 1 or be normalized
    weight_duration: float = 0.6  
    weight_reliability: float = 0.3 
    weight_transfers: float = 0.1
    def __post_init__(self):
        if not (0 <= self.weight_duration <= 1 and
                0 <= self.weight_reliability <= 1 and
                0 <= self.weight_transfers <=1):
            raise ValueError("Criteria weights must be between 0 and 1")
        # Check for sum close to 1 with tolerance
        total = self.weight_duration + self.weight_reliability + self.weight_transfers
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Criteria weights must sum to 1 (got {total})")

@dataclass
class ScoredJourney:
    """Wraps a Journey object with its calculated optimization scores."""
    journey: Journey
    predicted_arrival_time: datetime
    predicted_duration: timedelta
    # Reliability score (e.g., product of connection probabilities, 0.0 to 1.0)
    reliability_score: float = 1.0
    optimization_score:float = float("inf")
    # Make ScoredJourney sortable by optimization_score
    def __lt__(self, other:"ScoredJourney") -> bool:
        return self.optimization_score < other.optimization_score

@dataclass
class DisruptionInfo:
    """Represents information about a service disruption"""
    alert_id: Optional[str] = None 
    disruption_type: DisruptionType = DisruptionType.UNKNOWN
    affected_trips: List[str] = field(default_factory=list)
    affected_stops: List[str] = field(default_factory=list)
    affected_routes: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    message: Optional[str] = None
    severity: SeverityLevel = SeverityLevel.UNKNOWN