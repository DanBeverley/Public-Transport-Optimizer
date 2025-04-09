"""
Common utility functions for the data pipeline.

Includes helpers for configuration, timing, asynchronous operations, and data conversion.
"""
import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable, Coroutine, Optional, TypeVar, ParamSpec
from datetime import datetime, timezone, timedelta
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

def get_env_var(var_name:str, default:Optional[str] = None) -> Optional[str]:
    """Retrieves an enviroment variable"""
    value = os.getenv(var_name, default)
    if value is None:
        logger.warning(f"Enviroment variable '{var_name}' not set")
    return value

async def async_retry(retries:int = 3, delay_seconds:float=1.0, backoff_factor:float=2.0,
                      exceptions: tuple[type[Exception], ...] = (Exception, ),
                      ) -> Callable[[Callable[P, Coroutine[Any, Any, T]]],
                                     Callable[P, Coroutine[Any, Any, Optional[T]]]]:
    """
    Decorator for automatically retrying an async function if it raises specific exceptions.

    Args:
        retries: Maximum number of retries.
        delay_seconds: Initial delay between retries.
        backoff_factor: Multiplier for delay increase (exponential backoff).
        exceptions: Tuple of exception types to catch and retry on.

    Returns:
        A decorator function.
    """
    def decorator(func:Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, Coroutine[Any, Any, Optional[T]]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Optional[T]:
            current_delay = delay_seconds
            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries:
                        logger.error(f"Function '{func.__name__}' failed after {retries + 1} attempts. Error: {e}",
                                     exc_info = True)
                        return None
                    logger.warning(f"Attempt {attempt + 1}/{retries + 1} failed for '{func.__name__}'. Error: {e}"
                                   f"Retrying in {current_delay:.2f} seconds...")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor
            return None
        return wrapper
    return decorator
              
def get_current_utc_datetime() -> datetime:
    """return current datetime in UTC with timezone info"""
    return datetime.now(timezone.utc)

def is_timestamp_recent(timestamp: Optional[int],
                        max_staleness_seconds: int = 300) -> bool:
    """Check if a given Unix timestamp is recent compared to now"""
    if timestamp is None:
        return False
    now_ts = int(get_current_utc_datetime().timestamp())
    return (now_ts - timestamp) <= max_staleness_seconds

def parse_flexible_timestamp(ts_data:Any) -> Optional[datetime]:
    """Attempts to parse various timestamp formats into a timezone-aware UTC datetime"""
    if isinstance(ts_data, datetime):
        if ts_data.tzinfo is None:
            return ts_data.replace(tzinfo = timezone.utc) 
        return ts_data.astimezone(timezone.utc)
    if isinstance(ts_data, (int, float)):
        try:
            return datetime.fromtimestamp(ts_data, tz=timezone.utc)
        except (ValueError, OSError):
            logger.debug(f"Could not parse numeric timestamp: {ts_data}")
            return None
    if isinstance(ts_data, str):
        try:
            # Attempt ISO format
            dt = datetime.fromisoformat(ts_data.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            # TODO: Add more common formats
            logger.debug(f"Could not parse string timestamp: {ts_data}")
            return None
    return None

def safe_float(value:Any, default:Optional[float] = None) -> Optional[float]:
    """Safely conerts a value to a float, returning default on failure"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value:Any, default:Optional[int] = None) -> Optional[int]:
    """Safely converts a value to int, returning default on failure"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

# For performnace mesurement
def time_it(func:Callable[P, T]) -> Callable[P, T]:
    """Simple decorator to measure and log the execution time of a synchronous function"""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.debug(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

async def time_it_async(func:Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, Coroutine[Any, Any, T]]:
    """Simple decorator to measure and log the execution time of an asynchronous function"""
    async def wrapper(*args:P.args, **kwargs:P.kwargs) -> T:
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.debug(f"Async function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


# Constant
DEFAULT_CACHE_TTL_SECONDS = 60 * 5 # 5 minutes
DEFAULT_GTFS_RT_STALENESS_SECONDS = 60 * 3 # 3 minutes
MIN_LATITUDE, MAX_LATITUDE = -90.0, 90.0
MIN_LONGITUDE, MAX_LONGITUDE = -180.0, 180.0
# TODO: Move to a seperate config later 