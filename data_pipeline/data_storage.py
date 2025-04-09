"""
Handles asynchronous storage and retrieval of processed data.

Uses PostgreSQL (with PostGIS) for persistent storage of static GTFS,
historical data, and potentially structured real-time data snapshots.
Uses Redis for caching and storing frequently updated real-time information
like vehicle positions and current trip delays.
"""
import asyncio
import logging
import json
from typing import Any, Dict, List, Optional, Union, Sequence

import asyncpg # Direct async PostgreSQL driver
import redis.asyncio as redis # Async Redis client
import gtfs_kit as gk
import pandas as pd
from google.transit import gtfs_realtime_pb2

from .utils import (get_env_var, async_retry, time_it_async, parse_flexible_timestamp,
                    get_current_utc_datetime, DEFAULT_CACHE_TTL_SECONDS,
                    DEFAULT_GTFS_RT_STALENESS_SECONDS, safe_float, safe_int)

logger = logging.getLogger(__name__)

POSTGRES_DNS = get_env_var("POSTGRES_DSN", "postgresql://user:password@host:port/database")
REDIS_URL = get_env_var("REDIS_URL", "redis://localhost:6379/0")

# Connection pools for managing connections efficiently
_postgres_pool: Optional[asyncpg.Pool] = None
_redis_pool: Optional[redis.Redis] = None # redis-py's ConnectionPool handled internally

async def get_postgres_pool() -> asyncpg.Pool:
    """Initializes and returns the asyncpg connection pool"""
    global _postgres_pool
    if _postgres_pool is None:
        logger.info(f"Initializing PostgreSQL connection pool for DNS: {'...' + POSTGRES_DNS[-20:] if POSTGRES_DNS else 'None'}")
        if not POSTGRES_DNS:
            raise ValueError("POSTGRES_DNS enviroment variable not set")
        try:
            _postgres_pool = await asyncpg.create_pool(dns = POSTGRES_DNS,
                                                       min_size = 2, max_size = 10,
                                                       command_timeout = 60,
                                                       # TODO: also set up JSON and geometry types handling
                                                       init = _init_progress_connection)
            logger.info("PostgreSQL connection pool initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection pool: {e}", exc_info = True)
            raise
    return _postgres_pool

async def _init_postgres_connection(conn):
    """Initialize connection state, e.g, setting up type codecs"""
    # NOTE: Ensure the JSON is handled correctly
    await conn.set_type_codec('json', encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
    # TODO: PostGIS setup
    logger.debug(f"Initialized new PostgreSQL connection: {conn}")

async def get_redis_client() -> redis.Redis:
    """Initializes and returns the async Redis client"""
    global _redis_pool
    if _redis_pool is None:
        logger.info(f"Initializing Redis client for URL: {REDIS_URL}")
        if not REDIS_URL:
            raise ValueError("REDIS_URL enviroment variable not set")
        try:
            _redis_pool = redis.from_url(REDIS_URL, decode_responses=True,
                                         max_connections = 20)
            # Test connection
            await _redis_pool.ping()
            logger.info("Redis client initialized and connection verified")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}", exc_info = True)
            raise
    return _redis_pool

async def close_database_connections():
    """Close database connection pools"""
    global _postgres_pool, _redis_pool
    if _postgres_pool:
        logger.info("Closing PostgreSQL connection pool...")
        await _postgres_pool.close()
        _postgres_pool = None
        logger.info("PostgreSQL connection pool closed")
    if _redis_pool:
        logger.info("Closing Redis client connections...")
        await _redis_pool.close() # Close the underlying pool
        _redis_pool = None
        logger.into("Redis client connections failed")


# Static GTFS storage (PostgreSQL / PostGIS)
@time_it_async
@async_retry(retries=2, delay_seconds=2.0, exceptions=(asyncpg.PostgresError, OSError))
async def store_static_gtfs(feed:gk.Feed, schema: Optional[str] = 'gtfs'):
    """
    Stores a validated static GTFS feed into PostgreSQL using gtfs-kit's method.

    This leverages gtfs-kit's built-in capability, which likely uses SQLAlchemy
    under the hood. Connect using the asyncpg DSN.

    NOTE: gtfs-kit's `to_sql` or `to_postgis` might not be inherently async.
          This function wraps the potentially blocking call within an asyncio
          executor thread to avoid blocking the main event loop.

    Args:
        feed: The validated gtfs_kit.Feed object.
        schema: The PostgreSQL schema name to store tables in (default: 'gtfs').
    """
    if not POSTGRES_DNS:
        logger.error(f"Cannot store static GTFS: POSTGRES_DNS not configured")
        return False
    logger.info(f"Storing static GTFS feed to PostgreSQL schema '{schema}'...")
    loop = asyncio.get_running_loop()
    try:
        # Use to_postgis if spatial features are important and PostGIS is installed
        # Requires GeoDataFrames (geopandas), which gtfs-kit can create
        # Check if stops were converted to GeoDataFrame and geometry column exists
        if hasattr(feed, 'stops') and isinstance(feed.stops, pd.DataFrame) and 'geometry' in feed.stops.columns and not feed.stops['geometry'].isnull().all():
            logger.info("Stops have geometry data. Attempting to use to_postgis (requires PostGIS extension & GeoPandas).")
            # Ensure GeoPandas is available if attempting to use to_postgis
            try:
                import geopandas
                logger.debug("GeoPandas imported successfully.")
            except ImportError:
                logger.error("GeoPandas is required for to_postgis but not installed. Falling back to to_sql.")
                # Fallback to standard SQL storage
                await loop.run_in_executor(
                    None,
                    lambda: feed.to_sql(POSTGRES_DNS, schema=schema)
                )
            else:
                # GeoPandas is available, proceed with to_postgis
                await loop.run_in_executor(
                    None,  # Use default executor (ThreadPoolExecutor)
                    lambda: feed.to_postgis(POSTGRES_DNS, schema=schema)
                )
        else:
            logger.info("No valid geometry data found in stops or GeoPandas not used/available. Using standard to_sql.")
            # Fallback or primary method if not using spatial features directly in DB
            await loop.run_in_executor(
                None,
                lambda: feed.to_sql(POSTGRES_DNS, schema=schema)
            )

        logger.info(f"Successfully stored static GTFS feed tables to schema '{schema}'.")

        # Create indexes after tables are created
        await _create_gtfs_indexes(schema)

        return True

    except (RuntimeError, ValueError, ImportError, AttributeError) as e: # Catch errors from run_in_executor/gtfs-kit
        logger.error(f"Error during GTFS storage (potentially within gtfs-kit sync function): {e}", exc_info=True)
        return False
    except asyncpg.PostgresError as e:
        logger.error(f"Database error during GTFS index creation in schema '{schema}': {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error storing static GTFS feed to schema '{schema}': {e}", exc_info=True)
        return False

async def _create_gtfs_indexes(schema:str):
    """Creates potentially useful indexes on GTFS tables after loading"""
    pool = await get_postgres_pool()
    index_commands = [
        f"CREATE INDEX IF NOT EXISTS idx_{schema}_stops_stop_id ON {schema}.stops(stop_id);",
        f"CREATE INDEX IF NOT EXISTS idx_{schema}_trips_trip_id ON {schema}.trips(trip_id);",
        f"CREATE INDEX IF NOT EXISTS idx_{schema}_trips_route_id ON {schema}.trips(route_id);",
        f"CREATE INDEX IF NOT EXISTS idx_{schema}_stop_times_trip_id ON {schema}.stop_times(trip_id);",
        f"CREATE INDEX IF NOT EXISTS idx_{schema}_stop_times_stop_id ON {schema}.stop_times(stop_id);",
        f"CREATE INDEX IF NOT EXISTS idx_{schema}_stop_times_arrival_time ON {schema}.stop_times(arrival_time);",
        f"CREATE INDEX IF NOT EXISTS idx_{schema}_stop_times_departure_time ON {schema}.stop_times(departure_time);",
        # Add spatial index if using PostGIS and gtfs-kit didn't create one
        f"CREATE INDEX IF NOT EXISTS idx_{schema}_stops_geom ON {schema}.stops USING GIST (geometry);"
    ]
    async with pool.acquire() as conn:
        async with conn.transaction():
            logger.info(f"Attempting to create indexes in schema '{schema}'...")
            for cmd in index_commands:
                try:
                    await conn.execute(cmd)
                except asyncpg.UndefinedTableError:
                    logger.warning(f"Could not create index: Table for command '{cmd}' likely doesn't exist. Skipping.")
                    # Break if tables don't exist for this schema
                    break
                except asyncpg.DuplicateTableError: # Thrown if index already exists sometimes
                     logger.debug(f"Index for command '{cmd}' likely already exists. Skipping.")
                except Exception as e:
                    # Log other errors but continue trying other indexes
                    logger.warning(f"Failed to execute index command: {cmd}. Error: {e}")
            logger.info(f"Index creation process completed for schema '{schema}'.")

@time_it_async
@async_retry(retries=1, delay_seconds=0.5, exceptions = (redis.RedisError,))
async def update_vehicle_position(vehicle_update:gtfs_realtime_pb2.VehiclePosition):
    """
    Stores the latest vehicle position information in Redis.

    Uses a Redis Hash to store all attributes of a vehicle's position.
    Key: "vehicle:{vehicle_id}" or "vehicle:trip:{trip_id}"

    Args:
        vehicle_update: A VehiclePosition protobuf message.
    """
    redis_client = await get_redis_client()
    # Determine primary key: prefer vehicle_id if available
    vehicle_id = vehicle_update.vehicle.id if vehicle_update.vehicle.HasField("id") else None
    trip_id = vehicle_update.trip.trip_id if vehicle_update.trip.hasField("trip_id") else None
    route_id = vehicle_update.trip.route_id if vehicle_update.trip.hasField("route_id") else None

    if not vehicle_id and not trip_id:
        logger.warning("VehiclePostition update missing both vehicle_id and trip_id. Cannot store")
        return
    
    # Use vehicle_id if present, otherwise fallback to trip_id 
    # TODO: Find a better fallback
    redis_key = f"vehicle:{vehicle_id}" if vehicle_id else f"vehicle:trip:{trip_id}"
    # Use a short TTL as this data is highly volatile (e.g., 5-10 minutes)

    ttl_seconds = DEFAULT_GTFS_RT_STALENESS_SECONDS * 2

    position_data = {"latitude":safe_float(vehicle_update.position.latitude),
                     "longitude":safe_float(vehicle_update.position.longitude),
                     "bearing":safe_float(vehicle_update.position.bearing),
                     "speed":safe_float(vehicle_update.position.speed), # m/s
                     "timestamp":safe_int(vehicle_update.timestamp),
                     "trip_id":trip_id,
                     "route_id":route_id,
                     "vehicle_id":vehicle_id, # Store even if used in key
                     "stop_id":vehicle_update.stop_id if vehicle_update.HasField("stop_id") else None,
                     "current_status":gtfs_realtime_pb2.VehicleStopStatus.Name(vehicle_update.current_status),
                     "congestion_level":gtfs_realtime_pb2.CongestionLevel.Name(vehicle_update.congestion_level),
                     "occupancy_status":gtfs_realtime_pb2.OccupancyStatus.Name(vehicle_update.occupancy_status),
                     "last_updated":get_current_utc_datetime().isformat()}
    # Filter out None values before storing
    position_data_filtered = {k:v for k, v in position_data.items() if v is not None}
    try:
        # Use HSET to store multiple fields at once
        await redis_client.hset(redis_key, mapping = position_data_filtered)
        await redis_client.expire(redis_key, ttl_seconds)
        logger.debug(f"Stored vehicle position for key: {redis_key}")
    except redis.RedisError as e:
        logger.error(f"Redis error storing vehicle position for {redis_key}: {e}")
        raise # Re-raise for retry decorator

@time_it_async
@async_retry(retries=1, delay_seconds = 0.5, exceptions = (redis.RedisError,))
async def store_trip_update(trip_update: gtfs_realtime_pb2.TripUpdate):
    """
    Stores the latest trip update information (delays, schedule changes) in Redis.

    Stores data per trip. Could store all stop time updates as JSON or break down further.
    Key: "trip_update:{trip_id}"

    Args:
        trip_update: A TripUpdate protobuf message.
    """
    redis_client = await get_redis_client()
    trip_id = trip_update.trip.trip_id
    route_id = trip_update.trip.route_id

    if not trip_id:
        logger.warning("TripUpdate message missing trip_id. Cannot store")
        return
    redis_key = f"trip_update: {trip_id}"
    ttl_seconds = 60*30

    update_details = {"trip_id":trip_id,
                      "route_id":route_id,
                      "vehicle_id":trip_id.vehicle.id if trip_update.HasField("vehicle") else None,
                      "start_date":trip_update.trip.start_date if trip_update.trip.HasField("start_date") else None,
                      "schedule_relationship":gtfs_realtime_pb2.TripDescriptor.ScheduleRelationship.Name(trip_update.trip.schedule_relationship),
                      "timestamp":safe_int(trip_update.timestamp),
                      "delay":safe_int(trip_update.delay),
                      "last_updated": get_current_utc_datetime().isoformat(),
                      "stop_time_updates":[]}
    for stu in trip_update.stop_time_update:
        stop_update = {
            "stop_sequence": stu.stop_sequence if stu.HasField("stop_sequence") else None,
            "stop_id": stu.stop_id if stu.HasField("stop_id") else None,
            "arrival_delay": stu.arrival.delay if stu.HasField("arrival") and stu.arrival.HasField("delay") else None,
            "arrival_time": stu.arrival.time if stu.HasField("arrival") and stu.arrival.HasField("time") else None,
            "departure_delay": stu.departure.delay if stu.HasField("departure") and stu.departure.HasField("delay") else None,
            "departure_time": stu.departure.time if stu.HasField("departure") and stu.departure.HasField("time") else None,
            "schedule_relationship": gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.ScheduleRelationship.Name(stu.schedule_relationship),
            #TODO: Add uncertainty if needed: stu.arrival.uncertainty / stu.departure.uncertainty
        }
        # Filter out None values within the stop update
        update_details["stop_time_updates"].append({k: v for k, v in stop_update.items() if v is not None})

    try:
        # Store the entire structure as a JSON string in a single Redis key
        await redis_client.set(redis_key, json.dumps(update_details), ex=ttl_seconds)
        logger.debug(f"Stored trip update for trip: {trip_id}")
    except redis.RedisError as e:
        logger.error(f"Redis error storing trip update for {trip_id}: {e}")
        raise # Re-raise for retry

@time_it_async
@async_retry(retries = 1, delay_seconds = 0.5, exceptions=(redis.RedisError,))
async def store_alert(alert:gtfs_realtime_pb2.Alert):
    """
    Stores service alert information in Redis.

    Alerts are less frequent but important. Store them indexed by alert ID
    and potentially also by affected entity (route, stop).

    Args:
        alert: An Alert protobuf message.
    """
    redis_client = await get_redis_client()
    alert_id = f"alert: {hash(alert.SerializeToString())}"
    ttl_seconds = 60*60*24
    
    alert_data = {
        "cause": gtfs_realtime_pb2.Alert.Cause.Name(alert.cause),
        "effect": gtfs_realtime_pb2.Alert.Effect.Name(alert.effect),
        "url": alert.url.translation[0].text if alert.HasField("url") and alert.url.translation else None,
        "header_text": alert.header_text.translation[0].text if alert.HasField("header_text") and alert.header_text.translation else None,
        "description_text": alert.description_text.translation[0].text if alert.HasField("description_text") and alert.description_text.translation else None,
        "active_periods": [],
        "last_updated": get_current_utc_datetime().isoformat(),
    }

    max_end_time = 0
    for period in alert.active_period:
        start = safe_int(period.start)
        end = safe_int(period.end)
        alert_data["active_periods"].append({"start": start, "end": end})
        if end and end > max_end_time:
            max_end_time = end

    if max_end_time > 0:
        now_ts = int(get_current_utc_datetime().timestamp())
        effective_ttl = max(60, max_end_time - now_ts) # At least 1 minute
        ttl_seconds = min(ttl_seconds, effective_ttl) # Don't exceed default if end time is very far

    try:
        pipeline = redis_client.pipeline()
        # Store the main alert data
        pipeline.set(alert_id, json.dumps(alert_data), ex=ttl_seconds)

        # Link affected entities
        processed_entities = set()
        for entity_selector in alert.informed_entity:
            entity_key: Optional[str] = None
            if entity_selector.HasField("route_id"):
                entity_key = f"route:{entity_selector.route_id}:alerts"
            elif entity_selector.HasField("stop_id"):
                 entity_key = f"stop:{entity_selector.stop_id}:alerts"
            elif entity_selector.HasField("trip") and entity_selector.trip.HasField("trip_id"):
                 entity_key = f"trip:{entity_selector.trip.trip_id}:alerts"
            # Add other selectors (agency_id, route_type) if needed

            if entity_key and entity_key not in processed_entities:
                 # Use a Redis Set to store the IDs of alerts affecting this entity
                 pipeline.sadd(entity_key, alert_id)
                 # Set TTL on the entity link set as well, maybe slightly longer than alert
                 pipeline.expire(entity_key, ttl_seconds + 60)
                 processed_entities.add(entity_key)

        await pipeline.execute()
        logger.debug(f"Stored alert {alert_id} affecting {len(processed_entities)} entity types.")

        # TODO: adding alerts to a persistent log in PostgreSQL as well,
        # as Redis data is volatile.

    except redis.RedisError as e:
        logger.error(f"Redis error storing alert {alert_id}: {e}")
        raise

time_it_async
@async_retry(retries=1, delay_seconds=0.2, exceptions=(redis.RedisError,))
async def cache_data(key: str, value: Any, ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS):
    """
    Caches arbitrary data (serializable to JSON) in Redis.

    Args:
        key: The cache key.
        value: The data to cache (must be JSON serializable).
        ttl_seconds: Cache expiration time in seconds.
    """
    if value is None: # Don't cache None values explicitly unless intended
        logger.debug(f"Attempted to cache None value for key '{key}'. Skipping.")
        return False
    redis_client = await get_redis_client()
    try:
        # Serialize complex data structures to JSON
        serialized_value = json.dumps(value)
        await redis_client.set(key, serialized_value, ex=ttl_seconds)
        logger.debug(f"Cached data for key: {key} with TTL: {ttl_seconds}s")
        return True
    except TypeError as e:
        logger.error(f"Failed to serialize data for caching key '{key}': {e}. Ensure value is JSON serializable.")
        return False
    except redis.RedisError as e:
        logger.error(f"Redis error caching data for key '{key}': {e}")
        raise # Re-raise for retry

@time_it_async
@async_retry(retries=1, delay_seconds=0.2, exceptions=(redis.RedisError,))
async def get_cached_data(key: str) -> Optional[Any]:
    """
    Retrieves cached data from Redis.

    Args:
        key: The cache key.

    Returns:
        The deserialized data, or None if key not found, expired, or error occurred.
    """
    redis_client = await get_redis_client()
    try:
        cached_value = await redis_client.get(key)
        if cached_value:
            logger.debug(f"Cache HIT for key: {key}")
            try:
                # Deserialize from JSON
                return json.loads(cached_value)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to deserialize cached JSON data for key '{key}': {e}")
                # Optionally delete the corrupted key: await redis_client.delete(key)
                return None
        else:
            logger.debug(f"Cache MISS for key: {key}")
            return None
    except redis.RedisError as e:
        logger.error(f"Redis error retrieving cache for key '{key}': {e}")
        # Don't raise here, just return None on cache retrieval failure
        return None
