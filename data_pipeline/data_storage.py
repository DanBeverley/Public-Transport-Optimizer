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
                    get_current_utc_datetime, DEFAULT_CACHE_TTL_SECONDS, safe_float,
                    safe_int)

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

async def _create_gtfs_indexes(schema: str):
    """Creates recommended indexes on the GTFS tables for faster queries."""
    pool = await get_postgres_pool()
    if not pool:
        logger.error("Cannot create GTFS indexes: PostgreSQL pool not available.")
        return

    logger.info(f"Creating indexes for GTFS schema '{schema}'...")
    # Define indexes: (table, column or columns tuple)
    indexes_to_create = [
        ("stops", "stop_id"),
        ("routes", "route_id"),
        ("trips", "trip_id"),
        ("trips", "route_id"),
        ("trips", "service_id"),
        ("stop_times", "trip_id"),
        ("stop_times", "stop_id"),
        ("stop_times", ("arrival_time", "departure_time")), # Index arrival/departure
        ("calendar", "service_id"),
        ("calendar_dates", "service_id"),
        ("calendar_dates", "date"),
        # Add more indexes as needed based on query patterns
    ]

    async with pool.acquire() as conn:
        async with conn.transaction(): # Run index creation in a transaction
            for table, columns in indexes_to_create:
                # Ensure schema and table/column names are properly quoted
                safe_schema = asyncpg.utils.quote_ident(schema)
                safe_table = asyncpg.utils.quote_ident(table)
                
                if isinstance(columns, tuple):
                    # Multi-column index
                    column_names = ", ".join(asyncpg.utils.quote_ident(col) for col in columns)
                    index_name = f"idx_{table}_{'_'.join(columns)}"
                else:
                    # Single-column index
                    column_names = asyncpg.utils.quote_ident(columns)
                    index_name = f"idx_{table}_{columns}"
                
                safe_index_name = asyncpg.utils.quote_ident(index_name)
                
                sql = f"CREATE INDEX IF NOT EXISTS {safe_index_name} ON {safe_schema}.{safe_table} ({column_names})"
                try:
                    logger.debug(f"Executing: {sql}")
                    await conn.execute(sql)
                except asyncpg.PostgresError as e:
                    logger.warning(f"Failed to create index {safe_index_name} on {safe_schema}.{safe_table}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error creating index {safe_index_name}: {e}", exc_info=True)
                    # Decide if we should raise or just log and continue

    # Optional: Add spatial index for stops.geometry if using PostGIS
    try:
        async with pool.acquire() as conn:
            safe_schema = asyncpg.utils.quote_ident(schema)
            # Check if the geometry column exists before trying to index it
            geom_exists = await conn.fetchval(f"""
                SELECT EXISTS (
                    SELECT 1 
                    FROM information_schema.columns 
                    WHERE table_schema = $1 AND table_name = 'stops' AND column_name = 'geometry'
                );
            """, schema)

            if geom_exists:
                logger.info(f"Creating spatial index on stops.geometry in schema '{schema}'...")
                sql_spatial = f"CREATE INDEX IF NOT EXISTS {asyncpg.utils.quote_ident(f'idx_{schema}_stops_geom')} ON {safe_schema}.stops USING GIST (geometry);"
                await conn.execute(sql_spatial)
                logger.info("Spatial index created (or already existed).")
            else:
                 logger.debug("Stops geometry column not found, skipping spatial index.")
                 
    except asyncpg.PostgresError as e:
        logger.warning(f"Failed to create spatial index on {schema}.stops: {e}. Is PostGIS enabled?")
    except Exception as e:
        logger.error(f"Unexpected error creating spatial index: {e}", exc_info=True)

    logger.info(f"Finished creating indexes for GTFS schema '{schema}'.")

# --- Real-time Data Storage (Redis) ---
# ... (Keep existing code below this line) ...
