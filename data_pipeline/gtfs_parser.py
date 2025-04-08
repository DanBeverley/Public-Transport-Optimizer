"""
Handles parsing of static GTFS zip files and real-time GTFS-RT protocol buffer feeds.

This module utilizes gtfs-kit for efficient static GTFS processing and
httpx with google-transit-realtime-bindings for asynchronous GTFS-RT fetching and parsing.
"""
import logging
from pathlib import Path
from typing import Optional, Union

import gtfs_kit as gk
import httpx
from google.protobuf.message import DecodeError
from google.transit import gtfs_realtime_pb2

logging.basicConfig(level=logging.INFO, format="%(astime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# A default HTTP timeout
DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect = 5.0)

def load_static_gtfs(gtfs_source:Union[str, Path]) -> Optional[gk.Feed]:
    """
    Loads a static GTFS feed from a zip file or directory path.

    Uses gtfs-kit for parsing and initial validation.

    Args:
        gtfs_source: Path to the GTFS zip file or directory.

    Returns:
        A gtfs_kit.Feed object representing the parsed GTFS data, or None if loading fails.
    """
    try:
        logger.info(f"Attempting to load static GTFS from: {gtfs_source}")
        feed = gk.read_feed(gtfs_source, dist_units = "km")
        logger.info(f"Loaded static GTFS feed")
        return feed
    except FileNotFoundError:
        logger.error(f"GTFS source not found at: {gtfs_source}")
        return None
    except Exception as e:
        logger.error(f"Failed to load static GTFS from {gtfs_source}: {e}", exc_info=True)
        return None

async def process_realtime_feed(feed_url:str, session:httpx.AsyncClient,
                                 timeout:httpx.Timeout = DEFAULT_TIMEOUT)->Optional[gtfs_realtime_pb2.FeedMessage]:
    """
    Fetches and parses a GTFS-Realtime feed asynchronously.

    Args:
        feed_url: The URL of the GTFS-Realtime feed (protobuf format).
        session: An httpx.AsyncClient session for making the request.
        timeout: Request timeout configuration.

    Returns:
        A parsed gtfs_realtime_pb2.FeedMessage object, or None if fetching/parsing fails.
    """
    logger.debug(f"Fetching GTFS-RT feed from {feed_url}")
    headers = {"User-Agent":"AI-PublicTransportOptimizer/1.0"}
    try:
        response = await session.get(feed_url, timeout=timeout, headers=headers)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        content = await response.aread()
        if not content:
            logger.warning(f"Received empty response from GTFS-RT feed: {feed_url}")
            return None
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(content)
        logger.debug(f"Successfully parsed GTFS-RT feed from: {feed_url}")
        return feed
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching GTFS-RT feed {feed_url}: {e.response.status_code} - {e.response.text}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Network error fetching GTFS-RT feed {feed_url}: {e}")
        return None
    except DecodeError as e:
        logger.error(f"Error parsing protobuf for GTFS-RT feed {feed_url}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occured while processing GTFS-RT feed {feed_url}: {e}]", exc_info = True)
        return None
    