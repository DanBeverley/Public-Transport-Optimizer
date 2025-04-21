"""
Asynchronous API clients for fetching external data like traffic, weather, and events.

Uses httpx for efficient async requests. Requires API keys to be configured
(e.g., via environment variables or a config file).
"""
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import httpx
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
DEFAULT_RETRY_STATUS = {500, 502, 503, 504}

TRAFFIC_API_KEY = os.getenv("TRAFFIC_API_KEY") # e.g., Google Maps, HERE, TomTom
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY") # e.g., OpenWeatherMap
EVENTS_API_KEY = os.getenv("EVENTS_API_KEY")   # e.g., Ticketmaster, Eventbrite (if needed)
# Replace with actual API endpoints
TRAFFIC_API_BASE_URL = "https://api.example-traffic.com/v1/"
WEATHER_API_BASE_URL = "https://api.openweathermap.org/data/2.5/"
EVENTS_API_BASE_URL = "https://api.example-events.com/discovery/v2/"

async def _make_api_request(session:httpx.AsyncClient, method:str, url:str,
                            params:Optional[Dict[str, Any]] = None,
                            headers:Optional[Dict[str, str]] = None,
                            max_retries:int = 3,
                            retry_delay:float = 1.0,
                            timeout:httpx.Timeout = DEFAULT_TIMEOUT) -> Optional[Dict[str, Any]]:
    retries = 0
    while retries <= max_retries:
        try:
            response = await session.request(method, url, params = params, headers = headers,
                                             timeout = timeout)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP Error {e.response.status_code} for {e.request.url}. Response: {e.response.text}")
            if e.response.status_code in DEFAULT_RETRY_STATUS and retries < max_retries:
                retries += 1
                logger.info(f"Retrying request ({retries} / {max_retries})...")
                await asyncio.sleep(retry_delay * (2**retries))
            else:
                logger.error(f"Non-retryable HTTP status error or max retries reached for {url}")
                return None  # Give up after retries or for non-retryable errors (like 4xx)
        except httpx.RequestError as e:
            logger.error(f"Network error requesting {e.request.url}: {e}")
            # Network errors are often retryable
            if retries < max_retries:
                retries += 1
                logger.info(f"Retrying request ({retries}/{max_retries})...")
                await asyncio.sleep(retry_delay * (2**retries))
            else:
                logger.error(f"Max retries reached for network error at {url}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error during API request to {url}:{e}", exc_info = True)
            return None
    return None

async def get_traffic_conditions(api_key:Optional[str],
                                 area_bounds:Tuple[float, float, float, float], #(min_lat, min_lon, max_lat, max_lon)
                                 session:httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    """
    Fetches real-time traffic conditions for a given bounding box.

    NOTE: Adapt the URL, params, and response parsing based on your chosen Traffic API provider
          (e.g., Google Maps Directions/Matrix API, HERE Traffic API, TomTom Traffic API).

    Args:
        api_key: The API key for the traffic service.
        area_bounds: A tuple representing the bounding box (min_lat, min_lon, max_lat, max_lon).
        session: An httpx.AsyncClient session.

    Returns:
        A dictionary containing traffic data, or None if fetching fails.
    """
    if not api_key: 
        logger.warning("Traffic API key not provided. Skipping traffic fetch")
        return None
    # TODO: Adapt based on the actual API documentation
    request_url = f"{TRAFFIC_API_BASE_URL}flow"
    params = {"apiKey":api_key,
              "bbox":f"{area_bounds[0]}, {area_bounds[1]}, {area_bounds[2]}, {area_bounds[3]}",
              "units":"metric",
              # TODO: add more parameters (time, response fields)
              }
    headers = {"Accept":"application/json"}
    logger.info(f"Fetching traffic conditions for bounds: {area_bounds}")
    traffic_data = await _make_api_request(session, "GET", request_url, params = params,
                                           headers = headers)
    if traffic_data:
        logger.debug("Successfully fetched traffic data.")
        # TODO: add post-processing or validation of the response structure
        return traffic_data
    else:
        logger.error("Failed to fetch traffic conditions")
        return None

async def get_weather_forecast(api_key:Optional[str],
                               location:Tuple[float, float], # latitude, longtidue
                               session: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    """
    Fetches current weather and potentially forecast data for a specific location.

    Example uses OpenWeatherMap 'weather' endpoint. Adapt as needed for forecasts.

    Args:
        api_key: The API key for the weather service (e.g., OpenWeatherMap).
        location: A tuple representing the location (latitude, longitude).
        session: An httpx.AsyncClient session.

    Returns:
        A dictionary containing weather data, or None if fetching fails.
    """
    if not api_key:
        logger.warning("Weather API key not provided. Skipping weather fetch")
        return None
    lat, lon = location
    request_url = f"{WEATHER_API_BASE_URL}weather"
    # request_uri = f"{WEATHER_API_BASE_URL}forecast"" for forecast
    params = {"lat":lat,
              "lon":lon,
              "appid":api_key,
              "units":"metric" # Celsius, m/s
              }
    headers = {"Accept": "application/json"}
    logger.info(f"Fetching weather forecast for location: {location}")
    weather_data = await _make_api_request(session, "GET", request_url, params=params, headers=headers)
    if weather_data:
        logger.debug(f"Fetched weather data")
        return weather_data
    if weather_data and isinstance(weather_data, dict) and "main" in weather_data and "weather" in weather_data:
        try:
            extracted_data = {
                "temp_celsius":weather_data.get("main", {}).get("temp"),
                "condition":weather_data.get("weather", [{}])[0].get("description"),
                "wind_speed_mps":weather_data.get("wind", {}).get("speed"),
                "precipitation_1h_mm":weather_data.get("rain",{}).get("1h",0) + weather_data.get("snow", {}).get("1h", 0)
            }
            #TODO: Further validation on type/values
            logger.debug(f"Extracted weather: {extracted_data}")
            return extracted_data
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error extracting weather data fields: {e} from {weather_data}")
            return None
    else:
        logger.error("Failed to fetch weather forecast")
        return None

async def get_local_events(api_key:Optional[str],
                           area_bounds:Tuple[float, float, float, float],
                           time_window:Tuple[datetime, datetime],
                           session:httpx.AsyncClient,
                           max_events:int=50) -> Optional[List[Dict[str, Any]]]:
    """
    Fetches major public events within a given area and time window.

    NOTE: Adapt based on your chosen Events API provider (e.g., Ticketmaster, Eventbrite) or city-specific data source.
          Scraping city websites is another option but more complex and brittle.

    Args:
        api_key: The API key for the events service (if required).
        area_bounds: Bounding box (min_lat, min_lon, max_lat, max_lon).
        time_window: Tuple of (start_datetime, end_datetime).
        session: An httpx.AsyncClient session.
        max_events: Maximum number of events to fetch.

    Returns:
        A list of dictionaries, each representing an event, or None if fetching fails.
    """
    # Adapt base on actual API documentation
    request_url = f"{EVENTS_API_BASE_URL}events"
    start_iso, end_iso = time_window[0].isoformat(timespec='seconds') + 'Z', time_window[1].isoformat(timespec='seconds') + 'Z'
    params = {"apikey":api_key,
              "latlong":f"{(area_bounds[0] + area_bounds[2]) / 2},
                          {(area_bounds[1] + area_bounds[3]) / 2}",
              "radius":"50",
              "unit":"km",
              "startDatetime":start_iso,
              "endDatetime":end_iso,
              "sort":"date, asc",
              "size":max_events,
              # TOD: add parameters for filtering event types (sports, concerts)
              }
    headers = {"Accept":"application/json"}
    logger.info(f"Fetching local events for bounds: {area_bounds}, time:{time_window}")
    events_response = await _make_api_request(session, "GET", request_url, params = params,
                                              headers = headers)
    if events_response:
        logger.debug("Fetched events data")
        # Response structure varies greatly. Adapt parsing logic.
        # Example for Ticketmaster-like structure:
        events_list = events_response.get("_embedded", {}).get("events", [])
        if not isinstance(events_list, list):
             logger.warning(f"Events response format unexpected: {events_response}")
             return []
        extracted_events = []
        for event in events_list:
            try:
                name = event.get("name")
                event_url = event.get("url")
                start_time_str = event.get("dates", {}).get("start",{}).get("datetime")
                start_time = datetime.fromisoformat(start_time_str.replace("Z",'+00:00')) if start_time_str else None # Handle timezone

                venue_info = event.get("_embedded",{}).get("venues",[{}])[0]
                venue_name = venue_info.get("name")
                location = venue_info.get("location")
                latitude = float(location.get("latitude")) if location and "latitude" in location else None
                longitude = float(location.get("longitude")) if location and "longitude" in location else None
                extracted_events.append({"name":name,
                                         "start_time":start_time,
                                         "venue":venue_name,
                                         "latitude":latitude,
                                         "longitude":longitude,
                                         "url":event_url})
            except (AttributeError, KeyError, IndexError, TypeError, ValueError) as e:
                logger.warning(f"Could not fully parse event data: {event}. Error: {e}")
        logger.debug(f"Extracted {len(extracted_events)} events")
        return extracted_events

    else:
        logger.error("Failed to fetch local events.")
        return None  