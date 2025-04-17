"""
Constructs graph representations of the transit network for GNN models using PyTorch Geometric (PyG).

Builds a static graph where nodes are stops and edges represent connections
between consecutive stops on transit routes. Adds basic static node/edge features.
"""
import logging
from typing import Optional, Any, Dict, List, Tuple

import pandas as pd
import numpy as np

try:
    import torch
    import torch_geometric as pyg
    from torch_geometric.data import Data
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False
    class Data:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        def __repr__(self):
            return f"DummyData({self.__dict__})"

from . import config
from . import utils
from data_pipeline import gtfs_parser
import gtfs_kit as gk

logger = logging.getLogger(__name__)

# Static GTFS Loading
_static_gtfs_feed_cache:Dict[str, Optional[gk.Feed]] = {}

def load_static_gtfs_data(dataset_name:str = config.ACTIVE_DATASET_NAME) -> Optional[gk.Feed]:
    """
    Loads the static GTFS feed needed for graph construction for a specific dataset.
    Uses a simple cache. Assumes a GTFS source path can be determined.

    Args:
        dataset_name: The name of the dataset (e.g., 'MTA', 'HSL').

    Returns:
        A gtfs_kit Feed object or None.
    """
    if dataset_name in _static_gtfs_feed_cache:
        return _static_gtfs_feed_cache[dataset_name]
    logger.info(f"Attempting to load static GTFS data for graph construction (Dataset: {dataset_name})")
    try:
        ds_raw_dir = config.DATA_PIPELINE_RAW_OUTPUT_DIR / dataset_name.lower()
        # Look for common GTFS zip file names
        possible_paths = list(ds_raw_dir.glob("gtfs.zip")) + list(ds_raw_dir.glob(f"{dataset_name.lower()}_gtfs.zip"))
        if not possible_paths:
             # Look inside subdirectories common in GTFS archives (e.g., extracted folder)
             possible_paths = list(ds_raw_dir.glob("*/gtfs.zip")) + list(ds_raw_dir.glob(f"*/{dataset_name.lower()}_gtfs.zip"))

        if not possible_paths:
            logger.error(f"Could not find GTFS zip file in expected location: {ds_raw_dir}")
            _static_gtfs_feed_cache[dataset_name] = None
            return None

        gtfs_path = possible_paths[0] # Use the first one found
        logger.info(f"Found GTFS path: {gtfs_path}")

        feed = gtfs_parser.load_static_gtfs(gtfs_path) # Use function from data_pipeline
        if feed:
            logger.info(f"Loaded static GTFS for {dataset_name} successfully.")
            # Basic validation
            if feed.stops is None or feed.stop_times is None or feed.trips is None:
                 logger.error("Loaded GTFS feed is missing essential tables (stops, stop_times, trips).")
                 feed = None
        _static_gtfs_feed_cache[dataset_name] = feed
        return feed

    except Exception as e:
        logger.error(f"Failed to load static GTFS data for {dataset_name}: {e}", exc_info=True)
        _static_gtfs_feed_cache[dataset_name] = None
        return None
    
# Graph Construction Functions (PyG)
def build_static_graph(gtfs_feed:gk.Feed,
                    dataset_name:str = config.ACTIVE_DATASET_NAME) -> Optional[Data]:
    """
    Builds the base PyG Data object (nodes=stops, edges=route segments) from static GTFS data.

    Args:
        gtfs_feed: Loaded gtfs_kit Feed object.
        dataset_name: Name of the dataset (used for logging).

    Returns:
        A PyG Data object, or None if failed.
    """
    logger.info(f"Building static PyG graph structure for {dataset_name}")
    if not _PYG_AVAILABLE:
        logger.error("PyTorch Geometric (PyG) is required but not installed")
        return None
    if not gtfs_feed or gtfs_feed.stops is None or gtfs_feed.stop_times is None or gtfs_feed.trips is None:
        logger.error("Cannot build graph: GTFS feed or essential tables are missing")
        return None
    #1. Nodes: Stops
    stops_df = gtfs_feed.stops.copy().dropna(subset = ["stop_id"]) # Ensure no NaNs stop_ids
    stops_df["stop_id"] = stops_df["stop_id"].astype(str)
    num_nodes = len(stops_df)
    if num_nodes == 0:
        logger.error("No stops found in GTFS data")
        return None
    # Create mapping from stop_id to node index (0 to num_nodes-1)
    stop_id_to_idx = {stop_id:i for i, stop_id in enumerate(stops_df["stop_id"])}
    # Store mapping for later use (e.g., feature alignment)
    node_idx_to_stop_id = {i:stop_id for stop_id, i in stop_id_to_idx.items()}
    logger.info(f"Defined {num_nodes} nodes based on stops")

    #2. Edges: Connections between consecutive stops on trips
    edge_list_src = []
    edge_list_dst = []
    edge_data_list = [] 
    logger.info("Processig stop_times to create graph edges...")
    stop_times_df = gtfs_feed.stop_times.copy()
    # Merge with trips to get route_id etc
    if gtfs_feed.trips is not None:
        trips_df = gtfs_feed.trips[["trip_id", "route_id"]].copy()
        stop_times_df = pd.merge(stop_times_df, trips_df, on="trip_id", how="left")
    # Ensure required columns exist
    required_st_cols = ["trip_id", "stop_sequence", "stop_id", "departure_time", "arrival_time"]
    if not all (col in stop_times_df.columns for col in required_st_cols):
        logger.error(f"stopt_times table missing one or more required columns: {required_st_cols}")
        return None
    # Conver time to timedelta for calculation (handle potential > 24h)
    stop_times_df["departure_td"] = stop_times_df["departure_time"].apply(lambda x:gk.helpers.datestr_to_seconds(x, inverse = True) if pd.notna(x) else None)
    stop_times_df['arrival_td'] = stop_times_df['arrival_time'].apply(lambda x: gk.helpers.datestr_to_seconds(x, inverse=True) if pd.notna(x) else None)
    stop_times_df.dropna(subset=["departure_td", "arrival_td", "stop_id"], inplace = True)
    stop_times_df["stop_id"] = stop_times_df["stop_id"].astype(str)
    processed_edges = 0
    # Group by trip and sort by sequence
    grouped_st = stop_times_df.sort_values("stop_sequence").groupby("trip_id")
    for trip_id, group in grouped_st:
        stop_ids = group['stop_id'].tolist()
        dep_tds = group['departure_td'].tolist()
        arr_tds = group['arrival_td'].tolist()
        route_id = group['route_id'].iloc[0] if 'route_id' in group.columns else None
        for i in range(len(stop_ids) - 1):
            src_stop_id = stop_ids[i]
            dst_stop_id = stop_ids[i + 1]
            src_dep_td = dep_tds[i]
            dst_arr_td = arr_tds[i + 1]
            src_idx = stop_id_to_idx.get(src_stop_id)
            dst_idx = stop_id_to_idx.get(dst_stop_id)
            if src_idx is not None and dst_idx is not None:
                edge_list_src.append(src_idx)
                edge_list_dst.append(dst_idx)
                # Store data needed for edge attributes
                scheduled_time_secs = dst_arr_td - src_dep_td if dst_arr_td is not None and src_dep_td is not None else None
                edge_data_list.append({"src_stop_id":src_stop_id,
                                       "dst_stop_id":dst_stop_id,
                                       "trip_id":trip_id,
                                       "route_id":route_id,
                                       "scheduled_time_seconds":scheduled_time_secs})
                processed_edges += 1
            if not edge_list_src:
                logger.warning("No edges could be created from stop_times. Check data integrity")
                edge_index = torch.empty((2,0), dtype=torch.long)
            else:
                edge_index = torch.tensor([edge_list_src, edge_list_dst], dtype = torch.long)
            logger.info(f"Created {edge_index.shape[1]} directed edges")
            # 3. Create PyG Data object
    graph_data = Data(
        edge_index=edge_index,
        num_nodes=num_nodes,
        # Store mappings and edge data for feature assignment
        node_idx_to_stop_id=node_idx_to_stop_id,
        stop_id_to_idx=stop_id_to_idx,
        edge_data_list=edge_data_list # Raw data associated with edges by index
    )

    logger.info(f"Built basic PyG graph structure: {graph_data}")
    return graph_data


def add_static_node_features(graph_data: Data, gtfs_feed: gk.Feed) -> Data:
    """Adds static node features (location, type) to the PyG Data object."""
    logger.info("Adding static node features...")
    if not _PYG_AVAILABLE or not isinstance(graph_data, Data): return graph_data # Return unchanged if not PyG
    if not gtfs_feed or gtfs_feed.stops is None: return graph_data

    num_nodes = graph_data.num_nodes
    node_idx_to_stop_id = graph_data.node_idx_to_stop_id
    stops_df = gtfs_feed.stops.set_index('stop_id') # Index by stop_id for easy lookup
    node_feature_list = []
    feature_names = []
    
    # Example features: lat, lon, location_type (one-hot)
    has_location_type = "location_type" in stops_df.columns
    if has_location_type:
        # Get unique location type and create one-hot encoding map
        unique_loc_types = stops_df["location_type"].dropna().unique()
        loc_type_map = {ltype: i for i, ltype in enumerate(unique_loc_types)}
        num_loc_types = len(loc_type_map)
        logger.info(f"Found location types: {list(loc_type_map.keys())}")
        feature_names.extend([f"loc_type_{i}" for i in range(num_loc_types)])
    else: num_loc_types = 0

    feature_names.extend(['latitude', 'longitude']) # Add lat/lon names

    for node_idx in range(num_nodes):
        stop_id = node_idx_to_stop_id.get(node_idx)
        node_features = []
        if stop_id and stop_id in stops_df.index:
            stop_info = stops_df.loc[stop_id]
            # Location Type (One-Hot)
            if has_location_type:
                loc_type_one_hot = [0.0]*num_loc_types
                stop_loc_type = stop_info.get("location_type")
                if pd.notna(stop_loc_type) and stop_loc_type in loc_type_map:
                    loc_type_one_hot[loc_type_map[stop_loc_type]] = 1.0
                    node_features.extend(loc_type_one_hot)

            # Latitude/Longitude (consider scaling later)
            lat = stop_info.get('stop_lat', 0.0)
            lon = stop_info.get('stop_lon', 0.0)
            node_features.extend([lat, lon])
            # TODO: Add other static features (wheelchair_boarding etc.) with encoding
        else:
            logger.warning(f"Stop ID {stop_id} for node index {node_idx} not found in stops data. Using zeros for features.")
            node_features = [0.0] * (num_loc_types + 2) # +2 for lat/lon

        node_feature_list.append(node_features)

    if node_feature_list:
        graph_data.x = torch.tensor(node_feature_list, dtype=torch.float32)
        graph_data.node_feature_names = feature_names # Store feature names
        logger.info(f"Added static node features. Shape: {graph_data.x.shape}")
    else:
        logger.warning("No node features could be generated.")

    return graph_data



