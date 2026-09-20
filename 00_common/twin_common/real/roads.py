"""OpenStreetMap drive network, with a cache and a synthetic fallback (docs/04 section 7).

M04, M08, M12, M13 and M14 all route over this graph, so it must exist even with no network
and no OSMnx installed. Resolution order:

1. ``data/real_cache/roads.graphml`` - a previous download
2. ``data/bundled/roads.graphml`` - a small sample committed to git
3. OSMnx (skipped entirely when ``TWIN_OFFLINE=1``), then cached
4. a synthetic grid graph carrying the **same** R01-R10 and J01-J06 key IDs

Because the fallback keeps the key IDs, every downstream model works identically whichever
graph it got; only the geometry and the travel times differ. ``RoadNetwork.is_real`` says
which one is in use so a model can report ``degraded``.

OSMnx v2 note (docs/06 section 3): v2 removed the v1 keyword arguments, so the bounding box
is passed as a single ``bbox`` tuple. The call is wrapped in a version probe rather than
assuming either signature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

from ..config import assumptions, world
from ..logging import get_logger
from ..paths import bundled_dir, ensure_dir, is_offline, real_cache_dir

log = get_logger(__name__)

#: Half-width of the fetched bounding box, in metres.
DEFAULT_RADIUS_M = 2500.0
#: Metres per degree of latitude, for the bounding box maths.
METRES_PER_DEGREE_LAT = 111_320.0


@dataclass(frozen=True, slots=True)
class RoadNetwork:
    """A routable road graph plus the key-link mapping every model shares."""

    graph: nx.MultiDiGraph
    #: The D24 key link table, with osm_u/osm_v filled when the graph is real.
    key_links: pd.DataFrame
    source: str
    is_real: bool

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def free_flow_minutes(self, link_id: str) -> float:
        """Free-flow travel time of one key link, in minutes."""
        row = self.key_links.loc[self.key_links["link_id"] == link_id]
        if row.empty:
            raise KeyError(f"unknown key link {link_id!r}")
        length_km = float(row.iloc[0]["length_m"]) / 1000.0
        speed = max(float(row.iloc[0]["maxspeed_kmh"]), 1e-9)
        return length_km / speed * 60.0

    def capacity(self, link_id: str) -> float:
        row = self.key_links.loc[self.key_links["link_id"] == link_id]
        if row.empty:
            raise KeyError(f"unknown key link {link_id!r}")
        return float(row.iloc[0]["capacity_veh_hr"])


def _bbox(latitude: float, longitude: float, radius_m: float) -> tuple[float, float, float, float]:
    """(north, south, east, west) around a point."""
    d_lat = radius_m / METRES_PER_DEGREE_LAT
    d_lon = radius_m / (METRES_PER_DEGREE_LAT * math.cos(math.radians(latitude)))
    return latitude + d_lat, latitude - d_lat, longitude + d_lon, longitude - d_lon


def _synthetic_grid(latitude: float, longitude: float) -> nx.MultiDiGraph:
    """A small grid graph standing in for the real street network.

    Nodes are laid out on a 4x4 lattice around the venue, which is enough for shortest-path
    and BPR assignment to behave sensibly. Key links are mapped onto lattice edges so that
    R01-R10 and J01-J06 exist whichever graph is in use.
    """
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:4326"
    size = 4
    spacing_m = 700.0
    d_lat = spacing_m / METRES_PER_DEGREE_LAT
    d_lon = spacing_m / (METRES_PER_DEGREE_LAT * math.cos(math.radians(latitude)))

    for row in range(size):
        for column in range(size):
            node_id = row * size + column
            graph.add_node(
                node_id,
                x=longitude + (column - (size - 1) / 2) * d_lon,
                y=latitude + ((size - 1) / 2 - row) * d_lat,
            )
    lanes_default = 2
    speed_default = 40.0
    for row in range(size):
        for column in range(size):
            node_id = row * size + column
            if column + 1 < size:
                right = node_id + 1
                for u, v in ((node_id, right), (right, node_id)):
                    graph.add_edge(
                        u, v, length=spacing_m, lanes=lanes_default, maxspeed=speed_default
                    )
            if row + 1 < size:
                down = node_id + size
                for u, v in ((node_id, down), (down, node_id)):
                    graph.add_edge(
                        u, v, length=spacing_m, lanes=lanes_default, maxspeed=speed_default
                    )
    return graph


def _key_link_table(graph: nx.MultiDiGraph | None, is_real: bool) -> pd.DataFrame:
    """Build D24 ``key_links`` from world.yaml, mapping each ID onto a graph edge.

    The mapping is deterministic: key links are assigned to graph edges in sorted order, so
    the same graph always produces the same mapping.
    """
    world_cfg = world()
    lane_capacity = float(assumptions()["transport"]["lane_capacity_veh_hr"])
    edges = sorted(graph.edges(keys=True)) if graph is not None else []

    rows: list[dict[str, object]] = []
    specs = {**world_cfg["key_links"]}
    for index, (link_id, spec) in enumerate(specs.items()):
        lanes = int(spec["lanes"])
        if edges:
            u, v, key = edges[(index * 3) % len(edges)]
        else:
            u, v, key = "", "", 0
        rows.append(
            {
                "link_id": link_id,
                "name": str(spec["name"]),
                "lanes": lanes,
                "length_m": float(spec["length_m"]),
                "maxspeed_kmh": float(spec["maxspeed_kmh"]),
                "capacity_veh_hr": lanes * lane_capacity,
                "osm_u": str(u),
                "osm_v": str(v),
                "osm_key": int(key),
                "geometry": "",
                "is_synthetic": not is_real,
                "source": "osmnx" if is_real else "synthetic_grid",
            }
        )
    # The bridge is a key link too, so S12 can close it in the road model.
    for bridge_id, spec in world_cfg["bridges"].items():
        rows.append(
            {
                "link_id": bridge_id,
                "name": str(spec["name"]),
                "lanes": 2,
                "length_m": 220.0,
                "maxspeed_kmh": 30.0,
                "capacity_veh_hr": 2 * lane_capacity,
                "osm_u": "",
                "osm_v": "",
                "osm_key": 0,
                "geometry": "",
                "is_synthetic": not is_real,
                "source": "osmnx" if is_real else "synthetic_grid",
            }
        )
    return pd.DataFrame(rows)


def _load_graphml(path: Path) -> nx.MultiDiGraph | None:
    if not path.is_file():
        return None
    try:
        graph = nx.read_graphml(path)
    except Exception as exc:  # a truncated cache must not be fatal
        log.warning("ignoring unreadable road cache %s: %s", path, exc)
        return None
    log.info("using cached road graph %s (%d nodes)", path.name, graph.number_of_nodes())
    return nx.MultiDiGraph(graph)


def _fetch_osm(latitude: float, longitude: float, radius_m: float) -> nx.MultiDiGraph:
    """Download the drive network with OSMnx v2.

    v2 dropped the v1 ``north=/south=/east=/west=`` arguments in favour of one ``bbox``
    tuple (docs/06 section 3). Both are attempted so the code works either way rather than
    guessing at the installed version.
    """
    import osmnx as ox

    north, south, east, west = _bbox(latitude, longitude, radius_m)
    log.info("fetching the OSM drive network around %.4f, %.4f", latitude, longitude)
    try:
        # OSMnx v2 signature.
        return ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive")
    except TypeError:
        # OSMnx v1 signature.
        return ox.graph_from_bbox(
            north=north, south=south, east=east, west=west, network_type="drive"
        )


def get_road_network(
    *,
    latitude: float,
    longitude: float,
    radius_m: float = DEFAULT_RADIUS_M,
    use_cache: bool = True,
) -> RoadNetwork:
    """Load the road graph, from cache, OSM, or the synthetic grid fallback."""
    cache_path = real_cache_dir() / "roads.graphml"
    bundled_path = bundled_dir() / "roads.graphml"

    if use_cache:
        for path, label in ((cache_path, "osmnx (cached)"), (bundled_path, "osmnx (bundled)")):
            graph = _load_graphml(path)
            if graph is not None:
                return RoadNetwork(
                    graph=graph,
                    key_links=_key_link_table(graph, is_real=True),
                    source=label,
                    is_real=True,
                )

    if not is_offline():
        try:
            graph = _fetch_osm(latitude, longitude, radius_m)
            ensure_dir(cache_path.parent)
            # GraphML cannot hold nested attributes, so they are stringified on write.
            writable = nx.MultiDiGraph(graph)
            for _, _, data in writable.edges(data=True):
                for key, value in list(data.items()):
                    if isinstance(value, list | dict):
                        data[key] = str(value)
            nx.write_graphml(writable, cache_path)
            log.info(
                "cached road graph to %s (%d nodes, %d edges)",
                cache_path,
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )
            return RoadNetwork(
                graph=graph,
                key_links=_key_link_table(graph, is_real=True),
                source="osmnx",
                is_real=True,
            )
        except ImportError:
            log.warning("osmnx is not installed; using the synthetic grid graph")
        except Exception as exc:  # any network or Overpass problem falls back
            log.warning("OSM fetch failed (%s); using the synthetic grid graph", exc)
    else:
        log.info("TWIN_OFFLINE is set; skipping the OSM request")

    graph = _synthetic_grid(latitude, longitude)
    return RoadNetwork(
        graph=graph,
        key_links=_key_link_table(graph, is_real=False),
        source="synthetic_grid",
        is_real=False,
    )
