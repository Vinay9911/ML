"""Spatial layout: zones, assets, cameras and key links (docs/04 sections 2 and 5.1).

The venue is generated schematically around ``venue_center`` using local metric offsets in
UTM 43N, then converted to WGS84, so areas and buffers are computed in metres while the
output is map-ready lat/lon. If the user supplies ``00_common/config/zones.geojson`` those
polygons are used instead and only the attributes come from ``world.yaml``.

Zone geometry is a simple rectangle per zone, sized to match the configured ``area_m2``.
That is deliberate: the demo needs consistent areas, adjacency and camera coverage, not a
cartographically faithful site plan. Real polygons replace it without touching any model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import transform as shapely_transform

from ..config import assumptions, world, zones_geojson_path
from ..logging import get_logger

log = get_logger(__name__)

#: Schematic placement: (east_m, north_m) of each zone centre relative to venue_center.
#: Chosen so that the docs/04 section 2.1 adjacency list is geometrically plausible:
#: the river runs east-west along the south (Z01, Z02), the approach chain runs north,
#: and the hub sits furthest out.
ZONE_OFFSETS_M: dict[str, tuple[float, float]] = {
    "Z01": (-60.0, -220.0),  # Ghat A, river front
    "Z02": (120.0, -230.0),  # Ghat B, river front east
    "Z03": (0.0, -80.0),  # temple approach corridor, between ghats and Z04
    "Z04": (0.0, 60.0),  # main pedestrian approach
    "Z05": (0.0, 240.0),  # gate plaza / holding
    "Z06": (260.0, -330.0),  # bridge approach, east of Ghat B
    "Z07": (200.0, 120.0),  # market and food street
    "Z08": (-40.0, 520.0),  # transit and parking hub, furthest from the river
}

#: Where non-zone point entities sit relative to their zone centre, in metres.
#: Keeps gates on the outer edge and exits on the opposite side, so camera sectors and
#: walking distances are not degenerate.
ENTITY_JITTER_M: dict[str, tuple[float, float]] = {
    "gate": (0.0, 40.0),
    "exit": (0.0, -40.0),
    "bridge": (30.0, 0.0),
    "facility": (-35.0, 15.0),
    "service_point": (25.0, -20.0),
    "asset": (-25.0, -25.0),
    "camera": (0.0, 0.0),
    "parking_site": (-70.0, 30.0),
}


@dataclass(frozen=True, slots=True)
class Layout:
    """The generated spatial world, held in memory for the rest of the pipeline."""

    zones: pd.DataFrame
    assets: pd.DataFrame
    cameras: pd.DataFrame
    key_links: pd.DataFrame
    zones_geojson: dict[str, Any]
    #: zone_id -> shapely polygon in metres (UTM), for area and coverage maths.
    zone_polygons_m: dict[str, Polygon]

    @property
    def zone_ids(self) -> list[str]:
        return self.zones["zone_id"].tolist()

    def usable_area(self, zone_id: str) -> float:
        row = self.zones.loc[self.zones["zone_id"] == zone_id].iloc[0]
        return float(row["usable_area_m2"])

    def safe_capacity(self, zone_id: str) -> float:
        row = self.zones.loc[self.zones["zone_id"] == zone_id].iloc[0]
        return float(row["safe_capacity"])

    def zone_type(self, zone_id: str) -> str:
        row = self.zones.loc[self.zones["zone_id"] == zone_id].iloc[0]
        return str(row["type"])

    def entities_of(self, entity_type: str) -> pd.DataFrame:
        return self.assets.loc[self.assets["entity_type"] == entity_type]

    def entity_zone(self, entity_id: str) -> str | None:
        match = self.assets.loc[self.assets["entity_id"] == entity_id]
        if match.empty:
            return None
        value = match.iloc[0]["zone_id"]
        return None if pd.isna(value) or value == "" else str(value)


def _transformers(utm_epsg: int, wgs84_epsg: int) -> tuple[Transformer, Transformer]:
    to_metric = Transformer.from_crs(f"EPSG:{wgs84_epsg}", f"EPSG:{utm_epsg}", always_xy=True)
    to_wgs84 = Transformer.from_crs(f"EPSG:{utm_epsg}", f"EPSG:{wgs84_epsg}", always_xy=True)
    return to_metric, to_wgs84


def _rectangle(
    centre_x: float,
    centre_y: float,
    area_m2: float,
    aspect: float,
    *,
    portrait: bool = False,
) -> Polygon:
    """Axis-aligned rectangle of exactly ``area_m2`` around a centre point.

    ``aspect`` is the ratio of the long side to the short side. ``portrait`` runs the long
    side north-south instead of east-west, which matters because a corridor has to run
    along the direction it actually connects (see ``_SHAPE_BY_TYPE``).
    """
    long_side = math.sqrt(area_m2 * aspect)
    short_side = area_m2 / long_side
    width, height = (short_side, long_side) if portrait else (long_side, short_side)
    half_w, half_h = width / 2.0, height / 2.0
    return Polygon(
        [
            (centre_x - half_w, centre_y - half_h),
            (centre_x + half_w, centre_y - half_h),
            (centre_x + half_w, centre_y + half_h),
            (centre_x - half_w, centre_y + half_h),
        ]
    )


#: Zone shape per type: (aspect ratio, long axis runs north-south).
#: The orientation is not cosmetic. The docs/04 section 2.1 adjacency chain
#: Z08 - Z05 - Z04 - Z03 - Z01/Z02 runs north to south, so the two corridors must be
#: portrait; the ghats line the river along the south edge and the bridge approach reaches
#: east, so those are landscape. Getting this wrong points every camera across the narrow
#: axis of a corridor and collapses cctv_coverage (M23).
_SHAPE_BY_TYPE: dict[str, tuple[float, bool]] = {
    "ghat": (4.0, False),  # along the river, east-west
    "corridor": (6.0, True),  # along the approach chain, north-south
    "holding": (1.3, False),  # plaza, roughly square
    "bridge": (8.0, False),  # Z02 to the far bank, east-west
    "market": (3.0, True),  # food street off the Z04/Z05 spine
    "hub": (1.2, False),  # transit and parking, roughly square
}


def _fov_sector(
    x: float,
    y: float,
    bearing_deg: float,
    half_angle_deg: float,
    range_m: float,
    segments: int = 12,
) -> Polygon:
    """A camera field of view as a simple circular sector (docs/04 section 2.2).

    Bearing is measured from north, clockwise, which is the convention a surveyor would
    use; it is converted to the mathematical convention here.
    """
    start = math.radians(90.0 - bearing_deg - half_angle_deg)
    end = math.radians(90.0 - bearing_deg + half_angle_deg)
    points = [(x, y)]
    for index in range(segments + 1):
        angle = start + (end - start) * index / segments
        points.append((x + range_m * math.cos(angle), y + range_m * math.sin(angle)))
    return Polygon(points)


def build_layout(*, offline_cameras_share: float = 0.0, seed: int = 42) -> Layout:
    """Generate the spatial world.

    Args:
        offline_cameras_share: S09 ``camera_outage_share``. The picked cameras are chosen
            deterministically from the sorted ID list so the same share always takes out
            the same cameras for a given seed.
        seed: base seed, used only for the camera outage pick.
    """
    world_cfg = world()
    venue = world_cfg["venue"]
    centre_lat = float(venue["venue_center"]["lat"])
    centre_lon = float(venue["venue_center"]["lon"])
    to_metric, to_wgs84 = _transformers(int(venue["utm_epsg"]), int(venue["wgs84_epsg"]))
    origin_x, origin_y = to_metric.transform(centre_lon, centre_lat)

    design_density = float(assumptions()["crowd"]["design_density_p_m2"])
    user_geojson = zones_geojson_path()

    # ---------------------------------------------------------------- zones
    zone_polygons_m: dict[str, Polygon] = {}
    zone_rows: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []

    supplied: dict[str, Polygon] = {}
    if user_geojson is not None:
        log.info("using user-supplied zone polygons from %s", user_geojson)
        payload = json.loads(user_geojson.read_text(encoding="utf-8"))
        for feature in payload.get("features", []):
            zone_id = feature["properties"].get("zone_id")
            if zone_id:
                geom_wgs84 = shape(feature["geometry"])
                supplied[zone_id] = shapely_transform(
                    lambda x, y, z=None: to_metric.transform(x, y), geom_wgs84
                )

    for zone_id, spec in world_cfg["zones"].items():
        area_m2 = float(spec["area_m2"])
        usable_share = float(spec["usable_share"])
        if zone_id in supplied:
            polygon_m = supplied[zone_id]
            # Trust the drawn polygon for geometry, but keep the configured area for the
            # capacity maths so a rough sketch does not silently change safe capacity.
            log.debug("zone %s uses a supplied polygon (%.0f m2 drawn)", zone_id, polygon_m.area)
        else:
            east, north = ZONE_OFFSETS_M[zone_id]
            aspect, portrait = _SHAPE_BY_TYPE.get(str(spec["type"]), (2.0, False))
            polygon_m = _rectangle(
                origin_x + east, origin_y + north, area_m2, aspect, portrait=portrait
            )
        zone_polygons_m[zone_id] = polygon_m
        polygon_wgs84 = shapely_transform(lambda x, y, z=None: to_wgs84.transform(x, y), polygon_m)
        usable_area = area_m2 * usable_share
        adjacency = sorted(
            other
            for pair in world_cfg["adjacency"]
            for other in pair
            if zone_id in pair and other != zone_id
        )
        centroid = polygon_wgs84.centroid
        row = {
            "zone_id": zone_id,
            "name": str(spec["name"]),
            "type": str(spec["type"]),
            "area_m2": area_m2,
            "usable_area_m2": usable_area,
            "safe_capacity": usable_area * design_density,
            "low_lying": bool(spec["low_lying"]),
            "drainage_capacity_mm_hr": float(spec["drainage_capacity_mm_hr"]),
            "adjacency": json.dumps(adjacency),
            "attributes": json.dumps(spec.get("attributes") or {}),
            "geometry": polygon_wgs84.wkt,
            "centroid_lat": centroid.y,
            "centroid_lon": centroid.x,
            "is_synthetic": True,
            "source": "layout",
        }
        zone_rows.append(row)
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(polygon_wgs84),
                "properties": {
                    key: value
                    for key, value in row.items()
                    if key not in ("geometry", "is_synthetic", "source")
                }
                | {"is_synthetic": True, "source": "layout"},
            }
        )

    zones = pd.DataFrame(zone_rows)
    zones_geojson = {
        "type": "FeatureCollection",
        "name": "zones",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }

    # ---------------------------------------------------------------- assets
    def point_for(zone_id: str | None, entity_type: str, ordinal: int) -> tuple[float, float]:
        """Lat/lon for a point entity, spread around its zone centre."""
        if zone_id is None or zone_id not in zone_polygons_m:
            # Outside the venue (for example hospital H01): place it well north.
            offset_x, offset_y = 400.0 + 60.0 * ordinal, 900.0
            lon, lat = to_wgs84.transform(origin_x + offset_x, origin_y + offset_y)
            return lat, lon
        centre = zone_polygons_m[zone_id].centroid
        base_x, base_y = ENTITY_JITTER_M.get(entity_type, (0.0, 0.0))
        # Fan multiple entities of the same type around the zone centre.
        angle = 2.0 * math.pi * ordinal / 6.0
        spread = 18.0 * ordinal
        x = centre.x + base_x + spread * math.cos(angle)
        y = centre.y + base_y + spread * math.sin(angle)
        lon, lat = to_wgs84.transform(x, y)
        return lat, lon

    asset_rows: list[dict[str, Any]] = []

    def add_entity(
        entity_id: str,
        entity_type: str,
        zone_id: str | None,
        capacity: float,
        ordinal: int,
        attributes: dict[str, Any] | None = None,
        status: str = "active",
    ) -> None:
        lat, lon = point_for(zone_id, entity_type, ordinal)
        asset_rows.append(
            {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "zone_id": zone_id if zone_id is not None else "",
                "lat": lat,
                "lon": lon,
                "capacity": float(capacity),
                "status": status,
                "attributes": json.dumps(attributes or {}),
                "is_synthetic": True,
                "source": "layout",
            }
        )

    for ordinal, (gate_id, spec) in enumerate(world_cfg["gates"].items()):
        add_entity(
            gate_id,
            "gate",
            spec["zone_id"],
            float(spec["throughput_p_min"]),
            ordinal,
            {"secure": bool(spec["secure"]), "name": spec["name"]},
        )
    for ordinal, (exit_id, spec) in enumerate(world_cfg["exits"].items()):
        add_entity(
            exit_id,
            "exit",
            spec["zone_id"],
            float(spec["width_m"]),
            ordinal,
            {"width_m": float(spec["width_m"]), "name": spec["name"]},
            status=str(spec["status"]),
        )
    for ordinal, (bridge_id, spec) in enumerate(world_cfg["bridges"].items()):
        add_entity(
            bridge_id,
            "bridge",
            spec["zone_id"],
            float(spec["capacity_p_min"]),
            ordinal,
            {"width_m": float(spec["width_m"]), "name": spec["name"]},
            status=str(spec["status"]),
        )
    parking_capacity = assumptions()["transport"]["parking_capacity"]
    for ordinal, (parking_id, spec) in enumerate(world_cfg["parking_sites"].items()):
        add_entity(
            parking_id,
            "parking_site",
            spec["zone_id"],
            float(parking_capacity[parking_id]),
            ordinal,
            {"name": spec["name"], "link": spec["link"]},
        )
    for ordinal, (route_id, spec) in enumerate(world_cfg["shuttle_routes"].items()):
        add_entity(
            route_id,
            "route",
            None,
            0.0,
            ordinal,
            {
                "name": spec["name"],
                "from": spec["from"],
                "to": spec["to"],
                "length_km": float(spec["length_km"]),
                "kind": "shuttle",
            },
        )

    hospital_beds = assumptions()["medical"]["hospital_beds"]
    for ordinal, (hospital_id, spec) in enumerate(world_cfg["hospitals"].items()):
        zone_id = spec["zone_id"]
        add_entity(
            hospital_id,
            "facility",
            zone_id,
            float(hospital_beds.get(hospital_id, 0)),
            ordinal,
            {
                "name": spec["name"],
                "kind": "hospital",
                "inside_venue": bool(spec["inside_venue"]),
                "link": spec["link"],
            },
        )
    post_capacity = float(assumptions()["medical"]["first_aid_post_cases_per_hr_capacity"])
    for ordinal, (post_id, spec) in enumerate(world_cfg["medical_posts"].items()):
        add_entity(
            post_id, "facility", spec["zone_id"], post_capacity, ordinal, {"kind": "medical_post"}
        )
    for ordinal, (post_id, spec) in enumerate(world_cfg["police_posts"].items()):
        add_entity(post_id, "facility", spec["zone_id"], 0.0, ordinal, {"kind": "police_post"})
    for ordinal, (station_id, spec) in enumerate(world_cfg["fire_stations"].items()):
        add_entity(
            station_id,
            "facility",
            spec["zone_id"],
            0.0,
            ordinal,
            {
                "kind": "fire_station",
                "name": spec["name"],
                "link": spec["link"],
                "turnout_min": float(spec["turnout_min"]),
            },
        )
    for ordinal, (shelter_id, spec) in enumerate(world_cfg["shelters"].items()):
        add_entity(
            shelter_id,
            "facility",
            spec["zone_id"],
            float(spec["capacity_persons"]),
            ordinal,
            {"kind": "shelter"},
        )
    for ordinal, (site_id, spec) in enumerate(world_cfg["ambulance_staging"].items()):
        add_entity(
            site_id, "facility", spec["zone_id"], 0.0, ordinal, {"kind": "ambulance_staging"}
        )

    for ordinal, (cluster_id, spec) in enumerate(world_cfg["toilet_clusters"].items()):
        add_entity(
            cluster_id,
            "service_point",
            spec["zone_id"],
            float(spec["max_units"]),
            ordinal,
            {"kind": "toilet_cluster", "low_lying": bool(spec["low_lying"])},
        )
    persons_per_point = float(assumptions()["water"]["persons_per_water_point"])
    for ordinal, (point_id, spec) in enumerate(world_cfg["water_points"].items()):
        add_entity(
            point_id,
            "service_point",
            spec["zone_id"],
            persons_per_point,
            ordinal,
            {"kind": "water_point"},
        )
    bin_capacity = float(assumptions()["waste"]["bin_capacity_kg"])
    for ordinal, (group_id, spec) in enumerate(world_cfg["waste_bin_groups"].items()):
        add_entity(
            group_id,
            "service_point",
            spec["zone_id"],
            bin_capacity,
            ordinal,
            {"kind": "waste_bin_group"},
        )

    for ordinal, (substation_id, spec) in enumerate(world_cfg["substations"].items()):
        add_entity(
            substation_id,
            "asset",
            spec["feeds_zones"][0],
            float(spec["design_kw"]),
            ordinal,
            {"kind": "substation", "name": spec["name"], "feeds_zones": spec["feeds_zones"]},
        )
    generator_kw = float(assumptions()["power"]["generator_unit_kw"])
    for ordinal, (generator_id, spec) in enumerate(world_cfg["generators"].items()):
        add_entity(
            generator_id,
            "asset",
            spec["zone_id"],
            generator_kw,
            ordinal,
            {"kind": "generator", "substation": spec["substation"], "serves": spec["serves"]},
        )
    for ordinal, (pump_id, spec) in enumerate(world_cfg["pumps"].items()):
        add_entity(
            pump_id,
            "asset",
            spec["zone_id"],
            float(spec["design_lph"]),
            ordinal,
            {"kind": "pump", "substation": spec["substation"]},
        )
    tower_capacity = assumptions()["network"]["tower_capacity_mbps"]
    for ordinal, (tower_id, spec) in enumerate(world_cfg["network_towers"].items()):
        add_entity(
            tower_id,
            "asset",
            spec["zone_id"],
            float(tower_capacity[tower_id]),
            ordinal,
            {"kind": "network_tower", "substation": spec["substation"]},
        )

    assets = pd.DataFrame(asset_rows)

    # ---------------------------------------------------------------- cameras
    camera_cfg = world_cfg["cameras"]
    default_bitrate = float(camera_cfg["default_bitrate_mbps"])
    camera_rows: list[dict[str, Any]] = []
    for zone_id, placements in camera_cfg["placement"].items():
        zone_polygon = zone_polygons_m[zone_id]
        min_x, min_y, max_x, max_y = zone_polygon.bounds
        span_x, span_y = max_x - min_x, max_y - min_y
        # Mount cameras along the zone long axis, evenly spaced and inside the polygon, so a
        # thin corridor gets a line of cameras down its length rather than a ring that falls
        # outside it. Positions are at (i + 0.5) / n of the long side, on the centreline.
        along_y = span_y >= span_x
        for ordinal, (camera_id, mode, tower_id, bearing, half_angle, range_m) in enumerate(
            placements
        ):
            fraction = (ordinal + 0.5) / len(placements)
            if along_y:
                x = (min_x + max_x) / 2.0
                y = min_y + span_y * fraction
            else:
                x = min_x + span_x * fraction
                y = (min_y + max_y) / 2.0
            fov_m = _fov_sector(x, y, float(bearing), float(half_angle), float(range_m))
            covered_m = fov_m.intersection(zone_polygon)
            fov_wgs84 = shapely_transform(lambda px, py, z=None: to_wgs84.transform(px, py), fov_m)
            lon, lat = to_wgs84.transform(x, y)
            camera_rows.append(
                {
                    "camera_id": camera_id,
                    "zone_id": zone_id,
                    "mode": str(mode),
                    "bitrate_mbps": default_bitrate,
                    "tower_id": str(tower_id),
                    "status": "active",
                    "fov_wkt": fov_wgs84.wkt,
                    "roi_ground_area_m2": float(covered_m.area),
                    # A nominal calibration so M02 has a fallback when no homography exists.
                    "metres_per_pixel": float(range_m) / 1000.0,
                    "source_uri": "",
                    "lat": lat,
                    "lon": lon,
                    "is_synthetic": True,
                    "source": "layout",
                }
            )
    cameras = pd.DataFrame(camera_rows)

    # S09: take a deterministic share of cameras offline.
    if offline_cameras_share > 0:
        count = round(len(cameras) * float(offline_cameras_share))
        if count > 0:
            rng = np.random.default_rng(seed)
            picked = rng.choice(np.sort(cameras["camera_id"].to_numpy()), size=count, replace=False)
            cameras.loc[cameras["camera_id"].isin(picked), "status"] = "offline"
            log.info("S09 camera outage: %d of %d cameras offline", count, len(cameras))

    for _, row in cameras.iterrows():
        add_entity(
            str(row["camera_id"]),
            "camera",
            str(row["zone_id"]),
            float(row["bitrate_mbps"]),
            0,
            {"kind": "camera", "mode": row["mode"], "tower_id": row["tower_id"]},
            status=str(row["status"]),
        )
    assets = pd.DataFrame(asset_rows)

    # ---------------------------------------------------------------- key links
    lane_capacity = float(assumptions()["transport"]["lane_capacity_veh_hr"])
    link_rows: list[dict[str, Any]] = []
    for link_id, spec in world_cfg["key_links"].items():
        lanes = int(spec["lanes"])
        link_rows.append(
            {
                "link_id": link_id,
                "name": str(spec["name"]),
                "lanes": lanes,
                "length_m": float(spec["length_m"]),
                "maxspeed_kmh": float(spec["maxspeed_kmh"]),
                "capacity_veh_hr": lanes * lane_capacity,
                "osm_u": "",
                "osm_v": "",
                "osm_key": 0,
                "geometry": "",
                "is_synthetic": True,
                "source": "layout",
            }
        )
    # The bridge is also a key link for M04/M14 closure scenarios (S12).
    for bridge_id, spec in world_cfg["bridges"].items():
        link_rows.append(
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
                "is_synthetic": True,
                "source": "layout",
            }
        )
    key_links = pd.DataFrame(link_rows)

    log.info(
        "layout: %d zones, %d assets, %d cameras, %d key links",
        len(zones),
        len(assets),
        len(cameras),
        len(key_links),
    )
    return Layout(
        zones=zones,
        assets=assets,
        cameras=cameras,
        key_links=key_links,
        zones_geojson=zones_geojson,
        zone_polygons_m=zone_polygons_m,
    )
