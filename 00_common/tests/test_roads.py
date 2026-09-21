"""The road graph must load offline, degrade honestly, and survive a bad cache.

M04, M08, M12, M13 and M14 all route over this graph, so the fallback chain in
``twin_common.real.roads`` matters as much as the download itself. Every test here redirects
the cache directories at ``tmp_path``; none of them touch the real cache or the network.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from twin_common.real import roads

KEY_LINKS = tuple(f"R{n:02d}" for n in range(1, 11))


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch):
    """Point both cache roots at tmp_path so no test reads or writes the real one."""
    cache = tmp_path / "real_cache"
    bundled = tmp_path / "bundled"
    cache.mkdir()
    bundled.mkdir()
    monkeypatch.setattr(roads, "real_cache_dir", lambda: cache)
    monkeypatch.setattr(roads, "bundled_dir", lambda: bundled)
    return cache


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv("TWIN_OFFLINE", "1")


def _fake_osm_graph() -> nx.MultiDiGraph:
    """A tiny graph shaped like what OSMnx v2 returns, geometry and all.

    The geometry is the point: GraphML holds only scalars, so a shapely object on an edge is
    what broke the cache write.
    """
    from shapely.geometry import LineString

    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "epsg:4326"
    graph.add_node(1, x=73.79, y=20.00, street_count=3)
    graph.add_node(2, x=73.80, y=20.01, street_count=3)
    graph.add_edge(
        1,
        2,
        0,
        length=1200.0,
        maxspeed="50",
        lanes="2",
        geometry=LineString([(73.79, 20.00), (73.80, 20.01)]),
    )
    return graph


# ------------------------------------------------------------------ offline fallback
def test_offline_uses_the_synthetic_grid(offline) -> None:
    """TWIN_OFFLINE must never reach for the network, and must still return a graph."""
    net = roads.get_road_network(latitude=20.0063, longitude=73.7926)
    assert net.source == "synthetic_grid"
    assert net.is_real is False
    assert net.node_count > 0


def test_the_fallback_keeps_the_key_link_ids(offline) -> None:
    """Downstream models address links by ID, so the fallback must carry the same ones."""
    net = roads.get_road_network(latitude=20.0063, longitude=73.7926)
    assert set(KEY_LINKS) <= set(net.key_links["link_id"])


def test_the_fallback_is_flagged_synthetic(offline) -> None:
    """CLAUDE.md: every row carries is_synthetic, and the grid is not real data."""
    net = roads.get_road_network(latitude=20.0063, longitude=73.7926)
    assert net.key_links["is_synthetic"].all()


def test_free_flow_minutes_are_finite_on_the_fallback(offline) -> None:
    net = roads.get_road_network(latitude=20.0063, longitude=73.7926)
    for link_id in KEY_LINKS:
        minutes = net.free_flow_minutes(link_id)
        assert minutes > 0 and minutes < 600


# ------------------------------------------------------------------ the GraphML round trip
def test_a_graph_with_shapely_geometry_can_be_cached(tmp_path: Path) -> None:
    """The regression: a LineString edge attribute must not break the cache write.

    ``nx.write_graphml`` raises "does not support <class shapely...LineString> as data
    values", which used to abort the whole fetch.
    """
    path = tmp_path / "roads.graphml"
    roads._save_graphml(_fake_osm_graph(), path)
    assert path.is_file() and path.stat().st_size > 0


def test_a_cached_graph_reloads_with_numeric_lengths(tmp_path: Path) -> None:
    """``length`` must come back as a number: travel times are computed from it.

    A plain ``nx.read_graphml`` hands every attribute back as ``str``, which would turn the
    free-flow maths into string concatenation instead of failing loudly.
    """
    path = tmp_path / "roads.graphml"
    roads._save_graphml(_fake_osm_graph(), path)
    graph = roads._load_graphml(path)
    assert graph is not None
    lengths = [data["length"] for _, _, data in graph.edges(data=True)]
    assert lengths and all(isinstance(value, float) for value in lengths)


# ------------------------------------------------------------------ cache robustness
def test_a_corrupt_cache_is_ignored_rather_than_fatal(_isolated_cache, offline) -> None:
    """A half-written cache from a killed process must not take the service down."""
    (_isolated_cache / "roads.graphml").write_text("", encoding="utf-8")
    net = roads.get_road_network(latitude=20.0063, longitude=73.7926)
    assert net.source == "synthetic_grid"


def test_a_cache_write_failure_keeps_the_downloaded_graph(monkeypatch) -> None:
    """The regression that mattered: a caching problem must not discard a real download.

    Before this, the cache write sat inside the fetch ``try``, so a read-only disk - or any
    unserialisable attribute - silently downgraded a real OSM graph to the synthetic grid.
    """
    monkeypatch.setattr(roads, "_fetch_osm", lambda *a, **k: _fake_osm_graph())

    def explode(graph, path):
        raise OSError("read-only file system")

    monkeypatch.setattr(roads, "_save_graphml", explode)

    net = roads.get_road_network(latitude=20.0063, longitude=73.7926)
    assert net.is_real is True, "a cache failure must not fall back to the grid"
    assert net.source == "osmnx"


def test_a_failed_fetch_does_fall_back(monkeypatch) -> None:
    """The other side of the same coin: a genuine fetch failure degrades to the grid."""

    def explode(*args, **kwargs):
        raise RuntimeError("Overpass is down")

    monkeypatch.setattr(roads, "_fetch_osm", explode)
    net = roads.get_road_network(latitude=20.0063, longitude=73.7926)
    assert net.is_real is False
    assert net.source == "synthetic_grid"


def test_a_real_graph_is_not_flagged_synthetic(monkeypatch) -> None:
    monkeypatch.setattr(roads, "_fetch_osm", lambda *a, **k: _fake_osm_graph())
    net = roads.get_road_network(latitude=20.0063, longitude=73.7926)
    assert not net.key_links["is_synthetic"].any()


def test_the_cache_is_preferred_over_a_new_fetch(_isolated_cache, monkeypatch) -> None:
    """Step 1 of the documented order: a previous download wins over the network."""
    roads._save_graphml(_fake_osm_graph(), _isolated_cache / "roads.graphml")

    def explode(*args, **kwargs):
        raise AssertionError("the cache should have been used")

    monkeypatch.setattr(roads, "_fetch_osm", explode)
    net = roads.get_road_network(latitude=20.0063, longitude=73.7926)
    assert net.source == "osmnx (cached)"
    assert net.is_real is True


def test_use_cache_false_bypasses_the_cache(_isolated_cache, monkeypatch) -> None:
    roads._save_graphml(_fake_osm_graph(), _isolated_cache / "roads.graphml")
    calls: list[str] = []

    def record(*args, **kwargs):
        calls.append("fetched")
        return _fake_osm_graph()

    monkeypatch.setattr(roads, "_fetch_osm", record)
    roads.get_road_network(latitude=20.0063, longitude=73.7926, use_cache=False)
    assert calls == ["fetched"]
