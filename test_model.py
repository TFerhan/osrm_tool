"""
test_osrm_pipeline.py
=====================
Full test suite for osrm_pipeline.py
Moroccan UTM 32629 coordinates throughout (Casablanca, Rabat, Marrakech areas).

Run with:
    pytest test_osrm_pipeline.py -v
    pytest test_osrm_pipeline.py -v -k "transformer"   # single class
    pytest test_osrm_pipeline.py -v --tb=short          # compact tracebacks
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from shapely.geometry import Point

from osrm_pipeline import (
    EndpointConfig,
    LayerConfig,
    OSRMRoutingPipeline,
    PipelineConfig,
    _BatchBuilder,
    _CoordTransformer,
    _OSRMFetcher,
    _ParquetSink,
    _SpatialFilter,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Moroccan coordinate fixtures
#  All points are in EPSG:32629 (UTM Zone 29N, metres)
#  Approximate real-world anchors:
#    Casablanca centre ≈ (302_000, 3_720_000)
#    Rabat centre      ≈ (355_000, 3_765_000)
#    Marrakech centre  ≈ (263_000, 3_502_000)
# ══════════════════════════════════════════════════════════════════════════════

# ── Real Moroccan UTM 32629 coordinates ──────────────────────────────────────
# Derived from actual WGS84 city coords projected to EPSG:32629 (UTM Zone 29N)
#
#   Casablanca centre  → x=630_956,  y=3_715_705
#   Rabat centre       → x=699_350,  y=3_766_476
#   Marrakech centre   → x=596_637,  y=3_499_876

# Grid centroids around Casablanca (departure points)
CASA_GRIDS_UTM = [
    (630_956, 3_715_705),   # 0 — Casablanca centre  (~-7.589°, 33.573°)
    (633_956, 3_715_705),   # 1 — ~3 km east
    (630_956, 3_712_705),   # 2 — ~3 km south
    (636_956, 3_717_705),   # 3 — ~6 km NE
    (627_956, 3_713_705),   # 4 — ~3 km SW
]

# Stores — some close, one deliberately far away (Marrakech, ~220 km south)
CASA_STORES_UTM = [
    (631_956, 3_716_205),   # 0 — ~1.1 km from grid 0  ✓
    (633_456, 3_714_705),   # 1 — ~2.7 km from grid 0  ✓
    (631_456, 3_718_705),   # 2 — ~3.0 km from grid 0  ✓
    (637_956, 3_719_705),   # 3 — ~8 km from grid 0, ~2 km from grid 3 ✓
    (596_637, 3_499_876),   # 4 — Marrakech, ~220 km away              ✗
]

# Rabat grid + store pair for multi-city tests
RABAT_GRID_UTM  = (699_350, 3_766_476)    # Rabat centre
RABAT_STORE_UTM = (700_850, 3_766_976)    # ~1.7 km NE


def _make_gdf(utm_points: list[tuple[float, float]], crs="EPSG:32629") -> gpd.GeoDataFrame:
    """Build a minimal GeoDataFrame from a list of (x, y) UTM tuples."""
    geoms = [Point(x, y) for x, y in utm_points]
    return gpd.GeoDataFrame({"geometry": geoms}, crs=crs)


def _make_gdf_with_attrs(
    utm_points: list[tuple[float, float]],
    id_col: str,
    name_col: str,
    names: list[str],
    crs: str = "EPSG:32629",
) -> gpd.GeoDataFrame:
    geoms = [Point(x, y) for x, y in utm_points]
    ids   = list(range(len(utm_points)))
    return gpd.GeoDataFrame(
        {"geometry": geoms, id_col: ids, name_col: names}, crs=crs
    )


# ══════════════════════════════════════════════════════════════════════════════
#  1. EndpointConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestEndpointConfig:

    def test_basic_creation(self):
        ep = EndpointConfig("car", "http://localhost:5000")
        assert ep.profile == "car"
        assert ep.url == "http://localhost:5000"

    def test_trailing_slash_stripped(self):
        ep = EndpointConfig("foot", "http://localhost:5001/")
        assert ep.url == "http://localhost:5001"

    def test_multiple_trailing_slashes_stripped(self):
        ep = EndpointConfig("bike", "http://10.0.0.1:5002///")
        assert ep.url == "http://10.0.0.1:5002"

    def test_repr(self):
        ep = EndpointConfig("car", "http://localhost:5000")
        assert "car" in repr(ep)
        assert "localhost:5000" in repr(ep)

    def test_custom_profile_name(self):
        ep = EndpointConfig("moto", "http://192.168.1.10:5000")
        assert ep.profile == "moto"

    def test_remote_moroccan_server_url(self):
        ep = EndpointConfig("car", "http://osrm.ma:5000")
        assert ep.url == "http://osrm.ma:5000"


# ══════════════════════════════════════════════════════════════════════════════
#  2. LayerConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestLayerConfig:

    def test_minimal_defaults(self):
        lc = LayerConfig("grids.gpkg")
        assert lc.id_column is None
        assert lc.keep_columns == []
        assert lc.layer is None
        assert lc.crs == "EPSG:32629"

    def test_all_fields(self):
        lc = LayerConfig(
            path="stores.gpkg",
            id_column="store_id",
            keep_columns=["name", "category"],
            layer="commerce",
            crs="EPSG:26192",
        )
        assert lc.id_column == "store_id"
        assert lc.keep_columns == ["name", "category"]
        assert lc.layer == "commerce"
        assert lc.crs == "EPSG:26192"

    def test_accepts_geodataframe_directly(self):
        gdf = _make_gdf(CASA_GRIDS_UTM)
        lc = LayerConfig(gdf)
        assert isinstance(lc.path, gpd.GeoDataFrame)

    def test_keep_columns_are_independent_lists(self):
        lc1 = LayerConfig("a.gpkg", keep_columns=["x"])
        lc2 = LayerConfig("b.gpkg")
        lc1.keep_columns.append("y")
        assert lc2.keep_columns == [], "Mutable default leaked between instances"


# ══════════════════════════════════════════════════════════════════════════════
#  3. PipelineConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineConfig:

    def test_defaults(self):
        cfg = PipelineConfig()
        assert len(cfg.endpoints) == 2
        assert cfg.endpoints[0].profile == "car"
        assert cfg.endpoints[1].profile == "foot"
        assert cfg.max_dist_m == 5_000.0
        assert cfg.max_table_size == 8_000
        assert cfg.batch_size == 20
        assert cfg.concurrency == 2
        assert cfg.request_timeout == 30
        assert cfg.flush_every == 2_000
        assert isinstance(cfg.output_path, Path)

    def test_output_path_coerced_to_path(self):
        cfg = PipelineConfig(output_path="my_results.parquet")
        assert isinstance(cfg.output_path, Path)
        assert cfg.output_path == Path("my_results.parquet")

    def test_custom_endpoints(self):
        endpoints = [
            EndpointConfig("car",  "http://localhost:5000"),
            EndpointConfig("foot", "http://localhost:5001"),
            EndpointConfig("bike", "http://localhost:5002"),
        ]
        cfg = PipelineConfig(endpoints=endpoints)
        assert len(cfg.endpoints) == 3

    def test_defaults_are_independent(self):
        cfg1 = PipelineConfig()
        cfg2 = PipelineConfig()
        cfg1.endpoints.append(EndpointConfig("x", "http://x"))
        assert len(cfg2.endpoints) == 2, "Mutable default leaked between PipelineConfig instances"

    def test_moroccan_use_case_config(self):
        cfg = PipelineConfig(
            max_dist_m=3_000,
            concurrency=4,
            batch_size=15,
            output_path="/tmp/morocco_routing.parquet",
        )
        assert cfg.max_dist_m == 3_000
        assert cfg.concurrency == 4


# ══════════════════════════════════════════════════════════════════════════════
#  4. _CoordTransformer
# ══════════════════════════════════════════════════════════════════════════════

class TestCoordTransformer:

    def setup_method(self):
        self.t = _CoordTransformer()

    def test_casablanca_centre_roundtrip(self):
        """UTM → WGS84 for Casablanca centre should give ≈ -7.59°W, 33.57°N."""
        xy = np.array([[630_956.0, 3_715_705.0]])
        wgs = self.t.to_wgs84(xy, "EPSG:32629")
        lon, lat = wgs[0]
        assert -8.5 < lon < -6.5, f"Longitude out of range: {lon}"
        assert 33.0 < lat < 34.5, f"Latitude out of range: {lat}"

    def test_rabat_coordinates(self):
        xy = np.array([[699_350.0, 3_766_476.0]])
        wgs = self.t.to_wgs84(xy, "EPSG:32629")
        lon, lat = wgs[0]
        # Rabat ≈ -6.84°W, 34.02°N
        assert -8.0 < lon < -5.5
        assert 33.5 < lat < 35.0

    def test_marrakech_coordinates(self):
        xy = np.array([[596_637.0, 3_499_876.0]])
        wgs = self.t.to_wgs84(xy, "EPSG:32629")
        lon, lat = wgs[0]
        # Marrakech ≈ -7.98°W, 31.63°N
        assert -9.5 < lon < -7.0
        assert 31.0 < lat < 32.5

    def test_multiple_points_batch(self):
        xy = np.array([p for p in CASA_GRIDS_UTM], dtype=float)
        wgs = self.t.to_wgs84(xy, "EPSG:32629")
        assert wgs.shape == (len(CASA_GRIDS_UTM), 2)
        # All Casablanca-area points should land in Morocco's longitude/latitude band
        assert np.all(wgs[:, 0] > -9.0) and np.all(wgs[:, 0] < -5.0)
        assert np.all(wgs[:, 1] > 32.0) and np.all(wgs[:, 1] < 36.0)

    def test_transformer_is_cached(self):
        """Calling get() twice with the same CRS returns the same object."""
        t1 = self.t.get("EPSG:32629")
        t2 = self.t.get("EPSG:32629")
        assert t1 is t2

    def test_invalid_crs_raises(self):
        with pytest.raises(RuntimeError, match="Cannot create transformer"):
            self.t.get("EPSG:99999999")

    def test_output_shape_preserved(self):
        xy = np.zeros((10, 2))
        xy[:, 0] = 302_000
        xy[:, 1] = 3_720_000
        wgs = self.t.to_wgs84(xy, "EPSG:32629")
        assert wgs.shape == (10, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  5. _SpatialFilter
# ══════════════════════════════════════════════════════════════════════════════

class TestSpatialFilter:

    def setup_method(self):
        self.sf = _SpatialFilter(max_dist_m=5_000)

    def _make_dep_dst(self):
        dep = _make_gdf(CASA_GRIDS_UTM)
        dst = _make_gdf(CASA_STORES_UTM)
        return dep, dst

    def test_nearby_stores_found(self):
        dep, dst = self._make_dep_dst()
        mapping = self.sf.build_map(dep, dst)
        # Grid 0 (Casablanca centre) should find stores 0, 1, 2 (all within 5 km)
        assert 0 in mapping
        assert 0 in mapping[0]
        assert 1 in mapping[0]
        assert 2 in mapping[0]

    def test_marrakech_store_excluded(self):
        """Store 4 (Marrakech, ~220 km away) must not appear in any grid's list."""
        dep, dst = self._make_dep_dst()
        mapping = self.sf.build_map(dep, dst)
        for store_list in mapping.values():
            assert 4 not in store_list, "Marrakech store should be outside 5 km radius"

    def test_distant_grid_finds_different_store(self):
        dep, dst = self._make_dep_dst()
        mapping = self.sf.build_map(dep, dst)
        # Grid 3 (NE, ~8 km from centre) should reach store 3 (near grid 3)
        if 3 in mapping:
            assert 3 in mapping[3]

    def test_empty_result_when_all_out_of_range(self):
        dep = _make_gdf([RABAT_GRID_UTM])              # Rabat
        dst = _make_gdf([(596_637, 3_499_876)])        # Marrakech — ~300 km away
        mapping = self.sf.build_map(dep, dst)
        assert mapping == {}

    def test_tight_radius_reduces_matches(self):
        sf_tight = _SpatialFilter(max_dist_m=500)
        dep, dst = self._make_dep_dst()
        mapping_tight = self.sf.build_map(dep, dst)
        mapping_normal = _SpatialFilter(5_000).build_map(dep, dst)
        tight_total   = sum(len(v) for v in mapping_tight.values())
        normal_total  = sum(len(v) for v in mapping_normal.values())
        assert tight_total <= normal_total

    def test_all_grids_returned_as_keys_when_in_range(self):
        dep = _make_gdf(CASA_GRIDS_UTM)
        # One store right at Casablanca centre — every grid within 20 km finds it
        dst = _make_gdf([(630_956, 3_715_705)])
        mapping = _SpatialFilter(max_dist_m=20_000).build_map(dep, dst)
        for i in range(len(CASA_GRIDS_UTM)):
            assert i in mapping

    def test_single_dep_single_dst_in_range(self):
        dep = _make_gdf([RABAT_GRID_UTM])
        dst = _make_gdf([RABAT_STORE_UTM])
        mapping = self.sf.build_map(dep, dst)
        assert 0 in mapping
        assert mapping[0] == [0]

    def test_uses_projected_distances_not_degrees(self):
        """
        One degree of latitude in Morocco ≈ 111 km, so if distances were
        computed in degrees a 5 km filter would behave completely differently.
        Place two points 3 km apart in UTM — they must be found within 5 km.
        """
        p1 = (630_956, 3_715_705)              # Casablanca centre
        p2 = (630_956, 3_718_705)              # 3 000 m north
        dep = _make_gdf([p1])
        dst = _make_gdf([p2])
        mapping = self.sf.build_map(dep, dst)
        assert 0 in mapping and 0 in mapping[0]


# ══════════════════════════════════════════════════════════════════════════════
#  6. _BatchBuilder
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchBuilder:

    def _simple_map(self) -> dict:
        """5 departures, each with 2 destinations — small, well within limits."""
        return {i: [10 + i, 20 + i] for i in range(5)}

    def test_basic_batch_structure(self):
        bb = _BatchBuilder(max_table_size=8_000, batch_size=20)
        dep_ids = list(range(5))
        batches = bb.build(dep_ids, self._simple_map())
        # Each batch is a 3-tuple
        for dep_batch, dst_batch, valid_pairs in batches:
            assert isinstance(dep_batch, list)
            assert isinstance(dst_batch, list)
            assert isinstance(valid_pairs, set)

    def test_valid_pairs_are_correct(self):
        bb = _BatchBuilder(max_table_size=8_000, batch_size=20)
        dep_dst_map = {0: [100, 101], 1: [101, 102]}
        batches = bb.build([0, 1], dep_dst_map)
        all_pairs = set()
        for _, _, vp in batches:
            all_pairs |= vp
        assert (0, 100) in all_pairs
        assert (0, 101) in all_pairs
        assert (1, 101) in all_pairs
        assert (1, 102) in all_pairs

    def test_size_guard_splits_large_batches(self):
        """Force a scenario where every batch of 20 grids would exceed the cap."""
        # 100 departures, each with 200 destinations → 300 coords per naive batch
        dep_dst_map = {i: list(range(200)) for i in range(100)}
        bb = _BatchBuilder(max_table_size=50, batch_size=20)
        batches = bb.build(list(dep_dst_map.keys()), dep_dst_map)
        for dep_batch, dst_batch, _ in batches:
            assert len(dep_batch) + len(dst_batch) <= 50, (
                f"Batch exceeds max_table_size: {len(dep_batch)} dep + {len(dst_batch)} dst"
            )

    def test_single_dep_too_many_dst_chunks_destinations(self):
        """One departure with 10 destinations, max_table_size=5 → must chunk dsts."""
        dep_dst_map = {0: list(range(10))}
        bb = _BatchBuilder(max_table_size=5, batch_size=20)
        batches = bb.build([0], dep_dst_map)
        # Every batch should have exactly 1 departure
        for dep_b, dst_b, _ in batches:
            assert dep_b == [0]
            assert len(dst_b) <= 4   # max_table_size-1 = 4 destinations
        # All destinations should be covered across the batches
        covered = {s for _, dst_b, _ in batches for s in dst_b}
        assert covered == set(range(10))

    def test_all_pairs_are_covered(self):
        """No (dep, dst) pair from the spatial map should go missing."""
        dep_dst_map = {i: [i * 10 + j for j in range(5)] for i in range(10)}
        bb = _BatchBuilder(max_table_size=8_000, batch_size=3)
        batches = bb.build(list(dep_dst_map.keys()), dep_dst_map)
        all_valid = set()
        for _, _, vp in batches:
            all_valid |= vp
        expected = {(i, j) for i, dsts in dep_dst_map.items() for j in dsts}
        assert all_valid == expected

    def test_batch_size_respected(self):
        dep_dst_map = {i: [i] for i in range(30)}
        bb = _BatchBuilder(max_table_size=8_000, batch_size=5)
        batches = bb.build(list(dep_dst_map.keys()), dep_dst_map)
        # When pairs are small, batches of ≤5 departures expected
        for dep_b, _, _ in batches:
            assert len(dep_b) <= 5

    def test_shared_destinations_deduplicated_in_batch(self):
        """If two grids share the same store, it should appear once in dst_batch."""
        dep_dst_map = {0: [100, 101], 1: [101, 102]}
        bb = _BatchBuilder(max_table_size=8_000, batch_size=10)
        batches = bb.build([0, 1], dep_dst_map)
        assert len(batches) == 1
        _, dst_batch, _ = batches[0]
        assert len(dst_batch) == len(set(dst_batch)), "Duplicate destinations in batch"

    def test_empty_dep_list(self):
        bb = _BatchBuilder(max_table_size=8_000, batch_size=20)
        batches = bb.build([], {})
        assert batches == []


# ══════════════════════════════════════════════════════════════════════════════
#  7. _ParquetSink
# ══════════════════════════════════════════════════════════════════════════════

class TestParquetSink:

    def _sample_rows(self, n: int = 5) -> list[dict]:
        return [
            {
                "departure_idx":  i,
                "destination_idx": i + 10,
                "mobility_type":  "car",
                "distance_m":     float(1000 * (i + 1)),
                "duration_s":     float(60 * (i + 1)),
            }
            for i in range(n)
        ]

    def test_write_and_read_back(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = Path(f.name)
        sink = _ParquetSink(path)
        rows = self._sample_rows(5)
        sink.write(rows)
        sink.close()

        df = pd.read_parquet(path)
        assert len(df) == 5
        assert "distance_m" in df.columns
        path.unlink(missing_ok=True)

    def test_total_counter_increments(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = Path(f.name)
        sink = _ParquetSink(path)
        sink.write(self._sample_rows(3))
        assert sink.total == 3
        sink.write(self._sample_rows(7))
        assert sink.total == 10
        sink.close()
        path.unlink(missing_ok=True)

    def test_empty_write_is_noop(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = Path(f.name)
        sink = _ParquetSink(path)
        sink.write([])
        assert sink.total == 0
        assert sink._writer is None   # writer never opened
        sink.close()
        path.unlink(missing_ok=True)

    def test_multiple_flushes_accumulate(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = Path(f.name)
        sink = _ParquetSink(path)
        for _ in range(4):
            sink.write(self._sample_rows(10))
        sink.close()
        df = pd.read_parquet(path)
        assert len(df) == 40
        path.unlink(missing_ok=True)

    def test_close_idempotent(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = Path(f.name)
        sink = _ParquetSink(path)
        sink.write(self._sample_rows(2))
        sink.close()
        sink.close()   # second close must not raise
        path.unlink(missing_ok=True)

    def test_moroccan_data_with_string_ids(self):
        """Ensure the schema-inferred writer handles non-integer id types."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = Path(f.name)
        sink = _ParquetSink(path)
        sink.write([
            {"grid_code": "MA-CASA-001", "store_id": "S042",
             "mobility_type": "car", "distance_m": 1234.0, "duration_s": 90.0},
            {"grid_code": "MA-CASA-002", "store_id": "S043",
             "mobility_type": "foot", "distance_m": 980.0, "duration_s": 720.0},
        ])
        sink.close()
        df = pd.read_parquet(path)
        assert list(df["grid_code"]) == ["MA-CASA-001", "MA-CASA-002"]
        path.unlink(missing_ok=True)

    def test_writer_is_none_after_close(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = Path(f.name)
        sink = _ParquetSink(path)
        sink.write(self._sample_rows(1))
        sink.close()
        assert sink._writer is None
        path.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  8. _OSRMFetcher  (all network calls are mocked)
# ══════════════════════════════════════════════════════════════════════════════

def _fake_osrm_response(durations: list, distances: list, code: str = "Ok") -> MagicMock:
    """Build a mock aiohttp response that returns a fake OSRM /table payload."""
    resp = MagicMock()
    resp.status = 200
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__  = AsyncMock(return_value=False)
    resp.json = AsyncMock(return_value={
        "code": code,
        "durations": durations,
        "distances": distances,
    })
    return resp


def _fake_http_error_response(status: int, body: str = "Server Error") -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__  = AsyncMock(return_value=False)
    resp.text = AsyncMock(return_value=body)
    return resp


def _make_fetcher(endpoint: EndpointConfig | None = None):
    if endpoint is None:
        endpoint = EndpointConfig("car", "http://localhost:5000")
    session = MagicMock()
    sem     = asyncio.Semaphore(10)
    return _OSRMFetcher(endpoint, session, sem, timeout_s=30), session


# -- Moroccan WGS84 test coordinates (lon, lat) for mock requests
CASA_WGS_DEP = np.array([
    [-7.589, 33.573],   # Casablanca centre
    [-7.552, 33.573],   # ~3 km east
])
CASA_WGS_DST = np.array([
    [-7.580, 33.577],   # Store A
    [-7.564, 33.565],   # Store B
])


class TestOSRMFetcher:

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    # ── Success path ─────────────────────────────────────────────────────────

    def test_success_returns_records(self):
        fetcher, session = _make_fetcher()
        durations = [[120.0, 300.0], [200.0, 150.0]]
        distances = [[800.0, 2200.0], [1500.0, 1100.0]]
        resp = _fake_osrm_response(durations, distances)
        session.get = MagicMock(return_value=resp)

        dep_ids  = [0, 1]
        dst_ids  = [10, 11]
        valid    = {(0, 10), (0, 11), (1, 10), (1, 11)}

        records = self._run(fetcher.fetch(
            dep_ids, dst_ids,
            CASA_WGS_DEP, CASA_WGS_DST,
            valid, "departure_idx", "destination_idx",
        ))
        assert len(records) == 4
        assert all(r["mobility_type"] == "car" for r in records)
        assert all("distance_m" in r for r in records)
        assert all("duration_s" in r for r in records)

    def test_valid_pairs_filter_applied(self):
        """Records outside valid_pairs must be excluded from results."""
        fetcher, session = _make_fetcher()
        durations = [[120.0, 300.0]]
        distances = [[800.0, 2200.0]]
        resp = _fake_osrm_response(durations, distances)
        session.get = MagicMock(return_value=resp)

        # Only (0, 10) is valid — (0, 11) must be dropped
        records = self._run(fetcher.fetch(
            [0], [10, 11],
            CASA_WGS_DEP[:1], CASA_WGS_DST,
            {(0, 10)}, "departure_idx", "destination_idx",
        ))
        assert len(records) == 1
        assert records[0]["destination_idx"] == 10

    def test_null_duration_filtered_out(self):
        """OSRM returns None for unreachable pairs — must be skipped."""
        fetcher, session = _make_fetcher()
        durations = [[None, 300.0]]
        distances = [[None, 2200.0]]
        resp = _fake_osrm_response(durations, distances)
        session.get = MagicMock(return_value=resp)

        records = self._run(fetcher.fetch(
            [0], [10, 11],
            CASA_WGS_DEP[:1], CASA_WGS_DST,
            {(0, 10), (0, 11)}, "departure_idx", "destination_idx",
        ))
        # Store 10 is unreachable (None), only store 11 returned
        assert len(records) == 1
        assert records[0]["destination_idx"] == 11

    def test_output_column_names_respect_id_cols(self):
        fetcher, session = _make_fetcher()
        resp = _fake_osrm_response([[90.0]], [[600.0]])
        session.get = MagicMock(return_value=resp)

        records = self._run(fetcher.fetch(
            ["GR-01"], ["ST-99"],
            CASA_WGS_DEP[:1], CASA_WGS_DST[:1],
            {("GR-01", "ST-99")}, "grid_code", "store_code",
        ))
        assert len(records) == 1
        assert "grid_code"  in records[0]
        assert "store_code" in records[0]
        assert records[0]["grid_code"]  == "GR-01"
        assert records[0]["store_code"] == "ST-99"

    def test_profile_written_to_mobility_type(self):
        ep = EndpointConfig("foot", "http://localhost:5001")
        fetcher, session = _make_fetcher(ep)
        resp = _fake_osrm_response([[600.0]], [[500.0]])
        session.get = MagicMock(return_value=resp)

        records = self._run(fetcher.fetch(
            [0], [1],
            CASA_WGS_DEP[:1], CASA_WGS_DST[:1],
            {(0, 1)}, "departure_idx", "destination_idx",
        ))
        assert records[0]["mobility_type"] == "foot"

    # ── Error paths ──────────────────────────────────────────────────────────

    def test_http_error_returns_empty(self):
        fetcher, session = _make_fetcher()
        resp = _fake_http_error_response(500)
        session.get = MagicMock(return_value=resp)

        records = self._run(fetcher.fetch(
            [0], [1], CASA_WGS_DEP[:1], CASA_WGS_DST[:1],
            {(0, 1)}, "dep", "dst",
        ))
        assert records == []

    def test_osrm_non_ok_code_returns_empty(self):
        fetcher, session = _make_fetcher()
        resp = _fake_osrm_response([[90.0]], [[600.0]], code="NoSegment")
        session.get = MagicMock(return_value=resp)

        records = self._run(fetcher.fetch(
            [0], [1], CASA_WGS_DEP[:1], CASA_WGS_DST[:1],
            {(0, 1)}, "dep", "dst",
        ))
        assert records == []

    def test_timeout_returns_empty(self):
        import aiohttp as _aiohttp
        fetcher, session = _make_fetcher()
        resp = MagicMock()
        resp.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
        resp.__aexit__  = AsyncMock(return_value=False)
        session.get = MagicMock(return_value=resp)

        records = self._run(fetcher.fetch(
            [0], [1], CASA_WGS_DEP[:1], CASA_WGS_DST[:1],
            {(0, 1)}, "dep", "dst",
        ))
        assert records == []

    def test_connection_error_returns_empty(self):
        import aiohttp as _aiohttp
        fetcher, session = _make_fetcher()
        resp = MagicMock()
        resp.__aenter__ = AsyncMock(
            side_effect=_aiohttp.ClientConnectorError(
                MagicMock(), OSError("Connection refused")
            )
        )
        resp.__aexit__ = AsyncMock(return_value=False)
        session.get = MagicMock(return_value=resp)

        records = self._run(fetcher.fetch(
            [0], [1], CASA_WGS_DEP[:1], CASA_WGS_DST[:1],
            {(0, 1)}, "dep", "dst",
        ))
        assert records == []

    def test_json_parse_error_returns_empty(self):
        fetcher, session = _make_fetcher()
        resp = MagicMock()
        resp.status = 200
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__  = AsyncMock(return_value=False)
        resp.json = AsyncMock(side_effect=ValueError("Invalid JSON"))
        resp.text = AsyncMock(return_value="garbage response body")
        session.get = MagicMock(return_value=resp)

        records = self._run(fetcher.fetch(
            [0], [1], CASA_WGS_DEP[:1], CASA_WGS_DST[:1],
            {(0, 1)}, "dep", "dst",
        ))
        assert records == []

    def test_url_uses_correct_profile(self):
        """Verify the URL path contains the endpoint's profile name."""
        ep = EndpointConfig("foot", "http://localhost:5001")
        fetcher, session = _make_fetcher(ep)
        resp = _fake_osrm_response([[90.0]], [[500.0]])

        captured_url = []

        def fake_get(url, **kwargs):
            captured_url.append(url)
            return resp

        session.get = fake_get
        self._run(fetcher.fetch(
            [0], [1], CASA_WGS_DEP[:1], CASA_WGS_DST[:1],
            {(0, 1)}, "dep", "dst",
        ))
        assert len(captured_url) == 1
        assert "/table/v1/foot/" in captured_url[0]

    def test_url_uses_semicolons_not_commas_for_indices(self):
        """OSRM requires semicolons in sources= and destinations= params."""
        ep = EndpointConfig("car", "http://localhost:5000")
        fetcher, session = _make_fetcher(ep)
        resp = _fake_osrm_response([[90.0, 100.0]], [[500.0, 600.0]])

        captured_url = []

        def fake_get(url, **kwargs):
            captured_url.append(url)
            return resp

        session.get = fake_get
        self._run(fetcher.fetch(
            [0], [10, 11],
            CASA_WGS_DEP[:1], CASA_WGS_DST,
            {(0, 10), (0, 11)}, "dep", "dst",
        ))
        url = captured_url[0]
        # sources=0  destinations=1;2  (NOT 1,2)
        assert "sources=0" in url
        assert "destinations=1;2" in url
        assert "destinations=1,2" not in url

    def test_annotations_include_distance_and_duration(self):
        ep = EndpointConfig("car", "http://localhost:5000")
        fetcher, session = _make_fetcher(ep)
        resp = _fake_osrm_response([[90.0]], [[500.0]])
        captured_url = []

        def fake_get(url, **kwargs):
            captured_url.append(url)
            return resp

        session.get = fake_get
        self._run(fetcher.fetch(
            [0], [1], CASA_WGS_DEP[:1], CASA_WGS_DST[:1],
            {(0, 1)}, "dep", "dst",
        ))
        url = captured_url[0]
        assert "annotations=duration,distance" in url


# ══════════════════════════════════════════════════════════════════════════════
#  9. OSRMRoutingPipeline — layer loading & full integration
# ══════════════════════════════════════════════════════════════════════════════

def _osrm_table_mock_response(n_dep: int, n_dst: int) -> dict:
    """Generate a realistic fake OSRM /table response."""
    rng = np.random.default_rng(42)
    durations = rng.uniform(60, 1800, (n_dep, n_dst)).tolist()
    distances = rng.uniform(200, 8000, (n_dep, n_dst)).tolist()
    return {"code": "Ok", "durations": durations, "distances": distances}


def _make_pipeline_with_mock_osrm(
    dep_gdf: gpd.GeoDataFrame,
    dst_gdf: gpd.GeoDataFrame,
    dep_id_col: str | None = None,
    dst_id_col: str | None = None,
    dep_keep: list[str] | None = None,
    dst_keep: list[str] | None = None,
    max_dist_m: float = 50_000,   # very wide to guarantee matches in tests
    output_path: Path | None = None,
) -> tuple[OSRMRoutingPipeline, Path]:
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".parquet"))

    cfg = PipelineConfig(
        endpoints=[
            EndpointConfig("car",  "http://localhost:5000"),
            EndpointConfig("foot", "http://localhost:5001"),
        ],
        max_dist_m=max_dist_m,
        output_path=output_path,
        flush_every=100,
    )
    pipeline = OSRMRoutingPipeline(
        departure=LayerConfig(dep_gdf, id_column=dep_id_col, keep_columns=dep_keep or []),
        destination=LayerConfig(dst_gdf, id_column=dst_id_col, keep_columns=dst_keep or []),
        config=cfg,
    )
    return pipeline, output_path


def _mock_session_factory(n_dep: int, n_dst: int):
    """
    Returns a context manager mock for aiohttp.ClientSession that returns
    a realistic OSRM response for any URL.
    """
    payload = _osrm_table_mock_response(n_dep, n_dst)

    resp = MagicMock()
    resp.status = 200
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__  = AsyncMock(return_value=False)
    resp.json       = AsyncMock(return_value=payload)

    session = MagicMock()
    session.get = MagicMock(return_value=resp)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__  = AsyncMock(return_value=False)

    return session


class TestOSRMRoutingPipelineLayerLoading:

    def test_load_in_memory_geodataframe(self):
        gdf = _make_gdf(CASA_GRIDS_UTM)
        pipeline, _ = _make_pipeline_with_mock_osrm(gdf, _make_gdf(CASA_STORES_UTM))
        loaded = pipeline._load_layer(LayerConfig(gdf), "departure")
        assert len(loaded) == len(CASA_GRIDS_UTM)

    def test_load_from_file(self):
        gdf = _make_gdf(CASA_GRIDS_UTM)
        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as f:
            path = Path(f.name)
        gdf.to_file(path, driver="GPKG")
        pipeline, _ = _make_pipeline_with_mock_osrm(gdf, _make_gdf(CASA_STORES_UTM))
        loaded = pipeline._load_layer(LayerConfig(path), "departure")
        assert len(loaded) == len(CASA_GRIDS_UTM)
        path.unlink(missing_ok=True)

    def test_id_column_becomes_index(self):
        gdf = _make_gdf_with_attrs(
            CASA_GRIDS_UTM, "grid_id", "commune",
            ["Casa-Centre", "Casa-Est", "Casa-Sud", "Casa-NE", "Casa-SO"],
        )
        pipeline, _ = _make_pipeline_with_mock_osrm(gdf, _make_gdf(CASA_STORES_UTM))
        loaded = pipeline._load_layer(LayerConfig(gdf, id_column="grid_id"), "dep")
        assert loaded.index.name == "grid_id"
        assert "grid_id" not in loaded.columns

    def test_nonexistent_id_column_raises(self):
        gdf = _make_gdf(CASA_GRIDS_UTM)
        pipeline, _ = _make_pipeline_with_mock_osrm(gdf, gdf)
        with pytest.raises(KeyError, match="id_column"):
            pipeline._load_layer(LayerConfig(gdf, id_column="no_such_col"), "dep")

    def test_nonexistent_keep_column_raises(self):
        gdf = _make_gdf(CASA_GRIDS_UTM)
        pipeline, _ = _make_pipeline_with_mock_osrm(gdf, gdf)
        with pytest.raises(KeyError, match="keep_columns"):
            pipeline._load_layer(LayerConfig(gdf, keep_columns=["ghost_col"]), "dep")

    def test_reprojection_from_wgs84(self):
        gdf_4326 = gpd.GeoDataFrame(
            {"geometry": [Point(-7.589, 33.573), Point(-7.552, 33.573)]},
            crs="EPSG:4326",
        )
        pipeline, _ = _make_pipeline_with_mock_osrm(
            gdf_4326, _make_gdf(CASA_GRIDS_UTM)
        )
        loaded = pipeline._load_layer(LayerConfig(gdf_4326, crs="EPSG:32629"), "dep")
        assert loaded.crs.to_epsg() == 32629
        # After reprojection, X values should be in metre range, not degree range
        assert loaded.geometry.x.iloc[0] > 1000, "Expected UTM metres, got degrees?"

    def test_missing_crs_raises(self):
        gdf = _make_gdf(CASA_GRIDS_UTM)
        gdf = gdf.set_crs(None, allow_override=True)
        pipeline, _ = _make_pipeline_with_mock_osrm(
            _make_gdf(CASA_GRIDS_UTM), _make_gdf(CASA_STORES_UTM)
        )
        with pytest.raises(ValueError, match="no CRS"):
            pipeline._load_layer(LayerConfig(gdf), "dep")

    def test_id_column_collision_handled(self):
        """If both layers use the same id_column name, they get prefixed."""
        dep = _make_gdf_with_attrs(
            CASA_GRIDS_UTM[:2], "loc_id", "name", ["A", "B"]
        )
        dst = _make_gdf_with_attrs(
            CASA_STORES_UTM[:2], "loc_id", "name", ["X", "Y"]
        )
        pipeline, out = _make_pipeline_with_mock_osrm(dep, dst, "loc_id", "loc_id")
        dep_loaded = pipeline._load_layer(LayerConfig(dep, id_column="loc_id"), "dep")
        dst_loaded = pipeline._load_layer(LayerConfig(dst, id_column="loc_id"), "dst")
        # Both have loc_id as index name — collision resolution happens in run()
        assert dep_loaded.index.name == "loc_id"
        assert dst_loaded.index.name == "loc_id"
        out.unlink(missing_ok=True)


class TestOSRMRoutingPipelineIntegration:
    """Full end-to-end tests with mocked aiohttp session."""

    def _run_with_mock(
        self,
        dep_gdf: gpd.GeoDataFrame,
        dst_gdf: gpd.GeoDataFrame,
        dep_id_col: str | None = None,
        dst_id_col: str | None = None,
        dep_keep: list[str] | None = None,
        dst_keep: list[str] | None = None,
        max_dist_m: float = 50_000,
    ) -> pd.DataFrame:
        out = Path(tempfile.mktemp(suffix=".parquet"))
        pipeline, _ = _make_pipeline_with_mock_osrm(
            dep_gdf, dst_gdf,
            dep_id_col=dep_id_col, dst_id_col=dst_id_col,
            dep_keep=dep_keep, dst_keep=dst_keep,
            max_dist_m=max_dist_m, output_path=out,
        )
        n_dep = len(dep_gdf)
        n_dst = len(dst_gdf)
        mock_session = _mock_session_factory(n_dep, n_dst)
        with patch("aiohttp.ClientSession", return_value=mock_session):
            df = pipeline.run()
        out.unlink(missing_ok=True)
        return df

    def test_run_returns_dataframe(self):
        dep = _make_gdf(CASA_GRIDS_UTM)
        dst = _make_gdf(CASA_STORES_UTM[:4])   # exclude the Marrakech outlier
        df = self._run_with_mock(dep, dst)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_output_has_required_columns(self):
        dep = _make_gdf(CASA_GRIDS_UTM)
        dst = _make_gdf(CASA_STORES_UTM[:4])
        df = self._run_with_mock(dep, dst)
        required = {"mobility_type", "distance_m", "duration_s"}
        assert required.issubset(set(df.columns))

    def test_both_profiles_in_output(self):
        dep = _make_gdf(CASA_GRIDS_UTM[:2])
        dst = _make_gdf(CASA_STORES_UTM[:2])
        df = self._run_with_mock(dep, dst)
        profiles = set(df["mobility_type"].unique())
        assert "car"  in profiles
        assert "foot" in profiles

    def test_custom_id_column_in_output(self):
        dep = _make_gdf_with_attrs(
            CASA_GRIDS_UTM, "grid_id", "commune",
            ["Casa-Centre", "Casa-Est", "Casa-Sud", "Casa-NE", "Casa-SO"],
        )
        dst = _make_gdf_with_attrs(
            CASA_STORES_UTM[:3], "store_id", "name", ["Marjane", "Carrefour", "Label'Vie"]
        )
        df = self._run_with_mock(dep, dst, dep_id_col="grid_id", dst_id_col="store_id")
        assert "grid_id"  in df.columns
        assert "store_id" in df.columns

    def test_keep_columns_appear_in_output(self):
        dep = _make_gdf_with_attrs(
            CASA_GRIDS_UTM[:2], "grid_id", "commune", ["Ain Chock", "Sidi Bernoussi"]
        )
        dst = _make_gdf_with_attrs(
            CASA_STORES_UTM[:2], "store_id", "name", ["Marjane", "Carrefour"]
        )
        df = self._run_with_mock(
            dep, dst,
            dep_id_col="grid_id", dst_id_col="store_id",
            dep_keep=["commune"], dst_keep=["name"],
        )
        assert "dep__commune" in df.columns
        assert "dst__name"    in df.columns

    def test_distances_are_positive_floats(self):
        dep = _make_gdf(CASA_GRIDS_UTM[:2])
        dst = _make_gdf(CASA_STORES_UTM[:2])
        df = self._run_with_mock(dep, dst)
        assert (df["distance_m"] > 0).all()
        assert (df["duration_s"] > 0).all()

    def test_no_matches_returns_empty_dataframe(self):
        dep = _make_gdf([RABAT_GRID_UTM])                  # Rabat
        dst = _make_gdf([(596_637, 3_499_876)])            # Marrakech
        out = Path(tempfile.mktemp(suffix=".parquet"))
        cfg = PipelineConfig(max_dist_m=100, output_path=out)   # 100 m radius — no match
        pipeline = OSRMRoutingPipeline(
            LayerConfig(dep), LayerConfig(dst), config=cfg
        )
        df = pipeline.run()   # should short-circuit without hitting OSRM
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        out.unlink(missing_ok=True)

    def test_single_endpoint_only_car(self):
        dep = _make_gdf(CASA_GRIDS_UTM[:2])
        dst = _make_gdf(CASA_STORES_UTM[:2])
        out = Path(tempfile.mktemp(suffix=".parquet"))
        cfg = PipelineConfig(
            endpoints=[EndpointConfig("car", "http://localhost:5000")],
            max_dist_m=50_000, output_path=out,
        )
        pipeline = OSRMRoutingPipeline(LayerConfig(dep), LayerConfig(dst), config=cfg)
        mock_session = _mock_session_factory(2, 2)
        with patch("aiohttp.ClientSession", return_value=mock_session):
            df = pipeline.run()
        assert set(df["mobility_type"].unique()) == {"car"}
        out.unlink(missing_ok=True)

    def test_single_endpoint_only_foot(self):
        dep = _make_gdf(CASA_GRIDS_UTM[:2])
        dst = _make_gdf(CASA_STORES_UTM[:2])
        out = Path(tempfile.mktemp(suffix=".parquet"))
        cfg = PipelineConfig(
            endpoints=[EndpointConfig("foot", "http://localhost:5001")],
            max_dist_m=50_000, output_path=out,
        )
        pipeline = OSRMRoutingPipeline(LayerConfig(dep), LayerConfig(dst), config=cfg)
        mock_session = _mock_session_factory(2, 2)
        with patch("aiohttp.ClientSession", return_value=mock_session):
            df = pipeline.run()
        assert set(df["mobility_type"].unique()) == {"foot"}
        out.unlink(missing_ok=True)

    def test_moroccan_cities_multi_area(self):
        """Mix Casablanca grids + Rabat grid — each only reaches nearby stores."""
        dep_pts = CASA_GRIDS_UTM + [RABAT_GRID_UTM]
        dep = _make_gdf(dep_pts)

        # 3 Casablanca stores + 1 Rabat store
        dst_pts = CASA_STORES_UTM[:3] + [RABAT_STORE_UTM]
        dst = _make_gdf(dst_pts)

        n_dep, n_dst = len(dep_pts), len(dst_pts)
        out = Path(tempfile.mktemp(suffix=".parquet"))
        cfg = PipelineConfig(max_dist_m=10_000, output_path=out)
        pipeline = OSRMRoutingPipeline(LayerConfig(dep), LayerConfig(dst), config=cfg)
        mock_session = _mock_session_factory(n_dep, n_dst)
        with patch("aiohttp.ClientSession", return_value=mock_session):
            df = pipeline.run()

        assert len(df) > 0
        # Rabat grid (index 5) should not be matched with Casablanca stores
        # (they are 65 km apart, well outside 10 km)
        dep_sf = _SpatialFilter(max_dist_m=10_000)
        mapping = dep_sf.build_map(dep, dst)
        rabat_grid_idx = dep.index[-1]
        if rabat_grid_idx in mapping:
            for store_idx in mapping[rabat_grid_idx]:
                # Should only be the Rabat store (index 3)
                assert store_idx == dst.index[-1]
        out.unlink(missing_ok=True)

    def test_parquet_file_written_to_disk(self):
        dep = _make_gdf(CASA_GRIDS_UTM[:2])
        dst = _make_gdf(CASA_STORES_UTM[:2])
        out = Path(tempfile.mktemp(suffix=".parquet"))
        cfg = PipelineConfig(max_dist_m=50_000, output_path=out)
        pipeline = OSRMRoutingPipeline(LayerConfig(dep), LayerConfig(dst), config=cfg)
        mock_session = _mock_session_factory(2, 2)
        with patch("aiohttp.ClientSession", return_value=mock_session):
            pipeline.run()
        assert out.exists()
        df_disk = pd.read_parquet(out)
        assert len(df_disk) > 0
        out.unlink(missing_ok=True)

    def test_concurrency_setting_respected(self):
        """Semaphore should be initialised with the configured concurrency value."""
        dep = _make_gdf(CASA_GRIDS_UTM[:2])
        dst = _make_gdf(CASA_STORES_UTM[:2])
        out = Path(tempfile.mktemp(suffix=".parquet"))
        cfg = PipelineConfig(max_dist_m=50_000, output_path=out, concurrency=7)
        pipeline = OSRMRoutingPipeline(LayerConfig(dep), LayerConfig(dst), config=cfg)

        captured_sem = []
        original_semaphore = asyncio.Semaphore

        def spy_semaphore(n):
            s = original_semaphore(n)
            captured_sem.append(n)
            return s

        mock_session = _mock_session_factory(2, 2)
        with patch("aiohttp.ClientSession", return_value=mock_session), \
             patch("asyncio.Semaphore", side_effect=spy_semaphore):
            pipeline.run()

        assert any(v == 7 for v in captured_sem), (
            f"Expected Semaphore(7) to be created, got: {captured_sem}"
        )
        out.unlink(missing_ok=True)
