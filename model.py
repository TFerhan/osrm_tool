"""
osrm_pipeline.py — Portable OSRM table-routing pipeline
=========================================================

Quick-start
-----------
    from osrm_pipeline import OSRMRoutingPipeline, PipelineConfig, LayerConfig

    result = OSRMRoutingPipeline(
        departure=LayerConfig("grids.gpkg"),
        destination=LayerConfig("stores.gpkg", id_column="store_id"),
    ).run()

Full example with every option
-------------------------------
    from osrm_pipeline import (
        OSRMRoutingPipeline, PipelineConfig, LayerConfig, EndpointConfig
    )

    cfg = PipelineConfig(
        endpoints=[
            EndpointConfig("car",  "http://localhost:5000"),
            EndpointConfig("foot", "http://localhost:5001"),
        ],
        max_dist_m=3_000,
        max_table_size=8_000,
        batch_size=25,
        concurrency=4,
        request_timeout=45,
        output_path="my_results.parquet",
        flush_every=5_000,
    )

    dep = LayerConfig(
        path="grids.gpkg",
        id_column="grid_id",         # None → use DataFrame index
        keep_columns=["commune"],     # extra cols passed through to output
        crs="EPSG:32629",
    )
    dst = LayerConfig(
        path="stores.gpkg",
        id_column="store_id",
        keep_columns=["name", "type"],
        layer="stores_layer",         # for multi-layer GPKG
        crs="EPSG:32629",
    )

    pipeline = OSRMRoutingPipeline(dep, dst, config=cfg)
    df = pipeline.run()

Output schema
-------------
    <departure_id_col>   int/str  — departure identifier
    <destination_id_col> int/str  — destination identifier
    mobility_type        str      — endpoint profile name  ("car", "foot", …)
    distance_m           float64
    duration_s           float64
    [extra dep columns]           — from LayerConfig.keep_columns (prefixed dep__)
    [extra dst columns]           — from LayerConfig.keep_columns (prefixed dst__)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Transformer
from scipy.spatial import cKDTree
from tqdm.asyncio import tqdm as atqdm

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EndpointConfig:
    """One OSRM routing endpoint.

    Parameters
    ----------
    profile : Label written into the ``mobility_type`` output column.
    url     : Base URL of the osrm-routed instance (no trailing slash needed).
    """
    profile: str
    url: str

    def __post_init__(self) -> None:
        self.url = self.url.rstrip("/")

    def __repr__(self) -> str:
        return f"EndpointConfig(profile={self.profile!r}, url={self.url!r})"


@dataclass
class LayerConfig:
    """Configuration for one spatial layer (departure **or** destination).

    Parameters
    ----------
    path         : File path *or* an already-loaded GeoDataFrame.
                   Accepts anything ``geopandas.read_file`` understands
                   (GPKG, SHP, GeoJSON, FlatGeobuf, …).
    id_column    : Column whose values are used as the record identifier
                   in the output table.  ``None`` → the DataFrame index.
    keep_columns : Additional columns to carry through to the final output.
                   They will be prefixed with ``dep__`` / ``dst__``.
    layer        : For multi-layer files (e.g. GeoPackage), the layer name
                   or index passed directly to ``geopandas.read_file``.
    crs          : Target projected CRS (metres).  The layer is reprojected
                   automatically if it arrives in a different CRS.
    """
    path: Path | str | gpd.GeoDataFrame
    id_column: str | None = None
    keep_columns: list[str] = field(default_factory=list)
    layer: str | int | None = None
    crs: str = "EPSG:32629"


@dataclass
class PipelineConfig:
    """Pipeline-wide settings.

    Parameters
    ----------
    endpoints      : OSRM endpoints to query.  Default: car on :5000, foot on :5001.
    max_dist_m     : Spatial pre-filter radius in metres.
    max_table_size : Max coordinates per /table request; must match
                     ``--max-table-size`` on osrm-routed.
    batch_size     : Departure points per HTTP batch *before* the recursive
                     size-guard kicks in.
    concurrency    : Maximum simultaneous in-flight HTTP requests.
    request_timeout: Seconds before a request is abandoned.
    output_path    : Parquet file written incrementally.
    flush_every    : Rows buffered in RAM before flushing to disk.
    """
    endpoints: list[EndpointConfig] = field(default_factory=lambda: [
        EndpointConfig("car",  "http://localhost:5000"),
        EndpointConfig("foot", "http://localhost:5001"),
    ])
    max_dist_m:      float = 5_000.0
    max_table_size:  int   = 8_000
    batch_size:      int   = 20
    concurrency:     int   = 2
    request_timeout: int   = 30
    output_path:     Path  = field(default_factory=lambda: Path("routing_output.parquet"))
    flush_every:     int   = 2_000

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

class _CoordTransformer:
    """Cached CRS-to-WGS84 transformer."""

    def __init__(self) -> None:
        self._cache: dict[str, Transformer] = {}

    def get(self, from_crs: str) -> Transformer:
        if from_crs not in self._cache:
            try:
                self._cache[from_crs] = Transformer.from_crs(
                    from_crs, "EPSG:4326", always_xy=True
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot create transformer from {from_crs} → EPSG:4326: {exc}"
                ) from exc
        return self._cache[from_crs]

    def to_wgs84(self, xy: np.ndarray, from_crs: str) -> np.ndarray:
        """(N,2) projected metres → (N,2) [lon, lat] WGS84."""
        t = self.get(from_crs)
        try:
            lons, lats = t.transform(xy[:, 0], xy[:, 1])
            return np.column_stack([lons, lats])
        except Exception as exc:
            raise RuntimeError(f"Coordinate transform failed: {exc}") from exc


class _SpatialFilter:
    """KD-tree spatial pre-filter: departure → nearby destinations."""

    def __init__(self, max_dist_m: float) -> None:
        self.max_dist_m = max_dist_m

    def build_map(
        self,
        dep: gpd.GeoDataFrame,
        dst: gpd.GeoDataFrame,
    ) -> dict[Any, list[Any]]:
        """
        Returns
        -------
        dict mapping each departure index value to a list of destination
        index values within ``max_dist_m``.  Departures with no nearby
        destination are omitted entirely.
        """
        try:
            dep_xy = np.column_stack([dep.geometry.x.values, dep.geometry.y.values])
            dst_xy = np.column_stack([dst.geometry.x.values, dst.geometry.y.values])
        except Exception as exc:
            raise RuntimeError(
                "Failed to extract centroid coordinates — ensure both layers are in "
                f"a projected CRS (metres): {exc}"
            ) from exc

        try:
            tree   = cKDTree(dst_xy)
            nearby = tree.query_ball_point(dep_xy, r=self.max_dist_m, workers=-1)
        except Exception as exc:
            raise RuntimeError(f"KD-tree search failed: {exc}") from exc

        dep_ids = dep.index.tolist()
        dst_ids = dst.index.tolist()

        mapping: dict[Any, list[Any]] = {
            dep_ids[di]: [dst_ids[dl] for dl in dst_locals]
            for di, dst_locals in enumerate(nearby)
            if dst_locals
        }

        total_pairs = sum(len(v) for v in mapping.values())
        log.info(
            "Spatial pre-filter: %d departures have ≥1 destination within %.0f m "
            "| %d total pairs",
            len(mapping), self.max_dist_m, total_pairs,
        )
        print(
            f"Departures with ≥1 destination within {self.max_dist_m/1000:.1f} km : "
            f"{len(mapping):,}\n"
            f"Total (departure, destination) pairs                                  : "
            f"{total_pairs:,}"
        )
        return mapping


class _BatchBuilder:
    """
    Splits the departure list into URL-safe batches.

    Each batch satisfies:
        len(dep_batch) + len(unique_dst_for_batch) ≤ max_table_size

    If a *single* departure has so many nearby destinations that this
    constraint cannot be met, the destinations are chunked instead.
    """

    Batch = tuple[list[Any], list[Any], set[tuple[Any, Any]]]

    def __init__(self, max_table_size: int, batch_size: int) -> None:
        self.max_table_size = max_table_size
        self.batch_size     = batch_size

    def build(
        self,
        dep_ids: list[Any],
        dep_dst_map: dict[Any, list[Any]],
    ) -> list[Batch]:
        batches: list[_BatchBuilder.Batch] = []

        def _add(dep_batch: list[Any]) -> None:
            dst_set: set[Any] = set()
            for d in dep_batch:
                dst_set.update(dep_dst_map[d])
            dst_batch = sorted(dst_set)
            total = len(dep_batch) + len(dst_batch)

            if total > self.max_table_size:
                if len(dep_batch) == 1:
                    # Single departure with too many destinations — chunk dsts
                    dep = dep_batch[0]
                    max_dst = self.max_table_size - 1
                    all_dst = sorted(dep_dst_map[dep])
                    for s in range(0, len(all_dst), max_dst):
                        chunk = all_dst[s : s + max_dst]
                        batches.append(([dep], chunk, {(dep, d) for d in chunk}))
                    return
                mid = len(dep_batch) // 2
                _add(dep_batch[:mid])
                _add(dep_batch[mid:])
                return

            valid = {(dep, dst) for dep in dep_batch for dst in dep_dst_map[dep]}
            batches.append((dep_batch, dst_batch, valid))

        for start in range(0, len(dep_ids), self.batch_size):
            _add(dep_ids[start : start + self.batch_size])

        return batches


class _ParquetSink:
    """Incremental Parquet writer — write schema is inferred from first chunk."""

    def __init__(self, path: Path) -> None:
        self.path    = path
        self._writer: pq.ParquetWriter | None = None
        self.total   = 0

    def write(self, rows: list[dict]) -> None:
        if not rows:
            return
        try:
            chunk = pa.Table.from_pylist(rows)
        except Exception as exc:
            log.warning("Arrow conversion failed (%d rows dropped): %s", len(rows), exc)
            return
        try:
            if self._writer is None:
                self._writer = pq.ParquetWriter(
                    self.path, chunk.schema, compression="snappy"
                )
            self._writer.write_table(chunk)
            self.total += len(rows)
        except Exception as exc:
            log.warning("Parquet write failed (%d rows lost): %s", len(rows), exc)

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception as exc:
                log.warning("Parquet close error: %s", exc)
            finally:
                self._writer = None


class _OSRMFetcher:
    """Async /table fetcher for one endpoint profile."""

    def __init__(
        self,
        endpoint: EndpointConfig,
        session:  aiohttp.ClientSession,
        sem:      asyncio.Semaphore,
        timeout_s: int,
    ) -> None:
        self.endpoint  = endpoint
        self.session   = session
        self.sem       = sem
        self.timeout_s = timeout_s

    # ── URL helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _coord_str(lons: np.ndarray, lats: np.ndarray) -> str:
        """'lon,lat;lon,lat;…' at 5 d.p. precision."""
        return ";".join(f"{lo:.5f},{la:.5f}" for lo, la in zip(lons, lats))

    @staticmethod
    def _idx_str(indices: range | list[int]) -> str:
        return ";".join(map(str, indices))

    # ── Core request ─────────────────────────────────────────────────────────

    async def fetch(
        self,
        dep_ids:     list[Any],
        dst_ids:     list[Any],
        dep_wgs:     np.ndarray,     # (n_dep, 2) [lon, lat]
        dst_wgs:     np.ndarray,     # (n_dst, 2) [lon, lat]
        valid_pairs: set[tuple[Any, Any]],
        dep_id_col:  str,
        dst_id_col:  str,
    ) -> list[dict]:
        n_dep = len(dep_ids)
        n_dst = len(dst_ids)

        all_lons = np.concatenate([dep_wgs[:, 0], dst_wgs[:, 0]])
        all_lats = np.concatenate([dep_wgs[:, 1], dst_wgs[:, 1]])

        coords  = self._coord_str(all_lons, all_lats)
        src_str = self._idx_str(range(n_dep))
        dst_str = self._idx_str(range(n_dep, n_dep + n_dst))

        url = (
            f"{self.endpoint.url}/table/v1/{self.endpoint.profile}/{coords}"
            f"?sources={src_str}"
            f"&destinations={dst_str}"
            f"&annotations=duration,distance"
        )

        data: dict | None = None
        async with self.sem:
            try:
                async with self.session.get(url, allow_redirects=False) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        log.warning(
                            "[%s] HTTP %d | %d×%d | %.200s",
                            self.endpoint.profile, resp.status, n_dep, n_dst, body
                        )
                        return []
                    try:
                        data = await resp.json(content_type=None)
                    except Exception as exc:
                        raw = await resp.text()
                        log.warning(
                            "[%s] JSON parse error: %s | %.200s",
                            self.endpoint.profile, exc, raw
                        )
                        return []
            except asyncio.TimeoutError:
                log.warning(
                    "[%s] Timeout after %ds | %d×%d — try reducing batch_size",
                    self.endpoint.profile, self.timeout_s, n_dep, n_dst,
                )
                return []
            except aiohttp.ClientConnectorError as exc:
                log.warning("[%s] Cannot reach %s: %s", self.endpoint.profile, self.endpoint.url, exc)
                return []
            except Exception as exc:
                log.warning("[%s] Unexpected request error: %s", self.endpoint.profile, exc)
                return []

        if data is None or data.get("code") != "Ok":
            log.warning(
                "[%s] OSRM code=%s message=%s",
                self.endpoint.profile,
                data.get("code") if data else "N/A",
                data.get("message", "n/a") if data else "N/A",
            )
            return []

        durations = data.get("durations") or []
        distances = data.get("distances") or []
        records: list[dict] = []

        for i, dep_id in enumerate(dep_ids):
            dur_row  = durations[i] if i < len(durations) else []
            dist_row = distances[i] if i < len(distances) else []

            for j, dst_id in enumerate(dst_ids):
                if (dep_id, dst_id) not in valid_pairs:
                    continue
                dur  = dur_row[j]  if j < len(dur_row)  else None
                dist = dist_row[j] if j < len(dist_row) else None
                if dur is None or dist is None:
                    continue
                records.append({
                    dep_id_col: dep_id,
                    dst_id_col: dst_id,
                    "mobility_type": self.endpoint.profile,
                    "distance_m":    float(dist),
                    "duration_s":    float(dur),
                })

        return records


# ══════════════════════════════════════════════════════════════════════════════
#  Public pipeline class
# ══════════════════════════════════════════════════════════════════════════════

class OSRMRoutingPipeline:
    """
    End-to-end OSRM /table routing pipeline.

    Parameters
    ----------
    departure   : LayerConfig for the departure layer.
    destination : LayerConfig for the destination layer.
    config      : PipelineConfig with all tuning knobs.
                  If omitted, sensible defaults are used (car:5000, foot:5001,
                  5 km radius, 2 concurrent requests).

    Examples
    --------
    Minimal usage::

        df = OSRMRoutingPipeline(
            LayerConfig("grids.gpkg"),
            LayerConfig("stores.gpkg", id_column="store_id"),
        ).run()

    Custom endpoints + columns::

        cfg = PipelineConfig(
            endpoints=[EndpointConfig("bike", "http://10.0.0.5:5002")],
            max_dist_m=2_000,
            concurrency=6,
        )
        df = OSRMRoutingPipeline(
            LayerConfig("origins.gpkg",  id_column="origin_id",  keep_columns=["zone"]),
            LayerConfig("hospitals.gpkg", id_column="hosp_id",   keep_columns=["name"]),
            config=cfg,
        ).run()
    """

    def __init__(
        self,
        departure:   LayerConfig,
        destination: LayerConfig,
        config:      PipelineConfig | None = None,
    ) -> None:
        self.dep_cfg  = departure
        self.dst_cfg  = destination
        self.cfg      = config or PipelineConfig()

        self._transformer = _CoordTransformer()
        self._sp_filter   = _SpatialFilter(self.cfg.max_dist_m)
        self._batcher     = _BatchBuilder(self.cfg.max_table_size, self.cfg.batch_size)

    # ── Layer loading ─────────────────────────────────────────────────────────

    def _load_layer(self, layer_cfg: LayerConfig, role: str) -> gpd.GeoDataFrame:
        """Load, reproject, and normalise a GeoDataFrame."""
        if isinstance(layer_cfg.path, gpd.GeoDataFrame):
            gdf = layer_cfg.path.copy()
            log.info("[%s] Received in-memory GeoDataFrame (%d rows)", role, len(gdf))
        else:
            path = Path(layer_cfg.path)
            kwargs: dict[str, Any] = {}
            if layer_cfg.layer is not None:
                kwargs["layer"] = layer_cfg.layer
            try:
                gdf = gpd.read_file(path, **kwargs)
                log.info("[%s] Loaded %d rows from %s", role, len(gdf), path)
            except Exception as exc:
                raise RuntimeError(
                    f"[{role}] Cannot read file {path!r}: {exc}"
                ) from exc

        if gdf.crs is None:
            raise ValueError(
                f"[{role}] Layer has no CRS — assign one before passing to the pipeline."
            )

        target_epsg = int(layer_cfg.crs.split(":")[-1])
        if gdf.crs.to_epsg() != target_epsg:
            log.info(
                "[%s] Reprojecting from %s → %s", role, gdf.crs.to_string(), layer_cfg.crs
            )
            try:
                gdf = gdf.to_crs(layer_cfg.crs)
            except Exception as exc:
                raise RuntimeError(f"[{role}] Reprojection failed: {exc}") from exc

        # Promote id_column to index so internal logic always uses .index
        if layer_cfg.id_column is not None:
            if layer_cfg.id_column not in gdf.columns:
                raise KeyError(
                    f"[{role}] id_column={layer_cfg.id_column!r} not found in layer. "
                    f"Available columns: {list(gdf.columns)}"
                )
            gdf = gdf.set_index(layer_cfg.id_column, drop=True)

        # Validate keep_columns
        missing = [c for c in layer_cfg.keep_columns if c not in gdf.columns]
        if missing:
            raise KeyError(
                f"[{role}] keep_columns {missing} not found in layer. "
                f"Available: {list(gdf.columns)}"
            )

        return gdf

    # ── Coordinate prep ───────────────────────────────────────────────────────

    def _prepare_coords(
        self,
        gdf:     gpd.GeoDataFrame,
        id_list: list[Any],
        crs:     str,
    ) -> tuple[np.ndarray, np.ndarray, dict[Any, int]]:
        """
        For the given id_list, extract (N,2) projected XY and (N,2) WGS84,
        plus a position index {id → row_position}.
        """
        sub  = gdf.loc[id_list]
        xy   = np.column_stack([sub.geometry.x.values, sub.geometry.y.values])
        wgs  = self._transformer.to_wgs84(xy, crs)
        pos  = {gid: i for i, gid in enumerate(id_list)}
        return xy, wgs, pos

    # ── Async driver ──────────────────────────────────────────────────────────

    async def _run_async(
        self,
        dep:           gpd.GeoDataFrame,
        dst:           gpd.GeoDataFrame,
        dep_dst_map:   dict[Any, list[Any]],
        dep_id_col:    str,
        dst_id_col:    str,
    ) -> int:
        dep_needed = sorted(dep_dst_map.keys())
        dst_needed = sorted({d for dl in dep_dst_map.values() for d in dl})

        # Pre-compute WGS84 arrays for every point we will query
        try:
            _, dep_wgs, dep_pos = self._prepare_coords(dep, dep_needed, self.dep_cfg.crs)
            _, dst_wgs, dst_pos = self._prepare_coords(dst, dst_needed, self.dst_cfg.crs)
        except Exception as exc:
            raise RuntimeError(f"Coordinate preparation failed: {exc}") from exc

        # Build all batches upfront
        try:
            batches = self._batcher.build(dep_needed, dep_dst_map)
        except Exception as exc:
            raise RuntimeError(f"Batch construction failed: {exc}") from exc

        n_http = len(batches) * len(self.cfg.endpoints)
        print(
            f"\nBatches  : {len(batches):,}"
            f"  |  HTTP requests: {n_http:,} "
            f"({len(self.cfg.endpoints)} endpoint(s) × {len(batches):,} batches)\n"
        )

        connector = aiohttp.TCPConnector(
            limit=self.cfg.concurrency * 2,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=self.cfg.request_timeout)
        sem     = asyncio.Semaphore(self.cfg.concurrency)
        sink    = _ParquetSink(self.cfg.output_path)
        buf: list[dict] = []

        try:
            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout
            ) as session:

                # Build all coroutines for every (batch × endpoint) combination
                coros = []
                for dep_batch, dst_batch, valid_pairs in batches:
                    gp = [dep_pos[g] for g in dep_batch]
                    sp = [dst_pos[s] for s in dst_batch]

                    for endpoint in self.cfg.endpoints:
                        fetcher = _OSRMFetcher(endpoint, session, sem, self.cfg.request_timeout)
                        coros.append(
                            fetcher.fetch(
                                dep_batch, dst_batch,
                                dep_wgs[gp], dst_wgs[sp],
                                valid_pairs,
                                dep_id_col, dst_id_col,
                            )
                        )

                futs = [asyncio.ensure_future(c) for c in coros]

                with atqdm(total=len(futs), desc="OSRM requests", unit="req") as pbar:
                    for fut in asyncio.as_completed(futs):
                        try:
                            records = await fut
                        except Exception as exc:
                            log.warning("Coroutine raised unhandled error: %s", exc)
                            records = []

                        buf.extend(records)
                        pbar.update(1)

                        if len(buf) >= self.cfg.flush_every:
                            sink.write(buf)
                            buf.clear()

        except Exception as exc:
            log.warning("Session-level error: %s — flushing collected records", exc)
        finally:
            sink.write(buf)
            buf.clear()
            sink.close()

        return sink.total

    # ── Post-processing: attach keep_columns ─────────────────────────────────

    def _attach_extra_columns(
        self,
        result:   pd.DataFrame,
        dep:      gpd.GeoDataFrame,
        dst:      gpd.GeoDataFrame,
        dep_id_col: str,
        dst_id_col: str,
    ) -> pd.DataFrame:
        """
        Left-join the user-requested keep_columns back onto the result table.
        Departure columns are prefixed ``dep__``, destination columns ``dst__``.
        """
        if self.dep_cfg.keep_columns:
            dep_extra = (
                dep[self.dep_cfg.keep_columns]
                .rename(columns={c: f"dep__{c}" for c in self.dep_cfg.keep_columns})
                .reset_index()               # brings the index back as a column
                .rename(columns={dep.index.name: dep_id_col})
            )
            result = result.merge(dep_extra, on=dep_id_col, how="left")

        if self.dst_cfg.keep_columns:
            dst_extra = (
                dst[self.dst_cfg.keep_columns]
                .rename(columns={c: f"dst__{c}" for c in self.dst_cfg.keep_columns})
                .reset_index()
                .rename(columns={dst.index.name: dst_id_col})
            )
            result = result.merge(dst_extra, on=dst_id_col, how="left")

        return result

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Execute the full pipeline and return the routing result as a DataFrame.

        The Parquet file at ``config.output_path`` is also written incrementally
        throughout the run, so partial results survive a crash.

        Returns
        -------
        pd.DataFrame with columns:
            <dep_id_col>, <dst_id_col>, mobility_type, distance_m, duration_s,
            [dep__<col> for col in dep LayerConfig.keep_columns],
            [dst__<col> for col in dst LayerConfig.keep_columns]
        """
        t0 = time.time()

        # ── 1. Load layers ──────────────────────────────────────────────────
        print("── Loading layers ─────────────────────────────────────────")
        try:
            dep = self._load_layer(self.dep_cfg,  "departure")
            dst = self._load_layer(self.dst_cfg,  "destination")
        except Exception as exc:
            log.error("Layer loading failed: %s", exc)
            return pd.DataFrame()

        # Determine output column names (index.name after set_index, or default)
        dep_id_col = dep.index.name or "departure_idx"
        dst_id_col = dst.index.name or "destination_idx"
        # Avoid collision if both layers use the same id column name
        if dep_id_col == dst_id_col:
            dep_id_col = f"dep_{dep_id_col}"
            dst_id_col = f"dst_{dst_id_col}"

        print(
            f"  Departure   : {len(dep):,} rows  |  id → '{dep_id_col}'"
            + (f"  |  keep: {self.dep_cfg.keep_columns}" if self.dep_cfg.keep_columns else "")
        )
        print(
            f"  Destination : {len(dst):,} rows  |  id → '{dst_id_col}'"
            + (f"  |  keep: {self.dst_cfg.keep_columns}" if self.dst_cfg.keep_columns else "")
        )

        # ── 2. Spatial pre-filter ───────────────────────────────────────────
        print("\n── Spatial pre-filter ─────────────────────────────────────")
        try:
            dep_dst_map = self._sp_filter.build_map(dep, dst)
        except Exception as exc:
            log.error("Spatial pre-filter failed: %s", exc)
            return pd.DataFrame()

        if not dep_dst_map:
            print(
                "No departures have any destination within the distance threshold.\n"
                f"Check your data and PipelineConfig.max_dist_m ({self.cfg.max_dist_m} m)."
            )
            return pd.DataFrame()

        # ── 3. Async OSRM requests ──────────────────────────────────────────
        print("\n── OSRM routing ───────────────────────────────────────────")
        try:
            total = asyncio.run(
                self._run_async(dep, dst, dep_dst_map, dep_id_col, dst_id_col)
            )
        except Exception as exc:
            log.error("Async pipeline crashed: %s", exc)
            # Try to return whatever made it to disk before the crash
            if self.cfg.output_path.exists():
                print("Attempting to return partial results from disk…")
                try:
                    return pd.read_parquet(self.cfg.output_path)
                except Exception:
                    pass
            return pd.DataFrame()

        elapsed = time.time() - t0
        print(
            f"\n── Done in {elapsed / 60:.1f} min "
            f"| {total:,} records → {self.cfg.output_path}"
        )

        # ── 4. Read back and attach extra columns ───────────────────────────
        try:
            result = pd.read_parquet(self.cfg.output_path)
        except Exception as exc:
            log.warning("Could not read back Parquet file: %s", exc)
            return pd.DataFrame()

        if self.dep_cfg.keep_columns or self.dst_cfg.keep_columns:
            result = self._attach_extra_columns(result, dep, dst, dep_id_col, dst_id_col)

        return result
