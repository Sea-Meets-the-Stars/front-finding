"""Reading LLC4320 fields for ONE tile, from per-tile NetCDFs.

The global path (:mod:`front_finding.llc.source`) reads a whole 12960 x 17280
snapshot out of a zarr store on S3.  A tile build reads a single 720 x 720
block instead, computed by the preprocessing repo's ``generate-tile`` and
written one NetCDF per channel per snapshot.

Everything downstream is shape-agnostic -- ``find``, ``group`` and ``colocate``
take arrays -- so swapping this in for the global reader is the whole of a
tile build.  The products land in the same store, 720 x 720 instead of global.

Files are named by the run, not by ``generate-tile``'s own convention, so a
build can find them without knowing what hour resolves to what filename::

    {source.tile.root}/{YYYYMMDD_HHMMSS}/{channel}_tile{idx:03d}.nc

Which tile is set in the run YAML::

    source:
      tile:
        index: 330            # or i:/j: a rect pixel, or lon:/lat:
        root: "./tiles"
        mask_land: true
"""
import os
from typing import Tuple

import numpy as np
import xarray as xr

#: Rect grid, tile edge, tiles per rect row -- the numbering ``dbof.tiles`` uses.
RECT_SHAPE = (12960, 17280)
TILE = 720
TILES_I = RECT_SHAPE[1] // TILE

#: lat/lon per tile root.  Static across timesteps, so read once per process.
_GRID_CACHE = {}


def resolve_tile(cfg) -> Tuple[int, int, int]:
    """Resolve the configured tile to ``(tile_idx, i_rect, j_rect)``.

    Accepts ``index``, a rect pixel ``i``/``j``, or ``lon``/``lat``.  The
    geographic form is resolved by dbof against the grid store, so it needs
    network; the other two are arithmetic.

    The pixel returned is the tile's ORIGIN, whichever form was given.  dbof
    floors whatever pixel it is handed to the enclosing tile, so this changes
    nothing downstream -- but it means two configs naming the same tile produce
    the same provenance instead of remembering which pixel was typed.

    Args:
        cfg: The resolved run config (BuildJobConfig).
    """
    spec = cfg.source.tile
    if spec is None:
        raise ValueError("no 'source.tile' block in the run config")

    if 'index' in spec:
        idx = int(spec['index'])
        if not 0 <= idx < (RECT_SHAPE[0] // TILE) * TILES_I:
            raise ValueError(f"tile index must be 0..431, got {idx}")
        tj, ti = divmod(idx, TILES_I)
        return idx, ti * TILE, tj * TILE

    if 'i' in spec and 'j' in spec:
        i, j = int(spec['i']), int(spec['j'])
    elif 'lon' in spec and 'lat' in spec:
        from dbof.tiles.tile_utils import latlon_to_rect_ij
        i, j = latlon_to_rect_ij(float(spec['lon']), float(spec['lat']))
    else:
        raise ValueError(
            "'source.tile' needs one of: index, i+j, or lon+lat.  Got "
            f"{sorted(spec)}")

    if not (0 <= j < RECT_SHAPE[0] and 0 <= i < RECT_SHAPE[1]):
        raise ValueError(f"({j}, {i}) is outside the rect grid {RECT_SHAPE}")
    idx = (j // TILE) * TILES_I + (i // TILE)
    return idx, (i // TILE) * TILE, (j // TILE) * TILE


def tile_root(cfg) -> str:
    """Directory the run's tile NetCDFs live in.

    A relative path resolves against the run config, matching ``products.root``.
    """
    root = (cfg.source.tile or {}).get('root', 'tiles')
    if not os.path.isabs(root) and cfg.config_path:
        root = os.path.join(os.path.dirname(os.path.abspath(cfg.config_path)),
                            root)
    return os.path.normpath(root)


def tile_path(cfg, date: str, channel: str) -> str:
    """Path of one channel's tile NetCDF for one snapshot."""
    idx, _, _ = resolve_tile(cfg)
    return os.path.join(tile_root(cfg), date, f"{channel}_tile{idx:03d}.nc")


def available_channels(cfg, timestamp: str) -> set:
    """Channels whose tile NetCDF exists for *timestamp*."""
    date = cfg.date_prefixes[cfg.timestamps.index(timestamp)]
    folder = os.path.join(tile_root(cfg), date)
    if not os.path.isdir(folder):
        return set()
    idx, _, _ = resolve_tile(cfg)
    tail = f"_tile{idx:03d}.nc"
    return {f[:-len(tail)] for f in os.listdir(folder) if f.endswith(tail)}


def read_channel(cfg, timestamp: str, channel: str, subset: str = None,
                 ice_mask: bool = False, **kwargs) -> np.ndarray:
    """Read one channel of one snapshot for the configured tile.

    The signature matches :func:`front_finding.llc.source.read_channel` so the
    pipeline steps do not know which source they are on.  *subset* is accepted
    and ignored: a tile NetCDF holds one channel and names it.

    Args:
        cfg: The resolved run config (BuildJobConfig).
        timestamp (str): Snapshot, ``'YYYY-MM-DDTHH_MM_SS'``.
        channel (str): Channel name, e.g. ``'gradb2'``.
        subset (str): Unused.
        ice_mask (bool): NaN out ice-covered points, from the tile's own
            ``SIarea``.

    Returns:
        np.ndarray: ``(720, 720)``.
    """
    date = cfg.date_prefixes[cfg.timestamps.index(timestamp)]
    path = tile_path(cfg, date, channel)
    if not os.path.isfile(path):
        raise KeyError(f"no tile for {channel} at {date}: {path}")

    with xr.open_dataset(path) as ds:
        name = channel if channel in ds else list(ds.data_vars)[0]
        arr = np.asarray(ds[name].values, dtype=np.float32).squeeze()

    if ice_mask:
        mask_path = tile_path(cfg, date, 'SIarea')
        if not os.path.isfile(mask_path):
            raise KeyError(
                f"ice masking needs an SIarea tile at {date}: {mask_path}.  "
                f"Add 'icearea' to source.active_subsets and re-run the "
                f"gradb2 step.")
        with xr.open_dataset(mask_path) as ds:
            siarea = np.asarray(ds[list(ds.data_vars)[0]].values).squeeze()
        arr = np.where(siarea > 0, np.nan, arr)
    return arr


def read_latlon(cfg) -> Tuple[np.ndarray, np.ndarray]:
    """``(lat, lon)`` for the configured tile, from its own NetCDF coords.

    Every tile file of a tile carries the same ``YC``/``XC``, so the first one
    found serves, and it is cached for the life of the process.
    """
    root = tile_root(cfg)
    if root in _GRID_CACHE:
        return _GRID_CACHE[root]

    for date in cfg.date_prefixes:
        folder = os.path.join(root, date)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith('.nc'):
                continue
            with xr.open_dataset(os.path.join(folder, fname)) as ds:
                if 'YC' not in ds.coords or 'XC' not in ds.coords:
                    continue
                latlon = (np.asarray(ds['YC'].values).squeeze(),
                          np.asarray(ds['XC'].values).squeeze())
            _GRID_CACHE[root] = latlon
            return latlon

    raise FileNotFoundError(
        f"no tile NetCDF with XC/YC under {root}.  Run the 'gradb2' step "
        f"first -- the tile's coordinates come from the files it writes.")


def clear_grid_cache():
    """Drop the cached tile coordinates.  Mostly for tests."""
    _GRID_CACHE.clear()


def generate(cfg, channels, clobber: bool = False) -> list:
    """Compute the tile NetCDFs this run needs, skipping those that exist.

    One ``dbof.tiles.tile_utils.run_series`` call per channel: it resolves the
    tile and downloads the tile grid once, then loops the timestamps.  Paths
    are dictated here rather than taken from ``generate-tile``'s own naming, so
    :func:`tile_path` can find them again.

    Args:
        cfg: The resolved run config (BuildJobConfig).
        channels: Channel names to compute, e.g. ``['gradb2']``.
        clobber (bool): Recompute tiles that already exist.

    Returns:
        list: Paths written or already present, in ``channels`` order.
    """
    from dbof.tiles import tile_utils

    _, i_rect, j_rect = resolve_tile(cfg)
    mask_land = (cfg.source.tile or {}).get('mask_land', True)
    written = []

    for channel in channels:
        paths = [tile_path(cfg, d, channel) for d in cfg.date_prefixes]
        wanted = [(ts, p) for ts, p in zip(cfg.source.date_iterations, paths)
                  if clobber or not os.path.isfile(p)]
        if not wanted:
            print(f"  SKIP (tiles present)  {channel}")
            written.extend(paths)
            continue

        print(f"  GENERATE  {channel}  ({len(wanted)}/{len(paths)} snapshots)")
        for _, path in wanted:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        tile_utils.run_series(
            timestamps=[ts for ts, _ in wanted],
            i_rect=i_rect, j_rect=j_rect,
            property=channel,
            pipeline=cfg.pipeline,
            output_paths=[p for _, p in wanted],
            clobber=clobber,
            mask_land=mask_land,
        )
        written.extend(paths)
    return written
