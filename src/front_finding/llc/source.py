"""Reading LLC4320 fields straight out of the S3 zarr stores.

The fields this package works on are global (j, i) arrays produced by the
preprocessing repo and written to S3 as zarr.  They are read into memory here
and handed to the algorithms as numpy -- there is no intermediate file.

An earlier design exported each channel to a local NetCDF first.  Nothing
consumed those files except the very next step, and ``push`` never shipped
them, so they were ~900 MB of write-then-read per channel for no benefit.

The static grid (lat/lon) lives in its own store, unchanged across timesteps,
so it is fetched once per process and kept in memory.

"""
import numpy as np

from dbof.global_dataset_creation.subset_definitions import get_subset_definition
from dbof.global_dataset_creation.zarr_dataset_global import GlobalZarrDatasetReader
from dbof.global_dataset_creation.zarr_grid_global import GlobalGridZarrReader
from dbof.io.filesystems import create_s3_filesystems
from dbof.preprocessing.ice_mask import apply_ice_mask, load_siarea_mask


#: lat/lon for the run's grid store, keyed by store path.  The grid is static
#: and ~900 MB per variable, so it is fetched once and reused for the rest of
#: the process.  Deliberately in-memory only: nothing is written to disk.
_GRID_CACHE = {}


def _date_prefix(cfg, timestamp: str) -> str:
    """The store directory for *timestamp*, e.g. '20111204_000000'."""
    idx = cfg.timestamps.index(timestamp)
    return cfg.date_prefixes[idx]


def read_channel(cfg, timestamp: str, channel: str, subset: str,
                 ice_mask: bool = False,
                 ice_mask_dataset_name: str = 'icearea.zarr') -> np.ndarray:
    """Read one channel of one snapshot from S3 into memory.

    Parameters
    ----------
    cfg : front_finding.buildconfig.BuildJobConfig
        The resolved run config; locates the store.
    timestamp : str
        Snapshot, ``'YYYY-MM-DDTHH_MM_SS'``.  Must be one of ``cfg.timestamps``.
    channel : str
        Fully-expanded channel name, e.g. ``'gradb2'`` or ``'gradb2_sfc'``.
    subset : str
        The subset owning *channel*.  Passed in rather than looked up here:
        the resolver lives in properties.run, which imports this module.
    ice_mask : bool
        NaN out ice-covered points.  Needs ``icearea.zarr`` for the same run_id
        and date.
    ice_mask_dataset_name : str
        Store holding SIarea.  Only used when *ice_mask* is True.

    Returns
    -------
    np.ndarray
        ``(j, i)`` for the whole globe.
    """
    date_prefix = _date_prefix(cfg, timestamp)
    dataset_name = (cfg.source.dataset_name
                    or get_subset_definition(cfg.pipeline, subset)['dataset_name'])
    fs, _ = create_s3_filesystems(cfg.source.s3_endpoint)

    reader = GlobalZarrDatasetReader(
        bucket=cfg.source.bucket, folder=cfg.source.folder,
        run_id=cfg.run_id, dataset_name=dataset_name, fs=fs,
        date_prefix=date_prefix,
    )
    # Reads one of C chunks rather than the whole snapshot.
    arr = reader.get_channel_snapshot(channel)

    if ice_mask:
        mask = load_siarea_mask(
            bucket=cfg.source.bucket, folder=cfg.source.folder,
            run_id=cfg.run_id, date_prefix=date_prefix, fs=fs,
            dataset_name=ice_mask_dataset_name,
        )
        arr = apply_ice_mask(arr, mask)

    return arr


def read_channel_window(cfg, timestamp: str, channel: str, subset: str,
                        window) -> np.ndarray:
    """Read one channel of one snapshot over a window, straight from S3.

    The store's ``data`` array is chunked, so slicing it touches only the
    overlapping chunks -- a 720 x 720 tile costs a few MB rather than the
    snapshot's ~900 MB.  Ice masking is not applied: the mask is a global
    field, and a windowed read exists to avoid pulling one.

    Args:
        cfg: The resolved run config (BuildJobConfig).
        timestamp (str): Snapshot, ``'YYYY-MM-DDTHH_MM_SS'``.
        channel (str): Fully-expanded channel name.
        subset (str): The subset owning *channel*.
        window: ``(y0, y1, x0, x1)`` on the rect grid.

    Returns:
        np.ndarray: ``(y1 - y0, x1 - x0)``.
    """
    y0, y1, x0, x1 = window
    date_prefix = _date_prefix(cfg, timestamp)
    dataset_name = (cfg.source.dataset_name
                    or get_subset_definition(cfg.pipeline, subset)['dataset_name'])
    fs, _ = create_s3_filesystems(cfg.source.s3_endpoint)

    reader = GlobalZarrDatasetReader(
        bucket=cfg.source.bucket, folder=cfg.source.folder,
        run_id=cfg.run_id, dataset_name=dataset_name, fs=fs,
        date_prefix=date_prefix,
    )
    idx = reader.channel_names.index(channel)
    return np.asarray(reader.data[idx, y0:y1, x0:x1]).squeeze()


def available_channels(cfg, timestamp: str) -> set:
    """Every channel the run's active subsets hold for *timestamp*.

    Reads each store's metadata only -- no field data -- so it is cheap enough
    to call before deciding what to co-locate.
    """
    date_prefix = _date_prefix(cfg, timestamp)
    fs, _ = create_s3_filesystems(cfg.source.s3_endpoint)

    names = set()
    for subset in cfg.source.active_subsets:
        dataset_name = (cfg.source.dataset_name
                        or get_subset_definition(cfg.pipeline, subset)['dataset_name'])
        try:
            reader = GlobalZarrDatasetReader(
                bucket=cfg.source.bucket, folder=cfg.source.folder,
                run_id=cfg.run_id, dataset_name=dataset_name, fs=fs,
                date_prefix=date_prefix,
            )
        except Exception as exc:          # store absent for this subset/date
            print(f"  no store for {subset} at {date_prefix}: {exc}")
            continue
        names.update(reader.channel_names)
    return names


def read_latlon(cfg):
    """Return ``(lat, lon)`` for the global grid, as ``(j, i)`` arrays.

    Fetched from the static grid store and held for the life of the process --
    the grid does not vary with timestep, and a run touches it once per
    snapshot.  Nothing is cached to disk.
    """
    grid = cfg.source.grid
    key = (grid['bucket'], grid['folder'], grid['dataset_name'])
    if key not in _GRID_CACHE:
        fs, _ = create_s3_filesystems(grid['s3_endpoint'])
        reader = GlobalGridZarrReader(
            bucket=grid['bucket'], folder=grid['folder'],
            dataset_name=grid['dataset_name'], fs=fs,
        )
        print(f"Loading grid lat/lon from s3://{grid['bucket']}/"
              f"{grid['folder']}/{grid['dataset_name']}")
        _GRID_CACHE[key] = (reader.lat, reader.lon)
    return _GRID_CACHE[key]


def clear_grid_cache():
    """Drop the cached grid.  Mostly for tests."""
    _GRID_CACHE.clear()
