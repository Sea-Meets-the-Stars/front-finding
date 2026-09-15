"""
Front grouping algorithm.

Core computation for labeling connected front components, computing geometric
properties in parallel, and saving results. Parallel structure to
front_finding.finding.algorithms — file I/O and path setup live in the caller
(build_v1.py); this module handles the pure processing.
"""

import numpy as np
import pandas as pd

from multiprocessing import cpu_count, get_context

from front_finding.properties import group_labels, geometry


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Module-level globals for copy-on-write sharing across forked workers.
# Must be at module level to be picklable by multiprocessing.
# Set in group_fronts() before Pool creation; workers inherit them via fork,
# or receive them through _init_worker where fork is unavailable.
# ---------------------------------------------------------------------------
_GLOBAL_LABELED = None
_GLOBAL_LAT     = None
_GLOBAL_LON     = None


def _init_worker(labeled, lat, lon):
    """Seed the globals in a worker that did not inherit them.

    Only used when the fork start method is unavailable; see _worker_pool.
    """
    global _GLOBAL_LABELED, _GLOBAL_LAT, _GLOBAL_LON
    _GLOBAL_LABELED, _GLOBAL_LAT, _GLOBAL_LON = labeled, lat, lon


def _worker_pool(n_workers, labeled, lat, lon):
    """A Pool whose workers can see the label map and coordinates.

    Prefers fork, which shares those arrays copy-on-write -- at global-grid
    size they are gigabytes, and spawn would pickle a full copy per worker.
    fork is the default on Linux but not on macOS, so it is requested
    explicitly; where it does not exist the arrays are passed to an
    initializer instead, which is correct but pays that copy.
    """
    try:
        ctx = get_context('fork')
        return ctx.Pool(processes=n_workers)
    except ValueError:
        ctx = get_context('spawn')
        return ctx.Pool(processes=n_workers, initializer=_init_worker,
                        initargs=(labeled, lat, lon))


def _process_cutout_wrapper(args_tuple):
    """Multiprocessing worker — extracts cutout and calls process_single_front."""
    label, name, y0, y1, x0, x1, time_str, skip_curvature = args_tuple
    labeled_cutout = _GLOBAL_LABELED[y0:y1, x0:x1]
    mask = labeled_cutout == label
    return geometry.process_single_front(
        label=label, name=name,
        mask=mask,
        lat=_GLOBAL_LAT[y0:y1, x0:x1],
        lon=_GLOBAL_LON[y0:y1, x0:x1],
        time_str=time_str,
        y0=y0, y1=y1, x0=x0, x1=x1,
        skip_curvature=skip_curvature,
    )


def group_fronts(
    fronts_binary: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    timestamp: str,
    n_workers: int = None,
    skip_curvature: bool = False,
):
    """Label connected front components and measure each one, in parallel.

    Computes only; the caller persists what comes back.

    Parameters
    ----------
    fronts_binary : np.ndarray
        2D binary front field (True/1 = front pixel).
    lat, lon : np.ndarray
        2D coordinate grids, same shape as fronts_binary.
    timestamp : str
        Snapshot timestamp, ``YYYY-MM-DDTHH_MM_SS``.  Goes into each front's
        ID and its ``time`` column.
    n_workers : int, optional
        Parallel workers. Defaults to CPU count.
    skip_curvature : bool, optional
        Skip curvature calculation (~50% faster). Default False.

    Returns
    -------
    labeled : np.ndarray
        Integer label array, same shape as *fronts_binary*; 0 is background.
    df : pd.DataFrame
        One row per front: label, name, time, npix, bbox, centroid, length_km,
        orientation, num_branches, curvature.
    """
    global _GLOBAL_LABELED, _GLOBAL_LAT, _GLOBAL_LON

    n_workers = n_workers or cpu_count()

    labeled, n = group_labels.label_fronts(fronts_binary, connectivity=2,
                                           return_num=True)
    print(f"Labeled {n:,} fronts")

    properties = group_labels.get_front_properties(labeled)
    front_ids = group_labels.generate_front_ids(lat, lon, timestamp,
                                                properties=properties)

    # bbox per front, so each worker slices only its own window
    index = pd.DataFrame([
        {'label': int(lbl), 'name': name,
         'y0': int(properties[lbl]['bbox'][0]), 'x0': int(properties[lbl]['bbox'][1]),
         'y1': int(properties[lbl]['bbox'][2]), 'x1': int(properties[lbl]['bbox'][3])}
        for lbl, name in front_ids.items() if lbl in properties
    ])

    _GLOBAL_LABELED = labeled
    _GLOBAL_LAT     = lat
    _GLOBAL_LON     = lon

    time_str = timestamp.replace('_', ':')
    front_args = [
        (row.label, row.name, row.y0, row.y1, row.x0, row.x1, time_str,
         skip_curvature)
        for row in index.itertuples()
    ]
    chunksize = max(100, len(front_args) // (n_workers * 10))

    with _worker_pool(n_workers, labeled, lat, lon) as pool:
        results = [
            r for r in pool.imap_unordered(
                _process_cutout_wrapper, front_args, chunksize=chunksize)
            if r is not None
        ]
    print(f"Processed {len(results):,} fronts")

    df = pd.DataFrame(results)
    col_order = ['label', 'name', 'time', 'npix',
                 'y0', 'y1', 'x0', 'x1',
                 'centroid_lat', 'centroid_lon',
                 'length_km', 'orientation', 'num_branches',
                 'lat_min', 'lat_max', 'lon_min', 'lon_max',
                 'mean_curvature', 'curvature_direction']
    df = df[[c for c in col_order if c in df.columns]]

    return labeled, df
