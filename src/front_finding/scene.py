"""Crop a global build down to one tile.

A build's products are global: a 12960 x 17280 label map and ~117k fronts.  A
scene is one 720 x 720 tile of that -- the gradb2 field, the front map, and the
rows of the store's tables belonging to the fronts inside it.  Nothing is
recomputed; this is a window read and a table subset.

::

    fronts-scene --config configs/run/small_fronts_dataset_00.yaml --tile 330

Tiles are the 18 x 24 grid of 720 x 720 blocks the rect grid divides into,
indexed row-major -- the same numbering ``dbof.tiles`` uses, so tile 330 here
and ``generate-tile --lon -121.9 --lat 36.8`` cover the same pixels.

Labels keep their global values, so a front in a scene is the same front in the
build.  Fronts crossing the tile edge are cropped: their pixels stop at the
boundary while their table rows still describe the whole front.
"""
import argparse
import os
from typing import List, Tuple

import numpy as np
import xarray as xr

from front_finding import buildconfig
from front_finding.llc import source as llc_source
from front_finding.properties.run import channel_for_root, subset_for_channel
from front_finding.store import FrontStore

#: Rect grid, (rows, cols).
RECT_SHAPE = (12960, 17280)

#: Tile edge in pixels, and tiles per rect row.
TILE = 720
TILES_I = RECT_SHAPE[1] // TILE          # 24

#: Tiles in the rect grid.
N_TILES = (RECT_SHAPE[0] // TILE) * TILES_I      # 432


def tile_window(tile_idx: int) -> Tuple[int, int, int, int]:
    """Rect-grid window of a tile, ``(y0, y1, x0, x1)``.

    Args:
        tile_idx: Flat row-major tile index, 0..431.
    """
    if not 0 <= tile_idx < N_TILES:
        raise ValueError(f"tile_idx must be 0..{N_TILES - 1}, got {tile_idx}")
    tj, ti = divmod(int(tile_idx), TILES_I)
    return (tj * TILE, (tj + 1) * TILE, ti * TILE, (ti + 1) * TILE)


def crop(cfg, date: str, window: Tuple[int, int, int, int],
         store: FrontStore = None) -> xr.Dataset:
    """Crop one snapshot of a build to *window*.

    Args:
        cfg: The resolved run config (BuildJobConfig); locates the store and
            the source field.
        date: Snapshot group, ``YYYYMMDD_HHMMSS``.
        window: ``(y0, y1, x0, x1)`` on the rect grid.
        store: The store to read.  Defaults to the config's own.

    Returns:
        xr.Dataset: ``gradb2``, ``binary`` and ``labels`` on ``(j, i)``, plus
        one ``(front,)`` array per column of ``store.fronts`` for the fronts
        present in the window.  Label values are the build's.
    """
    y0, y1, x0, x1 = window
    store = store or FrontStore.open(cfg.store_url)
    if not store.has(date, 'group'):
        raise RuntimeError(f"[{date}] has no label map in {store.url}.  "
                           f"Run the 'group' step first.")
    timestamp = cfg.timestamps[cfg.date_prefixes.index(date)]

    labels = np.asarray(store.labels(date, window=window))
    binary = np.asarray(store.binary(date, window=window))

    channel = channel_for_root(cfg, cfg.finding.gradb2_root,
                               depth_suffix=cfg.finding.suffix)
    gradb2 = llc_source.read_channel_window(cfg, timestamp, channel,
                                            subset_for_channel(cfg, channel),
                                            window)

    present = np.unique(labels[labels > 0])
    fronts = store.fronts(date)
    fronts = fronts[fronts['label'].isin(present)].reset_index(drop=True)

    data = {'gradb2': (('j', 'i'), np.asarray(gradb2, dtype=np.float32)),
            'binary': (('j', 'i'), binary.astype(np.int8)),
            'labels': (('j', 'i'), labels.astype(np.int32))}
    for col in fronts.columns:
        values = fronts[col].to_numpy()
        if values.dtype == object:
            values = values.astype(str)
        elif values.dtype == np.float64:
            values = values.astype(np.float32)
        elif values.dtype == np.int64:
            values = values.astype(np.int32)
        data[col] = (('front',), values)

    return xr.Dataset(data, attrs={
        'date_prefix': date,
        'timestamp': timestamp.replace('_', ':'),
        'window': list(window),
        'rect_j_start': y0, 'rect_i_start': x0,
        'n_fronts': int(len(fronts)),
        'gradb2_channel': channel,
        'source_store': store.url,
        'build_version': cfg.finding.build_version,
        'finding_config': cfg.finding.config,
    })


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog='fronts-scene',
        description='Crop a global front build down to one tile.')
    p.add_argument('--config', required=True, help='The run YAML of the build.')
    p.add_argument('--date', default=None,
                   help='Snapshot, YYYYMMDD_HHMMSS.  Default: the first.')
    p.add_argument('--tile', type=int, default=None, help='Tile index, 0..431.')
    p.add_argument('--window', type=int, nargs=4, default=None,
                   metavar=('Y0', 'Y1', 'X0', 'X1'),
                   help='Explicit rect-grid window instead of --tile.')
    p.add_argument('--output', default='.',
                   help='Output file, or a directory to name one in.')
    p.add_argument('--npy', action='store_true',
                   help='Also write the label map on its own, as .npy.')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if (args.tile is None) == (args.window is None):
        raise SystemExit("pass exactly one of --tile or --window")
    window = tile_window(args.tile) if args.tile is not None else tuple(args.window)

    cfg = buildconfig.load_config(args.config)
    store = FrontStore.open(cfg.store_url)
    date = args.date or store.dates[0]
    ds = crop(cfg, date, window, store=store)
    print(f"{ds.attrs['n_fronts']} fronts in {window} of {store.url}")

    out = args.output
    if os.path.isdir(out):
        name = (f"tile{args.tile:03d}" if args.tile is not None
                else "j{}-{}_i{}-{}".format(*window))
        out = os.path.join(out, f"fronts_{name}_{date}.nc")
    ds.to_netcdf(out)
    print(f"wrote {out}")

    if args.npy:
        npy = os.path.splitext(out)[0] + '_labels.npy'
        np.save(npy, ds['labels'].values)
        print(f"wrote {npy}")


if __name__ == '__main__':
    main()
