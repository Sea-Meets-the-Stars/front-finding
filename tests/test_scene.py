"""Unit tests for cropping a build to one tile.

A scene is a window read plus a table subset, so what these pin is that the
two agree: every label in the cropped raster has a row, and no row belongs to a
front outside the window.
"""
import argparse
import textwrap
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from front_finding import buildconfig
from front_finding.cli import scene
from front_finding.llc import publish as llc_publish
from front_finding.finding import run as finding_run
from front_finding.llc import source as llc_source
from front_finding.properties import run as prun
from front_finding.store import FrontStore

N = 192
WINDOW = (40, 140, 40, 140)
DATE = '20111204_000000'

_YAML = """
source:
  pipeline: "SURF"
  run:
    run_id: "itest"
  data:
    date_iterations:
      - '2011-12-04 00:00:00'
  active_subsets: [frontal_structure]
products:
  root: "products"
finding:
  build_version: "STEST"
  config: "D"
"""


# -- tile windows -----------------------------------------------------------

def test_tile_330_is_the_monterey_bay_block():
    # Pinned because a config in the preprocessing repo names this tile.
    assert scene.tile_window(330) == (9360, 10080, 12960, 13680)


@pytest.mark.parametrize("idx", [0, 1, 23, 24, 330, scene.N_TILES - 1])
def test_every_tile_is_720_square_and_inside_the_grid(idx):
    y0, y1, x0, x1 = scene.tile_window(idx)
    assert (y1 - y0, x1 - x0) == (scene.TILE, scene.TILE)
    assert 0 <= y0 < y1 <= scene.RECT_SHAPE[0]
    assert 0 <= x0 < x1 <= scene.RECT_SHAPE[1]


def test_tiles_tile_the_grid_without_gaps():
    starts = sorted(scene.tile_window(i)[0::2] for i in range(scene.N_TILES))
    assert len(set(map(tuple, starts))) == scene.N_TILES


def test_an_out_of_range_tile_raises():
    with pytest.raises(ValueError):
        scene.tile_window(scene.N_TILES)


# -- cropping ---------------------------------------------------------------

def _gradb2(n=N):
    """Three gaussian ridges, so config D has a maximum to sharpen onto."""
    f = np.full((n, n), 1e-3, dtype=np.float32)
    x = np.arange(n)
    for col in (n // 4, n // 2, 3 * n // 4):
        profile = np.exp(-0.5 * ((x - col) / 1.2) ** 2).astype(np.float32)
        f[10:n - 10, :] = np.maximum(f[10:n - 10, :], profile[None, :])
    return f


@pytest.fixture
def cfg(tmp_path):
    """The run config a scene is cut from."""
    path = tmp_path / 'run.yaml'
    path.write_text(textwrap.dedent(_YAML))
    return buildconfig.load_config(str(path))


@pytest.fixture
def built(cfg, tmp_path, monkeypatch):
    """A real store with find, group and colocate done; S3 reads stubbed."""
    gradb2 = _gradb2()
    lat = np.repeat(np.linspace(-60, 60, N)[:, None], N, axis=1)
    lon = np.repeat(np.linspace(-180, 180, N)[None, :], N, axis=0)

    for mod in (llc_source, finding_run.llc_source, prun.llc_source):
        monkeypatch.setattr(mod, 'read_channel', lambda *a, **k: gradb2)
        monkeypatch.setattr(mod, 'read_latlon', lambda cfg_: (lat, lon))
        monkeypatch.setattr(mod, 'available_channels',
                            lambda cfg_, ts: {'gradb2'})
    monkeypatch.setattr(
        scene.llc_source, 'read_channel_window',
        lambda cfg_, ts, ch, sub, w: gradb2[w[0]:w[1], w[2]:w[3]])

    store = FrontStore.open(cfg.store_url, mode='w')
    ts = cfg.timestamps[0]
    finding_run.find_gradb2_fronts(cfg, store, ts, DATE, 'D',
                                   gradb2_field='gradb2',
                                   gradb2_subset='frontal_structure')
    prun.group_fronts(cfg, store, ts, DATE)
    prun.colocate_fronts(cfg, store, ts, DATE, property_names=['gradb2'],
                         stats=['mean', 'std'])
    return cfg, store, tmp_path


@pytest.fixture
def ds(built):
    cfg, store, _ = built
    return scene.crop(cfg, DATE, WINDOW, store=store)


def test_rasters_are_the_window(ds, built):
    _cfg, store, _ = built
    ny, nx = WINDOW[1] - WINDOW[0], WINDOW[3] - WINDOW[2]
    for name in ('gradb2', 'binary', 'labels'):
        assert ds[name].shape == (ny, nx), name
    np.testing.assert_array_equal(
        ds['labels'].values,
        np.asarray(store.labels(DATE))[WINDOW[0]:WINDOW[1],
                                       WINDOW[2]:WINDOW[3]])


def test_labels_keep_their_global_values(ds, built):
    _cfg, store, _ = built
    present = np.unique(ds['labels'].values)
    present = set(present[present > 0])
    assert present <= set(store.geometry(DATE)['label'])


def test_the_table_holds_exactly_the_fronts_in_the_window(ds):
    present = np.unique(ds['labels'].values)
    assert set(present[present > 0]) == set(ds['label'].values)
    assert ds.attrs['n_fronts'] == ds.sizes['front']


def test_colocated_properties_come_along(ds):
    assert 'gradb2_mean' in ds and 'gradb2_std' in ds
    assert np.isfinite(ds['gradb2_mean'].values).all()


def test_geometry_comes_along(ds):
    for col in ('length_km', 'orientation', 'centroid_lat', 'npix'):
        assert col in ds, col


def test_the_binary_map_matches_the_labels(ds):
    np.testing.assert_array_equal(ds['binary'].values.astype(bool),
                                  ds['labels'].values > 0)


def test_the_window_is_recorded(ds):
    assert tuple(ds.attrs['window']) == WINDOW
    assert ds.attrs['rect_j_start'] == WINDOW[0]
    assert ds.attrs['date_prefix'] == DATE
    assert ds.attrs['finding_config'] == 'D'


def test_an_empty_window_gives_no_fronts(built):
    cfg, store, _ = built
    empty = scene.crop(cfg, DATE, (0, 8, 0, 8), store=store)
    assert empty.sizes['front'] == 0


def test_a_snapshot_without_a_label_map_is_refused(built, tmp_path):
    cfg, _store, _ = built
    fresh = FrontStore.open(str(tmp_path / 'other.zarr'), mode='w')
    fresh.write_binary(DATE, np.zeros((N, N), dtype=bool))
    with pytest.raises(RuntimeError, match="group"):
        scene.crop(cfg, DATE, WINDOW, store=fresh)


def test_it_round_trips_through_netcdf(ds, tmp_path):
    path = tmp_path / 'scene.nc'
    ds.to_netcdf(path)
    reread = xr.open_dataset(path)
    np.testing.assert_array_equal(reread['labels'].values, ds['labels'].values)
    assert reread.sizes['front'] == ds.sizes['front']


# -- where it writes ----------------------------------------------------------

def test_the_scene_lands_beside_the_store_by_default(built, monkeypatch):
    cfg, store, tmp_path = built
    monkeypatch.setattr(scene.FrontStore, 'open',
                        classmethod(lambda cls, url, **kw: store))
    monkeypatch.setattr(scene.buildconfig, 'load_config', lambda *a, **k: cfg)
    scene.main(['--config', 'ignored.yaml', '--window', '40', '140', '40', '140',
                '--npy', '--no-push'])
    out = Path(cfg.products_root) / 'scenes' / f'fronts_j40-140_i40-140_{DATE}.nc'
    assert out.is_file()
    assert out.with_name(out.stem + '_labels.npy').is_file()


# -- publishing ---------------------------------------------------------------

def test_scenes_publish_beside_the_store(cfg):
    """Same bucket and run as the source fields, under the build's own key."""
    store = llc_publish.store_s3_prefix(cfg)
    scenes = llc_publish.scene_s3_prefix(cfg)
    assert store.endswith('/STEST/SURF/fronts.zarr')
    assert scenes.endswith('/STEST/SURF/scenes')
    assert store.rsplit('/', 1)[0] == scenes.rsplit('/', 1)[0]


def test_a_dry_run_names_the_keys_without_uploading(cfg, tmp_path):
    local = tmp_path / 'fronts_tile330_20111204_000000.nc'
    local.write_bytes(b'not really a netcdf')
    uris = llc_publish.push_files(cfg, [str(local)],
                                  llc_publish.scene_s3_prefix(cfg),
                                  dry_run=True)
    assert uris == [f"s3://{llc_publish.scene_s3_prefix(cfg)}/{local.name}"]


def _run_scene(monkeypatch, cfg, store, argv):
    """Run the CLI with the store stubbed, capturing what it would upload."""
    monkeypatch.setattr(scene.FrontStore, 'open',
                        classmethod(lambda cls, url, **kw: store))
    monkeypatch.setattr(scene.buildconfig, 'load_config', lambda *a, **k: cfg)
    pushed = {}
    monkeypatch.setattr(llc_publish, 'push_files',
                        lambda cfg_, paths, prefix, **kw: pushed.update(
                            paths=list(paths), prefix=prefix) or [])
    scene.main(argv)
    return pushed


def test_a_scene_publishes_itself(built, monkeypatch):
    """No flag needed -- the scene lands on S3 beside the build."""
    cfg, store, _ = built
    pushed = _run_scene(monkeypatch, cfg, store,
                        ['--config', 'x.yaml', '--window', '40', '140',
                         '40', '140', '--npy'])
    assert pushed['prefix'] == llc_publish.scene_s3_prefix(cfg)
    assert len(pushed['paths']) == 2                      # the .nc and the .npy
    assert pushed['paths'][0].endswith('.nc')
    assert pushed['paths'][1].endswith('_labels.npy')


def test_no_push_keeps_it_local(built, monkeypatch):
    cfg, store, _ = built
    pushed = _run_scene(monkeypatch, cfg, store,
                        ['--config', 'x.yaml', '--window', '40', '140',
                         '40', '140', '--no-push'])
    assert pushed == {}


# -- CLI --------------------------------------------------------------------

def test_a_tile_and_a_window_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        scene.main(['--config', 'x.yaml', '--tile', '330',
                    '--window', '0', '1', '0', '1'])
    with pytest.raises(SystemExit):
        scene.main(['--config', 'x.yaml'])


def test_the_cli_parses_a_window():
    args = scene.parse_args(['--config', 'x.yaml',
                             '--window', '0', '720', '0', '720'])
    assert args.window == [0, 720, 0, 720] and args.tile is None
