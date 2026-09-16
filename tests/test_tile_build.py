"""A build that runs on one tile instead of the globe.

The tile path is a source swap: fields come from per-tile NetCDFs rather than
the global zarr stores, and everything below -- find, group, colocate, the
store -- is the same code on smaller arrays.  These tests pin the swap (that
the readers dispatch, and that the naming round-trips) and then run the real
three stages end to end on a synthetic tile directory.
"""
import os
import textwrap

import numpy as np
import pytest
import xarray as xr
import yaml

from front_finding import buildconfig
from front_finding.cli import build_fronts
from front_finding.llc import source as llc_source
from front_finding.llc import tiles as tile_source
from front_finding.store import FrontStore

N = 96                      # a stand-in for 720, so the stages run fast
DATES = ('20120629_000000', '20120629_010000')
TIMESTAMPS = ('2012-06-29T00_00_00', '2012-06-29T01_00_00')

_YAML = """
source:
  pipeline: "OSN"
  run:
    run_id: "tile330"
  tile:
    index: 330
    root: "tiles"
  active_subsets: [frontal_structure]
  data:
    date_iterations:
      - '2012-06-29 00:00:00'
      - '2012-06-29 01:00:00'
products:
  root: "output"
finding:
  build_version: "TILE330"
  config: "D"
"""


def _gradb2(n=N):
    """Three gaussian ridges, so config D has a maximum to sharpen onto."""
    f = np.full((n, n), 1e-3, dtype=np.float32)
    x = np.arange(n)
    for col in (n // 4, n // 2, 3 * n // 4):
        profile = np.exp(-0.5 * ((x - col) / 1.2) ** 2).astype(np.float32)
        f[6:n - 6, :] = np.maximum(f[6:n - 6, :], profile[None, :])
    return f


def _write_tile(path, name, values):
    """A NetCDF shaped like one generate-tile writes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lat = np.repeat(np.linspace(36.0, 37.0, values.shape[0])[:, None],
                    values.shape[1], axis=1)
    lon = np.repeat(np.linspace(-123.0, -122.0, values.shape[1])[None, :],
                    values.shape[0], axis=0)
    xr.Dataset({name: (('j', 'i'), values)},
               coords={'YC': (('j', 'i'), lat), 'XC': (('j', 'i'), lon),
                       'tile_index': 330, 'face_index': 10}).to_netcdf(path)


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / 'run.yaml'
    path.write_text(textwrap.dedent(_YAML))
    tile_source.clear_grid_cache()
    return buildconfig.load_config(str(path))


@pytest.fixture
def tiled(cfg):
    """A config whose gradb2 tiles exist on disk."""
    for date in DATES:
        _write_tile(tile_source.tile_path(cfg, date, 'gradb2'), 'gradb2',
                    _gradb2())
    return cfg


# -- the tile is resolved -----------------------------------------------------

def test_a_tile_config_is_in_tile_mode(cfg):
    assert cfg.is_tile is True
    assert tile_source.resolve_tile(cfg)[0] == 330


def test_a_global_config_is_not(tmp_path):
    path = tmp_path / 'g.yaml'
    path.write_text(textwrap.dedent(_YAML).replace(
        "  tile:\n    index: 330\n    root: \"tiles\"\n", ""))
    assert buildconfig.load_config(str(path)).is_tile is False


def test_a_rect_pixel_resolves_to_its_tile(cfg):
    cfg.source.raw['tile'] = {'i': 13320, 'j': 9720}
    # Any pixel inside the tile, floored to the tile's origin.
    assert tile_source.resolve_tile(cfg) == (330, 12960, 9360)


def test_an_index_resolves_to_the_same_pixel(cfg):
    assert tile_source.resolve_tile(cfg) == (330, 12960, 9360)


def test_an_unusable_tile_block_raises(cfg):
    cfg.source.raw['tile'] = {'root': 'tiles'}
    with pytest.raises(ValueError, match="index, i\\+j, or lon\\+lat"):
        tile_source.resolve_tile(cfg)
    cfg.source.raw['tile'] = {'index': 999}
    with pytest.raises(ValueError, match="0..431"):
        tile_source.resolve_tile(cfg)


def test_tile_paths_are_named_by_channel_and_snapshot(cfg):
    path = tile_source.tile_path(cfg, DATES[0], 'gradb2')
    assert path.endswith(os.path.join('tiles', DATES[0], 'gradb2_tile330.nc'))
    assert os.path.isabs(path)          # resolved against the config


# -- reading ------------------------------------------------------------------

def test_a_channel_is_read_from_its_tile(tiled):
    arr = tile_source.read_channel(tiled, TIMESTAMPS[0], 'gradb2')
    assert arr.shape == (N, N)
    np.testing.assert_allclose(arr, _gradb2(), rtol=1e-6)


def test_an_absent_channel_says_which_file_is_missing(tiled):
    with pytest.raises(KeyError, match="density"):
        tile_source.read_channel(tiled, TIMESTAMPS[0], 'density')


def test_available_channels_lists_what_is_on_disk(tiled):
    assert tile_source.available_channels(tiled, TIMESTAMPS[0]) == {'gradb2'}
    _write_tile(tile_source.tile_path(tiled, DATES[0], 'density'), 'density',
                np.ones((N, N), dtype=np.float32))
    assert tile_source.available_channels(tiled, TIMESTAMPS[0]) == {
        'gradb2', 'density'}


def test_coordinates_come_from_the_tile_itself(tiled):
    lat, lon = tile_source.read_latlon(tiled)
    assert lat.shape == lon.shape == (N, N)
    assert 36.0 <= lat.min() and lat.max() <= 37.0
    assert -123.0 <= lon.min() and lon.max() <= -122.0


def test_coordinates_are_refused_before_any_tile_exists(cfg):
    with pytest.raises(FileNotFoundError, match="gradb2"):
        tile_source.read_latlon(cfg)


def test_the_ice_mask_needs_its_own_tile(tiled):
    with pytest.raises(KeyError, match="SIarea"):
        tile_source.read_channel(tiled, TIMESTAMPS[0], 'gradb2', ice_mask=True)
    _write_tile(tile_source.tile_path(tiled, DATES[0], 'SIarea'), 'SIarea',
                np.tile([1.0] * (N // 2) + [0.0] * (N // 2), (N, 1)
                        ).astype(np.float32))
    arr = tile_source.read_channel(tiled, TIMESTAMPS[0], 'gradb2',
                                   ice_mask=True)
    assert np.isnan(arr[:, :N // 2]).all()
    assert np.isfinite(arr[:, N // 2:]).all()


# -- the readers dispatch -----------------------------------------------------

def test_the_global_reader_hands_a_tile_run_over(tiled):
    arr = llc_source.read_channel(tiled, TIMESTAMPS[0], 'gradb2',
                                  'frontal_structure')
    np.testing.assert_allclose(arr, _gradb2(), rtol=1e-6)
    assert llc_source.available_channels(tiled, TIMESTAMPS[0]) == {'gradb2'}
    assert llc_source.read_latlon(tiled)[0].shape == (N, N)



# -- generation ---------------------------------------------------------------

def test_dbof_provides_the_series_entry_point():
    """The call generate() makes.  Pins the contract with the tiles pipeline."""
    from dbof.tiles import tile_utils
    if not hasattr(tile_utils, 'run_series'):
        pytest.skip("this dbof predates tile_utils.run_series")
    import inspect
    params = inspect.signature(tile_utils.run_series).parameters
    for name in ('timestamps', 'i_rect', 'j_rect', 'property', 'pipeline',
                 'output_paths', 'clobber', 'mask_land'):
        assert name in params, name



def test_generation_skips_tiles_that_exist(tiled, monkeypatch):
    calls = []
    monkeypatch.setattr('dbof.tiles.tile_utils.run_series',
                        lambda **kw: calls.append(kw) or [],
                        raising=False)
    tile_source.generate(tiled, ['gradb2'])
    assert calls == []


def test_generation_asks_for_the_snapshots_that_are_missing(cfg, monkeypatch):
    _write_tile(tile_source.tile_path(cfg, DATES[0], 'gradb2'), 'gradb2',
                _gradb2())
    calls = []
    monkeypatch.setattr('dbof.tiles.tile_utils.run_series',
                        lambda **kw: calls.append(kw) or [],
                        raising=False)
    tile_source.generate(cfg, ['gradb2'])

    assert len(calls) == 1
    kw = calls[0]
    assert kw['timestamps'] == ['2012-06-29 01:00:00']
    assert kw['property'] == 'gradb2'
    assert kw['pipeline'] == 'OSN'
    assert (kw['j_rect'], kw['i_rect']) == (9360, 12960)
    assert kw['output_paths'] == [tile_source.tile_path(cfg, DATES[1], 'gradb2')]


def test_clobber_regenerates_everything(tiled, monkeypatch):
    calls = []
    monkeypatch.setattr('dbof.tiles.tile_utils.run_series',
                        lambda **kw: calls.append(kw) or [],
                        raising=False)
    tile_source.generate(tiled, ['gradb2'], clobber=True)
    assert len(calls[0]['timestamps']) == 2


def test_the_tile_key_is_not_handed_to_dbof(cfg):
    with buildconfig.materialized_source(cfg) as path:
        block = yaml.safe_load(open(path))
    assert 'tile' not in block
    assert block['pipeline'] == 'OSN'          # the rest survives


# -- end to end ---------------------------------------------------------------

@pytest.fixture
def built(tiled, monkeypatch):
    """find, group and colocate run for real against the tile directory."""
    for date in DATES:
        _write_tile(tile_source.tile_path(tiled, date, 'density'), 'density',
                    (_gradb2() * 3.0 + 1025.0).astype(np.float32))
    monkeypatch.setattr(build_fronts, 'expand_property_roots',
                        lambda roots, cfg_: ['gradb2', 'density'])
    monkeypatch.setattr(build_fronts, 'all_property_roots',
                        lambda cfg_, exclude=None: ['gradb2', 'density'])
    monkeypatch.setattr(tile_source, 'generate',
                        lambda cfg_, channels, clobber=False: [])
    build_fronts.run(tiled, ['gradb2', 'find', 'group', 'colocate'])
    return tiled, FrontStore.open(tiled.store_url)


def test_every_step_finishes_for_every_snapshot(built):
    _cfg, store = built
    status = store.status().set_index('date')
    for date in DATES:
        assert list(status.loc[date]) == ['done', 'done', 'done']


def test_the_products_are_tile_shaped(built):
    _cfg, store = built
    assert store.labels(DATES[0]).shape == (N, N)
    assert store.binary(DATES[0]).shape == (N, N)


def test_rasters_are_chunked_no_larger_than_they_are(built):
    _cfg, store = built
    assert store.labels(DATES[0]).chunks == (N, N)


def test_fronts_were_found_and_measured(built):
    _cfg, store = built
    fronts = store.fronts(DATES[0])
    assert len(fronts) >= 3
    assert (fronts['length_km'] > 0).all()
    # Coordinates came from the tile, so centroids sit in the tile.
    assert fronts['centroid_lat'].between(36.0, 37.0).all()
    assert fronts['centroid_lon'].between(-123.0, -122.0).all()


def test_the_other_channels_were_colocated(built):
    _cfg, store = built
    props = store.properties(DATES[0])
    assert 'density_mean' in props and 'gradb2_mean' in props
    assert np.isfinite(props['density_mean']).all()


def test_the_build_records_which_tile_it_was(built):
    _cfg, store = built
    assert store.attrs['tile_index'] == 330
    assert store.attrs['pipeline'] == 'OSN'


def test_push_is_refused_on_a_tile_run(tiled):
    with pytest.raises(ValueError, match="tile run has none"):
        build_fronts.run(tiled, ['push'])
