"""Integration test: fields in memory -> fronts -> groups -> per-front table.

Runs the real find / group / colocate stages against a synthetic field, writing
to a real zarr store in tmp_path.  Only the S3 boundary is stubbed --
``read_channel`` and ``read_latlon`` return arrays instead of fetching them.
Everything below that is the production code path, so a break in how the three
stages hand off through the store shows up here.
"""
import os
import textwrap

import numpy as np
import pytest

from front_finding import buildconfig
from front_finding.finding import run as finding_run
from front_finding.llc import source as llc_source
from front_finding.properties import run as prun
from front_finding.store import FrontStore

TIMESTAMP = '2011-12-04T00_00_00'
RUN_ID = 'itest'
FINDING_CONFIG = 'D'
N = 192

#: Everything SURF's frontal_structure subset carries.
CHANNELS = ['gradb2', 'gradsalt2', 'gradtheta2', 'gradeta2', 'gradrho2',
            'turner_angle', 'density', 'buoyancy']

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
  build_version: "ITEST"
  config: "D"
"""


def _gradb2(n=N):
    """Three well-separated ridges, gaussian in cross-section.

    Config D sharpens onto the gradient maximum, and a flat-topped ridge has no
    maximum to find -- it erodes to a topological minimum instead.
    """
    f = np.full((n, n), 1e-3, dtype=np.float32)
    x = np.arange(n)
    for col in (n // 4, n // 2, 3 * n // 4):
        profile = np.exp(-0.5 * ((x - col) / 1.2) ** 2).astype(np.float32)
        f[10:n - 10, :] = np.maximum(f[10:n - 10, :], profile[None, :])
    return f


@pytest.fixture
def run_cfg(tmp_path, monkeypatch):
    """A config plus a stubbed store: fields and grid served from memory."""
    path = tmp_path / 'run.yaml'
    path.write_text(textwrap.dedent(_YAML))
    cfg = buildconfig.load_config(str(path))

    gradb2 = _gradb2()
    fields = {ch: (gradb2 if ch == 'gradb2'
                   else (gradb2 * (1.0 + 0.1 * i)).astype(np.float32))
              for i, ch in enumerate(CHANNELS)}
    lat = np.repeat(np.linspace(-60, 60, N)[:, None], N, axis=1)
    lon = np.repeat(np.linspace(-180, 180, N)[None, :], N, axis=0)

    def _read_channel(cfg_, timestamp, channel, subset, **kw):
        if channel not in fields:
            raise KeyError(channel)
        return fields[channel]

    for mod in (llc_source, finding_run.llc_source, prun.llc_source):
        monkeypatch.setattr(mod, 'read_channel', _read_channel)
        monkeypatch.setattr(mod, 'read_latlon', lambda cfg_: (lat, lon))
        monkeypatch.setattr(mod, 'available_channels',
                            lambda cfg_, ts: set(CHANNELS))

    yield cfg, FrontStore.open(cfg.store_url, mode='w')


DATE = '20111204_000000'


def _find(cfg, store):
    finding_run.find_gradb2_fronts(
        cfg, store, TIMESTAMP, DATE, FINDING_CONFIG,
        gradb2_field='gradb2', gradb2_subset='frontal_structure')


# ---------------------------------------------------------------------------
#  find
# ---------------------------------------------------------------------------

def test_find_writes_a_binary_front_map(run_cfg):
    cfg, store = run_cfg
    _find(cfg, store)
    assert store.has(DATE, 'find')
    fronts = store.binary(DATE)[:]
    assert fronts.shape == (N, N)
    assert fronts.dtype == bool
    assert fronts.any(), 'the detector found nothing in a field with ridges'


def test_find_keeps_the_unprocessed_threshold(run_cfg):
    """Post-processing only narrows, so the kept mask must be the wider one."""
    cfg, store = run_cfg
    _find(cfg, store)
    raw = store.binary_unprocessed(DATE)[:]
    assert raw.shape == store.binary(DATE)[:].shape
    assert raw.sum() > store.binary(DATE)[:].sum()
    assert store.step_attrs(DATE, 'find')['n_unprocessed_px'] == int(raw.sum())


def test_find_skips_the_unprocessed_raster_when_switched_off(run_cfg):
    cfg, store = run_cfg
    cfg = buildconfig.replace(cfg, finding=buildconfig.replace(
        cfg.finding, save_unprocessed_binary=False))
    _find(cfg, store)
    assert store.has(DATE, 'find')
    with pytest.raises(KeyError):
        store.binary_unprocessed(DATE)
    assert 'n_unprocessed_px' not in store.step_attrs(DATE, 'find')


def test_find_is_idempotent_without_clobber(run_cfg):
    cfg, store = run_cfg
    _find(cfg, store)
    first = store.step_attrs(DATE, 'find')['done']
    _find(cfg, store)
    assert store.step_attrs(DATE, 'find')['done'] == first   # skipped


def test_the_store_lands_under_the_configured_root(run_cfg):
    cfg, store = run_cfg
    assert cfg.store_url.startswith(cfg.products.root)
    assert f'/{cfg.run_dir}/' in cfg.store_url               # ITEST/SURF


def test_find_records_what_it_used(run_cfg):
    cfg, store = run_cfg
    _find(cfg, store)
    attrs = store.step_attrs(DATE, 'find')
    assert attrs['config'] == FINDING_CONFIG
    assert attrs['gradb2_channel'] == 'gradb2'
    assert attrs['n_front_px'] == int(store.binary(DATE)[:].sum())


# ---------------------------------------------------------------------------
#  group
# ---------------------------------------------------------------------------

@pytest.fixture
def grouped(run_cfg):
    cfg, store = run_cfg
    _find(cfg, store)
    prun.group_fronts(cfg, store, TIMESTAMP, DATE, n_workers=1)
    return cfg, store


def test_group_writes_the_label_map_and_geometry(grouped):
    cfg, store = grouped
    assert store.has(DATE, 'group')
    assert store.labels(DATE).shape == (N, N)
    assert len(store.geometry(DATE)) == 3


def test_label_map_matches_the_binary_map(grouped):
    cfg, store = grouped
    labeled = store.labels(DATE)[:]
    binary = store.binary(DATE)[:]
    assert labeled.shape == binary.shape
    assert np.array_equal(labeled > 0, binary)
    assert labeled.max() == 3, 'three ridges should label as three fronts'
    assert labeled.dtype == np.int32


def test_geometry_table_has_one_row_per_front(grouped):
    cfg, store = grouped
    df = store.geometry(DATE)
    assert len(df) == 3
    for col in ('label', 'name', 'npix', 'length_km', 'orientation'):
        assert col in df.columns
    assert (df['length_km'] > 0).all()


def test_geometry_column_order_survives_the_round_trip(grouped):
    """zarr lists arrays alphabetically; the table must not be reordered."""
    cfg, store = grouped
    assert list(store.geometry(DATE).columns)[:4] == [
        'label', 'name', 'time', 'npix']


def test_group_takes_its_coordinates_from_the_grid(grouped):
    """Latitudes must span the stubbed grid, not some default."""
    cfg, store = grouped
    df = store.geometry(DATE)
    assert df['centroid_lat'].between(-60, 60).all()
    assert df['centroid_lon'].between(-180, 180).all()


# ---------------------------------------------------------------------------
#  colocate
# ---------------------------------------------------------------------------

def test_colocate_joins_property_fields_onto_the_fronts(grouped):
    """The stage that turns fronts into a per-front feature row."""
    cfg, store = grouped
    prun.colocate_fronts(cfg, store, TIMESTAMP, DATE,
                         property_names=CHANNELS, percentiles=[90],
                         properties_dilation_radius=0, clobber=True)
    df = store.properties(DATE)
    assert len(df) == 3
    for col in ('flabel', 'npix', 'gradb2_mean', 'gradb2_std',
                'gradb2_median', 'gradb2_p90', 'turner_angle_mean'):
        assert col in df.columns


def test_colocate_masked_to_front_pixels_records_the_flag(grouped):
    cfg, store = grouped
    prun.colocate_fronts(cfg, store, TIMESTAMP, DATE, property_names=['gradb2'],
                         properties_dilation_radius=3, clobber=True,
                         dilate_only_to_front_pixels=True)
    assert store.step_attrs(DATE, 'colocate')['dilate_only_to_front_pixels']


def test_colocate_masked_needs_the_unprocessed_raster(grouped):
    cfg, store = grouped
    del store.root[DATE]['binary_unprocessed']
    with pytest.raises(RuntimeError, match='save_unprocessed_binary'):
        prun.colocate_fronts(cfg, store, TIMESTAMP, DATE,
                             property_names=['gradb2'], clobber=True,
                             dilate_only_to_front_pixels=True)


def test_colocate_honours_the_configured_stats(grouped):
    cfg, store = grouped
    prun.colocate_fronts(cfg, store, TIMESTAMP, DATE, property_names=['gradb2'],
                         stats=['mean', 'max'], clobber=True)
    cols = set(store.properties(DATE).columns)
    assert {'gradb2_mean', 'gradb2_max'} <= cols
    assert 'gradb2_median' not in cols          # not asked for
    assert store.step_attrs(DATE, 'colocate')['stats'] == ['mean', 'max']


def test_colocate_records_the_nan_policy_it_used(grouped):
    cfg, store = grouped
    prun.colocate_fronts(cfg, store, TIMESTAMP, DATE, property_names=['gradb2'],
                         nan_policy='propagate', clobber=True)
    assert store.step_attrs(DATE, 'colocate')['nan_policy'] == 'propagate'


def test_colocate_reads_only_the_columns_asked_for(grouped):
    cfg, store = grouped
    prun.colocate_fronts(cfg, store, TIMESTAMP, DATE,
                         property_names=CHANNELS, properties_dilation_radius=0)
    assert list(store.properties(DATE, columns=['gradb2_mean']).columns) == [
        'gradb2_mean']


def test_colocate_skips_channels_the_stores_lack(grouped):
    cfg, store = grouped
    prun.colocate_fronts(cfg, store, TIMESTAMP, DATE,
                         property_names=['gradb2', 'NoSuchField'],
                         skip_missing=True, properties_dilation_radius=0, clobber=True)
    df = store.properties(DATE)
    assert 'gradb2_mean' in df.columns
    assert not any(c.startswith('NoSuchField') for c in df.columns)


def test_colocate_raises_on_an_absent_channel_by_default(grouped):
    cfg, store = grouped
    with pytest.raises(KeyError, match='NoSuchField'):
        prun.colocate_fronts(cfg, store, TIMESTAMP, DATE,
                             property_names=['NoSuchField'], clobber=True)


# ---------------------------------------------------------------------------
#  The join that ties the stages together
# ---------------------------------------------------------------------------

def test_fronts_joins_geometry_to_properties(grouped):
    """group writes `label`, colocate writes `flabel` -- they must agree."""
    cfg, store = grouped
    prun.colocate_fronts(cfg, store, TIMESTAMP, DATE,
                         property_names=['gradb2'], properties_dilation_radius=0)
    geom, props = store.geometry(DATE), store.properties(DATE)
    assert sorted(geom['label']) == sorted(props['flabel'])

    joined = store.fronts(DATE)
    assert len(joined) == len(geom)
    assert 'flabel' not in joined.columns          # folded into `label`
    assert 'gradb2_mean' in joined.columns


def test_fronts_returns_geometry_alone_before_colocation(grouped):
    """A half-built store is still readable."""
    cfg, store = grouped
    assert not store.has(DATE, 'colocate')
    assert list(store.fronts(DATE).columns) == list(store.geometry(DATE).columns)


def test_dataset_spans_the_snapshots_that_are_done(grouped):
    cfg, store = grouped
    store.write_raster('20111206_180000', 'binary',
                       np.zeros((8, 8), dtype=bool))   # started, not grouped
    ds = store.dataset()
    assert set(ds['date']) == {DATE}                   # the other is skipped
    assert len(ds) == 3


def test_the_store_is_the_only_thing_written(grouped):
    """No NetCDF, no parquet sidecars, no coords file -- one store."""
    cfg, store = grouped
    written = sorted(os.listdir(os.path.join(cfg.products.root, cfg.run_dir)))
    assert written == ['fronts.zarr']
