"""Integration test: fields in memory -> binary fronts -> groups -> per-front table.

Runs the real find / group / colocate stages against a synthetic field on a
temporary filesystem.  Only the S3 boundary is stubbed -- ``read_channel`` and
``read_latlon`` return arrays instead of fetching them.  Everything below that
is the production code path, so a break in the filename conventions that tie
the three stages together shows up here.
"""
import os
import textwrap

import numpy as np
import pytest

from front_finding import buildconfig
from front_finding.finding import io as finding_io
from front_finding.finding import run as finding_run
from front_finding.llc import io as llc_io
from front_finding.llc import source as llc_source
from front_finding.properties import io as properties_io
from front_finding.properties import run as prun

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

    llc_io.clear_run_layout()
    llc_io.set_fronts_path(cfg.products.root)
    llc_io.set_run_layout(cfg.run_dir, file_tag=cfg.run_id)
    yield cfg
    llc_io.clear_run_layout()


def _find(cfg):
    finding_run.find_gradb2_fronts(
        cfg, TIMESTAMP, FINDING_CONFIG, cfg.run_id,
        gradb2_field='gradb2', gradb2_subset='frontal_structure')


def _product(kind):
    return properties_io.get_global_front_output_path(
        llc_io.fronts_dir(RUN_ID, TIMESTAMP), TIMESTAMP.replace('_', ':'),
        kind, f'{RUN_ID}_bfronts')


# ---------------------------------------------------------------------------
#  find
# ---------------------------------------------------------------------------

def test_find_writes_a_binary_front_map(run_cfg):
    _find(run_cfg)
    path = finding_io.binary_filename(TIMESTAMP, FINDING_CONFIG, RUN_ID)
    assert os.path.isfile(path)
    fronts = np.load(path)
    assert fronts.shape == (N, N)
    assert fronts.any(), 'the detector found nothing in a field with ridges'


def test_find_is_idempotent_without_clobber(run_cfg):
    _find(run_cfg)
    path = finding_io.binary_filename(TIMESTAMP, FINDING_CONFIG, RUN_ID)
    mtime = os.path.getmtime(path)
    _find(run_cfg)
    assert os.path.getmtime(path) == mtime      # skipped, not rewritten


def test_products_land_under_the_configured_root(run_cfg):
    _find(run_cfg)
    path = finding_io.binary_filename(TIMESTAMP, FINDING_CONFIG, RUN_ID)
    assert path.startswith(run_cfg.products.root)
    assert f'/{run_cfg.run_dir}/' in path       # ITEST/SURF


# ---------------------------------------------------------------------------
#  group
# ---------------------------------------------------------------------------

@pytest.fixture
def grouped(run_cfg):
    _find(run_cfg)
    prun.group_fronts(run_cfg, TIMESTAMP, FINDING_CONFIG, run_cfg.run_id,
                      n_workers=1)
    return run_cfg


def test_group_writes_the_label_map_and_tables(grouped):
    for kind in ('label_map', 'front_index', 'geometry', 'metadata'):
        assert os.path.isfile(_product(kind)), f'{kind} was not written'


def test_label_map_matches_the_binary_map(grouped):
    labeled = np.load(_product('label_map'))
    binary = np.load(finding_io.binary_filename(TIMESTAMP, FINDING_CONFIG, RUN_ID))
    assert labeled.shape == binary.shape
    assert np.array_equal(labeled > 0, binary.astype(bool))
    assert labeled.max() == 3, 'three ridges should label as three fronts'


def test_geometry_table_has_one_row_per_front(grouped):
    df = properties_io.load_front_index(_product('geometry'))
    assert len(df) == 3
    for col in ('label', 'name', 'npix', 'length_km', 'orientation'):
        assert col in df.columns
    assert (df['length_km'] > 0).all()


def test_group_takes_its_coordinates_from_the_grid_store(grouped):
    """Latitudes must span the stubbed grid, not some default."""
    df = properties_io.load_front_index(_product('geometry'))
    assert df['centroid_lat'].between(-60, 60).all()
    assert df['centroid_lon'].between(-180, 180).all()


# ---------------------------------------------------------------------------
#  colocate
# ---------------------------------------------------------------------------

def test_colocate_joins_property_fields_onto_the_fronts(grouped):
    """The stage that turns fronts into a per-front feature row."""
    prun.colocate_fronts(grouped, TIMESTAMP, FINDING_CONFIG, grouped.run_id,
                         property_names=CHANNELS, percentiles=[90],
                         dilation_radius=0, clobber=True)

    df = properties_io.load_front_index(_product('properties'))
    assert len(df) == 3
    for col in ('flabel', 'npix', 'gradb2_mean', 'gradb2_std',
                'gradb2_median', 'gradb2_p90', 'turner_angle_mean'):
        assert col in df.columns


def test_colocate_skips_channels_the_stores_lack(grouped):
    prun.colocate_fronts(grouped, TIMESTAMP, FINDING_CONFIG, grouped.run_id,
                         property_names=['gradb2', 'NoSuchField'],
                         skip_missing=True, dilation_radius=0, clobber=True)
    df = properties_io.load_front_index(_product('properties'))
    assert 'gradb2_mean' in df.columns
    assert not any(c.startswith('NoSuchField') for c in df.columns)


def test_colocate_raises_on_an_absent_channel_by_default(grouped):
    with pytest.raises(KeyError, match='NoSuchField'):
        prun.colocate_fronts(grouped, TIMESTAMP, FINDING_CONFIG,
                             grouped.run_id, property_names=['NoSuchField'],
                             clobber=True)


# ---------------------------------------------------------------------------
#  The join that ties the stages together
# ---------------------------------------------------------------------------

def test_geometry_and_property_tables_describe_the_same_fronts(grouped):
    """group writes `label`, colocate writes `flabel` -- they must agree."""
    prun.colocate_fronts(grouped, TIMESTAMP, FINDING_CONFIG, grouped.run_id,
                         property_names=['gradb2'], dilation_radius=0,
                         clobber=True)

    geom = properties_io.load_front_index(_product('geometry'))
    props = properties_io.load_front_index(_product('properties'))

    assert sorted(geom['label']) == sorted(props['flabel'])
    merged = geom.merge(props, left_on='label', right_on='flabel')
    assert len(merged) == len(geom)
    assert (merged['npix_x'] == merged['npix_y']).all()


def test_nothing_is_staged_to_disk_but_products(grouped):
    """No NetCDF, no coords file -- only the products themselves."""
    written = sorted(os.listdir(llc_io.fronts_dir(RUN_ID, TIMESTAMP)))
    assert written, 'the run wrote nothing'
    assert not any(f.endswith('.nc') for f in written), written
    for f in written:
        assert f.endswith(('.npy', '.parquet', '.json')), f
