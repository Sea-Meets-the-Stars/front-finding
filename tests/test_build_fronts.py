"""Contract tests for the build-fronts workflow.

These are *offline* tests: no S3, no OSN, no data.  They pin the interfaces
that ``fronts/runs/prototypes/one_full/build_fronts.py`` relies on -- both inside
``front_finding`` and across the boundary into the ``llc4320-native-grid-preprocessing``
(``dbof``) package.  If the preprocessing repo changes shape underneath us,
these fail fast and tell us exactly what moved.

Run with::

    pytest fronts/tests/test_build_fronts.py -v
"""
import os
import inspect
import re
import textwrap

import numpy as np
import pytest

from dbof.cli import generate_global, run_all_subsets, zarr_to_netcdf
from dbof.global_dataset_creation import check_existence
from dbof.global_dataset_creation.config import default_output_folder
from dbof.global_dataset_creation.iterations import (
    date_to_run_id, prefix_to_filename_date,
)
from dbof.global_dataset_creation.subset_definitions import (
    expand_channels_with_suffixes, get_subset_definition, valid_subsets,
)

from front_finding import buildconfig
from front_finding.cli import build_fronts
from front_finding.finding import io as finding_io
from front_finding.finding import run as finding_run
from front_finding.llc import io as llc_io
from front_finding.llc import meta as llc_meta
from front_finding.llc import source as llc_source
from front_finding.llc import publish as llc_publish
from front_finding.properties import run as prun

PIPELINES = ("SURF", "OSN", "DEPTH")


# ===========================================================================
#  Fixtures -- throwaway configs, no S3 and no data
# ===========================================================================

_SURF_YAML = """
source:
  pipeline: "SURF"
  run:
    run_id: "V5test"
  data:
    date_iterations:
      - '2012-11-09 12:00:00'
      - '2012-11-10 06:00:00'
  output:
    s3_endpoint: "https://s3-west.nrp-nautilus.io"
    bucket: "dbof/"
  active_subsets:
    - frontal_structure
    - kinematic
    - frontogenesis
    - native_fields
    - surface_wind
    - icearea
finding:
  build_version: "V5"
  config: "D"
  ice_mask_find: false
  ice_mask_props: true
  percentiles: [90]
products:
  root: "products"
"""

_DEPTH_YAML = """
source:
  pipeline: "DEPTH"
  run:
    run_id: "V5depth"
  data:
    date_iterations:
      - '2012-11-09 12:00:00'
  active_subsets:
    - frontal_structure
    - stratification
    - mixing_parameters
    - icearea
  depth_suffixes: [sfc, z25m]
products:
  root: "products"
"""


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


@pytest.fixture
def surf_path(tmp_path):
    return _write(tmp_path, "run_surf.yaml", _SURF_YAML)


@pytest.fixture
def depth_path(tmp_path):
    return _write(tmp_path, "run_depth.yaml", _DEPTH_YAML)


@pytest.fixture
def surf_cfg(surf_path):
    """Loaded the way the CLI loads it -- build_version from the driver."""
    return buildconfig.load_config(
        surf_path, build_version=build_fronts.BUILD_VERSION)


@pytest.fixture
def depth_cfg(depth_path):
    return buildconfig.load_config(
        depth_path, build_version=build_fronts.BUILD_VERSION)


# ===========================================================================
#  The contract with the preprocessing repo
# ===========================================================================

def test_all_three_pipelines_are_known():
    """SURF / OSN / DEPTH each resolve to a subset table."""
    for pipeline in PIPELINES:
        assert valid_subsets(pipeline), f"{pipeline} has no subsets"
    with pytest.raises(ValueError):
        valid_subsets("NOPE")


def test_default_output_folder_per_pipeline():
    """The S3 folder build_fronts reads from is pipeline-derived, not hardcoded."""
    assert default_output_folder("SURF") == "surface_fields/"
    assert default_output_folder("OSN") == "surface_fields/"
    assert default_output_folder("DEPTH") == "depth_fields/"


def test_frontal_structure_exists_in_every_pipeline():
    """Step 1 only ever generates frontal_structure -- it must exist everywhere."""
    for pipeline in PIPELINES:
        assert "frontal_structure" in valid_subsets(pipeline)


def test_icearea_exists_in_every_pipeline():
    """Ice masking needs icearea.zarr for the same run_id, in any pipeline."""
    for pipeline in PIPELINES:
        assert "icearea" in valid_subsets(pipeline)


def _channels_for(pipeline, subset, depth_suffixes=None):
    defn = get_subset_definition(pipeline, subset)
    if depth_suffixes and "depth_suffixes" in defn:
        defn["depth_suffixes"] = depth_suffixes
    return (list(defn.get("model_data_feature_channels") or [])
            + expand_channels_with_suffixes(
                defn.get("compute_features_channels") or [],
                defn.get("depth_suffixes"),
                defn.get("extra_channels")))


def test_gradb2_channel_name_depends_on_pipeline():
    """The gradb2 channel is bare on SURF/OSN and suffixed on DEPTH.

    Any driver that names the channel literally therefore works on one pipeline
    and fails on the others with a missing file.
    """
    for pipeline in ("SURF", "OSN"):
        chans = _channels_for(pipeline, "frontal_structure")
        assert "gradb2" in chans
        assert "gradb2_sfc" not in chans

    depth = _channels_for("DEPTH", "frontal_structure",
                          depth_suffixes=["sfc", "z25m"])
    assert "gradb2_sfc" in depth
    assert "gradb2" not in depth


def test_surface_frontal_structure_has_extra_channels():
    """SURF/OSN frontal_structure carries density + buoyancy; DEPTH does not.

    Exporting the whole subset in step 1 would therefore build 8 NetCDFs on
    SURF (21 on DEPTH with 4 suffixes) when only gradb2 is needed.
    """
    surf = _channels_for("SURF", "frontal_structure")
    assert {"density", "buoyancy"} <= set(surf)
    assert len(surf) == 8

    depth = _channels_for("DEPTH", "frontal_structure",
                          depth_suffixes=["sfc", "z25m", "mld", "mld_mean"])
    assert {"density", "buoyancy"} & set(depth) == set()
    assert len(depth) == 21


def test_depth_channel_roster_is_discovered_not_assumed():
    """subset_definitions is the only place the DEPTH channel list lives.

    R_ib, Wstar and rossby_number are easy to miss by hand.  build_fronts reads the
    roster rather than restating it; this test is the canary for a roster that
    grows again.
    """
    roots = set()
    for subset in valid_subsets("DEPTH"):
        defn = get_subset_definition("DEPTH", subset)
        roots |= set(defn.get("compute_features_channels") or [])
        roots |= set(defn.get("model_data_feature_channels") or [])
        roots |= set(defn.get("extra_channels") or [])
    assert {"R_ib", "Wstar", "rossby_number"} <= roots


def test_run_all_subsets_exposes_the_flags_the_driver_uses():
    """build_fronts shells out to ``python -m dbof.cli.run_all_subsets``."""
    src = inspect.getsource(run_all_subsets._parse_args)
    flags = set(re.findall(r'"(--[a-z-]+)"', src))
    required = {
        "--config", "--netcdf-base", "--pipeline", "--subsets", "--run-id",
        "--ice-mask", "--clobber", "--clobber-export",
        "--generate-only", "--export-only", "--dry-run",
    }
    assert required <= flags, f"missing: {sorted(required - flags)}"


def test_generate_global_main_signature():
    """Step 4 may call generate_global.main() directly for a single subset."""
    params = inspect.signature(generate_global.main).parameters
    assert {"config_file", "run_id", "subset", "pipeline", "clobber"} <= set(params)


def test_zarr_reader_serves_one_channel_of_one_snapshot():
    """Fields are read straight from the store, so this is the contract.

    A store holds one snapshot, and a channel must be fetchable on its own --
    reading all C channels to use one would be prohibitive at global size.
    """
    from dbof.global_dataset_creation.zarr_dataset_global import (
        GlobalZarrDatasetReader)
    params = inspect.signature(GlobalZarrDatasetReader.__init__).parameters
    assert {"bucket", "folder", "run_id", "dataset_name", "fs",
            "date_prefix"} <= set(params)
    assert hasattr(GlobalZarrDatasetReader, "get_channel_snapshot")


def test_grid_store_carries_latlon():
    """group_fronts takes its coordinates from the static grid store."""
    from dbof.global_dataset_creation.zarr_grid_global import GlobalGridZarrReader
    for prop in ("lat", "lon", "XC", "YC"):
        assert hasattr(GlobalGridZarrReader, prop)


def test_output_filename_requires_a_single_date():
    """Guard rail: many dates + one output filename is an error upstream.

    Exports must therefore be per-timestamp -- handing the whole
    ``date_iterations`` list to zarr_to_netcdf raises for any config with more
    than one date.
    """
    src = inspect.getsource(zarr_to_netcdf.main)
    assert "--output-filename can only be used when converting a single" in src


def test_date_helpers_roundtrip():
    """fronts derives its timestamps with the producer's own helpers."""
    prefix = date_to_run_id("2012-11-09 12:00:00")
    assert prefix == "20121109_120000"
    assert prefix_to_filename_date(prefix) == "2012-11-09T12_00_00"


# ===========================================================================
#  Step 1 builds gradb2 and nothing else
# ===========================================================================

class _Spy:
    """Record calls instead of making them.

    Pass *log* and *name* to also append each call to a shared list, so tests
    can assert the order two different spies were called in.
    """

    def __init__(self, result=None, log=None, name=None):
        self.calls = []
        self.result = result
        self.log = log
        self.name = name

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.log is not None:
            self.log.append(self.name)
        return self.result

    @property
    def kwargs(self):
        assert len(self.calls) == 1, f"expected 1 call, got {len(self.calls)}"
        return self.calls[0][1]

    @property
    def args(self):
        assert len(self.calls) == 1, f"expected 1 call, got {len(self.calls)}"
        return self.calls[0][0]


@pytest.fixture(autouse=True)
def _reset_layout():
    """Keep the module-level run layout from leaking between tests."""
    llc_io.clear_run_layout()
    yield
    llc_io.clear_run_layout()


@pytest.fixture
def spies(monkeypatch, tmp_path):
    """Neutralise every side-effecting call build_fronts makes."""
    order = []
    s = {
        "generate": _Spy(log=order, name="generate"),
        "read": _Spy(result=None, log=order, name="read"),
        "find": _Spy(log=order, name="find"),
        "group": _Spy(log=order, name="group"),
        "colocate": _Spy(log=order, name="colocate"),
        "push": _Spy(result=[], log=order, name="push"),
        "order": order,
    }

    # generate_for_channels runs for real, so the tests exercise its subset
    # loop.  Only its S3 lookup is stubbed -- plan_zarr's verdict per store,
    # defaulting to "nothing is built yet".  A test wanting existing stores
    # sets spies["state"].result to check_existence.ZARR_FULL.
    state = _Spy(result=check_existence.ZARR_MISSING, log=order, name="state")
    monkeypatch.setattr(prun.check_existence, "plan_zarr", state)
    monkeypatch.setattr(prun, "create_s3_filesystems",
                        lambda endpoint: (None, None))
    s["state"] = state

    # Fields are read straight from the stores now; spy at that boundary.
    monkeypatch.setattr(llc_source, "read_channel", s["read"])
    monkeypatch.setattr(prun.llc_source, "available_channels",
                        lambda cfg, ts: set(prun.expand_property_roots(
                            prun.all_property_roots(cfg), cfg)))

    monkeypatch.setattr(prun, "generate_global_dataset", s["generate"])
    monkeypatch.setattr(build_fronts, "generate_global_dataset", s["generate"])
    monkeypatch.setattr(build_fronts, "find_gradb2_fronts", s["find"])
    monkeypatch.setattr(build_fronts, "group_fronts", s["group"])
    monkeypatch.setattr(build_fronts, "colocate_fronts", s["colocate"])
    monkeypatch.setattr(build_fronts.llc_publish, "push_run", s["push"])
    return s


def test_step1_builds_only_the_gradb2_subset(spies, surf_cfg):
    """Step 1 generates the subset that owns gradb2 -- and nothing else."""
    build_fronts.run(surf_cfg, ['gradb2'])
    kw = spies["generate"].kwargs
    assert kw["subsets"] == ["frontal_structure"]
    assert kw["generate_only"] is True          # never exports from here


def test_step1_generate_writes_into_this_builds_directory(spies, surf_cfg):
    """netcdf_base matches step 4's, so the two agree on where products live."""
    build_fronts.run(surf_cfg, ['gradb2'])
    args, _ = spies["generate"].calls[0]
    assert args[1] == llc_io.run_root("V5test")      # .../Fronts/V5/SURF


def test_step1_asks_only_about_the_channel_it_reads(spies, surf_cfg):
    """The whole point: gradb2, not frontal_structure's other 7 channels.

    Asking about the full subset would classify a store written before
    'density' was added upstream as INCOMPLETE, and generate_global raises on
    an incomplete store instead of rebuilding it -- abandoning, in the same
    breath, any date that really was missing.
    """
    build_fronts.run(surf_cfg, ['gradb2'])
    (_fs, store, channels), _kw = spies["state"].calls[0]
    assert channels == ["gradb2"]
    assert store.endswith("frontal_structure.zarr")


def test_step1_checks_one_date_and_believes_it(spies, surf_cfg):
    """A run's dates hold the same channels, so one lookup answers for all.

    surf_cfg has two dates; the store is still classified exactly once, on
    the first of them.
    """
    build_fronts.run(surf_cfg, ['gradb2'])
    assert len(spies["state"].calls) == 1
    first_prefix = surf_cfg.date_prefixes[0]
    assert f"/{first_prefix}/" in spies["state"].calls[0][0][1]


def test_step1_skips_generate_when_the_store_already_has_gradb2(
        spies, surf_cfg):
    """An out-of-date store that still holds gradb2 is left alone."""
    spies["state"].result = check_existence.ZARR_FULL

    build_fronts.run(surf_cfg, ['gradb2'])

    assert spies["generate"].calls == []         # run_all_subsets not invoked


def test_step1_generate_gets_the_config_it_was_given(spies, surf_cfg):
    """No narrowed copy is written -- generate_global skips complete dates."""
    build_fronts.run(surf_cfg, ['gradb2'])
    args, _ = spies["generate"].calls[0]
    assert args[0] == surf_cfg


def test_step1_adds_icearea_only_when_masking_gradb2(spies, tmp_path):
    """The mask is read from icearea.zarr, so that store has to exist too."""
    build_fronts.run(buildconfig.load_config(_write(tmp_path, "unmasked.yaml", _SURF_YAML),
        build_version=build_fronts.BUILD_VERSION), ['gradb2'])
    assert [c[0][2] for c in spies["state"].calls] == [["gradb2"]]

    spies["state"].calls.clear()
    spies["generate"].calls.clear()
    masked = _write(tmp_path, "masked_gen.yaml",
                    _SURF_YAML.replace("ice_mask_find: false",
                                       "ice_mask_find: true"))
    build_fronts.run(buildconfig.load_config(masked,
        build_version=build_fronts.BUILD_VERSION), ['gradb2'])
    assert [c[0][2] for c in spies["state"].calls] == [["gradb2"], ["SIarea"]]
    # One generate call per subset, each scoped to that subset alone.
    assert [c[1]["subsets"] for c in spies["generate"].calls] == [
        ["frontal_structure"], ["icearea"]]


def test_step1_reads_no_fields(spies, surf_cfg):
    """Step 1 only ensures the store exists -- the field is read by `find`."""
    build_fronts.run(surf_cfg, ['gradb2'])
    assert spies["read"].calls == []


def test_step1_uses_the_depth_channel_name_on_depth(spies, depth_cfg):
    """The channel it asks the store for carries the depth suffix."""
    build_fronts.run(depth_cfg, ['gradb2'])
    assert [c[0][2] for c in spies["state"].calls] == [["gradb2_sfc"]]


def test_step1_does_not_colocate_or_find(spies, surf_cfg):
    build_fronts.run(surf_cfg, ['gradb2'])
    assert spies["find"].calls == []
    assert spies["group"].calls == []
    assert spies["colocate"].calls == []


def test_step4_covers_every_subset_and_colocates(spies, surf_cfg):
    """Step 4 is where the other subsets are finally paid for."""
    build_fronts.run(surf_cfg, ['colocate'])
    kw = spies["generate"].kwargs
    assert "subsets" not in kw or kw["subsets"] is None   # -> all active_subsets
    assert len(spies["colocate"].calls) == 2              # one per date


def test_step4_generates_but_never_exports(spies, surf_cfg):
    """Co-location builds the stores it needs, then reads them directly."""
    build_fronts.run(surf_cfg, ['colocate'])
    assert spies["generate"].kwargs["generate_only"] is True


def test_step1_generates_before_find_reads(spies, surf_cfg):
    """Order matters: the store must exist before anything opens it.

    A read resolves to zarr.open_group(mode='r'), which raises on a store that
    is not there and nothing catches it.
    """
    build_fronts.run(surf_cfg, ['gradb2', 'find'])
    order = spies["order"]
    assert order.index("generate") < order.index("find")


def test_steps_2_and_3_generate_nothing(spies, surf_cfg):
    """Finding and grouping are pure consumers of step 1's output."""
    build_fronts.run(surf_cfg, ['find'])
    build_fronts.run(surf_cfg, ['group'])
    assert spies["generate"].calls == []
    assert spies["read"].calls == []
    assert len(spies["find"].calls) == 2
    assert len(spies["group"].calls) == 2


def test_step2_reads_the_pipeline_correct_gradb2_field(spies, surf_cfg, depth_cfg):
    build_fronts.run(surf_cfg, ['find'])
    assert spies["find"].calls[0][1]["gradb2_field"] == "gradb2"
    spies["find"].calls.clear()
    build_fronts.run(depth_cfg, ['find'])
    assert spies["find"].calls[0][1]["gradb2_field"] == "gradb2_sfc"


# ===========================================================================
#  Pipeline selection
# ===========================================================================

def test_channel_for_root_resolves_per_pipeline(surf_cfg, depth_cfg):
    assert prun.channel_for_root(surf_cfg, "gradb2") == "gradb2"
    assert prun.channel_for_root(depth_cfg, "gradb2") == "gradb2_sfc"
    assert prun.channel_for_root(depth_cfg, "gradb2",
                                 depth_suffix="z25m") == "gradb2_z25m"


def test_channel_for_root_rejects_a_suffix_the_config_does_not_build(depth_cfg):
    with pytest.raises(ValueError, match="finding.suffix"):
        prun.channel_for_root(depth_cfg, "gradb2", depth_suffix="mld")


def test_channel_for_root_rejects_an_unknown_root(surf_cfg):
    with pytest.raises(ValueError, match="not produced by any active subset"):
        prun.channel_for_root(surf_cfg, "N2")     # DEPTH-only


def test_subset_for_channel(surf_cfg, depth_cfg):
    assert prun.subset_for_channel(surf_cfg, "gradb2") == "frontal_structure"
    assert prun.subset_for_channel(depth_cfg, "gradb2_sfc") == "frontal_structure"


def test_all_property_roots_follows_the_pipeline(surf_cfg, depth_cfg):
    """The root list follows the pipeline, with no overlap between the two."""
    surf = set(prun.all_property_roots(surf_cfg))
    assert {"gradb2", "density", "buoyancy", "rossby_number"} <= surf
    assert not ({"N2", "Ri", "ertel_pv", "KE"} & surf)   # DEPTH-only

    depth = set(prun.all_property_roots(depth_cfg))
    assert {"N2", "R_ib", "gradb2"} <= depth
    assert not ({"density", "buoyancy"} & depth)        # SURF-only


def test_all_property_roots_always_expand_cleanly(surf_cfg, depth_cfg):
    """Derived roots round-trip through expand_property_roots without raising.

    A hand-written root list drifts out of the pipeline it was written for and
    raises ValueError on every root the active subsets do not produce.
    """
    for cfg in (surf_cfg, depth_cfg):
        roots = prun.all_property_roots(cfg)
        channels = prun.expand_property_roots(roots, cfg)
        assert len(channels) >= len(roots)


def test_exclude_roots_are_dropped(surf_cfg):
    roots = prun.all_property_roots(surf_cfg, exclude=["density", "buoyancy"])
    assert "density" not in roots and "buoyancy" not in roots
    assert "gradb2" in roots


def test_load_config_merges_defaults_with_the_yaml(surf_cfg, depth_cfg):
    assert surf_cfg.pipeline == "SURF"
    assert surf_cfg.run_id == "V5test"
    assert surf_cfg.timestamps == ["2012-11-09T12_00_00", "2012-11-10T06_00_00"]
    assert surf_cfg.date_prefixes == ["20121109_120000", "20121110_060000"]
    assert surf_cfg.finding.config == "D"          # from the YAML
    assert surf_cfg.finding.percentiles == [90]    # from the YAML
    assert surf_cfg.finding.suffix == "sfc"        # FindingConfig default
    assert surf_cfg.finding.exclude_roots == []    # FindingConfig default

    # A config with no finding: block still gets every default.
    assert depth_cfg.finding == buildconfig.FindingConfig(
        build_version=build_fronts.BUILD_VERSION)
    assert depth_cfg.source.depth_suffixes == ["sfc", "z25m"]


# ===========================================================================
#  The ice mask is a per-step toggle
# ===========================================================================

def _spy_read(monkeypatch):
    """Capture read_channel calls made by the real stage functions."""
    spy = _Spy(result=np.zeros((8, 8), dtype=np.float32))
    monkeypatch.setattr(llc_source, "read_channel", spy)
    monkeypatch.setattr(finding_run.llc_source, "read_channel", spy)
    monkeypatch.setattr(prun.llc_source, "read_channel", spy)
    return spy


def _neuter_find(monkeypatch, tmp_path):
    llc_io.set_fronts_path(str(tmp_path / "F"))
    monkeypatch.setattr(finding_run.finding_algorithms, "fronts_from_gradb2",
                        lambda arr, **kw: np.zeros_like(arr, dtype=bool))
    monkeypatch.setattr(finding_run.finding_io, "save_binary_fronts",
                        lambda *a, **k: None)


def test_find_reads_unmasked_by_default(monkeypatch, tmp_path, depth_cfg):
    """depth_cfg has no finding: block at all."""
    spy = _spy_read(monkeypatch)
    _neuter_find(monkeypatch, tmp_path)
    finding_run.find_gradb2_fronts(
        depth_cfg, depth_cfg.timestamps[0], "D", depth_cfg.run_id,
        gradb2_field="gradb2_sfc", gradb2_subset="frontal_structure")
    assert spy.calls[0][1]["ice_mask"] is False


def test_find_honours_ice_mask_find(monkeypatch, tmp_path):
    """ice_mask_find travels to the read, which pulls icearea.zarr."""
    cfg = buildconfig.load_config(
        _write(tmp_path, "masked.yaml",
               _SURF_YAML.replace("ice_mask_find: false",
                                  "ice_mask_find: true")),
        build_version=build_fronts.BUILD_VERSION)
    spy = _spy_read(monkeypatch)
    _neuter_find(monkeypatch, tmp_path)
    finding_run.find_gradb2_fronts(
        cfg, cfg.timestamps[0], "D", cfg.run_id,
        gradb2_field="gradb2", gradb2_subset="frontal_structure")
    assert spy.calls[0][1]["ice_mask"] is True


def test_find_and_props_masks_are_independent(surf_cfg):
    """surf_cfg sets them differently -- they are separate knobs."""
    assert surf_cfg.finding.ice_mask_find is False
    assert surf_cfg.finding.ice_mask_props is True


def test_read_channel_targets_one_store_and_one_date(monkeypatch, surf_cfg):
    """The read is scoped to a single snapshot's store.

    Store location comes from the config; date_prefix pins one snapshot, so a
    100-date config still reads one field at a time.
    """
    seen = {}

    class _Reader:
        def __init__(self, **kw):
            seen.update(kw)

        def get_channel_snapshot(self, channel):
            seen["channel"] = channel
            return "field"

    monkeypatch.setattr(llc_source, "GlobalZarrDatasetReader", _Reader)
    monkeypatch.setattr(llc_source, "create_s3_filesystems",
                        lambda endpoint: (None, None))

    out = llc_source.read_channel(surf_cfg, "2012-11-09T12_00_00", "gradb2",
                                  "frontal_structure")
    assert out == "field"
    assert seen["bucket"] == "dbof/"
    assert seen["run_id"] == "V5test"
    assert seen["dataset_name"] == "frontal_structure.zarr"
    assert seen["date_prefix"] == "20121109_120000"
    assert seen["channel"] == "gradb2"


def test_read_channel_follows_the_store_folder(monkeypatch, tmp_path):
    """source.output.folder locates the stores; SURF would resolve elsewhere."""
    seen = {}

    class _Reader:
        def __init__(self, **kw):
            seen.update(kw)

        def get_channel_snapshot(self, channel):
            return None

    monkeypatch.setattr(llc_source, "GlobalZarrDatasetReader", _Reader)
    monkeypatch.setattr(llc_source, "create_s3_filesystems",
                        lambda endpoint: (None, None))

    cfg = buildconfig.load_config(_write(
        tmp_path, "elsewhere.yaml",
        _SURF_YAML.replace('    bucket: "dbof/"',
                           '    bucket: "dbof/"\n'
                           '    folder: "globals_for_cutouts/"')))
    llc_source.read_channel(cfg, "2012-11-09T12_00_00", "gradb2",
                            "frontal_structure")
    assert seen["folder"] == "globals_for_cutouts/"


def test_grid_latlon_is_fetched_once_per_process(monkeypatch, surf_cfg):
    """The grid is static and large -- read once, kept in memory, never on disk."""
    calls = []

    class _GridReader:
        def __init__(self, **kw):
            calls.append(kw)
            self.lat, self.lon = "LAT", "LON"

    monkeypatch.setattr(llc_source, "GlobalGridZarrReader", _GridReader)
    monkeypatch.setattr(llc_source, "create_s3_filesystems",
                        lambda endpoint: (None, None))
    llc_source.clear_grid_cache()

    assert llc_source.read_latlon(surf_cfg) == ("LAT", "LON")
    assert llc_source.read_latlon(surf_cfg) == ("LAT", "LON")
    assert len(calls) == 1                      # cached, not re-fetched
    assert calls[0]["folder"] == "LLC4320_GRID_2D"
    llc_source.clear_grid_cache()


# ===========================================================================
#  The shipped 100-timestep config
# ===========================================================================

#: The config shipped with the repo -- tests/ sits next to configs/.
_RUN_CFG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "run", "run_v5_100_timesteps.yaml")


@pytest.mark.skipif(not os.path.exists(_RUN_CFG), reason="config not present")
def test_shipped_config_is_coherent():
    cfg = buildconfig.load_config(_RUN_CFG)
    assert cfg.pipeline == "SURF"
    assert len(cfg.source.date_iterations) == 100
    assert len(set(cfg.source.date_iterations)) == 100      # no duplicates
    assert "frontal_structure" in cfg.source.active_subsets

    # Points at the stores under s3://dbof/globals_for_cutouts/v2_2_01/.
    assert cfg.run_id == "v2_2_01"
    assert cfg.finding.build_version == "V5"
    assert cfg.run_dir == "V5/SURF"
    assert cfg.source.folder == "globals_for_cutouts/"
    assert cfg.source.bucket == "dbof/"

    # Steps 1-3 resolve without touching S3.
    channel = prun.channel_for_root(cfg, cfg.finding.gradb2_root,
                                    depth_suffix=cfg.finding.suffix)
    assert channel == "gradb2"
    assert prun.subset_for_channel(cfg, channel) == "frontal_structure"

    # The finding config it names actually exists.
    from front_finding.finding import config as find_config
    assert os.path.isfile(find_config.config_filename(cfg.finding.config))


@pytest.mark.skipif(not os.path.exists(_RUN_CFG), reason="config not present")
def test_shipped_config_dates_match_the_transfer_config():
    """Every date is one of the timesteps sitting in LLC4320_RAW/SURFACE."""
    cfg = buildconfig.load_config(_RUN_CFG)
    for date in cfg.source.date_iterations:
        prefix = date_to_run_id(date)               # raises if out of range
        assert len(prefix) == 15 and prefix[8] == "_"


def test_generate_global_dataset_builds_the_right_command(monkeypatch, surf_cfg):
    """The subprocess argv is the contract with run_all_subsets."""
    spy = _Spy()
    monkeypatch.setattr(prun.subprocess, "run", spy)
    prun.generate_global_dataset(
        surf_cfg, "/base", subsets=["frontal_structure", "icearea"],
        generate_only=True, ice_mask=True, pipeline="SURF", run_id="V5test")
    cmd = spy.args[0]
    assert cmd[1:4] == ["-m", "dbof.cli.run_all_subsets", "--config"]
    assert "--generate-only" in cmd
    assert "--ice-mask" in cmd
    assert cmd[cmd.index("--subsets") + 1:cmd.index("--subsets") + 3] == \
        ["frontal_structure", "icearea"]
    assert cmd[cmd.index("--pipeline") + 1] == "SURF"
    assert cmd[cmd.index("--run-id") + 1] == "V5test"


# ===========================================================================
#  Output layout
# ===========================================================================

def test_layout_splits_directory_from_filename_tag(tmp_path):
    """Products sit under the build; filenames name the source dataset."""
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    llc_io.set_run_layout("V5/SURF", file_tag="v2_2_01")

    path = finding_io.binary_filename("2011-12-04T00_00_00", "D", "v2_2_01")
    assert path == str(tmp_path / "Fronts" / "V5" / "SURF" / "20111204_000000"
                       / "LLC4320_2011-12-04T00_00_00_v2_2_01_bfronts.npy")


def test_layout_applies_to_the_binary_fronts_file(tmp_path):
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    llc_io.set_run_layout("V5/SURF", file_tag="v2_2_01")
    path = finding_io.binary_filename("2011-12-04T00_00_00", "D", "v2_2_01")
    assert os.path.dirname(path).endswith("V5/SURF/20111204_000000")
    assert os.path.basename(path) == \
        "LLC4320_2011-12-04T00_00_00_v2_2_01_bfronts.npy"


def test_run_root_is_the_level_above_the_timestamps(tmp_path):
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    llc_io.set_run_layout("V5/SURF", file_tag="v2_2_01")
    root = llc_io.run_root("v2_2_01")
    assert root == str(tmp_path / "Fronts" / "V5" / "SURF")
    assert llc_io.fronts_dir("v2_2_01", "2011-12-04T00_00_00").startswith(root)


def test_without_a_layout_the_version_drives_both(tmp_path):
    """Callers that never set a layout keep the flat run_id/ directory."""
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    path = finding_io.binary_filename("2011-12-04T00_00_00", "D", "V4")
    assert path == str(tmp_path / "Fronts" / "V4" / "20111204_000000"
                       / "LLC4320_2011-12-04T00_00_00_V4_bfronts.npy")


def test_driver_sets_the_layout_from_the_config(spies, surf_cfg, tmp_path):
    build_fronts.run(surf_cfg, ['gradb2'])
    assert llc_io.run_root("V5test").endswith("products/V5/SURF")


# ===========================================================================
#  The label map written by step 3 is the one step 4 reads
# ===========================================================================

def test_label_map_tag_matches_between_group_and_colocate(tmp_path):
    """Both sides derive the run tag from the binary-fronts filename.

    group_fronts names its outputs after the .npy it was handed; colocate must
    resolve the same name or the label map is never found.
    """
    from front_finding.properties import algorithms as prop_algorithms
    from front_finding.properties import io as properties_io

    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    llc_io.set_run_layout("V5/SURF", file_tag="v2_2_01")

    fronts_file = finding_io.binary_filename("2011-12-04T00_00_00", "D",
                                             "v2_2_01")
    time_str, run_tag, _ = prop_algorithms._parse_fronts_filename(fronts_file)
    assert run_tag == "v2_2_01_bfronts"

    written = properties_io.get_global_front_output_path(
        tmp_path, time_str, "label_map", run_tag)
    assert written.name == \
        "labeled_fronts_global_20111204T00_00_00_v2_2_01_bfronts.npy"


# ===========================================================================
#  Pushing products back to S3
# ===========================================================================

def test_s3_prefix_lands_beside_the_source_stores(surf_cfg, tmp_path):
    cfg = buildconfig.load_config(_write(
        tmp_path, "src.yaml",
        _SURF_YAML.replace('    bucket: "dbof/"',
                           '    bucket: "dbof/"\n'
                           '    folder: "globals_for_cutouts/"')))
    prefix = llc_publish.fronts_s3_prefix(cfg, "2011-12-04T00_00_00")
    assert prefix == "dbof/globals_for_cutouts/V5test/20111204_000000/Fronts"


def test_only_front_products_are_listed(tmp_path):
    d = tmp_path / "ts"
    d.mkdir()
    for name in ("LLC4320_2011-12-04T00_00_00_v2_2_01_bfronts.npy",
                 "labeled_fronts_global_20111204T00_00_00_v2_2_01_bfronts.npy",
                 "front_index_20111204T00_00_00_v2_2_01_bfronts.parquet",
                 "global_front_geometry_20111204T00_00_00_v2_2_01_bfronts.parquet",
                 "front_properties_20111204T00_00_00_v2_2_01_bfronts.parquet",
                 "metadata_20111204T00_00_00_v2_2_01_bfronts.json",
                 "LLC4320_2011-12-04T00_00_00_gradb2_v2_2_01.nc",   # excluded
                 "scratch.txt"):                                     # excluded
        (d / name).touch()

    found = [os.path.basename(f) for f in llc_publish.list_products(str(d))]
    assert len(found) == 6
    assert not any(f.endswith(".nc") for f in found)
    assert "scratch.txt" not in found


def test_push_ignores_a_co_tenant_dataset(tmp_path):
    """Two datasets share the directory; each push must carry only its own.

    Every product name embeds the file tag, so the S3 prefix for one run never
    receives the other run's files.
    """
    d = tmp_path / "ts"
    d.mkdir()
    for name in ("LLC4320_2011-12-04T00_00_00_v2_2_01_bfronts.npy",
                 "front_index_20111204T00_00_00_v2_2_01_bfronts.parquet",
                 "LLC4320_2011-12-04T00_00_00_v2_00_2_bfronts.npy",
                 "front_index_20111204T00_00_00_v2_00_2_bfronts.parquet",
                 "LLC4320_2011-12-04T00_00_00_v2_2_012_bfronts.npy"):
        (d / name).touch()

    mine = [os.path.basename(f)
            for f in llc_publish.list_products(str(d), file_tag="v2_2_01")]
    assert len(mine) == 2
    assert all("_v2_2_01_" in f for f in mine)
    assert not any("v2_2_012" in f for f in mine)     # not a prefix match
    assert len(llc_publish.list_products(str(d), file_tag="v2_00_2")) == 2
    assert len(llc_publish.list_products(str(d))) == 5   # unfiltered


def test_push_uploads_products_and_skips_existing(monkeypatch, tmp_path):
    cfg = buildconfig.load_config(_write(
        tmp_path, "src.yaml",
        _SURF_YAML.replace('    bucket: "dbof/"',
                           '    bucket: "dbof/"\n'
                           '    folder: "globals_for_cutouts/"')))
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    llc_io.set_run_layout("V5/SURF", file_tag="v2_2_01")
    ts = "2011-12-04T00_00_00"
    d = llc_io.fronts_dir("v2_2_01", ts, generate=True)
    open(os.path.join(d, f"LLC4320_{ts}_v2_2_01_bfronts.npy"), "w").close()
    open(os.path.join(d, f"LLC4320_{ts}_gradb2_v2_2_01.nc"), "w").close()
    open(os.path.join(d, f"LLC4320_{ts}_v2_00_2_bfronts.npy"), "w").close()

    class _FS:
        def __init__(self, existing=()):
            self.put_calls = []
            self.existing = set(existing)

        def exists(self, key):
            return key in self.existing

        def put(self, local, key):
            self.put_calls.append((local, key))

    fs = _FS()
    out = llc_publish.push_timestamp(cfg, ts, "v2_2_01", fs=fs)
    # the .nc is not pushed, and neither is the co-tenant's .npy
    assert len(fs.put_calls) == 1
    local, key = fs.put_calls[0]
    assert key == ("dbof/globals_for_cutouts/V5test/20111204_000000/Fronts/"
                   f"LLC4320_{ts}_v2_2_01_bfronts.npy")
    assert out == [f"s3://{key}"]

    fs2 = _FS(existing={key})
    llc_publish.push_timestamp(cfg, ts, "v2_2_01", fs=fs2)
    assert fs2.put_calls == []                         # skipped
    llc_publish.push_timestamp(cfg, ts, "v2_2_01", fs=fs2, clobber=True)
    assert len(fs2.put_calls) == 1                     # clobber forces it


def test_step5_pushes_every_timestamp(spies, surf_cfg):
    build_fronts.run(surf_cfg, ['push'])
    args, kwargs = spies["push"].calls[0]
    assert args[1] == ["2012-11-09T12_00_00", "2012-11-10T06_00_00"]
    assert kwargs["version"] == "V5test"
    assert spies["read"].calls == []                   # push only


# ===========================================================================
#  The run descriptor
# ===========================================================================

def test_meta_filename_names_its_source():
    name = llc_meta.meta_filename("V5", "SURF", "globals_for_cutouts/",
                                  "v2_2_01")
    assert name == "fronts_meta_V5_SURF_from_globals_for_cutouts_v2_2_01.meta"


def test_meta_is_written_at_the_run_root_and_is_readable(tmp_path):
    import yaml as _yaml
    cfg_path = _write(
        tmp_path, "src.yaml",
        _SURF_YAML.replace('    bucket: "dbof/"',
                           '    bucket: "dbof/"\n'
                           '    folder: "globals_for_cutouts/"'))
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    llc_io.set_run_layout("V5/SURF", file_tag="V5test")

    cfg = buildconfig.load_config(cfg_path)
    path = llc_meta.write_run_meta(cfg,
                                   extra={"gradb2_channel": "gradb2",
                                          "gradb2_subset": "frontal_structure"})
    assert os.path.dirname(path).endswith("V5/SURF")
    assert os.path.basename(path) == \
        "fronts_meta_V5_SURF_from_globals_for_cutouts_V5test.meta"

    doc = _yaml.safe_load(open(path))
    assert doc["build"]["pipeline"] == "SURF"
    assert doc["source"]["folder"] == "globals_for_cutouts/"
    assert doc["source"]["run_id"] == "V5test"
    assert "globals_for_cutouts" in doc["source"]["store_uri"]
    assert doc["fronts"]["gradb2_channel"] == "gradb2"
    assert doc["fronts"]["finding_config"] == "D"
    assert doc["fronts"]["ice_mask_props"] is True
    assert doc["dates"]["n"] == 2
    assert set(doc["code"]) == {"front_finding_git", "dbof_git"}


def test_step1_writes_the_descriptor(spies, surf_cfg):
    build_fronts.run(surf_cfg, ['gradb2'])
    root = llc_io.run_root("V5test")
    metas = [f for f in os.listdir(root) if f.endswith(".meta")]
    assert metas == ["fronts_meta_V5_SURF_from_surface_fields_V5test.meta"]


# ===========================================================================
#  Generalising across pipelines and naming schemes
# ===========================================================================

_DEPTH_VX_YAML = """
source:
  pipeline: "DEPTH"
  run:
    run_id: "V5"
  data:
    date_iterations:
      - '2012-11-09 12:00:00'
  output:
    bucket: "dbof/"
  active_subsets: [frontal_structure, stratification, icearea]
  depth_suffixes: [sfc, z25m]
finding:
  build_version: "V5"
  config: "D"
  suffix: "sfc"
products:
  root: "products"
"""

_SURF_DOTTED_YAML = """
source:
  pipeline: "SURF"
  run:
    run_id: "v2_00_2"
  data:
    date_iterations:
      - '2012-11-09 12:00:00'
  output:
    bucket: "dbof/"
    folder: "globals_for_cutouts/"
  active_subsets: [frontal_structure, icearea]
finding:
  build_version: "v2_00"
  config: "A"
products:
  root: "products"
"""


def _paths_for(cfg_path, tmp_path):
    """Every path a run touches, resolved without any I/O."""
    from dbof.global_dataset_creation.config import default_output_folder
    from dbof.global_dataset_creation.zarr_dataset_global import make_run_prefix

    cfg = buildconfig.load_config(cfg_path)
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    llc_io.set_run_layout(cfg.run_dir, file_tag=cfg.run_id)

    channel = prun.channel_for_root(cfg, cfg.finding.gradb2_root,
                                    depth_suffix=cfg.finding.suffix)
    subset = prun.subset_for_channel(cfg, channel)
    folder = cfg.source.folder
    ts = cfg.timestamps[0]
    return cfg, {
        "channel": channel,
        "store": make_run_prefix(cfg.source.bucket, folder, cfg.run_id,
                                 f"{subset}.zarr",
                                 date_prefix=cfg.date_prefixes[0]),
        "bfronts": finding_io.binary_filename(ts, cfg.finding.config,
                                              cfg.run_id),
        "push": "s3://" + llc_publish.fronts_s3_prefix(cfg, ts),
        "meta": llc_meta.meta_filename(cfg.finding.build_version, cfg.pipeline,
                                       folder, cfg.run_id),
    }


def test_depth_pipeline_with_vx_naming(tmp_path):
    """DEPTH, run_id == build_version, folder from the pipeline default."""
    cfg_path = _write(tmp_path, "depth_vx.yaml", _DEPTH_VX_YAML)
    cfg, p = _paths_for(cfg_path, tmp_path)

    assert cfg.run_dir == "V5/DEPTH"
    assert p["channel"] == "gradb2_sfc"                  # suffixed on DEPTH
    assert p["store"] == \
        "s3://dbof/depth_fields/V5/20121109_120000/frontal_structure.zarr"
    assert p["bfronts"].endswith(
        "Fronts/V5/DEPTH/20121109_120000/"
        "LLC4320_2012-11-09T12_00_00_V5_bfronts.npy")
    assert p["push"] == "s3://dbof/depth_fields/V5/20121109_120000/Fronts"
    assert p["meta"] == "fronts_meta_V5_DEPTH_from_depth_fields_V5.meta"


def test_surf_pipeline_with_dotted_naming(tmp_path):
    """A run_id full of underscores, and a folder that is not the default."""
    cfg_path = _write(tmp_path, "surf_dotted.yaml", _SURF_DOTTED_YAML)
    cfg, p = _paths_for(cfg_path, tmp_path)

    assert cfg.run_dir == "v2_00/SURF"
    assert p["channel"] == "gradb2"                      # bare on SURF
    assert p["store"] == ("s3://dbof/globals_for_cutouts/v2_00_2/"
                          "20121109_120000/frontal_structure.zarr")
    assert p["bfronts"].endswith(
        "Fronts/v2_00/SURF/20121109_120000/"
        "LLC4320_2012-11-09T12_00_00_v2_00_2_bfronts.npy")
    assert p["push"] == ("s3://dbof/globals_for_cutouts/v2_00_2/"
                         "20121109_120000/Fronts")
    assert p["meta"] == \
        "fronts_meta_v2_00_SURF_from_globals_for_cutouts_v2_00_2.meta"


def test_underscored_run_id_survives_the_filename_parser(tmp_path):
    """A run_id like 'v2_00_2' must round-trip out of the .npy filename.

    group_fronts and colocate both recover the run tag by parsing the binary
    fronts filename, so an underscore-heavy tag must not be truncated.
    """
    from front_finding.properties import algorithms as prop_algorithms

    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    llc_io.set_run_layout("v2_00/SURF", file_tag="v2_00_2")
    fronts_file = finding_io.binary_filename("2012-11-09T12_00_00", "A",
                                             "v2_00_2")
    time_str, run_tag, raw = prop_algorithms._parse_fronts_filename(fronts_file)
    assert run_tag == "v2_00_2_bfronts"
    assert time_str == "2012-11-09T12:00:00"
    assert raw == "2012-11-09T12_00_00"


def test_build_version_comes_from_the_driver_not_the_config(tmp_path):
    """The output directory is a property of the code that made the products.

    A config may name any dataset; everything build_fronts writes still lands
    under V5/, so the layout cannot drift between runs or between people.
    """
    cfg_path = _write(tmp_path, "claims_otherwise.yaml",
                      _SURF_DOTTED_YAML.replace('build_version: "v2_00"',
                                                'build_version: "SOMETHING_ELSE"'))
    cfg = buildconfig.load_config(cfg_path, build_version=build_fronts.BUILD_VERSION)
    assert build_fronts.BUILD_VERSION == "V5"
    assert cfg.run_dir == "V5/SURF"
    assert cfg.run_id == "v2_00_2"             # the source is still recorded


def test_driver_run_dir_is_the_same_for_every_source(spies, tmp_path):
    for run_id in ("v2_2_01", "v2_00_2"):
        cfg = _write(tmp_path, f"{run_id}.yaml",
                     _SURF_YAML.replace('run_id: "V5test"', f'run_id: "{run_id}"'))
        build_fronts.run(buildconfig.load_config(cfg,
        build_version=build_fronts.BUILD_VERSION), ['gradb2'])
        assert llc_io.run_root(run_id).endswith("products/V5/SURF")


def test_two_source_datasets_do_not_overwrite_each_other(tmp_path):
    """Same build + pipeline, different run_id -> distinct filenames.

    They share a directory; the filename tag is what keeps them apart.
    """
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    names = []
    for run_id in ("v2_2_01", "v2_2_02"):
        llc_io.set_run_layout("V5/SURF", file_tag=run_id)
        names.append(finding_io.binary_filename("2012-11-09T12_00_00", "D",
                                                run_id))
    a, b = names
    assert os.path.dirname(a) == os.path.dirname(b)           # shared dir
    assert a != b                                             # distinct files


def test_every_pipeline_resolves_a_full_path_set(tmp_path):
    """Smoke: SURF, OSN and DEPTH all resolve end to end.

    The store the products are pushed to is always the store they were read
    from, whether the folder came from the pipeline default or a YAML override.
    """
    from dbof.global_dataset_creation.config import default_output_folder

    for pipeline in PIPELINES:
        body = _DEPTH_VX_YAML if pipeline == "DEPTH" else _SURF_DOTTED_YAML
        body = body.replace('pipeline: "DEPTH"', f'pipeline: "{pipeline}"')
        body = body.replace('pipeline: "SURF"', f'pipeline: "{pipeline}"')
        cfg_path = _write(tmp_path, f"{pipeline}.yaml", body)
        cfg, p = _paths_for(cfg_path, tmp_path)
        folder = cfg.source.folder.strip("/")

        assert cfg.pipeline == pipeline
        assert p["channel"].startswith("gradb2")
        assert p["meta"].startswith(
            f"fronts_meta_{cfg.finding.build_version}_{pipeline}_from_{folder}_")
        # read-from and push-to share a prefix: same bucket, folder, run_id, date
        assert p["store"].startswith(f"s3://dbof/{folder}/{cfg.run_id}/")
        assert p["push"] == (f"s3://dbof/{folder}/{cfg.run_id}/"
                             f"{cfg.date_prefixes[0]}/Fronts")
        # local products stay under this build, never under the source run_id
        assert f"/{cfg.run_dir}/" in p["bfronts"]
