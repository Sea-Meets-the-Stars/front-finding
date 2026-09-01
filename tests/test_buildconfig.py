"""Unit tests for the run configuration.

The two blocks are modelled differently on purpose -- ``finding:`` is this
repo's schema and is validated strictly, ``source:`` is dbof's and is passed
through.  Most of these tests exist to pin that asymmetry, because getting it
wrong fails in opposite directions: a strict ``source:`` rejects valid dbof
configs, a loose ``finding:`` silently ignores typos.
"""
import textwrap

import pytest
import yaml

from front_finding import buildconfig
from front_finding.cli import build_fronts


#: A `products:` block is required, so it is part of the minimum.  Tests that
#: append extra `source:` keys insert them before it.
MINIMAL = """
source:
  pipeline: "SURF"
  run:
    run_id: "v1"
  data:
    date_iterations:
      - '2012-11-09 12:00:00'
  active_subsets: [frontal_structure]
products:
  root: "products"
"""


def _with_source(extra: str) -> str:
    """MINIMAL with *extra* added inside the source: block."""
    return MINIMAL.replace("products:\n", extra.lstrip("\n") + "products:\n")


def _cfg(tmp_path, body=MINIMAL, name="run.yaml", **kw):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return buildconfig.load_config(str(p), **kw)


# ---------------------------------------------------------------------------
#  Reading
# ---------------------------------------------------------------------------

def test_minimal_config_resolves(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.pipeline == "SURF"
    assert cfg.run_id == "v1"
    assert cfg.date_prefixes == ["20121109_120000"]
    assert cfg.timestamps == ["2012-11-09T12_00_00"]
    assert cfg.run_dir == "V5/SURF"


def test_pipeline_is_upper_cased(tmp_path):
    assert _cfg(tmp_path, MINIMAL.replace('"SURF"', '"surf"')).pipeline == "SURF"


def test_finding_block_is_optional(tmp_path):
    """Every finding key has a default, so a config may omit the block."""
    assert _cfg(tmp_path).finding == buildconfig.FindingConfig()


def test_finding_values_override_the_defaults(tmp_path):
    cfg = _cfg(tmp_path, MINIMAL + """
finding:
  config: "A"
  percentiles: [10, 90]
  ice_mask_props: true
""")
    assert cfg.finding.config == "A"
    assert cfg.finding.percentiles == [10, 90]
    assert cfg.finding.ice_mask_props is True
    assert cfg.finding.suffix == "sfc"          # untouched default


def test_run_dir_follows_build_version_and_pipeline(tmp_path):
    cfg = _cfg(tmp_path, MINIMAL + '\nfinding:\n  build_version: "V9"\n')
    assert cfg.run_dir == "V9/SURF"


# ---------------------------------------------------------------------------
#  Overrides
# ---------------------------------------------------------------------------

def test_build_version_override_wins_over_the_file(tmp_path):
    cfg = _cfg(tmp_path, MINIMAL + '\nfinding:\n  build_version: "IGNORED"\n',
               build_version="V5")
    assert cfg.finding.build_version == "V5"


def test_run_id_override_wins_and_does_not_mutate_the_file(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text(textwrap.dedent(MINIMAL))
    assert buildconfig.load_config(str(p), run_id="other").run_id == "other"
    assert yaml.safe_load(p.read_text())["source"]["run"]["run_id"] == "v1"


# ---------------------------------------------------------------------------
#  Validation -- strict where it is our schema, permissive where it is dbof's
# ---------------------------------------------------------------------------

def test_missing_source_block_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="'source:' block is required"):
        _cfg(tmp_path, "finding:\n  config: 'D'\n")


@pytest.mark.parametrize("drop, message", [
    ('  pipeline: "SURF"\n', "source.pipeline"),
    ('    run_id: "v1"\n', "source.run.run_id"),
    ("      - '2012-11-09 12:00:00'\n", "source.data.date_iterations"),
])
def test_required_source_keys_are_checked(tmp_path, drop, message):
    with pytest.raises(ValueError, match=message):
        _cfg(tmp_path, MINIMAL.replace(drop, ""))


def test_unknown_finding_key_is_rejected(tmp_path):
    """finding: is our schema -- a typo there is a bug, not a passthrough."""
    with pytest.raises(ValueError, match="finding_confg"):
        _cfg(tmp_path, MINIMAL + '\nfinding:\n  finding_confg: "D"\n')


def test_unknown_source_keys_are_accepted(tmp_path):
    """source: is dbof's schema -- keys we have no opinion about must load.

    These are all real dbof keys this repo does not model.  Rejecting them
    would mean every dbof schema addition breaks front-finding.
    """
    cfg = _cfg(tmp_path, _with_source("""
  runtime:
    dask_memory_limit: "40GB"
  output:
    dataset_name: "custom.zarr"
"""))
    assert cfg.source.dataset_name == "custom.zarr"


# ---------------------------------------------------------------------------
#  Source accessors
# ---------------------------------------------------------------------------

def test_folder_falls_back_to_the_pipeline_default(tmp_path):
    from dbof.global_dataset_creation.config import default_output_folder
    assert _cfg(tmp_path).source.folder == default_output_folder("SURF")


def test_explicit_folder_wins(tmp_path):
    cfg = _cfg(tmp_path, _with_source('  output:\n    folder: "mine/"\n'))
    assert cfg.source.folder == "mine/"


def test_singular_active_subset_is_honoured(tmp_path):
    """Older configs name one subset with the singular key."""
    cfg = _cfg(tmp_path, MINIMAL.replace(
        "  active_subsets: [frontal_structure]", '  active_subset: kinematic'))
    assert cfg.source.active_subsets == ["kinematic"]


def test_no_subsets_falls_back_to_every_valid_one(tmp_path):
    from dbof.global_dataset_creation.subset_definitions import valid_subsets
    cfg = _cfg(tmp_path, MINIMAL.replace(
        "  active_subsets: [frontal_structure]\n", ""))
    assert set(cfg.source.active_subsets) == set(valid_subsets("SURF"))


# ---------------------------------------------------------------------------
#  Materialisation -- the file handed to dbof
# ---------------------------------------------------------------------------

def test_materialized_source_is_the_block_at_the_top_level(tmp_path):
    """dbof reads pipeline/run/data at the top level, not under source:."""
    cfg = _cfg(tmp_path)
    with buildconfig.materialized_source(cfg) as path:
        raw = yaml.safe_load(open(path))
    assert raw["pipeline"] == "SURF"
    assert raw["run"]["run_id"] == "v1"
    assert raw["data"]["date_iterations"] == ["2012-11-09 12:00:00"]
    assert "source" not in raw and "finding" not in raw


def test_materialized_source_keeps_keys_we_do_not_model(tmp_path):
    cfg = _cfg(tmp_path, _with_source("""
  runtime:
    dask_memory_limit: "40GB"
    zarr_async_concurrency: 8
"""))
    with buildconfig.materialized_source(cfg) as path:
        raw = yaml.safe_load(open(path))
    assert raw["runtime"]["dask_memory_limit"] == "40GB"


def test_materialized_source_is_cleaned_up(tmp_path):
    import os
    cfg = _cfg(tmp_path)
    with buildconfig.materialized_source(cfg) as path:
        assert os.path.exists(path)
    assert not os.path.exists(path)


def test_materialized_source_is_accepted_by_dbof(tmp_path):
    """The contract that makes the nesting safe."""
    from dbof.cli import generate_global
    cfg = _cfg(tmp_path)
    with buildconfig.materialized_source(cfg) as path:
        resolved, _ = generate_global._resolve_job_config(config_file=path)
    assert resolved.pipeline == "SURF"
    assert resolved.run.run_id == "v1"


# ---------------------------------------------------------------------------
#  Step parsing (CLI)
# ---------------------------------------------------------------------------

def test_all_expands_to_every_step():
    assert build_fronts.parse_steps("all") == list(buildconfig.STEPS)


def test_steps_run_in_pipeline_order_however_they_are_given():
    assert build_fronts.parse_steps("push,find,gradb2") == \
        ["gradb2", "find", "push"]


def test_duplicate_steps_collapse():
    assert build_fronts.parse_steps("find,find") == ["find"]


def test_unknown_step_is_rejected():
    with pytest.raises(ValueError, match="Unknown step"):
        build_fronts.parse_steps("find,frobnicate")
