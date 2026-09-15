"""Typed run configuration for the front-finding build.

A run config has two top-level blocks:

``source:``
    Where the global LLC4320 fields come from.  These are *dbof's* keys, not
    this repo's.  They are kept verbatim so the block can be written straight
    back out as a standalone dbof config (see :func:`materialized_source`) when
    a store needs building.

``finding:``
    How fronts are found and what gets measured.  This repo's own knobs.

The two are modelled differently on purpose.  :class:`FindingConfig` is strict
-- it is this repo's schema, so an unknown key there is a typo and should
raise.  :class:`SourceConfig` is not: it wraps the raw block and exposes only
the values front_finding actually reads, so a dbof key this repo has no opinion
about (``runtime.dask_memory_limit``, ``data.k_levels``, ...) passes through to
dbof untouched instead of failing to load here.

The YAML is parsed once by :func:`load_config` into a frozen
:class:`BuildJobConfig`, which is then passed around; nothing else parses a run
config.  Argument handling for the CLI lives with the CLI, in
:mod:`front_finding.cli.build_fronts`.

Not to be confused with :mod:`front_finding.finding.config`, which loads the
*front-detection* parameter files (window, threshold, thinning) selected by
``finding.config`` here.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import os
import tempfile
import yaml

from dbof.global_dataset_creation.config import default_output_folder
from dbof.global_dataset_creation.iterations import (
    date_to_run_id,
    prefix_to_filename_date,
)
from dbof.global_dataset_creation.subset_definitions import valid_subsets

#: Statistics the colocate step can compute per channel.
SUPPORTED_STATS = frozenset(['mean', 'std', 'median', 'min', 'max', 'count',
                             'skew'])

#: Pipeline steps, in execution order.
STEPS = ('gradb2', 'find', 'group', 'colocate', 'push')

#: Steps that can be forced to redo work already recorded as done.  'gradb2'
#: is absent: whether a source store needs rebuilding is dbof's existence
#: check to make, not this repo's.
CLOBBERABLE_STEPS = ('find', 'group', 'colocate', 'push')


@dataclass(frozen=True)
class SourceConfig:
    """The dbof half of a run config: where the global fields come from.

    Wraps the raw ``source:`` block rather than modelling it field by field.
    The block is dbof's schema and is handed back to dbof verbatim, so keys
    this repo does not read -- including ones added upstream after this was
    written -- must survive untouched rather than raise.  The properties below
    are the values front_finding itself needs.
    """
    #: The ``source:`` block exactly as it appeared in the YAML.
    raw: dict

    # -- identity -----------------------------------------------------------
    @property
    def pipeline(self) -> str:
        return (self.raw.get('pipeline') or '').upper()

    @property
    def run_id(self) -> str:
        """The source dataset tag, stamped into every output filename."""
        return (self.raw.get('run') or {}).get('run_id')

    @property
    def log_dir(self) -> str:
        return (self.raw.get('run') or {}).get('log_dir', './logs/')

    # -- store location -----------------------------------------------------
    @property
    def _output(self) -> dict:
        return self.raw.get('output') or {}

    @property
    def s3_endpoint(self) -> str:
        return self._output.get('s3_endpoint',
                                'https://s3-west.nrp-nautilus.io')

    @property
    def bucket(self) -> str:
        return self._output.get('bucket', 'dbof/')

    @property
    def folder(self) -> str:
        """Store folder, falling back to dbof's per-pipeline default."""
        return self._output.get('folder') or default_output_folder(self.pipeline)

    @property
    def dataset_name(self) -> Optional[str]:
        """Explicit store-name override; None means use the subset definition."""
        return self._output.get('dataset_name')

    # -- what to read -------------------------------------------------------
    @property
    def active_subsets(self) -> List[str]:
        """Subsets the run reads.  Falls back to the singular ``active_subset``
        (used by older configs), then to every subset valid for the pipeline.
        """
        active = self.raw.get('active_subsets')
        if not active:
            single = self.raw.get('active_subset')
            active = [single] if single else valid_subsets(self.pipeline)
        return list(active)

    @property
    def grid(self) -> dict:
        """Location of the static grid store, which carries lat/lon.

        Mirrors dbof's GridAccessConfig.  The grid is static across timesteps
        and lives apart from the snapshot stores, so it has its own location.
        """
        g = self.raw.get('grid') or {}
        return {
            's3_endpoint': g.get('s3_endpoint', self.s3_endpoint),
            'bucket': g.get('bucket', 'dbof'),
            'folder': g.get('folder', 'LLC4320_GRID_2D'),
            'dataset_name': g.get('dataset_name', 'llc4320_grid.zarr'),
        }

    @property
    def depth_suffixes(self) -> Optional[List[str]]:
        return self.raw.get('depth_suffixes')

    @property
    def date_iterations(self) -> List[str]:
        return list((self.raw.get('data') or {}).get('date_iterations') or [])

    @property
    def date_prefixes(self) -> List[str]:
        """``YYYYMMDD_HHMMSS`` -- the store and product directory names."""
        return [date_to_run_id(d) for d in self.date_iterations]

    @property
    def timestamps(self) -> List[str]:
        """``YYYY-MM-DDTHH_MM_SS`` -- the form used in every product filename."""
        return [prefix_to_filename_date(p) for p in self.date_prefixes]


@dataclass(frozen=True)
class FindingConfig:
    """The front-finding half of a run config.  This repo's schema, so strict.

    Every key has a default, which makes the whole ``finding:`` block optional.
    """
    #: Products land under {fronts_root}/{build_version}/{pipeline}/.
    build_version: str = 'V5'
    #: Selects front_finding/finding/configs/finding_config_{X}.yaml.
    config: str = 'D'
    gradb2_root: str = 'gradb2'
    #: Which depth suffix to find fronts in; no effect on SURF.
    suffix: str = 'sfc'
    #: Mask gradb2 before finding fronts (gradb2 step).
    ice_mask_find: bool = False
    #: Mask the property fields the fronts are co-located with (colocate step).
    ice_mask_props: bool = False
    #: Pixels each front is dilated by before its property statistics are
    #: taken, so they describe a band around the front rather than the
    #: skeleton itself.  0 samples the front pixels alone.  Affects the
    #: colocate step only -- the stored geometry is untouched.
    properties_dilation_radius: int = 1
    #: Statistics per co-located channel; see SUPPORTED_STATS.  None keeps
    #: colocation's own default of mean/std/median.
    properties_stats: Optional[List[str]] = None
    #: How co-location treats NaN: 'omit' drops it, 'propagate' lets one NaN
    #: pixel make the whole front's statistic NaN.  The fields carry NaN over
    #: land.
    properties_nan_policy: str = 'omit'
    #: Thickness beyond that band described separately, as the store's
    #: cross_properties table.  0 skips it.  This one may overlap neighbouring
    #: fronts -- it is the front's surroundings, not the front.
    properties_cross_front_radius: int = 0
    #: Restrict the front's own band to pixels the threshold flagged, so it
    #: follows the gradient ridge rather than a disc.  Needs the store's
    #: binary_unprocessed raster.  The cross-front band stays unmasked and
    #: absorbs whatever the band gives up.
    dilate_only_to_front_pixels: bool = False
    #: Extra percentile columns per co-located channel, beyond mean/std/median.
    percentiles: List[int] = field(default_factory=lambda: [25, 75, 90])
    #: Property roots to leave out of co-location.
    exclude_roots: List[str] = field(default_factory=list)
    #: Keep the threshold output alongside the finished front map, as the
    #: store's binary_unprocessed raster.  Denser than binary and costs
    #: proportionally more on disk.  Defaults on: the pipeline wants it, while
    #: fronts_from_gradb2 itself still defaults to returning one array.
    save_unprocessed_binary: bool = True
    #: Per-step overwrite, keyed by step name.  A step not named here skips
    #: work the store already records as done; see CLOBBERABLE_STEPS.
    clobber: Dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class ProductsConfig:
    """Where this build writes.

    Front products are files -- a binary map, a label map, parquet tables --
    and each step reads the previous step's output, so they need somewhere on
    disk.  ``push`` copies the finished set to S3; this is the working area,
    not an archive.
    """
    root: str


@dataclass(frozen=True)
class BuildJobConfig:
    """A fully-resolved run.  Build it with :func:`load_config`."""
    source: SourceConfig
    finding: FindingConfig
    products: ProductsConfig
    #: Path the config was read from.  Recorded in the run's .meta descriptor
    #: for provenance -- nothing re-reads it.
    config_path: Optional[str] = None

    # -- passthroughs to the source block -----------------------------------
    @property
    def pipeline(self) -> str:
        return self.source.pipeline

    @property
    def run_id(self) -> str:
        return self.source.run_id

    @property
    def timestamps(self) -> List[str]:
        return self.source.timestamps

    @property
    def date_prefixes(self) -> List[str]:
        return self.source.date_prefixes

    def clobber(self, step: str) -> bool:
        """Whether *step* should redo work the store already records as done."""
        return bool(self.finding.clobber.get(step, False))

    @property
    def products_root(self) -> str:
        """Directory holding this build's products: ``{root}/{run_dir}``."""
        return os.path.join(self.products.root, self.run_dir)

    @property
    def store_url(self) -> str:
        """The build's zarr store -- every product this run writes."""
        return os.path.join(self.products_root, 'fronts.zarr')

    @property
    def run_dir(self) -> str:
        """Sub-path holding this build's products: ``{build_version}/{pipeline}``.

        Products are organised by the build that made them; filenames keep the
        source run_id, so a file always names the dataset it came from.
        """
        return f'{self.finding.build_version}/{self.pipeline}'


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_config(path: str, build_version: str = None,
                run_id: str = None) -> BuildJobConfig:
    """Read a run YAML into a :class:`BuildJobConfig`.

    Parameters
    ----------
    path : str
        Path to the run YAML.
    build_version : str, optional
        Overrides ``finding.build_version``.  A driver passes its own version so
        the output directory is a property of the code that made the products,
        not of the dataset they were made from.
    run_id : str, optional
        Overrides ``source.run.run_id``.

    Raises
    ------
    ValueError
        If the ``source:`` block or a key this repo depends on is missing, or an
        unknown key appears in the ``finding:`` block.  Keys under ``source:``
        that this repo does not read are not validated -- they belong to dbof.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    src = raw.get('source')
    if not src:
        raise ValueError(
            f"A 'source:' block is required in {path}.  It holds the dbof keys "
            f"describing where the global fields come from: pipeline, run, "
            f"output, runtime, active_subsets, data.date_iterations."
        )
    src = dict(src)

    if run_id is not None:
        src['run'] = {**(src.get('run') or {}), 'run_id': run_id}

    source = SourceConfig(raw=src)
    if not source.raw.get('pipeline'):
        raise ValueError(f"'source.pipeline' must be set in {path}")
    if not source.run_id:
        raise ValueError(f"'source.run.run_id' must be set in {path}")
    if not source.date_iterations:
        raise ValueError(f"'source.data.date_iterations' must be set in {path}")

    finding_raw = dict(raw.get('finding') or {})
    unknown = set(finding_raw) - set(FindingConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(
            f"Unknown key(s) in the 'finding:' block of {path}: {sorted(unknown)}.  "
            f"Valid keys: {sorted(FindingConfig.__dataclass_fields__)}"
        )
    finding = FindingConfig(**finding_raw)
    bad = set(finding.clobber) - set(CLOBBERABLE_STEPS)
    if bad:
        raise ValueError(
            f"Unknown step(s) in the 'finding.clobber:' block of {path}: "
            f"{sorted(bad)}.  Clobberable steps: {list(CLOBBERABLE_STEPS)}."
        )
    non_bool = {k: v for k, v in finding.clobber.items()
                if not isinstance(v, bool)}
    if non_bool:
        raise ValueError(
            f"'finding.clobber' values must be true/false in {path}, got "
            f"{non_bool}."
        )
    if finding.properties_cross_front_radius < 0:
        raise ValueError(
            f"'finding.properties_cross_front_radius' must be >= 0 in {path}, "
            f"got {finding.properties_cross_front_radius}.  0 skips the "
            f"cross-front pass."
        )
    if finding.properties_dilation_radius < 0:
        raise ValueError(
            f"'finding.properties_dilation_radius' must be >= 0 in {path}, "
            f"got {finding.properties_dilation_radius}.  Co-location treats "
            f"anything below 1 as no dilation, so a negative value silently "
            f"means 0."
        )
    unknown_stats = set(finding.properties_stats or ()) - SUPPORTED_STATS
    if unknown_stats:
        raise ValueError(
            f"Unknown stat(s) in 'finding.properties_stats' of {path}: "
            f"{sorted(unknown_stats)}.  Supported: {sorted(SUPPORTED_STATS)}."
        )
    if finding.properties_stats is not None and not finding.properties_stats:
        raise ValueError(
            f"'finding.properties_stats' is empty in {path}.  Omit the key to "
            f"take colocation's default; an empty list would co-locate every "
            f"channel and compute nothing from it."
        )
    if finding.properties_nan_policy not in ('omit', 'propagate'):
        raise ValueError(
            f"'finding.properties_nan_policy' must be 'omit' or 'propagate' "
            f"in {path}, got {finding.properties_nan_policy!r}."
        )
    if build_version:
        finding = replace(finding, build_version=build_version)

    products_raw = raw.get('products') or {}
    root = products_raw.get('root')
    if not root:
        raise ValueError(
            f"'products.root' must be set in {path}.  It is the directory this "
            f"build writes to -- each step reads the previous step's output, so "
            f"the products need somewhere on disk before 'push' ships them."
        )
    # Relative paths resolve against the config file, not the working
    # directory, so the same config behaves identically from the CLI, a
    # notebook, or anywhere else.
    if not os.path.isabs(root):
        root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), root))
    products = ProductsConfig(root=root)

    return BuildJobConfig(source=source, finding=finding, products=products,
                          config_path=path)


@contextmanager
def materialized_source(cfg: BuildJobConfig):
    """Yield a path to the ``source:`` block written as a standalone dbof config.

    ``dbof.cli.run_all_subsets`` takes a ``--config`` path and reads its keys at
    the top level, so the nested block has to be unwrapped into a file of its
    own before being handed over.  The raw block is dumped, not a rebuild from
    :class:`SourceConfig`, so keys this repo does not model reach dbof intact.

    The file is deleted on exit; the run config it came from is the record of
    what was used, and is already named in the run's ``.meta`` descriptor.
    """
    fd, path = tempfile.mkstemp(prefix='dbof_source_', suffix='.yaml')
    try:
        with os.fdopen(fd, 'w') as fh:
            yaml.safe_dump(cfg.source.raw, fh, sort_keys=False,
                           default_flow_style=False)
        yield path
    finally:
        os.unlink(path)
