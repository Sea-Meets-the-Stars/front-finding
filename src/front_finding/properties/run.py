""" High-level routines to run bits and pieces of front_finding.properties
"""
import sys
import subprocess

import numpy as np
from tqdm import tqdm

from dbof.global_dataset_creation import check_existence
from dbof.global_dataset_creation.subset_definitions import (
    get_subset_definition, expand_channels_with_suffixes,
)
from dbof.global_dataset_creation.zarr_dataset_global import make_run_prefix
from dbof.io.filesystems import create_s3_filesystems

from front_finding import buildconfig
from front_finding.llc import source as llc_source

from front_finding.properties import algorithms as prop_algorithms
from front_finding.properties import colocation


def generate_global_dataset(cfg, products_root: str,
                            ice_mask: bool = False, clobber: bool = False,
                            clobber_export: bool = False,
                            subsets: list = None, pipeline: str = None,
                            run_id: str = None,
                            generate_only: bool = False,
                            export_only: bool = False,
                            dry_run: bool = False):
    """Generate + export subsets via ``dbof.run_all_subsets``.

    Thin wrapper around the preprocessing batch driver (a CLI entry point),
    run as a subprocess in the current interpreter's environment.  Pipeline,
    run_id, subsets, dates, and depth_suffixes all come from *cfg* unless
    overridden here.  Existing subset/date zarr stores are skipped unless
    clobbering.

    Args:
        cfg: The resolved run config (BuildJobConfig).
        products_root (str): Passed through as --netcdf-base; required by the
            CLI but unused, since nothing exports NetCDF.
        ice_mask (bool): NaN-mask ice-covered points during export.
        clobber (bool): Force BOTH phases — regenerate the zarr stores AND
            re-export every channel, even if they exist.
        clobber_export (bool): Force re-export of every channel NetCDF from the
            existing zarr stores, WITHOUT regenerating the stores.
        subsets (list, optional): Only process these subsets, overriding
            ``active_subsets`` in the YAML.  Used by build_v5 step 1 to build
            just the frontal-structure store.
        pipeline (str, optional): Override the ``pipeline`` key in the YAML.
        run_id (str, optional): Override ``run.run_id`` in the YAML.
        generate_only (bool): Build the zarr stores, skip the NetCDF export.
        export_only (bool): Export NetCDFs from existing stores, skip generate.
        dry_run (bool): Log the plan without doing anything.
    """
    # dbof reads its keys at the top level of the file it is given, so the run
    # config's 'source:' block is written out on its own for the call.  The
    # temp file must outlive the subprocess, so the whole call sits inside the
    # context manager.
    with buildconfig.materialized_source(cfg) as source_config:
        cmd = [sys.executable, '-m', 'dbof.cli.run_all_subsets',
               '--config', source_config, '--netcdf-base', products_root]
        if pipeline:
            cmd += ['--pipeline', pipeline]
        if run_id:
            cmd += ['--run-id', run_id]
        if subsets:
            cmd += ['--subsets'] + list(subsets)
        if ice_mask:
            cmd.append('--ice-mask')
        if clobber:
            cmd.append('--clobber')
        if clobber_export:
            cmd.append('--clobber-export')
        if generate_only:
            cmd.append('--generate-only')
        if export_only:
            cmd.append('--export-only')
        if dry_run:
            cmd.append('--dry-run')
        print('Running: ' + ' '.join(cmd))
        subprocess.run(cmd, check=True)


def generate_for_channels(cfg, products_root: str,
                          channels_by_subset: dict, run_id: str = None):
    """Build the stores that do not already provide the channels wanted.

    The narrow counterpart to :func:`generate_global_dataset`, for a step that
    reads a couple of channels rather than a whole pipeline.  Per subset, the
    store is classified with ``check_existence.plan_zarr`` against **only**
    *channels* -- not the subset's full channel list -- and generated only if
    it comes up short.

    Asking the narrow question is the point.  A store written before a channel
    was added upstream is ``ZARR_INCOMPLETE`` by the subset's own standard and
    still perfectly good for, say, gradb2.  Handing it to ``generate_global``
    anyway is worse than wasteful: that pre-flight raises on the first
    incomplete store it sees, before generating anything, so one stale store
    would abandon the genuinely missing ones too.

    A run's dates are produced together and hold the same channels, so the
    config's FIRST date is checked and its verdict taken for all of them --
    one metadata GET per subset rather than one per subset x date.  The
    trade-off: a half-finished transfer whose early dates are complete reads
    as ready, and its later dates fail at export instead.

    Parameters
    ----------
    cfg : front_finding.buildconfig.BuildJobConfig
        The resolved run config.
    products_root : str
        Passed to ``run_all_subsets`` as --netcdf-base.  Nothing here exports
        NetCDF any more, and --generate-only ignores it, but the CLI still
        requires the flag.
    channels_by_subset : dict
        ``{subset_name: [channel, ...]}`` -- what the caller will read.
    run_id : str, optional
        Override the run_id used to locate the stores on S3.
    """
    tag = run_id or cfg.run_id
    _, fs = create_s3_filesystems(cfg.source.s3_endpoint)

    for subset, channels in channels_by_subset.items():
        store = make_run_prefix(
            cfg.source.bucket, cfg.source.folder, tag,
            get_subset_definition(cfg.pipeline, subset)['dataset_name'],
            date_prefix=cfg.date_prefixes[0])
        state = check_existence.plan_zarr(fs, store, list(channels))
        if state == check_existence.ZARR_FULL:
            print(f"  SKIP (store serves {', '.join(channels)})  {subset}")
            continue
        print(f"  GENERATE  {subset}  ({state})")
        generate_global_dataset(cfg, products_root,
                                subsets=[subset], generate_only=True)


def colocate_fronts(cfg, store, timestamp: str, date: str,
                    property_names: list,
                    stats: list = None, percentiles: list = None,
                    min_npix: int = 1, nan_policy: str = 'omit',
                    properties_dilation_radius: int = 1,
                    properties_cross_front_radius: int = 0,
                    dilate_only_to_front_pixels: bool = False,
                    clobber: bool = False,
                    skip_missing: bool = False):
    """Sample the property fields onto the fronts; one row per front.

    Reads the label map this build's ``group`` step wrote and the fields from
    the source zarr stores, and writes the property table back to the store.

    Args:
        cfg: The resolved run config (BuildJobConfig); locates the source stores.
        store: The FrontStore this build reads from and writes to.
        timestamp (str): Snapshot timestamp, e.g. '2011-12-04T00_00_00'.
        date (str): Snapshot group in the store, ``YYYYMMDD_HHMMSS``.
        property_names (list): Fully-expanded channel names to co-locate.
        stats (list, optional): Statistics per property.
            Defaults to ['mean', 'std', 'median'].
        percentiles (list, optional): Extra percentiles, e.g. [10, 90].
        min_npix (int): Minimum front size in pixels.
        nan_policy (str): 'omit' drops NaN, 'propagate' lets one NaN
            pixel make a whole front's statistic NaN.  Recorded in the
            store, since it changes what every column means.
        properties_dilation_radius (int): Pixels to dilate each front before
            sampling, so the statistics cover a band rather than the skeleton.
        properties_cross_front_radius (int): Thickness beyond that band to
            describe separately, as a ``cross_properties`` table.  0 skips it.
            Unlike the band, this one may overlap neighbouring fronts.
        dilate_only_to_front_pixels (bool): Restrict the front's own band to
            pixels the threshold flagged, so it follows the gradient ridge
            rather than a disc.  Needs ``binary_unprocessed`` in the store.
            The cross-front band is left unmasked and absorbs whatever the
            band gives up.
        clobber (bool): Redo the step even if it has already run.
        skip_missing (bool): Drop channels the source stores do not hold
            instead of raising.

    Ice masking follows ``cfg.finding.ice_mask_props``.
    """
    if store.has(date, 'colocate') and not clobber:
        print(f"[{date}] already co-located; pass clobber=True to redo")
        return
    if not store.has(date, 'group'):
        raise RuntimeError(
            f"[{date}] no label map in {store.url}.  Run the 'group' step first."
        )

    # Fail before the channel reads, which are the expensive part.
    front_pixel_mask = None
    if dilate_only_to_front_pixels:
        try:
            front_pixel_mask = np.asarray(store.binary_unprocessed(date))
        except KeyError:
            raise RuntimeError(
                f"[{date}] dilate_only_to_front_pixels needs the "
                f"binary_unprocessed raster, which {store.url} does not have.  "
                f"Set finding.save_unprocessed_binary and re-run the 'find' step."
            ) from None

    # Check the source stores hold the channels before any heavy work --
    # one metadata read per subset, no field data.
    available = llc_source.available_channels(cfg, timestamp)
    missing = [name for name in property_names if name not in available]
    if missing:
        if not skip_missing:
            raise KeyError(
                f"{len(missing)} channel(s) are not in the stores for "
                f"{timestamp}: {missing}.  Generate the subset that owns them, "
                f"or pass skip_missing=True to co-locate what exists."
            )
        print(f"WARNING: skipping {len(missing)} channel(s) absent from the "
              f"stores: {missing}")
        property_names = [n for n in property_names if n not in missing]
        if not property_names:
            print("None of the requested channels are present; nothing to "
                  "co-locate. Returning.")
            return

    labeled = store.labels(date)[:]

    property_arrays = {}
    reads = tqdm(property_names, desc="Reading channels", unit="ch")
    for name in reads:
        reads.set_postfix_str(name)
        property_arrays[name] = np.asarray(llc_source.read_channel(
            cfg, timestamp, name, subset_for_channel(cfg, name),
            ice_mask=cfg.finding.ice_mask_props)).squeeze()

    print(f"Co-locating {len(property_arrays)} properties with "
          f"{(labeled > 0).sum():,} front pixels "
          f"(properties_dilation_radius={properties_dilation_radius})...")
    df = colocation.colocate_fronts_with_properties(
        labeled_fronts=labeled,
        properties=property_arrays,
        stats=stats,
        percentiles=percentiles,
        min_npix=min_npix,
        nan_policy=nan_policy,
        dilation_radius=properties_dilation_radius,
        front_pixel_mask=front_pixel_mask,
    )
    print(f"Co-located {len(df):,} fronts")

    cross = None
    if properties_cross_front_radius > 0:
        print(f"Describing surroundings out to "
              f"{properties_dilation_radius + properties_cross_front_radius} px "
              f"(cross_front_radius={properties_cross_front_radius})...")
        cross = colocation.cross_front_properties(
            labeled_fronts=labeled,
            core_labels=colocation._dilate_labeled_array(
                labeled, df['flabel'].to_numpy(),
                properties_dilation_radius, mask=front_pixel_mask),
            properties=property_arrays,
            flabels=df['flabel'].to_numpy(),
            dilation_radius=properties_dilation_radius,
            cross_front_radius=properties_cross_front_radius,
            stats=stats,
            percentiles=percentiles,
            nan_policy=nan_policy,
        )
        print(f"Cross-front band: median {int(cross['npix'].median()):,} px "
              f"per front")

    store.write_properties(date, df, cross_properties=cross,
                           channels=list(property_names),
                           percentiles=percentiles,
                           properties_dilation_radius=properties_dilation_radius,
                           properties_cross_front_radius=properties_cross_front_radius,
                           dilate_only_to_front_pixels=dilate_only_to_front_pixels,
                           stats=list(stats) if stats else None,
                           min_npix=min_npix, nan_policy=nan_policy)


def _resolve_channel_maps(cfg):
    """Resolve channel ↔ subset mappings from the thin global config.

    The ``subsets:`` block no longer lives in the YAML; the canonical channel
    lists live in ``dbof.global_dataset_creation.subset_definitions``, keyed by
    pipeline.  Depth (compute) channels are expanded with the active
    ``depth_suffixes`` (the YAML override wins; otherwise the per-subset
    default is used).  ``model_data_feature_channels`` and ``extra_channels``
    are never suffixed.

    Parameters
    ----------
    cfg : front_finding.buildconfig.BuildJobConfig
        The resolved run config.

    Returns
    -------
    (channel_to_subset, root_to_expanded) : tuple[dict, dict]
        ``channel_to_subset`` maps every fully-expanded channel name to its
        subset.  ``root_to_expanded`` maps each *root* (base) name to the list
        of expanded channel names it produces under the active config.
    """
    pipeline = cfg.pipeline

    # depth_suffixes: an explicit YAML key overrides the per-subset default,
    # but ONLY for subsets that actually carry a depth_suffixes key -- this
    # mirrors dbof.run_all_subsets (it applies the override only when
    # "depth_suffixes" in defn), so surface-only subsets (surface_wind,
    # icearea) keep bare channels.
    suffix_override = cfg.source.depth_suffixes   # None if absent

    # Restrict to the subsets the run actually produces, if listed.
    active = cfg.source.active_subsets

    channel_to_subset = {}
    root_to_expanded = {}
    for subset_name in active:
        defn = get_subset_definition(pipeline, subset_name)

        if suffix_override and ('depth_suffixes' in defn):
            eff_suffixes = suffix_override
        else:
            eff_suffixes = defn.get('depth_suffixes')

        compute = defn.get('compute_features_channels') or []
        model = defn.get('model_data_feature_channels') or []
        extra = defn.get('extra_channels') or []

        # Compute channels get suffix-expanded; model/extra stay bare.
        for base in compute:
            expanded = expand_channels_with_suffixes([base], eff_suffixes, None)
            root_to_expanded[base] = expanded
            for ch in expanded:
                channel_to_subset[ch] = subset_name
        for ch in list(model) + list(extra):
            root_to_expanded[ch] = [ch]
            channel_to_subset[ch] = subset_name

    return channel_to_subset, root_to_expanded


def expand_property_roots(property_roots: list, cfg) -> list:
    """Expand property *roots* into fully-suffixed channel names.

    Lets a caller list root names like ``'relative_vorticity'``
    and receive every variant the active config produces
    (``relative_vorticity_sfc``, ``relative_vorticity_mld``, ...), while
    channels that carry no suffix (``coriolis_f``, ``mixed_layer_depth``, native
    model fields) pass through unchanged.

    Parameters
    ----------
    property_roots : list of str
        Root/base channel names.  Already-expanded names are accepted too.
    cfg : front_finding.buildconfig.BuildJobConfig
        The resolved run config.

    Returns
    -------
    list of str
        Fully-expanded channel names, order-preserving and de-duplicated.

    Raises
    ------
    ValueError
        If a root is unknown to the active pipeline/subsets.
    """
    channel_to_subset, root_to_expanded = _resolve_channel_maps(cfg)

    expanded, seen = [], set()
    unknown = []
    for root in property_roots:
        if root in root_to_expanded:
            names = root_to_expanded[root]
        elif root in channel_to_subset:
            names = [root]            # already an expanded channel name
        else:
            unknown.append(root)
            continue
        for ch in names:
            if ch not in seen:
                seen.add(ch)
                expanded.append(ch)

    if unknown:
        raise ValueError(
            f"These property roots are not in any active subset of "
            f"{cfg.config_path}: {unknown}"
        )
    return expanded


# ===========================================================================
#  Pipeline-aware config helpers
# ===========================================================================
#
#  Channel names and subset membership both depend on the pipeline: SURF and OSN
#  emit a bare 'gradb2', DEPTH emits 'gradb2_sfc', and the depth-resolved
#  subsets (stratification, ertel_pv, ...) have no surface equivalent at all.
#  Everything below derives from the pipeline + active_subsets in the YAML, so a
#  single driver runs on all three and picks up channels the moment they land in
#  subset_definitions.

#: Defaults for the optional ``build:`` block in a run YAML.
def channel_for_root(cfg, root: str,
                     depth_suffix: str = 'sfc') -> str:
    """Resolve a root name to the ONE channel name this config produces.

    ``gradb2`` -> ``'gradb2'`` on SURF/OSN, ``'gradb2_sfc'`` on DEPTH.  Raises
    rather than guessing if the root is not produced by the active subsets.

    Parameters
    ----------
    cfg : front_finding.buildconfig.BuildJobConfig
        The resolved run config.
    root : str
        Base channel name, e.g. ``'gradb2'``.
    depth_suffix : str
        Which suffix to pick when the root expands to several (DEPTH only).

    Returns
    -------
    str
        The fully-expanded channel name.
    """
    channel_to_subset, root_to_expanded = _resolve_channel_maps(cfg)

    if root in root_to_expanded:
        expanded = root_to_expanded[root]
    elif root in channel_to_subset:
        return root                      # already an expanded channel name
    else:
        raise ValueError(
            f"Root '{root}' is not produced by any active subset of "
            f"{cfg.config_path}.  Available roots: {sorted(root_to_expanded)}")

    if len(expanded) == 1:
        return expanded[0]

    want = f'{root}_{depth_suffix}'
    if want not in expanded:
        raise ValueError(
            f"Root '{root}' expands to {expanded} under {cfg.config_path}, "
            f"which does not include '{want}'.  Set finding.suffix to one of "
            f"{[c.split(root + '_')[-1] for c in expanded]}.")
    return want


def subset_for_channel(cfg, channel: str) -> str:
    """Return the dbof subset that produces *channel* under this config."""
    channel_to_subset, _ = _resolve_channel_maps(cfg)
    if channel not in channel_to_subset:
        raise ValueError(
            f"Channel '{channel}' is not produced by any active subset of "
            f"{cfg.config_path}.")
    return channel_to_subset[channel]


def all_property_roots(cfg, exclude: list = None) -> list:
    """Every property root the active subsets produce, in config order.

    Derived from ``subset_definitions``, so the set follows the pipeline and a
    channel added upstream is co-located automatically.

    Parameters
    ----------
    cfg : front_finding.buildconfig.BuildJobConfig
        The resolved run config.
    exclude : list, optional
        Roots to leave out (e.g. heavy fields you don't want co-located).

    Returns
    -------
    list of str
        Root names, suitable for :func:`expand_property_roots`.
    """
    _, root_to_expanded = _resolve_channel_maps(cfg)
    drop = set(exclude or [])
    return [r for r in root_to_expanded if r not in drop]


def group_fronts(cfg, store, timestamp: str, date: str,
                 n_workers: int = None, skip_curvature: bool = False,
                 clobber: bool = False):
    """Label connected components and measure each front's geometry.

    Reads the binary map this build's ``find`` step wrote, and writes the label
    map and geometry table back alongside it.  Coordinates come from the static
    grid store on S3.

    Args:
        cfg: The resolved run config (BuildJobConfig).
        store: The FrontStore this build reads from and writes to.
        timestamp (str): Snapshot timestamp, e.g. '2011-12-04T00_00_00'.
        date (str): Snapshot group in the store, ``YYYYMMDD_HHMMSS``.
        n_workers (int, optional): Parallel workers. Defaults to CPU count.
        skip_curvature (bool): Skip curvature calculation (~50% faster).
        clobber (bool): Redo the step even if it has already run.
    """
    if store.has(date, 'group') and not clobber:
        print(f"[{date}] fronts already grouped; pass clobber=True to redo")
        return
    if not store.has(date, 'find'):
        raise RuntimeError(
            f"[{date}] no binary front map in {store.url}.  Run the 'find' "
            f"step first."
        )

    fronts_binary = store.binary(date)[:]
    lat, lon = llc_source.read_latlon(cfg)

    labeled, geometry = prop_algorithms.group_fronts(
        fronts_binary, lat, lon, timestamp,
        n_workers=n_workers, skip_curvature=skip_curvature,
    )
    store.write_group(date, labeled, geometry, skip_curvature=skip_curvature)


