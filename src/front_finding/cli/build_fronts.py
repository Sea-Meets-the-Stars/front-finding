"""Front finding and co-location for LLC4320.

Steps
-----
gradb2
    Build the store that owns gradb2 if it does not already hold it.
find
    Threshold gradb2 into a binary front map.
group
    Label the fronts and measure their geometric properties.
colocate
    Build the remaining subsets, export their channels, co-locate.
push
    Copy the front products back to S3, next to the stores they came from.
    Runs at the end of every build unless --no-push; name it in --steps to
    publish without building anything.

``gradb2``, ``find`` and ``group`` are self-contained: a front-binary map costs
one subset.  ``colocate`` is the only step that needs the other fields.

Fields are read straight from the S3 zarr stores into memory; nothing is
staged to disk except this build's own products.

Data production is delegated to the preprocessing repo (``dbof``); this driver
finds, groups and co-locates.

Products are organised by the build that made them; filenames keep the source
run_id, so a file always names the dataset it came from::

    {products.root}/{build_version}/{pipeline}/{date_prefix}/

Usage
-----
::

    build-fronts --config configs/run/run_v5_100_timesteps.yaml
    build-fronts --config configs/run/run_v5_100_timesteps.yaml --steps gradb2
    build-fronts --config configs/run/run_v5_100_timesteps.yaml --steps find,group
"""
import argparse
import os
from typing import List

from front_finding import buildconfig
from front_finding.llc import publish as llc_publish
from front_finding.store import FrontStore
from front_finding.finding.run import find_gradb2_fronts
from front_finding.properties.run import (
    all_property_roots,
    channel_for_root,
    colocate_fronts,
    expand_property_roots,
    generate_for_channels,
    generate_global_dataset,
    group_fronts,
    subset_for_channel,
)

#: Products land under {products.root}/{BUILD_VERSION}/{pipeline}/, whatever
#: dataset they were made from -- the source is recorded in the filenames and
#: the .meta descriptor instead.
BUILD_VERSION = 'V5'


def _resolve_gradb2(cfg):
    """gradb2's fully-suffixed channel name and the subset that owns it."""
    channel = channel_for_root(cfg, cfg.finding.gradb2_root,
                               depth_suffix=cfg.finding.suffix)
    return channel, subset_for_channel(cfg, channel)


def _colocation_channels(cfg):
    """Build every missing store, then resolve the channels to co-locate."""
    generate_global_dataset(cfg, cfg.products_root, generate_only=True)
    names = expand_property_roots(
        all_property_roots(cfg, exclude=cfg.finding.exclude_roots), cfg)
    print(f'Resolved {len(names)} channels to co-locate')
    return names


def run(cfg, steps, push: bool = True):
    """Execute *steps* (already ordered and validated) for *cfg*.

    Args:
        cfg: The resolved run config (BuildJobConfig).
        steps: Steps to run, in pipeline order.
        push (bool): Publish the store to S3 when the build finishes.  On by
            default: a build's products belong beside the fields they were
            made from, and push skips keys already there, so a re-run costs a
            listing rather than an upload.
    """
    gradb2_channel, gradb2_subset = _resolve_gradb2(cfg)

    # 'a' so a re-run adds to the build rather than discarding it; the store
    # records what has already been done per snapshot.
    store = FrontStore.open(cfg.store_url, mode='a')
    store.set_build_attrs(
        build_version=cfg.finding.build_version, pipeline=cfg.pipeline,
        run_id=cfg.run_id, config_file=os.path.abspath(cfg.config_path),
        source_bucket=cfg.source.bucket, source_folder=cfg.source.folder,
        finding_config=cfg.finding.config,
        gradb2_channel=gradb2_channel, gradb2_subset=gradb2_subset,
        dates=list(cfg.source.date_iterations),
    )

    print(f'pipeline={cfg.pipeline}  run_id={cfg.run_id}  '
          f'dates={len(cfg.timestamps)}  gradb2={gradb2_channel} '
          f'(subset={gradb2_subset})  finding_config={cfg.finding.config}')
    print(f'steps={steps}')
    print(f'store -> {cfg.store_url}')

    if 'gradb2' in steps:
        # generate_for_channels() asks only about the channels named here, so a
        # store that predates a channel added upstream counts as ready rather
        # than being rebuilt.  Nothing is exported: later steps read the store.
        wanted = {gradb2_subset: [gradb2_channel]}
        if cfg.finding.ice_mask_find:
            wanted['icearea'] = ['SIarea']       # the mask is read from it
        generate_for_channels(cfg, cfg.products_root, wanted, run_id=cfg.run_id)

    # Resolved once for all timestamps: it builds every missing store.
    property_names = _colocation_channels(cfg) if 'colocate' in steps else None

    per_timestamp = [s for s in ('find', 'group', 'colocate') if s in steps]
    if per_timestamp:
        for timestamp, date in zip(cfg.timestamps, cfg.date_prefixes):
            print(f'[{timestamp}]')
            if 'find' in steps:
                find_gradb2_fronts(cfg, store, timestamp, date,
                                   cfg.finding.config,
                                   gradb2_field=gradb2_channel,
                                   gradb2_subset=gradb2_subset,
                                   clobber=cfg.clobber('find'))
            if 'group' in steps:
                group_fronts(cfg, store, timestamp, date,
                             clobber=cfg.clobber('group'))
            if 'colocate' in steps:
                # skip_missing: co-locate whatever the stores hold rather than
                # dying on a channel whose subset was never generated.
                colocate_fronts(
                    cfg, store, timestamp, date,
                    property_names=property_names,
                    stats=cfg.finding.properties_stats,
                    nan_policy=cfg.finding.properties_nan_policy,
                    percentiles=cfg.finding.percentiles,
                    properties_dilation_radius=(
                        cfg.finding.properties_dilation_radius),
                    properties_cross_front_radius=(
                        cfg.finding.properties_cross_front_radius),
                    dilate_only_to_front_pixels=(
                        cfg.finding.dilate_only_to_front_pixels),
                    skip_missing=True,
                    clobber=cfg.clobber('colocate'))

    if push or 'push' in steps:
        llc_publish.push_run(cfg, store, clobber=cfg.clobber('push'))


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------

def parse_steps(value: str) -> List[str]:
    """Parse a ``--steps`` value into an ordered, de-duplicated step list."""
    if value.strip().lower() == 'all':
        return list(buildconfig.STEPS)
    names = [s.strip() for s in value.split(',') if s.strip()]
    unknown = [s for s in names if s not in buildconfig.STEPS]
    if unknown:
        raise ValueError(
            f"Unknown step(s): {unknown}.  Valid steps: {list(buildconfig.STEPS)}"
        )
    # Always execute in pipeline order, whatever order they were given in.
    return [s for s in buildconfig.STEPS if s in names]


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog='build-fronts',
        description='Find, group and co-locate ocean fronts in LLC4320 fields.',
    )
    p.add_argument('--config', required=True, help='Path to the run YAML.')
    p.add_argument('--steps', default='all',
                   help=f"Comma-separated steps to run, in any order "
                        f"(they always execute in pipeline order), or 'all'.  "
                        f"Choices: {', '.join(buildconfig.STEPS)}.  Default: all.")
    p.add_argument('--run-id', default=None,
                   help='Override source.run.run_id from the config.')
    p.add_argument('--build-version', default=None,
                   help='Override finding.build_version from the config.')
    p.add_argument('--no-push', action='store_true',
                   help='Leave the products on local disk instead of '
                        'publishing them to S3 when the build finishes.')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    steps = parse_steps(args.steps)
    cfg = buildconfig.load_config(args.config,
                                  build_version=args.build_version or BUILD_VERSION,
                                  run_id=args.run_id)
    run(cfg, steps, push=not args.no_push)


if __name__ == '__main__':
    main()
