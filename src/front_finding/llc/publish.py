"""Push front products back to the S3 store they were derived from.

Each timestamp's outputs land in a ``Fronts/`` folder alongside the zarr
stores they came from::

    s3://{bucket}/{folder}/{run_id}/{YYYYMMDD_HHMMSS}/frontal_structure.zarr
    s3://{bucket}/{folder}/{run_id}/{YYYYMMDD_HHMMSS}/Fronts/
        LLC4320_{ts}_{tag}_bfronts.npy
        labeled_fronts_global_{ts}_{tag}_bfronts.npy
        front_index_{ts}_{tag}_bfronts.parquet
        global_front_geometry_{ts}_{tag}_bfronts.parquet
        front_properties_{ts}_{tag}_bfronts.parquet
        metadata_{ts}_{tag}_bfronts.json

The S3 location is read from the same config that drove the run, so products
cannot land next to the wrong dataset.
"""
import fnmatch
import os
import re


from dbof.io.filesystems import create_s3_filesystems

from front_finding.llc import io as llc_io


#: Filename patterns treated as front products -- everything this build makes.
#: Source fields are not among them: they are read straight from the zarr
#: stores and never staged, so there is nothing of theirs to upload.
PRODUCT_PATTERNS = (
    '*_bfronts.npy',
    'labeled_fronts_global_*.npy',
    'front_index_*.parquet',
    'global_front_geometry_*.parquet',
    'front_properties_*.parquet',
    'metadata_*.json',
    'metadata_properties_*.json',
)

DEFAULT_SUBFOLDER = 'Fronts'


def _s3_settings(cfg) -> dict:
    """Resolve bucket / folder / run_id / endpoint from a run config."""
    return {
        's3_endpoint': cfg.source.s3_endpoint,
        'bucket': cfg.source.bucket.strip().strip('/'),
        'folder': cfg.source.folder.strip().strip('/'),
        'run_id': cfg.run_id,
    }


def fronts_s3_prefix(cfg, timestamp: str,
                     subfolder: str = DEFAULT_SUBFOLDER,
                     run_id: str = None) -> str:
    """Return the S3 key prefix (no scheme) for one timestamp's products."""
    s3 = _s3_settings(cfg)
    date_prefix = llc_io._format_timestamp(timestamp)
    return '/'.join([s3['bucket'], s3['folder'], run_id or s3['run_id'],
                     date_prefix, subfolder])


def list_products(local_dir: str, patterns: tuple = PRODUCT_PATTERNS,
                  file_tag: str = None) -> list:
    """Return the front-product files in *local_dir*, sorted by name.

    A directory is keyed on the build and the pipeline, so products derived
    from two different source datasets can sit side by side.  *file_tag*
    narrows the result to one of them -- without it, a push would carry a
    co-tenant's files into the wrong S3 prefix.

    Parameters
    ----------
    local_dir : str
        Directory holding one timestamp's products.
    patterns : tuple
        Filename globs to match.
    file_tag : str, optional
        Keep only files carrying this tag, e.g. ``'v2_2_01'``.  Matched as a
        whole underscore-delimited component, so ``'v2_2_01'`` does not also
        select ``'v2_2_012'``.
    """
    if not os.path.isdir(local_dir):
        return []
    hits = [f for f in os.listdir(local_dir)
            if any(fnmatch.fnmatch(f, pat) for pat in patterns)]
    if file_tag:
        keep = re.compile(rf'_{re.escape(file_tag)}(?=[_.])')
        hits = [f for f in hits if keep.search(f)]
    return sorted(os.path.join(local_dir, f) for f in hits)


def push_timestamp(cfg, timestamp: str, version: str,
                   subfolder: str = DEFAULT_SUBFOLDER,
                   patterns: tuple = PRODUCT_PATTERNS,
                   run_id: str = None, clobber: bool = False,
                   dry_run: bool = False, fs=None) -> list:
    """Upload one timestamp's front products to S3.

    Parameters
    ----------
    cfg : front_finding.buildconfig.BuildJobConfig
        The resolved run config.  Supplies the S3 endpoint, bucket, folder and
        run_id, so the destination always matches the source dataset.
    timestamp : str
        Snapshot timestamp, e.g. '2011-12-04T00_00_00'.
    version : str
        Run tag, resolved through the active layout to locate the local
        directory (see :func:`front_finding.llc.io.set_run_layout`).
    subfolder : str
        Folder created under the timestamp prefix.  Defaults to ``'Fronts'``.
    patterns : tuple
        Filename globs to upload.  Defaults to :data:`PRODUCT_PATTERNS`.
    run_id : str, optional
        Override the destination run_id from the config.
    clobber : bool
        Overwrite keys that already exist.  Default skips them.
    dry_run : bool
        Report what would be uploaded without touching S3.
    fs : fsspec filesystem, optional
        Reuse a synchronous S3 filesystem across calls.

    Returns
    -------
    list of str
        ``s3://`` URIs that now hold this timestamp's products.
    """
    local_dir = llc_io.fronts_dir(version, timestamp)
    files = list_products(local_dir, patterns,
                          file_tag=llc_io._resolve_file_tag(version))
    if not files:
        print(f"  no front products in {local_dir} — nothing to push")
        return []

    prefix = fronts_s3_prefix(cfg, timestamp, subfolder, run_id)
    if fs is None and not dry_run:
        _, fs = create_s3_filesystems(_s3_settings(cfg)['s3_endpoint'])

    written = []
    for path in files:
        key = f"{prefix}/{os.path.basename(path)}"
        uri = f"s3://{key}"
        if dry_run:
            print(f"  [DRY RUN] {path} -> {uri}")
            written.append(uri)
            continue
        if not clobber and fs.exists(key):
            print(f"  SKIP (exists)  {uri}")
            written.append(uri)
            continue
        print(f"  PUT  {os.path.basename(path)} -> {uri}")
        fs.put(path, key)
        written.append(uri)
    return written


def push_run(cfg, timestamps: list, version: str,
             subfolder: str = DEFAULT_SUBFOLDER,
             patterns: tuple = PRODUCT_PATTERNS,
             run_id: str = None, clobber: bool = False,
             dry_run: bool = False) -> list:
    """Upload every timestamp's front products, reusing one S3 connection."""
    fs = None
    if not dry_run:
        _, fs = create_s3_filesystems(_s3_settings(cfg)['s3_endpoint'])

    written = []
    for timestamp in timestamps:
        print(f"[{timestamp}]")
        written.extend(push_timestamp(
            cfg, timestamp, version, subfolder=subfolder,
            patterns=patterns, run_id=run_id, clobber=clobber,
            dry_run=dry_run, fs=fs))
    print(f"Pushed {len(written)} file(s) across {len(timestamps)} timestamp(s)")
    return written
