"""Push the front store back to the S3 dataset it was derived from.

A build's products are one zarr store, so publishing copies that store to a
``Fronts/`` prefix beside the source stores it was built from::

    s3://{bucket}/{folder}/{run_id}/{YYYYMMDD_HHMMSS}/frontal_structure.zarr
    s3://{bucket}/{folder}/{run_id}/Fronts/{build_version}/{pipeline}/fronts.zarr

The store spans every snapshot, so it sits beside the date directories rather
than inside one, and carries the build version and pipeline in its key so two
builds of the same source dataset cannot overwrite each other.

The destination is read from the same config that drove the run, so products
cannot land next to the wrong dataset.

Snapshots publish independently: pushing one date uploads that group plus the
store's root metadata, so a long run can publish as it goes rather than only at
the end.
"""
import os

from dbof.io.filesystems import create_s3_filesystems

#: Prefix products are published under, beside the source stores.
DEFAULT_SUBFOLDER = 'Fronts'


def _s3_settings(cfg) -> dict:
    """Resolve bucket / folder / run_id / endpoint from a run config."""
    return {
        's3_endpoint': cfg.source.s3_endpoint,
        'bucket': cfg.source.bucket.strip().strip('/'),
        'folder': cfg.source.folder.strip().strip('/'),
        'run_id': cfg.run_id,
    }


def store_s3_prefix(cfg, subfolder: str = DEFAULT_SUBFOLDER,
                    run_id: str = None) -> str:
    """The S3 key prefix (no scheme) this build's store publishes to."""
    s3 = _s3_settings(cfg)
    return '/'.join([s3['bucket'], s3['folder'], run_id or s3['run_id'],
                     subfolder, cfg.run_dir, 'fronts.zarr'])


def _local_store_path(store) -> str:
    """The store's directory on disk, or None if it is already remote."""
    url = str(store.url)
    if '://' in url and not url.startswith('file://'):
        return None
    return url[len('file://'):] if url.startswith('file://') else url


def store_files(store_path: str, date: str = None) -> list:
    """Files to upload, as ``(local_path, key_suffix)`` pairs.

    With *date*, that snapshot's group plus the store's root metadata -- enough
    for the published store to be readable even when only some snapshots have
    been pushed.  Without it, the whole store.
    """
    if not os.path.isdir(store_path):
        return []

    roots = []
    if date is not None:
        # Root metadata first: a group is unreadable without it.
        roots += [(os.path.join(store_path, f), f)
                  for f in sorted(os.listdir(store_path))
                  if os.path.isfile(os.path.join(store_path, f))]
        walk_from = os.path.join(store_path, date)
        if not os.path.isdir(walk_from):
            return roots
    else:
        walk_from = store_path

    out = list(roots)
    for dirpath, _, filenames in os.walk(walk_from):
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            out.append((full, os.path.relpath(full, store_path)))
    return out


def push_timestamp(cfg, store, date: str, subfolder: str = DEFAULT_SUBFOLDER,
                   run_id: str = None, clobber: bool = False,
                   dry_run: bool = False, fs=None) -> list:
    """Upload one snapshot's group, plus the store's root metadata.

    Parameters
    ----------
    cfg : front_finding.buildconfig.BuildJobConfig
        The resolved run config.  Supplies the S3 endpoint, bucket, folder and
        run_id, so the destination always matches the source dataset.
    store : front_finding.store.FrontStore
        The build's store.  Must be local -- there is nothing to push from a
        store that already lives on S3.
    date : str
        Snapshot group, ``YYYYMMDD_HHMMSS``.
    subfolder : str
        Prefix under the run_id.  Defaults to ``'Fronts'``.
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
        ``s3://`` URIs now holding this snapshot's data.
    """
    store_path = _local_store_path(store)
    if store_path is None:
        print(f"  store is already remote ({store.url}) — nothing to push")
        return []

    files = store_files(store_path, date)
    if not files:
        print(f"  nothing for {date} in {store_path} — nothing to push")
        return []

    prefix = store_s3_prefix(cfg, subfolder, run_id)
    if fs is None and not dry_run:
        _, fs = create_s3_filesystems(_s3_settings(cfg)['s3_endpoint'])

    written, skipped = [], 0
    for path, rel in files:
        key = f"{prefix}/{rel}"
        uri = f"s3://{key}"
        if dry_run:
            written.append(uri)
            continue
        # Root metadata is rewritten as snapshots are added, so it always goes
        # up; chunks are immutable once written and can be skipped.
        is_root = os.sep not in rel
        if not clobber and not is_root and fs.exists(key):
            skipped += 1
            written.append(uri)
            continue
        fs.put(path, key)
        written.append(uri)

    verb = '[DRY RUN] would upload' if dry_run else 'uploaded'
    print(f"  {verb} {len(written) - skipped} file(s)"
          + (f", skipped {skipped} already present" if skipped else ""))
    return written


def push_run(cfg, store, dates: list = None,
             subfolder: str = DEFAULT_SUBFOLDER, run_id: str = None,
             clobber: bool = False, dry_run: bool = False) -> list:
    """Publish the store, one snapshot at a time, over one S3 connection."""
    dates = list(dates) if dates is not None else store.dates
    fs = None
    if not dry_run:
        _, fs = create_s3_filesystems(_s3_settings(cfg)['s3_endpoint'])

    print(f"Publishing to s3://{store_s3_prefix(cfg, subfolder, run_id)}")
    written = []
    for date in dates:
        print(f"[{date}]")
        written.extend(push_timestamp(
            cfg, store, date, subfolder=subfolder, run_id=run_id,
            clobber=clobber, dry_run=dry_run, fs=fs))
    print(f"Pushed {len(written)} file(s) across {len(dates)} snapshot(s)")
    return written
