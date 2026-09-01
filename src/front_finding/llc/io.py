""" Basic I/O routines for the LLC analysis """

import os


# ---------------------------------------------------------------------------
# Module-level configurable root path for all Fronts I/O
# ---------------------------------------------------------------------------
_fronts_root = None

# Run layout.  ``_run_dir`` is the sub-path under the root that holds one
# build's products; ``_file_tag`` is the suffix stamped into every filename.
# They are separate so a build can be organised by its own version while the
# files stay named after the dataset they were derived from.  Both fall back to
# the ``version`` argument when unset.
_run_dir = None
_file_tag = None


def set_run_layout(run_dir: str, file_tag: str = None):
    """Set the output sub-path and filename tag for this run.

    ::

        set_fronts_path('/.../LLC/Fronts')
        set_run_layout('V5/SURF', file_tag='v2_2_01')
        # -> /.../LLC/Fronts/V5/SURF/20111204_000000/

    Parameters
    ----------
    run_dir : str
        Sub-path under the Fronts root, e.g. ``'V5/SURF'``.
    file_tag : str, optional
        Filename suffix, e.g. the source ``run_id``.  Defaults to *run_dir*.
    """
    global _run_dir, _file_tag
    _run_dir = run_dir
    _file_tag = file_tag if file_tag is not None else run_dir


def clear_run_layout():
    """Fall back to using the ``version`` argument for both path and tag."""
    global _run_dir, _file_tag
    _run_dir = None
    _file_tag = None


def _resolve_run_dir(version: str) -> str:
    return _run_dir if _run_dir is not None else version


def _resolve_file_tag(version: str) -> str:
    return _file_tag if _file_tag is not None else version


def run_root(version: str = None, generate: bool = False) -> str:
    """Return the directory holding all timestamps for this run.

    ``{fronts_path}/{run_dir}`` -- the level above the per-timestamp folders,
    where run-wide files such as the ``.meta`` descriptor live.
    """
    d = os.path.join(get_fronts_path(), _resolve_run_dir(version))
    if generate:
        os.makedirs(d, exist_ok=True)
    return d


def set_fronts_path(path:str):
    """Set the root directory for all Fronts I/O products.

    All output files are organised as::

        PATH / V{version} / YYYYMMDD_HHMMSS / <filename>

    Call once at the start of a script, e.g.::

        from front_finding.llc import io as llc_io
        llc_io.set_fronts_path('/mnt/tank/Oceanography/data/OGCM/LLC/Fronts')

    Parameters
    ----------
    path : str
        Root directory for Fronts products (the ``PATH`` component).
    """
    global _fronts_root
    _fronts_root = path

def get_fronts_path() -> str:
    """Return the current Fronts root directory.

    Set from the config's ``products.root`` via :func:`set_fronts_path`;
    :func:`front_finding.cli.build_fronts.run` does that for a run.
    """
    if _fronts_root is None:
        raise RuntimeError(
            'No products root set.  Call set_fronts_path() with the config\'s '
            'products.root -- build_fronts.run() does this for you.'
        )
    return _fronts_root


def _format_timestamp(timestamp: str) -> str:
    """Convert a timestamp string to the directory-name format YYYYMMDD_HHMMSS.

    Accepts formats like '2012-11-09T12_00_00' or '2012-11-09T12:00:00'.

    Examples
    --------
    >>> _format_timestamp('2012-11-09T12_00_00')
    '20121109_120000'
    """
    # Strip dashes and the 'T' separator
    s = timestamp.replace('-', '').replace('T', '_')
    # At this point we have e.g. '20121109_12_00_00'; collapse to YYYYMMDD_HHMMSS
    parts = s.split('_')
    # parts[0] = YYYYMMDD, rest = HH, MM, SS  (or already HHMMSS)
    date_part = parts[0]
    time_part = ''.join(parts[1:]).replace(':', '')
    return f'{date_part}_{time_part}'


def fronts_dir(version: str, timestamp: str, generate: bool = False) -> str:
    """Build the run + timestamped output directory.

    Returns ``PATH / {version} / YYYYMMDD_HHMMSS`` and creates it if
    it does not exist.

    Parameters
    ----------
    version : str
        Run tag, used **verbatim** as the directory name.  This is the
        ``run_id`` (e.g. 'Vtest', 'vtest', 'global_DEPTH_test01').
        No ``V`` is prepended.
    timestamp : str
        Snapshot timestamp (e.g. '2012-11-09T12_00_00').
    generate : bool, optional
        Generate the directory if it does not exist. Defaults to False.

    Returns:
    --------
        str: The path to the directory.
    """
    ts_dir = _format_timestamp(timestamp)
    d = os.path.join(get_fronts_path(), _resolve_run_dir(version), ts_dir)
    if generate:
        os.makedirs(d, exist_ok=True)
    # Return
    return d
