"""The front store: everything a build produces, in one zarr store.

Layout
------
One store per build, with a group per snapshot::

    {products.root}/{build_version}/{pipeline}/fronts.zarr/
      .zattrs                       build + source provenance
      20111204_000000/
        .zattrs                     which steps have run here, and with what
        binary      (j, i) bool     front pixels          <- find
        labels      (j, i) int32    connected components  <- group
        geometry/   one 1-D array per column              <- group
        properties/ one 1-D array per column              <- colocate
      20111206_180000/
        ...

Rasters are chunked at :data:`CHUNK`, which divides the LLC4320 rectangular
grid exactly (12 x 16 chunks).  A window read touches only the chunks it
overlaps, so cropping one front out of the global grid costs milliseconds and
never materialises the whole array.

Tables are stored a column at a time, so reading one property does not pull the
other forty-nine.

Crash safety
------------
Zarr writes are not atomic and a build can die halfway through a snapshot.  The
arrays are therefore written *first* and the step is marked done in the group's
attributes *last* -- see :meth:`FrontStore.write_table` and friends.  A step
that was interrupted leaves arrays behind but no ``done`` marker, so:

* :meth:`FrontStore.has` reports it as not done, and the step re-runs and
  overwrites, and
* :meth:`FrontStore.status` shows it as ``partial`` rather than hiding it.

Never infer completion from an array existing.  Ask :meth:`has`.

Reading a store from elsewhere
------------------------------
Nothing here is private to the pipeline; the store is the published artifact::

    from front_finding.store import FrontStore

    store = FrontStore.open("s3://dbof/.../fronts.zarr")
    store.status()                          # what finished, per snapshot
    store.fronts(store.dates[0])            # one row per front
    store.labels(date, window=(y0, y1, x0, x1))   # just that window
    store.dataset()                         # every snapshot, concatenated
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import zarr

#: Raster chunking.  1080 divides both 12960 and 17280 exactly.
CHUNK = (1080, 1080)

#: Steps that write here, in pipeline order.
STEPS = ("find", "group", "colocate")

#: Which arrays each step is responsible for.  Used by status() to tell a
#: finished step from one that died partway through.
_STEP_OUTPUTS = {
    "find": ("binary",),
    "group": ("labels", "geometry"),
    "colocate": ("properties",),
}

#: cross_properties is absent from _STEP_OUTPUTS on purpose: it is written
#: only when a cross-front radius is configured, so its absence is not a
#: half-finished colocate.
_TABLES = ("geometry", "properties", "cross_properties")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _with_key(columns, key: str) -> list:
    """*columns* with *key* prepended if it is not already there."""
    columns = list(columns)
    return columns if key in columns else [key, *columns]


class FrontStore:
    """Read/write access to one build's front products.

    Open with :meth:`open`; the constructor takes an already-open zarr group.
    """

    def __init__(self, root: zarr.Group, url: str):
        self.root = root
        self.url = url

    # -- opening ------------------------------------------------------------

    @classmethod
    def open(cls, url: str, mode: str = "r", storage_options: dict = None
             ) -> "FrontStore":
        """Open a store.

        Parameters
        ----------
        url : str
            Local path or a URI fsspec understands (``s3://...``).
        mode : {'r', 'a', 'w'}
            ``'r'`` read-only, ``'a'`` read/write creating if absent, ``'w'``
            create fresh, discarding anything there.
        storage_options : dict, optional
            Passed to the filesystem, e.g. ``{"endpoint_url": ...}`` for S3.
        """
        root = zarr.open_group(url, mode=mode,
                               storage_options=storage_options or None)
        return cls(root, url)

    # -- discovery ----------------------------------------------------------

    @property
    def attrs(self) -> dict:
        """Build and source provenance for the whole store."""
        return dict(self.root.attrs)

    @property
    def dates(self) -> List[str]:
        """Snapshot groups present, as ``YYYYMMDD_HHMMSS``, sorted."""
        return sorted(k for k, v in self.root.groups())

    def has(self, date: str, step: str) -> bool:
        """Has *step* completed for *date*?

        True only when the step wrote its ``done`` marker, which happens after
        its arrays are on disk.  An interrupted step is False even though some
        of its arrays exist.
        """
        if date not in self.root:
            return False
        steps = dict(self.root[date].attrs).get("steps", {})
        return bool(steps.get(step, {}).get("done"))

    def step_attrs(self, date: str, step: str) -> dict:
        """What *step* recorded for *date* -- parameters, counts, timestamp."""
        if date not in self.root:
            return {}
        return dict(self.root[date].attrs).get("steps", {}).get(step, {})

    def status(self) -> pd.DataFrame:
        """One row per snapshot, one column per step.

        Values are ``done`` (finished), ``partial`` (arrays present but no
        completion marker -- an interrupted run, safe to re-run), or
        ``missing``.
        """
        rows = []
        for date in self.dates:
            grp = self.root[date]
            row = {"date": date}
            for step in STEPS:
                if self.has(date, step):
                    row[step] = "done"
                elif any(name in grp for name in _STEP_OUTPUTS[step]):
                    row[step] = "partial"
                else:
                    row[step] = "missing"
            rows.append(row)
        return pd.DataFrame(rows, columns=["date", *STEPS])

    def pending(self, step: str, dates: Iterable[str] = None) -> List[str]:
        """Dates where *step* still needs to run (missing or interrupted)."""
        return [d for d in (dates if dates is not None else self.dates)
                if not self.has(d, step)]

    # -- rasters ------------------------------------------------------------

    def _raster(self, date: str, name: str,
                window: Tuple[int, int, int, int] = None):
        arr = self.root[date][name]
        if window is None:
            return arr
        y0, y1, x0, x1 = window
        return arr[y0:y1, x0:x1]

    def binary(self, date: str, window=None):
        """The binary front map.  ``window=(y0, y1, x0, x1)`` reads a crop.

        Without a window this returns the lazy zarr array, not a numpy copy --
        the global grid is 224 million cells.
        """
        return self._raster(date, "binary", window)

    def binary_unprocessed(self, date: str, window=None):
        """Front pixels as the threshold found them, before post-processing.

        Sharpening, thinning, cropping and spur removal all narrow this, so it
        is the widest candidate set the detector ever holds.  Written from the
        run that produced ``binary``; raises KeyError on a store built before
        it was kept.  See :meth:`binary` for the window semantics.
        """
        return self._raster(date, "binary_unprocessed", window)

    def labels(self, date: str, window=None):
        """The labelled front map; 0 is background.  See :meth:`binary`."""
        return self._raster(date, "labels", window)

    # -- tables -------------------------------------------------------------

    def _table(self, date: str, name: str,
               columns: Sequence[str] = None) -> pd.DataFrame:
        grp = self.root[date][name]
        # Column order is carried in attrs: zarr lists arrays alphabetically,
        # which would silently reorder every table on read.
        order = dict(grp.attrs).get("columns") or sorted(grp.array_keys())
        wanted = list(columns) if columns is not None else order
        missing = [c for c in wanted if c not in grp]
        if missing:
            raise KeyError(
                f"{name} for {date} has no column(s) {missing}.  "
                f"Available: {sorted(grp.array_keys())}"
            )
        return pd.DataFrame({c: grp[c][:] for c in wanted})

    def geometry(self, date: str, columns=None) -> pd.DataFrame:
        """Per-front shape: length, orientation, curvature, bbox, centroid."""
        return self._table(date, "geometry", columns)

    def properties(self, date: str, columns=None) -> pd.DataFrame:
        """Per-front field statistics: ``{channel}_{stat}`` for each channel."""
        return self._table(date, "properties", columns)

    def cross_properties(self, date: str, columns=None) -> pd.DataFrame:
        """Statistics of each front's surroundings, outside its own band.

        Same columns as :meth:`properties`.  Present only when the run set a
        cross-front radius; raises KeyError otherwise.
        """
        return self._table(date, "cross_properties", columns)

    def has_cross_properties(self, date: str) -> bool:
        """Whether this snapshot has a cross-front table."""
        return (self.has(date, "colocate")
                and "cross_properties" in self.root[date])

    def fronts(self, date: str, geometry_columns=None,
               property_columns=None, cross=False) -> pd.DataFrame:
        """Geometry joined to properties -- one row per front.

        The join is ``geometry.label == properties.flabel``; ``flabel`` is
        dropped, and ``label`` is the key.  Returns geometry alone when
        co-location has not run.

        With *cross*, the cross-front table joins too, its columns prefixed
        ``cross_``.  The two tables share column names by design -- the prefix
        exists only here, so a front's own band and its surroundings can sit
        side by side.
        """
        # The join keys have to survive a column selection, or the merge has
        # nothing to join on.
        if geometry_columns is not None:
            geometry_columns = _with_key(geometry_columns, "label")
        if property_columns is not None:
            property_columns = _with_key(property_columns, "flabel")

        geom = self.geometry(date, geometry_columns)
        if not self.has(date, "colocate"):
            return geom
        props = self.properties(date, property_columns)
        if "flabel" in props.columns:
            props = props.rename(columns={"flabel": "label"})
        joined = geom.merge(props, on="label", suffixes=("", "_prop"))
        if cross and self.has_cross_properties(date):
            other = self.cross_properties(date)
            other = other.rename(columns={
                c: ("label" if c == "flabel" else f"cross_{c}")
                for c in other.columns})
            joined = joined.merge(other, on="label")
        return joined

    def dataset(self, dates: Iterable[str] = None, geometry_columns=None,
                property_columns=None) -> pd.DataFrame:
        """Every snapshot's fronts in one frame, with a ``date`` column.

        Snapshots that have not been grouped yet are skipped.
        """
        frames = []
        for date in (dates if dates is not None else self.dates):
            if not self.has(date, "group"):
                continue
            df = self.fronts(date, geometry_columns, property_columns)
            df.insert(0, "date", date)
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # -- writing ------------------------------------------------------------
    #
    # Arrays first, completion marker last.  See the module docstring.

    def set_build_attrs(self, **attrs) -> None:
        """Record build and source provenance on the store root."""
        self.root.attrs.update(attrs)

    def _group(self, date: str) -> zarr.Group:
        return self.root.require_group(date)

    def _mark_done(self, date: str, step: str, **info) -> None:
        """Stamp *step* complete.  Called only after its arrays are written."""
        grp = self._group(date)
        steps = dict(grp.attrs).get("steps", {})
        steps[step] = {"done": _now(), **info}
        grp.attrs["steps"] = steps

    def write_raster(self, date: str, name: str, arr: np.ndarray) -> None:
        """Write one raster.  Does not mark any step done."""
        grp = self._group(date)
        if name in grp:
            del grp[name]
        # A tile build's rasters are smaller than one global chunk; zarr would
        # take CHUNK verbatim and record a chunk grid larger than the array.
        chunks = tuple(min(c, n) for c, n in zip(CHUNK, arr.shape))
        z = grp.create_array(name, shape=arr.shape, dtype=arr.dtype,
                             chunks=chunks)
        z[:] = arr

    def write_table(self, date: str, name: str, df: pd.DataFrame) -> None:
        """Write one table, a column per array.  Does not mark a step done.

        float64 columns are stored as float32: these are statistics of float32
        fields, so the extra precision is noise.  int64 becomes int32, which
        still holds far more fronts than a snapshot can produce.
        """
        if name not in _TABLES:
            raise ValueError(f"Unknown table {name!r}; expected one of {_TABLES}")
        grp = self._group(date)
        if name in grp:
            del grp[name]
        tbl = grp.require_group(name)
        tbl.attrs["columns"] = list(df.columns)
        for col in df.columns:
            values = df[col].to_numpy()
            if values.dtype == np.float64:
                values = values.astype(np.float32)
            elif values.dtype == np.int64:
                values = values.astype(np.int32)
            dtype = str if values.dtype == object else values.dtype
            z = tbl.create_array(col, shape=values.shape, dtype=dtype)
            z[:] = values

    def write_binary(self, date: str, arr: np.ndarray,
                     unprocessed: np.ndarray = None, **info) -> None:
        """Write the binary front map and mark ``find`` done.

        *unprocessed* is the pre-post-processing mask; both rasters land before
        the marker, so an interrupted run reads as partial either way.
        """
        self.write_raster(date, "binary", arr.astype(bool))
        if unprocessed is not None:
            self.write_raster(date, "binary_unprocessed",
                              unprocessed.astype(bool))
            info["n_unprocessed_px"] = int(np.count_nonzero(unprocessed))
        self._mark_done(date, "find", n_front_px=int(np.count_nonzero(arr)),
                        **info)

    def write_group(self, date: str, labels: np.ndarray,
                    geometry: pd.DataFrame, **info) -> None:
        """Write the label map and geometry table, then mark ``group`` done.

        Both land before the marker, so an interrupted write leaves the step
        re-runnable rather than half-claimed.
        """
        self.write_raster(date, "labels", labels.astype(np.int32))
        self.write_table(date, "geometry", geometry)
        self._mark_done(date, "group", n_fronts=int(len(geometry)), **info)

    def write_properties(self, date: str, properties: pd.DataFrame,
                         cross_properties: pd.DataFrame = None,
                         **info) -> None:
        """Write the property tables and mark ``colocate`` done.

        Both tables land before the marker, so an interrupted run reads as
        partial rather than as a colocate that simply had no cross radius.
        """
        self.write_table(date, "properties", properties)
        if cross_properties is not None:
            self.write_table(date, "cross_properties", cross_properties)
        elif "cross_properties" in self.root[date]:
            del self.root[date]["cross_properties"]   # no stale table on re-run
        self._mark_done(date, "colocate", n_fronts=int(len(properties)), **info)
