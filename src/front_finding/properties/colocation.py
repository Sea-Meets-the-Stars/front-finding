"""
Co-locate labeled fronts with mapped property fields and compute per-front statistics.
"""

from __future__ import annotations

from functools import partial
from typing import Dict

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm
from scipy.stats import skew

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _skew(values: np.ndarray, nan_policy: str = 'omit') -> float:
    """Fisher skewness of a flat array, or NaN where it is undefined.

    scipy has no nan-aware counterpart the way numpy does, and its own
    nan_policy leaves two cases that are ordinary here: it returns 0.0 for two
    points, and warns about catastrophic cancellation on a band of constant
    value.  Both are reported as NaN instead -- a front too small or too flat
    to have a third moment has no skewness, not a skewness of zero.
    """
    v = values[np.isfinite(values)] if nan_policy == 'omit' else values
    if v.size < 3 or not np.isfinite(v).all() or np.ptp(v) == 0:
        return np.nan
    return float(skew(v))


#: Vectorised over labels, so far faster than labeled_comprehension -- but only
#: for the stats scipy.ndimage implements.  A stat absent here falls back.
_NDIMAGE_FUNC = {
    'mean':   ndimage.mean,
    'std':    ndimage.standard_deviation,
    'median': ndimage.median,
    'min':    ndimage.minimum,
    'max':    ndimage.maximum,
}

#: Plain counterparts of _NANFUNC, for nan_policy='propagate' on a flat array.
_PLAINFUNC = {
    'mean':   np.mean,
    'std':    np.std,
    'median': np.median,
    'min':    np.min,
    'max':    np.max,
    'skew':   partial(_skew, nan_policy='propagate'),
}

_NANFUNC = {
    'mean':   np.nanmean,
    'std':    np.nanstd,
    'median': np.nanmedian,
    'min':    np.nanmin,
    'max':    np.nanmax,
    'skew':   partial(_skew, nan_policy='omit'),
}


def cross_front_properties(
    labeled_fronts: np.ndarray,
    core_labels: np.ndarray,
    properties: Dict[str, np.ndarray],
    flabels: np.ndarray,
    dilation_radius: int,
    cross_front_radius: int,
    stats=None,
    percentiles=None,
    nan_policy: str = 'omit',
) -> pd.DataFrame:
    """Statistics of the surroundings of each front, outside its own band.

    The band sampled here is everything within ``dilation_radius +
    cross_front_radius`` of the front, minus the pixels co-location already
    assigned to it.  Unlike that first pass this one is deliberately NOT a
    partition: neighbouring fronts and their bands fall inside it, and two
    fronts closer than twice the outer radius share pixels.

    That overlap is why the work is per front.  ``_dilate_labeled_array``
    gets away with one global distance transform because every pixel belongs
    to exactly one front, which is what ``labeled_comprehension`` needs; here
    the distance transform measures distance to one front alone, so it has to
    be redone inside each front's bounding box.

    Parameters
    ----------
    labeled_fronts : np.ndarray
        Integer label array where 0 is background.
    core_labels : np.ndarray
        The dilated label array the first pass sampled, from
        :func:`_dilate_labeled_array`.  Its pixels are excluded here.
    properties : dict
        Property arrays with the same shape as *labeled_fronts*.
    flabels : np.ndarray
        Fronts to describe, in the order the rows should come out.
    dilation_radius, cross_front_radius : int
        Inner band already sampled, and the thickness added beyond it.
    stats, percentiles, nan_policy
        As for :func:`colocate_fronts_with_properties`.

    Returns
    -------
    pd.DataFrame
        One row per front: ``flabel``, ``npix`` (pixels in the cross-front
        band), and ``{prop}_{stat}``.  ``npix`` counts the band geometrically;
        land inside it is dropped by *nan_policy* rather than by the count.
    """
    if stats is None:
        stats = ['mean', 'std', 'median']
    if nan_policy not in ("propagate", "omit"):
        raise ValueError("nan_policy must be 'propagate' or 'omit'")

    outer = dilation_radius + cross_front_radius
    ny, nx = labeled_fronts.shape
    # One pass for every front's bounding box, rather than a scan per front.
    boxes = ndimage.find_objects(labeled_fronts)

    npix = np.zeros(len(flabels), dtype=np.int64)
    columns = {}
    for name in properties:
        for stat in stats:
            columns[f'{name}_{stat}'] = np.full(len(flabels), np.nan)
        for pct in (percentiles or []):
            tag = f'{int(pct)}' if pct == int(pct) else f'{pct}'
            columns[f'{name}_p{tag}'] = np.full(len(flabels), np.nan)

    reduce = _NANFUNC if nan_policy == 'omit' else _PLAINFUNC
    pct_fn = np.nanpercentile if nan_policy == 'omit' else np.percentile

    for i, lbl in enumerate(tqdm(flabels, desc='Cross-front band',
                                 unit='front')):
        box = boxes[lbl - 1]
        if box is None:                      # label absent from the raster
            continue
        ys = max(box[0].start - outer, 0), min(box[0].stop + outer, ny)
        xs = max(box[1].start - outer, 0), min(box[1].stop + outer, nx)
        view = (slice(*ys), slice(*xs))

        dist = ndimage.distance_transform_edt(labeled_fronts[view] != lbl)
        band = (dist <= outer) & (core_labels[view] != lbl)
        npix[i] = int(band.sum())
        if not npix[i]:
            continue

        for name, arr in properties.items():
            values = arr[view][band].astype(np.float64)
            if nan_policy == 'omit' and not np.isfinite(values).any():
                continue                     # all land; leave the row NaN
            for stat in stats:
                columns[f'{name}_{stat}'][i] = reduce[stat](values)
            for pct in (percentiles or []):
                tag = f'{int(pct)}' if pct == int(pct) else f'{pct}'
                columns[f'{name}_p{tag}'][i] = pct_fn(values, pct)

    return pd.DataFrame({'flabel': flabels, 'npix': npix, **columns})


def _dilate_labeled_array(
    labeled_fronts: np.ndarray,
    valid_labels: np.ndarray,
    dilation_radius: int,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    Expand each valid front outward by *dilation_radius* pixels.

    Parameters
    ----------
    labeled_fronts : np.ndarray
        Integer label array where 0 is background.
    valid_labels : np.ndarray
        Labels to keep and dilate.
    dilation_radius : int
        Dilation radius in pixels.
    mask : np.ndarray, optional
        Boolean array limiting where a front may expand to.  Only the pixels
        gained by dilating are masked -- a front's own pixels are copied from
        *labeled_fronts* and always survive, so a front can never end up with
        fewer sampled pixels than it has skeleton.

    Returns
    -------
    np.ndarray
        Labeled array with valid fronts expanded into nearby background pixels.
    """

    # Mark pixels belonging to fronts we want to keep
    # This allows filtering, e.g. by min_npix
    max_lbl = int(labeled_fronts.max())
    lookup  = np.zeros(max_lbl + 1, dtype=bool)
    lookup[valid_labels] = True
    valid_mask = lookup[labeled_fronts]        

    # For each background pixel, find distance to nearest valid front pixel
    background = ~valid_mask
    dist, nearest_idx = ndimage.distance_transform_edt(
        background, return_indices=True
    )

    # Start from the original array, but remove invalid fronts/background
    dilated = labeled_fronts.copy()
    dilated[background] = 0                      

    # Background pixels within radius inherit nearest valid front label
    expand_mask = background & (dist <= dilation_radius)
    if mask is not None:
        expand_mask &= mask
    dilated[expand_mask] = labeled_fronts[
        nearest_idx[0][expand_mask],
        nearest_idx[1][expand_mask],
    ]

    return dilated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def colocate_fronts_with_properties(
    labeled_fronts: np.ndarray,
    properties: Dict[str, np.ndarray],
    stats=None,
    percentiles=None,
    min_npix: int = 1,
    nan_policy: str = 'omit',
    dilation_radius: int = 0,
    front_pixel_mask: np.ndarray = None,
) -> pd.DataFrame:
    """Co-locate fronts with mapped property fields.
    Parameters
    ----------
    labeled_fronts : np.ndarray
        Integer label array where 0 is background.
    properties : dict
        Property arrays with the same shape as labeled_fronts.
    stats : list, optional
        Statistics to compute for each property.
        Any combination of ``'mean'``, ``'std'``, ``'skew'``, ``'median'``, ``'min'``, ``'max'``,``'count'``.
    percentiles : sequence, optional
        Percentiles to compute for each property, e.g. ``[10, 25, 75, 90]``
    min_npix : int, optional
        Minimum front size to keep.
    nan_policy : {'omit', 'propagate'}
        How to handle NaNs.  Defaults to 'omit', matching every caller.
    dilation_radius : int, optional
        Number of pixels to dilate each retained front before computing stats.
        Default is 0 (no dilation).
    front_pixel_mask : np.ndarray, optional
        Limits the dilation to these pixels -- the unprocessed threshold map,
        so a band follows the gradient ridge's real width instead of a disc.

    Returns
    -------
    pd.DataFrame
        One row per front.  Columns:
        - flabel
        - npix
        - {prop}_{stat}
        - {prop}_p{pct}

    Usage
    --------
    Basic usage with two property arrays: 
        -include percentiles 
        -ignore NaN (e.g. land pixels)
        -dilate fronts by 5 pixels

    >>> labeled = np.load('labeled_fronts_global_20121109T12_00_00.npy')
    >>> df = colocate_fronts_with_properties(
    ...     labeled,
    ...     properties={
    ...         'relative_vorticity': np.load('relative_vorticity.npy'),
    ...         'rossby_number':        np.load('rossby_number.npy'),
    ...     },
    ...     dilation_radius=5,
    ...     stats=['mean', 'std'],
    ...     percentiles=[10, 90],
    ...     min_npix=5,
    ...     nan_policy='omit', 
    ... )

    """

    # Input validation
    # ------------------------------------------------------------------
    if stats is None:
        stats = ['mean', 'std', 'median']
    
    if nan_policy not in ("propagate", "omit"):
        raise ValueError("nan_policy must be 'propagate' or 'omit'")

    for prop_name, prop_arr in properties.items():
        if prop_arr.shape != labeled_fronts.shape:
            raise ValueError(f"{prop_name} shape does not match labeled_fronts")


    # identify front labels and original pixel counts
    # ------------------------------------------------------------------
    all_labels, all_counts = np.unique(labeled_fronts, return_counts=True)

    # define all fronts
    bg_mask  = all_labels > 0
    flabels  = all_labels[bg_mask]
    npix     = all_counts[bg_mask]

    if len(flabels) == 0:
        return pd.DataFrame()

    # define fronts large enough to keep
    keep     = npix >= min_npix
    flabels  = flabels[keep]
    npix     = npix[keep]

    if len(flabels) == 0:
        return pd.DataFrame()


    # Build the label array used for statistics
    # ------------------------------------------------------------------
    if dilation_radius > 0:
        stat_labels = _dilate_labeled_array(labeled_fronts, flabels,
                                            dilation_radius,
                                            mask=front_pixel_mask)
    else:
        # Restrict to valid labels only (zero out filtered-out fronts)
        if len(flabels) < len(all_labels[bg_mask]):
            max_lbl = int(labeled_fronts.max())
            lut = np.zeros(max_lbl + 1, dtype=labeled_fronts.dtype)
            lut[flabels] = flabels
            stat_labels = lut[labeled_fronts]
        else:
            stat_labels = labeled_fronts


    # Compute statistics
    # ------------------------------------------------------------------
    result = {'flabel': flabels, 'npix':   npix, }

    for prop_name, prop_arr in tqdm(properties.items(), total=len(properties),
                                    desc='Front properties', unit='ch'):
        prop_float = prop_arr.astype(np.float64)

        for stat in stats:
            col = f'{prop_name}_{stat}'
            if stat == 'count':
                result[col] = npix.copy()
                continue

            if nan_policy == 'omit':
                nan_fn = _NANFUNC[stat]
                values = ndimage.labeled_comprehension(
                    prop_float,
                    labels=stat_labels,
                    index=flabels,
                    func=nan_fn,
                    out_dtype=np.float64,
                    default=np.nan,
                )
            elif stat in _NDIMAGE_FUNC:       # 'propagate', vectorised
                values = _NDIMAGE_FUNC[stat](
                    prop_float,
                    labels=stat_labels,
                    index=flabels,
                )
            else:                             # 'propagate', no ndimage form
                values = ndimage.labeled_comprehension(
                    prop_float,
                    labels=stat_labels,
                    index=flabels,
                    func=_PLAINFUNC[stat],
                    out_dtype=np.float64,
                    default=np.nan,
                )

            result[col] = np.asarray(values, dtype=np.float64)

        # Percentiles — always use nan-aware path
        if percentiles is not None:
            for pct in percentiles:
                pct_label = f'{int(pct)}' if pct == int(pct) else f'{pct}'
                col = f'{prop_name}_p{pct_label}'

                if nan_policy == 'omit':
                    pct_fn = lambda x, q=pct: np.nanpercentile(x, q)
                else:
                    pct_fn = lambda x, q=pct: np.percentile(x, q)

                values = ndimage.labeled_comprehension(
                    prop_float,
                    labels=stat_labels,
                    index=flabels,
                    func=pct_fn,
                    out_dtype=np.float64,
                    default=np.nan,
                )
                result[col] = np.asarray(values, dtype=np.float64)

    return pd.DataFrame(result)
