""" Front finding algorithms """
import numpy as np

from skimage import morphology

from front_finding.finding import pyboa
from front_finding.finding.sharpen import global_sharpen_pq
from front_finding.finding.despur import prune_short_spurs

def fronts_from_gradb2(gradb2, window:int=40, thin:bool=False,
                      rm_weak:float=None, dilate_radius:int=0,
                      sharpen:bool=False, 
                      despur:bool=False,
                      Lspur:int=None,
                      connectivity:int=2, 
                      threshold:float=90,
                      thresh_mode:str='generic', 
                      n_workers:int=None,
                      min_size:int=7, hole_max_size:int=4,
                      return_unprocessed:bool=False,
                      verbose:bool=False,
                      debug:bool=False):
    """
    Identifies and processes fronts from a gradient field (gradb2).

    Parameters:
    -----------
    gradb2 : ndarray
        The input gradient field from which fronts are to be identified.
    window : int, optional, default=40
        The window size used for thresholding in the front detection algorithm.
    thin : bool, optional, default=False
        If True, thins the detected fronts to single-pixel width.
    sharpen : bool, optional, default=False
        If True, sharpens the detected fronts on gradb2 to single-pixel width.
    despur : bool, optional, default=False
        If True, removes spurs from the detected fronts.
    Lspur : int, optional
        Maximum spur length in pixels (measured as branch-distance).
        Branches with distance <= Lspur are removed.
        Passed to prune_short_spurs()
    rm_weak : float, optional, default=None
        If provided, removes weak segments where gradb2 values are below this threshold.
    dilate_radius : int, optional, default=0
        If > 0, dilates the cropped fronts by this many pixels.  A radius of 1
        is a 4-connected cross, which is what skimage dilates by when given no
        footprint.  Note that a final thinning follows when thin or sharpen is
        set, which undoes most of the dilation.
    hole_max_size : int, optional, default=4
        Largest enclosed hole filled during cropping, in pixels.  This is the
        smallest closed front -- an eddy -- the pipeline can keep: fill a
        ring's interior and the final thinning collapses it to a point.
    min_size : int, optional, default=7
        Minimum component size in pixels.  Applied during cropping and again
        at the very end, after thinning and spur removal have shrunk what
        cropping measured.
    thresh_mode : str, optional, default=generic
        Thresholding mode
    n_workers : int, optional, 
        Number of workers for parallel calculations
    threshold : float, optional, default=90
        Percentile used in the cropping function to determine size threshold.
    verbose : bool, optional, default=False
        If True, prints verbose output.
    debug : bool, optional, default=False
    return_unprocessed : bool, optional, default=False
        Also return the threshold output, before any of the operations above.

    Returns:
    --------
    ndarray
        The processed front field after applying the specified operations.
    ndarray
        The threshold output, only when return_unprocessed is set.  Every
        later stage narrows or reshapes this, so it is the widest set of
        candidate front pixels the algorithm ever holds.
    """

    # Threshold
    if verbose:
        print(f'Thresholding with window size {window} and threshold {threshold} and mode {thresh_mode}')
    res_frnt_np = pyboa.front_thresh(gradb2, wndw=window, prcnt=threshold,
        mode=thresh_mode, n_workers=n_workers)

    # Kept before rm_weak, so this is the threshold's own verdict on which
    # pixels are front -- everything after here narrows or reshapes it.
    unprocessed = res_frnt_np.copy()

    if rm_weak is not None:
        res_frnt_np &= gradb2 > rm_weak

    if sharpen:
        res_frnt_np = global_sharpen_pq(res_frnt_np, gradb2,
                                   protect_endpoints=True)

    if thin:
        if verbose:
            print(f'There are {np.sum(res_frnt_np)} front pixels before thinning')
            print('Thinning...')
        res_frnt_np = morphology.thin(res_frnt_np)
        if verbose:
            print(f'There are {np.sum(res_frnt_np)} front pixels after thinning')

    if min_size > 0:
        if verbose:
            print(f'Cropping with minimum size {min_size} and connectivity {connectivity}')
        # This also fills in small holes and 
        #   requires a second thinning step (if thin=True)
        res_frnt_crop = pyboa.cropping(res_frnt_np, min_size=min_size,
                                   connectivity=connectivity,
                                   hole_max_size=hole_max_size)
    else:
        res_frnt_crop = res_frnt_np

    if dilate_radius > 0:
        res_frnt_crop = morphology.dilation(res_frnt_crop,
                                            morphology.disk(dilate_radius))

    # Thin a final time
    if thin or sharpen:
        if verbose:
            print('Thinning a final time...')
        res_frnt_crop = morphology.thin(res_frnt_crop)
        if verbose:
            print(f'There are {np.sum(res_frnt_crop)} front pixels after final thinning')

    if despur:
        res_frnt_crop = prune_short_spurs(res_frnt_crop, Lspur=Lspur)
        if verbose:
            print(f'There are {np.sum(res_frnt_crop)} front pixels after '
                  f'removing spurs shorter than {Lspur}')

    # The final thinning and spur pruning both shrink components after
    # cropping's size filter has already run, so a front can end up below
    # min_size with nothing left to catch it.  Filter once more at the end so
    # min_size means what the config says it means.
    if min_size > 0:
        res_frnt_crop = morphology.remove_small_objects(
            res_frnt_crop, min_size=min_size, connectivity=connectivity)
        if verbose:
            print(f'There are {np.sum(res_frnt_crop)} front pixels after '
                  f'dropping components smaller than {min_size}')

    if return_unprocessed:
        return res_frnt_crop, unprocessed
    return res_frnt_crop
