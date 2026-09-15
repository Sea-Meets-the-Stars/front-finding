"""Unit tests for the detection stage: threshold -> sharpen -> thin -> despur.

These exercise the pure array code on synthetic fields, so they need neither
dbof nor S3.  The contract-level tests in test_build_fronts.py cover the
workflow around them; these cover what the algorithms actually compute.
"""
import numpy as np
from scipy import ndimage
from skimage import morphology
import pytest

from front_finding.finding import algorithms, despur, pyboa, sharpen


# ---------------------------------------------------------------------------
#  Fixtures -- fields with a front whose position we know exactly
# ---------------------------------------------------------------------------

#: Window used throughout.  It has to be wide relative to the front -- see
#: test_a_front_wider_than_the_window_tail_is_invisible for why.
WNDW = 32


@pytest.fixture
def ridge():
    """128x128 of low gradient with one bright 3-px ridge at column 64."""
    field = np.full((128, 128), 1e-3)
    field[:, 63:66] = 1.0          # three columns wide, so thinning has work
    return field


@pytest.fixture
def two_ridges():
    """Two separated vertical ridges -- two connected components."""
    field = np.full((128, 128), 1e-3)
    field[:, 31:34] = 1.0
    field[:, 95:98] = 1.0
    return field


# ---------------------------------------------------------------------------
#  Local-percentile thresholding
# ---------------------------------------------------------------------------

def test_front_thresh_finds_the_ridge(ridge):
    out = pyboa.front_thresh(ridge, wndw=WNDW, prcnt=90, mode='generic')
    assert out.dtype == bool
    assert out.shape == ridge.shape
    # The ridge is selected; the flat background is not.
    assert np.array_equal(np.flatnonzero(out.any(axis=0)), [63, 64, 65])
    assert out[:, :60].sum() == 0


def test_front_thresh_modes_agree(ridge):
    """generic / vectorized / pool must not disagree about where fronts are."""
    generic = pyboa.front_thresh(ridge, wndw=WNDW, prcnt=90, mode='generic')
    vector = pyboa.front_thresh(ridge, wndw=WNDW, prcnt=90, mode='vectorized')
    pool = pyboa.front_thresh(ridge, wndw=WNDW, prcnt=90, mode='pool', n_workers=2)
    assert np.array_equal(generic, vector)
    assert np.array_equal(generic, pool)


def test_front_thresh_on_a_flat_field_selects_nothing_useful():
    """A field with no structure has no ridge to sit above its own percentile."""
    out = pyboa.front_thresh(np.ones((32, 32)), wndw=8, prcnt=90, mode='generic')
    assert out.sum() == 0


def test_a_front_wider_than_the_window_tail_is_invisible():
    """A front filling more than (100 - prcnt)% of the window vanishes.

    The threshold is the local percentile and the comparison is strict, so a
    front occupying more than the tail becomes its own percentile and fails
    ``> threshold``.  At prcnt=90 the front must cover under 10% of the window
    -- which is why the shipped configs pair a 3-px-scale front with window=64.
    """
    field = np.full((64, 64), 1e-3)
    field[:, 28:37] = 1.0                     # 9 of 16 window columns
    assert pyboa.front_thresh(field, wndw=16, prcnt=90, mode='generic').sum() == 0


def test_higher_percentile_is_never_more_permissive(ridge):
    loose = pyboa.front_thresh(ridge, wndw=WNDW, prcnt=80, mode='generic')
    tight = pyboa.front_thresh(ridge, wndw=WNDW, prcnt=95, mode='generic')
    assert tight.sum() <= loose.sum()


# ---------------------------------------------------------------------------
#  Cropping (small-object removal)
# ---------------------------------------------------------------------------

def test_cropping_drops_objects_below_min_size():
    binary = np.zeros((32, 32), dtype=bool)
    binary[5, 5] = True                       # 1 px -- noise
    binary[10:25, 10] = True                  # 15 px -- a real front
    out = pyboa.cropping(binary, min_size=7, connectivity=2)
    assert not out[5, 5]
    assert out[10:25, 10].any()


# ---------------------------------------------------------------------------
#  Spur removal
# ---------------------------------------------------------------------------

def test_a_closed_front_survives_cropping():
    """A ring is an eddy, not noise.

    cropping() fills enclosed holes, and a closed front encloses one.  Fill it
    and the following thinning collapses the ring to a point, so skimage's own
    default of 64 px silently destroys every eddy under roughly 9 px across.
    """
    yy, xx = np.mgrid[-20:21, -20:21]
    dist = np.hypot(yy, xx)
    ring = morphology.thin((dist >= 2.5) & (dist <= 3.5))   # interior 21 px
    assert ring.sum() >= 7

    kept = morphology.thin(pyboa.cropping(ring, min_size=7, connectivity=2))
    assert kept.sum() >= 7, 'the ring collapsed -- an eddy was destroyed'
    assert ndimage.label(~kept)[1] > 1, 'the ring is no longer closed'


def test_hole_filling_still_cleans_pinholes():
    """The reason hole filling exists: a 1-px hole is threshold noise."""
    band = np.zeros((21, 21), dtype=bool)
    band[8:13, 4:17] = True
    band[10, 10] = False                       # a pinhole
    assert pyboa.cropping(band, min_size=7, connectivity=2)[10, 10]


def test_no_front_survives_below_min_size():
    """min_size is the final word, not just cropping's opinion.

    A small closed loop passes cropping's size filter, then cropping fills its
    hole into a solid disc and the final thinning reduces that to a single
    pixel.  Without a size filter at the end, that 1-px component reaches the
    output -- which is why a real run produced fronts of npix=1 under
    min_size=7.
    """
    from skimage import measure
    field = np.full((128, 128), 1e-3)
    field[:, 100:103] = 1.0                   # a long ridge, so output is not empty
    yy, xx = np.mgrid[0:128, 0:128]
    dist = np.hypot(yy - 40, xx - 40)
    field[(dist >= 3.5) & (dist <= 4.5)] = 1.0    # a loop that collapses to 1 px

    out = algorithms.fronts_from_gradb2(
        field, window=WNDW, threshold=90, thresh_mode='generic',
        thin=True, min_size=7)
    sizes = np.bincount(measure.label(out, connectivity=2).ravel())[1:]
    assert sizes.size, 'the ridge should survive'
    assert sizes.min() >= 7, f'components below min_size reached the output: {sizes}'


def test_prune_short_spurs_removes_a_stub_and_keeps_the_spine():
    skel = np.zeros((40, 40), dtype=bool)
    skel[5:35, 20] = True                     # 30 px spine
    skel[20, 21:25] = True                    # 4 px spur off the middle
    out = despur.prune_short_spurs(skel, Lspur=6)
    assert out[5:35, 20].sum() >= 28           # spine survives
    assert out[20, 22:25].sum() == 0           # stub gone


def test_prune_short_spurs_keeps_a_branch_longer_than_lspur():
    skel = np.zeros((40, 40), dtype=bool)
    skel[5:35, 20] = True
    skel[20, 21:34] = True                    # 13 px -- a real branch
    out = despur.prune_short_spurs(skel, Lspur=6)
    assert out[20, 25:33].any()


# ---------------------------------------------------------------------------
#  Sharpening
# ---------------------------------------------------------------------------

def test_global_sharpen_narrows_the_band_onto_the_ridge(ridge):
    binary = ridge > 0.5                      # the full 3-wide band
    out = sharpen.global_sharpen_pq(binary, ridge, protect_endpoints=True)
    assert out.sum() < binary.sum()           # it got thinner
    # and it stayed on the ridge rather than wandering into the background
    assert out[:, :63].sum() == 0 and out[:, 66:].sum() == 0


def test_global_sharpen_preserves_connectivity(two_ridges):
    from skimage import measure
    binary = two_ridges > 0.5
    out = sharpen.global_sharpen_pq(binary, two_ridges, protect_endpoints=True)
    before = measure.label(binary, connectivity=2).max()
    after = measure.label(out, connectivity=2).max()
    assert after == before == 2                # neither split nor merged


# ---------------------------------------------------------------------------
#  The assembled detector
# ---------------------------------------------------------------------------

def test_fronts_from_gradb2_thins_to_one_pixel(ridge):
    out = algorithms.fronts_from_gradb2(
        ridge, window=WNDW, threshold=90, thresh_mode='generic',
        thin=True, min_size=5)
    assert out.dtype == bool
    # Every row that has a front pixel has exactly one.
    counts = out.sum(axis=1)
    assert set(np.unique(counts[counts > 0])) == {1}


def test_fronts_from_gradb2_finds_both_fronts(two_ridges):
    from skimage import measure
    out = algorithms.fronts_from_gradb2(
        two_ridges, window=WNDW, threshold=90, thresh_mode='generic',
        thin=True, min_size=5)
    assert measure.label(out, connectivity=2).max() == 2


def test_min_size_zero_disables_cropping(ridge):
    """Config Z relies on this to inspect the raw threshold."""
    cropped = algorithms.fronts_from_gradb2(
        ridge, window=WNDW, threshold=90, thresh_mode='generic',
        thin=False, min_size=7)
    raw = algorithms.fronts_from_gradb2(
        ridge, window=WNDW, threshold=90, thresh_mode='generic',
        thin=False, min_size=0)
    assert raw.sum() >= cropped.sum()


def test_rm_weak_drops_pixels_below_the_floor(ridge):
    """A floor above the whole field leaves nothing behind."""
    out = algorithms.fronts_from_gradb2(
        ridge, window=WNDW, threshold=90, thresh_mode='generic',
        rm_weak=10.0, min_size=0)
    assert out.sum() == 0


def test_prune_short_spurs_handles_an_empty_field():
    """A fully land- or ice-masked tile has no fronts, and must not crash.

    skan cannot build a graph from an empty skeleton -- without a guard this
    fails inside scipy.sparse with 'index pointer size 0 should be 1'.
    """
    out = despur.prune_short_spurs(np.zeros((32, 32), dtype=bool), Lspur=10)
    assert out.shape == (32, 32)
    assert out.sum() == 0
