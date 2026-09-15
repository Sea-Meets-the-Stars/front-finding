"""Unit tests for labelling, geometry and co-location.

Pure array code -- no dbof, no S3.  Co-location is the stage that turns fronts
into a per-front table, so its statistics and its dilation are pinned here in
detail.
"""
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from scipy.stats import skew

from front_finding.properties import colocation, geometry, group_labels


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_fronts():
    """32x32 binary with two disjoint vertical fronts, 10 px each."""
    b = np.zeros((32, 32), dtype=bool)
    b[5:15, 8] = True
    b[5:15, 24] = True
    return b


@pytest.fixture
def latlon():
    """A regular 1-degree grid, so distances are easy to reason about."""
    lat = np.repeat(np.arange(32, dtype=float)[:, None], 32, axis=1)
    lon = np.repeat(np.arange(32, dtype=float)[None, :], 32, axis=0)
    return lat, lon


# ---------------------------------------------------------------------------
#  Labelling
# ---------------------------------------------------------------------------

def test_label_fronts_counts_components(two_fronts):
    labeled, n = group_labels.label_fronts(two_fronts, connectivity=2,
                                           return_num=True)
    assert n == 2
    assert set(np.unique(labeled)) == {0, 1, 2}


def test_diagonal_pixels_join_under_connectivity_2():
    b = np.zeros((10, 10), dtype=bool)
    b[2, 2] = b[3, 3] = b[4, 4] = True
    _, n8 = group_labels.label_fronts(b, connectivity=2, return_num=True)
    _, n4 = group_labels.label_fronts(b, connectivity=1, return_num=True)
    assert n8 == 1 and n4 == 3


def test_get_front_properties_reports_bbox_and_centroid(two_fronts):
    labeled = group_labels.label_fronts(two_fronts, connectivity=2)
    props = group_labels.get_front_properties(labeled)
    assert set(props) == {1, 2}
    for p in props.values():
        min_row, min_col, max_row, max_col = p['bbox']
        assert max_row - min_row == 10          # 10 px tall
        assert max_col - min_col == 1           # 1 px wide


def test_front_ids_carry_the_timestamp_and_are_unique(two_fronts, latlon):
    lat, lon = latlon
    labeled = group_labels.label_fronts(two_fronts, connectivity=2)
    props = group_labels.get_front_properties(labeled)
    ids = group_labels.generate_front_ids(
        lat, lon, 'LLC4320_2012-11-09T12_00_00_v1_bfronts.npy', properties=props)
    assert len(set(ids.values())) == 2          # distinct
    assert all('2012' in v for v in ids.values())


# ---------------------------------------------------------------------------
#  Geometry
# ---------------------------------------------------------------------------

def test_haversine_matches_a_known_degree_of_latitude():
    """One degree of latitude is ~111 km anywhere on the sphere."""
    d = geometry.haversine_distance(0.0, 0.0, 1.0, 0.0)
    assert 110.0 < d < 112.0


def test_haversine_is_zero_for_a_point():
    assert geometry.haversine_distance(45.0, 30.0, 45.0, 30.0) == pytest.approx(0.0)


def test_front_length_grows_with_the_front(latlon):
    lat, lon = latlon
    short = np.zeros((32, 32), dtype=bool); short[5:10, 8] = True
    long_ = np.zeros((32, 32), dtype=bool); long_[5:25, 8] = True
    assert (geometry.calculate_front_length(long_, lat, lon)
            > geometry.calculate_front_length(short, lat, lon))


def test_branch_points_zero_on_a_line_and_nonzero_on_a_tee():
    line = np.zeros((20, 20), dtype=bool); line[5:15, 10] = True
    tee = line.copy(); tee[10, 11:16] = True
    assert geometry.calculate_branch_points(line) == 0
    assert geometry.calculate_branch_points(tee) >= 1


def test_orientation_distinguishes_vertical_from_horizontal():
    vert = np.zeros((20, 20), dtype=bool); vert[5:15, 10] = True
    horiz = np.zeros((20, 20), dtype=bool); horiz[10, 5:15] = True
    assert geometry.calculate_front_orientation(vert) != \
        geometry.calculate_front_orientation(horiz)


def test_process_single_front_returns_the_expected_columns(latlon):
    lat, lon = latlon
    mask = np.zeros((32, 32), dtype=bool); mask[5:15, 8] = True
    props = geometry.process_single_front(
        1, 'front_a', mask, lat, lon, '2012-11-09T12:00:00',
        y0=5, y1=15, x0=8, x1=9)
    assert props['label'] == 1 and props['npix'] == 10
    for key in ('centroid_lat', 'centroid_lon', 'length_km',
                'num_branches', 'orientation'):
        assert key in props


# ---------------------------------------------------------------------------
#  Co-location -- one row per front
# ---------------------------------------------------------------------------

def test_colocate_returns_one_row_per_front(two_fronts):
    labeled = group_labels.label_fronts(two_fronts, connectivity=2)
    df = colocation.colocate_fronts_with_properties(
        labeled, {'p': np.ones_like(labeled, dtype=float)})
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df['flabel']) == [1, 2]
    assert list(df['npix']) == [10, 10]


def test_statistics_are_computed_per_front(two_fronts):
    """Each front sees only its own pixels."""
    labeled = group_labels.label_fronts(two_fronts, connectivity=2)
    prop = np.zeros_like(labeled, dtype=float)
    prop[labeled == 1] = 2.0
    prop[labeled == 2] = 8.0
    df = colocation.colocate_fronts_with_properties(
        labeled, {'p': prop}, stats=['mean', 'std', 'median', 'min', 'max'])
    assert list(df['p_mean']) == [2.0, 8.0]
    assert list(df['p_min']) == [2.0, 8.0]
    assert list(df['p_max']) == [2.0, 8.0]
    assert list(df['p_std']) == [0.0, 0.0]


def test_percentile_columns_are_named_and_correct(two_fronts):
    labeled = group_labels.label_fronts(two_fronts, connectivity=2)
    prop = np.zeros_like(labeled, dtype=float)
    prop[5:15, 8] = np.arange(10)              # 0..9 along front 1
    prop[5:15, 24] = np.arange(10)
    df = colocation.colocate_fronts_with_properties(
        labeled, {'p': prop}, stats=['median'], percentiles=[25, 90])
    assert 'p_p25' in df.columns and 'p_p90' in df.columns
    assert df['p_median'].iloc[0] == pytest.approx(4.5)
    assert df['p_p25'].iloc[0] == pytest.approx(2.25)


def test_min_npix_drops_small_fronts():
    b = np.zeros((32, 32), dtype=bool)
    b[5:15, 8] = True                          # 10 px
    b[20, 20] = True                           # 1 px
    labeled = group_labels.label_fronts(b, connectivity=2)
    df = colocation.colocate_fronts_with_properties(
        labeled, {'p': np.ones_like(labeled, dtype=float)}, min_npix=5)
    assert len(df) == 1 and df['npix'].iloc[0] == 10


def test_nan_policy_omit_ignores_land(two_fronts):
    """propagate lets a NaN poison the front; omit steps over it."""
    labeled = group_labels.label_fronts(two_fronts, connectivity=2)
    prop = np.ones_like(labeled, dtype=float)
    prop[5, 8] = np.nan                        # one bad pixel on front 1
    poisoned = colocation.colocate_fronts_with_properties(
        labeled, {'p': prop}, stats=['mean'], nan_policy='propagate')
    clean = colocation.colocate_fronts_with_properties(
        labeled, {'p': prop}, stats=['mean'], nan_policy='omit')
    assert np.isnan(poisoned['p_mean'].iloc[0])
    assert clean['p_mean'].iloc[0] == pytest.approx(1.0)


def test_dilation_samples_the_field_around_the_front(two_fronts):
    """This is what makes the row describe a neighbourhood, not just a line."""
    labeled = group_labels.label_fronts(two_fronts, connectivity=2)
    prop = np.zeros_like(labeled, dtype=float)
    prop[labeled > 0] = 1.0                    # on-front value
    prop[5:15, 9] = 5.0                        # one column to the right

    on_front = colocation.colocate_fronts_with_properties(
        labeled, {'p': prop}, stats=['max'], dilation_radius=0)
    around = colocation.colocate_fronts_with_properties(
        labeled, {'p': prop}, stats=['max'], dilation_radius=1)
    assert on_front['p_max'].iloc[0] == 1.0    # never saw the neighbour
    assert around['p_max'].iloc[0] == 5.0      # dilation reached it


def test_dilation_does_not_merge_adjacent_fronts():
    """Two fronts 4 px apart, dilated by 1, must stay two rows."""
    b = np.zeros((32, 32), dtype=bool)
    b[5:15, 10] = True
    b[5:15, 14] = True
    labeled = group_labels.label_fronts(b, connectivity=2)
    df = colocation.colocate_fronts_with_properties(
        labeled, {'p': np.ones_like(labeled, dtype=float)}, dilation_radius=1)
    assert len(df) == 2


def test_npix_is_the_undilated_count(two_fronts):
    """npix describes the front, not the sampled neighbourhood."""
    labeled = group_labels.label_fronts(two_fronts, connectivity=2)
    df = colocation.colocate_fronts_with_properties(
        labeled, {'p': np.ones_like(labeled, dtype=float)}, dilation_radius=3)
    assert list(df['npix']) == [10, 10]


def test_mismatched_property_shape_is_rejected(two_fronts):
    labeled = group_labels.label_fronts(two_fronts, connectivity=2)
    with pytest.raises(ValueError, match='shape'):
        colocation.colocate_fronts_with_properties(
            labeled, {'p': np.ones((8, 8))})


def test_every_layer_defaults_to_omit(two_fronts):
    """The default disagreed between layers; land must not poison a front."""
    for fn in (colocation.colocate_fronts_with_properties,
               colocation.cross_front_properties):
        assert (inspect.signature(fn).parameters['nan_policy'].default
                == 'omit'), fn.__name__
    labeled = group_labels.label_fronts(two_fronts, connectivity=2)
    prop = np.ones_like(labeled, dtype=float)
    prop[5, 8] = np.nan                             # one land pixel on front 1
    out = colocation.colocate_fronts_with_properties(
        labeled, {'p': prop}, stats=['mean'])       # no nan_policy given
    assert out['p_mean'].iloc[0] == pytest.approx(1.0)   # omit stepped over it


# ---------------------------------------------------------------------------
#  Skew -- the one stat scipy.ndimage cannot do
# ---------------------------------------------------------------------------

def test_skew_matches_scipy_on_clean_data(two_fronts):
    labeled = group_labels.label_fronts(two_fronts, connectivity=2)
    rng = np.random.default_rng(0)
    prop = rng.gamma(2.0, size=labeled.shape)          # genuinely skewed
    out = colocation.colocate_fronts_with_properties(
        labeled, {'p': prop}, stats=['skew'])
    assert out.loc[0, 'p_skew'] == pytest.approx(skew(prop[labeled == 1]))


@pytest.mark.parametrize('policy', ['omit', 'propagate'])
def test_skew_works_under_both_nan_policies(two_fronts, policy):
    """scipy.ndimage has no skew, so propagate must fall back, not KeyError."""
    labeled = group_labels.label_fronts(two_fronts, connectivity=2)
    rng = np.random.default_rng(0)
    prop = rng.gamma(2.0, size=labeled.shape)
    out = colocation.colocate_fronts_with_properties(
        labeled, {'p': prop}, stats=['skew'], nan_policy=policy)
    assert np.isfinite(out['p_skew']).all()


def test_skew_propagates_a_single_nan(two_fronts):
    labeled = group_labels.label_fronts(two_fronts, connectivity=2)
    rng = np.random.default_rng(0)
    prop = rng.gamma(2.0, size=labeled.shape)
    prop[5, 8] = np.nan                                # one land pixel, front 1
    propagated = colocation.colocate_fronts_with_properties(
        labeled, {'p': prop}, stats=['skew'], nan_policy='propagate')
    omitted = colocation.colocate_fronts_with_properties(
        labeled, {'p': prop}, stats=['skew'], nan_policy='omit')
    assert np.isnan(propagated['p_skew'].iloc[0])
    assert np.isfinite(omitted['p_skew'].iloc[0])
    assert np.isfinite(propagated['p_skew'].iloc[1])   # the clean front survives


def test_skew_of_a_flat_band_is_nan(two_fronts):
    """Zero variance is 0/0; scipy warns and returns garbage, we say undefined."""
    labeled = group_labels.label_fronts(two_fronts, connectivity=2)
    out = colocation.colocate_fronts_with_properties(
        labeled, {'p': np.ones_like(labeled, dtype=float)}, stats=['skew'])
    assert np.isnan(out['p_skew']).all()


def test_skew_of_too_few_points_is_nan():
    """scipy returns 0.0 for two points -- a real number for an undefined value."""
    assert np.isnan(colocation._skew(np.array([1.0, 5.0])))
    assert np.isnan(colocation._skew(np.array([1.0])))
    assert np.isfinite(colocation._skew(np.array([1.0, 2.0, 9.0])))


def test_skew_emits_no_warnings_on_the_degenerate_cases():
    """These fire constantly on real data; they must not reach the log."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for v in (np.ones(20), np.full(20, np.nan), np.array([1.0, 2.0])):
            assert np.isnan(colocation._skew(v))


def test_skew_is_reachable_from_the_cross_front_pass():
    lab = np.zeros((40, 40), np.int32)
    lab[10, 5:35] = 1
    core = colocation._dilate_labeled_array(lab, np.array([1]), 1)
    rng = np.random.default_rng(0)
    out = colocation.cross_front_properties(
        lab, core, {'p': rng.gamma(2.0, size=lab.shape)}, np.array([1]), 1, 6,
        stats=['skew'])
    assert np.isfinite(out.loc[0, 'p_skew'])


def test_bad_nan_policy_is_rejected(two_fronts):
    labeled = group_labels.label_fronts(two_fronts, connectivity=2)
    with pytest.raises(ValueError, match='nan_policy'):
        colocation.colocate_fronts_with_properties(
            labeled, {'p': np.ones_like(labeled, dtype=float)},
            nan_policy='sometimes')


def test_no_fronts_gives_an_empty_frame():
    labeled = np.zeros((16, 16), dtype=int)
    df = colocation.colocate_fronts_with_properties(
        labeled, {'p': np.ones((16, 16))})
    assert df.empty


# ---------------------------------------------------------------------------
#  Cross-front properties -- the surroundings, which may overlap
# ---------------------------------------------------------------------------

def _pair(gap=6):
    """Two parallel fronts *gap* rows apart, plus their step-1 core."""
    lab = np.zeros((40, 40), np.int32)
    lab[10, 5:35] = 1
    lab[10 + gap, 5:35] = 2
    core = colocation._dilate_labeled_array(lab, np.array([1, 2]), 1)
    return lab, core


def test_the_mask_limits_where_a_front_dilates_to():
    lab = np.zeros((20, 20), np.int32)
    lab[10, 5:15] = 1
    mask = np.zeros((20, 20), bool)
    mask[9:12, :] = True                      # one row either side only
    free = colocation._dilate_labeled_array(lab, np.array([1]), 3)
    held = colocation._dilate_labeled_array(lab, np.array([1]), 3, mask=mask)
    assert (held == 1).sum() < (free == 1).sum()
    assert not ((held == 1) & ~mask & (lab != 1)).any()


def test_the_mask_never_drops_a_fronts_own_pixels():
    """Hole-filled pixels are not in the threshold map; they must still count."""
    lab = np.zeros((20, 20), np.int32)
    lab[10, 5:15] = 1
    mask = np.zeros((20, 20), bool)           # excludes the front itself
    held = colocation._dilate_labeled_array(lab, np.array([1]), 2, mask=mask)
    assert (held == 1).sum() == int((lab == 1).sum())


def test_masked_statistics_ignore_pixels_outside_the_mask():
    lab = np.zeros((20, 20), np.int32)
    lab[10, 5:15] = 1
    field = np.ones((20, 20)); field[13, :] = 1000.0     # 3 px away
    mask = np.zeros((20, 20), bool); mask[9:12, :] = True
    out = colocation.colocate_fronts_with_properties(
        lab, {'p': field}, stats=['max'], dilation_radius=3,
        front_pixel_mask=mask)
    assert out.loc[0, 'p_max'] == 1.0


def test_cross_band_includes_neighbouring_fronts():
    """The whole point of the second pass: it is not a partition."""
    lab, core = _pair()
    field = np.ones((40, 40))
    field[lab == 2] = 1000.0                  # marker on the neighbour
    out = colocation.cross_front_properties(
        lab, core, {'p': field}, np.array([1, 2]), 1, 8, stats=['max'])
    assert out.loc[0, 'p_max'] == 1000.0      # front 1 saw front 2


def test_cross_band_excludes_the_fronts_own_core():
    lab, core = _pair()
    field = np.ones((40, 40))
    field[core == 1] = -5.0                   # marker on front 1's own band
    out = colocation.cross_front_properties(
        lab, core, {'p': field}, np.array([1]), 1, 8, stats=['min'])
    assert out.loc[0, 'p_min'] == 1.0         # the -5 never reached it


def test_cross_bands_of_two_fronts_overlap():
    """Two fronts closer than twice the outer radius share pixels."""
    lab, core = _pair(gap=6)
    masks = []
    for label in (1, 2):
        dist = ndimage.distance_transform_edt(lab != label)
        masks.append((dist <= 9) & (core != label))
    assert (masks[0] & masks[1]).sum() > 0


def test_cross_npix_counts_the_band():
    lab, core = _pair()
    out = colocation.cross_front_properties(
        lab, core, {'p': np.ones((40, 40))}, np.array([1]), 1, 8)
    dist = ndimage.distance_transform_edt(lab != 1)
    assert out.loc[0, 'npix'] == int(((dist <= 9) & (core != 1)).sum())


def test_cross_band_of_all_land_is_nan():
    """nan_policy='omit' handles land; the row stays NaN rather than erroring."""
    lab, core = _pair()
    field = np.full((40, 40), np.nan)
    out = colocation.cross_front_properties(
        lab, core, {'p': field}, np.array([1]), 1, 8, stats=['mean'])
    assert out.loc[0, 'npix'] > 0             # the band is still there
    assert np.isnan(out.loc[0, 'p_mean'])


def test_cross_columns_match_the_front_property_columns():
    """Same schema, so core and surroundings are directly comparable."""
    lab, core = _pair()
    field = np.random.default_rng(0).random((40, 40))
    kw = dict(stats=['mean', 'std'], percentiles=[90])
    near = colocation.colocate_fronts_with_properties(
        lab, {'p': field}, dilation_radius=1, **kw)
    far = colocation.cross_front_properties(
        lab, core, {'p': field}, np.array([1, 2]), 1, 8, **kw)
    assert list(near.columns) == list(far.columns)
