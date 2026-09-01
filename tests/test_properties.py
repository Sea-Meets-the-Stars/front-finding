"""Unit tests for labelling, geometry and co-location.

Pure array code -- no dbof, no S3.  Co-location is the stage that turns fronts
into a per-front table, so its statistics and its dilation are pinned here in
detail.
"""
import numpy as np
import pandas as pd
import pytest

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
