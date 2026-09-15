"""Unit tests for the front store.

The store is the published artifact, so these pin the contract outside programs
bind to: discovery, window reads, column selection, and the crash-safety rule
that a step is done only when it says so.
"""
import os

import numpy as np
import pandas as pd
import pytest

from front_finding.store import CHUNK, STEPS, FrontStore

D1 = "20111204_000000"
D2 = "20111206_180000"


@pytest.fixture
def store(tmp_path):
    return FrontStore.open(str(tmp_path / "fronts.zarr"), mode="w")


def _binary(n=64):
    b = np.zeros((n, n), dtype=bool)
    b[10:30, 20] = True
    b[10:30, 40] = True
    return b


def _geometry(n=2):
    return pd.DataFrame({
        "label": np.arange(1, n + 1, dtype="int64"),
        "name": [f"20111204TT000000_{i}.0N_2.0E" for i in range(n)],
        "npix": np.full(n, 20, dtype="int64"),
        "length_km": np.linspace(10.0, 20.0, n),
    })


def _cross(n=2):
    return pd.DataFrame({
        "flabel": np.arange(1, n + 1, dtype="int64"),
        "gradb2_mean": np.linspace(9e-13, 2e-12, n),
    })


def _properties(n=2):
    return pd.DataFrame({
        "flabel": np.arange(1, n + 1, dtype="int64"),
        "gradb2_mean": np.linspace(1e-13, 5e-13, n),
    })


def _fill(store, date=D1):
    b = _binary()
    store.write_binary(date, b, config="D")
    store.write_group(date, (b * 1).astype(np.int32), _geometry())
    store.write_properties(date, _properties())


# ---------------------------------------------------------------------------
#  Discovery
# ---------------------------------------------------------------------------

def test_a_new_store_has_no_snapshots(store):
    assert store.dates == []
    assert store.status().empty
    assert store.has(D1, "find") is False


def test_status_tracks_each_step(store):
    b = _binary()
    store.write_binary(D1, b)
    assert store.status().iloc[0].to_dict() == {
        "date": D1, "find": "done", "group": "missing", "colocate": "missing"}
    store.write_group(D1, (b * 1).astype(np.int32), _geometry())
    assert store.status().iloc[0]["group"] == "done"
    store.write_properties(D1, _properties())
    assert store.status().iloc[0]["colocate"] == "done"


def test_dates_are_sorted(store):
    store.write_binary(D2, _binary())
    store.write_binary(D1, _binary())
    assert store.dates == [D1, D2]


def test_build_attrs_round_trip(store):
    store.set_build_attrs(build_version="V5", pipeline="SURF", run_id="v1")
    assert store.attrs["pipeline"] == "SURF"


def test_step_attrs_record_the_parameters(store):
    store.write_binary(D1, _binary(), config="D", gradb2_channel="gradb2")
    a = store.step_attrs(D1, "find")
    assert a["config"] == "D"
    assert a["gradb2_channel"] == "gradb2"
    assert a["n_front_px"] == 40
    assert a["done"]


def test_step_attrs_of_an_absent_snapshot_is_empty(store):
    assert store.step_attrs("29990101_000000", "find") == {}


# ---------------------------------------------------------------------------
#  Crash safety -- arrays first, marker last
# ---------------------------------------------------------------------------

def test_arrays_without_a_marker_are_partial_not_done(store):
    """The rule the whole design rests on: never infer done from an array."""
    store.write_raster(D1, "binary", _binary())
    assert store.has(D1, "find") is False
    assert store.status().iloc[0]["find"] == "partial"


def test_pending_returns_interrupted_and_missing_steps(store):
    store.write_raster(D1, "binary", _binary())      # interrupted
    store.write_binary(D2, _binary())                # done
    assert store.pending("find") == [D1]
    assert set(store.pending("group")) == {D1, D2}


def test_rerunning_a_step_replaces_what_was_there(store):
    store.write_binary(D1, _binary())
    store.write_group(D1, np.zeros((64, 64), dtype=np.int32), _geometry(2))
    store.write_group(D1, np.zeros((64, 64), dtype=np.int32), _geometry(5))
    assert len(store.geometry(D1)) == 5              # no stale rows left


# ---------------------------------------------------------------------------
#  Rasters
# ---------------------------------------------------------------------------

def test_rasters_are_chunked_and_downcast(store):
    b = _binary()
    store.write_binary(D1, b)
    store.write_group(D1, (b * 1).astype(np.int64), _geometry())
    assert store.binary(D1).dtype == bool
    assert store.labels(D1).dtype == np.int32        # int64 narrowed
    assert store.labels(D1).chunks == CHUNK


def test_binary_unprocessed_round_trips(store):
    b = _binary()
    raw = b.copy(); raw[10:30, 21] = True          # wider before thinning
    store.write_binary(D1, b, unprocessed=raw)
    assert store.binary_unprocessed(D1)[:].sum() > store.binary(D1)[:].sum()
    assert store.step_attrs(D1, "find")["n_unprocessed_px"] == int(raw.sum())


def test_binary_unprocessed_is_absent_when_not_written(store):
    """Stores built before it was kept must fail loudly, not silently."""
    store.write_binary(D1, _binary())
    with pytest.raises(KeyError):
        store.binary_unprocessed(D1)


def test_a_raster_without_a_window_stays_lazy(store):
    """The global grid is 224 million cells; reading it must be deliberate."""
    store.write_binary(D1, _binary())
    assert not isinstance(store.binary(D1), np.ndarray)


def test_a_window_returns_just_that_crop(store):
    store.write_binary(D1, _binary())
    crop = store.binary(D1, window=(10, 30, 19, 21))
    assert isinstance(crop, np.ndarray)
    assert crop.shape == (20, 2)
    assert crop[:, 1].all()                          # the front at column 20


# ---------------------------------------------------------------------------
#  Tables
# ---------------------------------------------------------------------------

def test_column_order_survives(store):
    """zarr lists arrays alphabetically; the table must not be reordered."""
    _fill(store)
    assert list(store.geometry(D1).columns) == list(_geometry().columns)


def test_columns_can_be_selected(store):
    _fill(store)
    assert list(store.geometry(D1, columns=["length_km"]).columns) == ["length_km"]


def test_an_unknown_column_names_what_is_available(store):
    _fill(store)
    with pytest.raises(KeyError, match="length_km"):
        store.geometry(D1, columns=["nope"])


def test_tables_are_downcast(store):
    _fill(store)
    grp = store.root[D1]["geometry"]
    assert grp["length_km"].dtype == np.float32       # float64 narrowed
    assert grp["npix"].dtype == np.int32              # int64 narrowed


def test_string_columns_round_trip(store):
    _fill(store)
    assert store.geometry(D1)["name"].iloc[0].startswith("20111204TT000000")


def test_an_unknown_table_is_rejected(store):
    with pytest.raises(ValueError, match="Unknown table"):
        store.write_table(D1, "nonsense", _geometry())


# ---------------------------------------------------------------------------
#  The joined view
# ---------------------------------------------------------------------------

def test_fronts_joins_on_the_label(store):
    _fill(store)
    joined = store.fronts(D1)
    assert len(joined) == 2
    assert "flabel" not in joined.columns            # folded into `label`
    assert "gradb2_mean" in joined.columns


def test_fronts_degrades_to_geometry_before_colocation(store):
    b = _binary()
    store.write_binary(D1, b)
    store.write_group(D1, (b * 1).astype(np.int32), _geometry())
    assert list(store.fronts(D1).columns) == list(_geometry().columns)


def test_dataset_concatenates_and_labels_each_snapshot(store):
    _fill(store, D1)
    _fill(store, D2)
    ds = store.dataset()
    assert len(ds) == 4
    assert list(ds.columns)[0] == "date"
    assert set(ds["date"]) == {D1, D2}


def test_dataset_skips_snapshots_that_are_not_grouped(store):
    _fill(store, D1)
    store.write_binary(D2, _binary())                # found, not grouped
    assert set(store.dataset()["date"]) == {D1}


def test_dataset_of_an_empty_store_is_empty(store):
    assert store.dataset().empty


# ---------------------------------------------------------------------------
#  Reopening
# ---------------------------------------------------------------------------

def test_a_reopened_store_sees_everything(store, tmp_path):
    store.set_build_attrs(run_id="v1")
    _fill(store)
    reopened = FrontStore.open(str(tmp_path / "fronts.zarr"))
    assert reopened.dates == [D1]
    assert reopened.attrs["run_id"] == "v1"
    assert all(reopened.has(D1, s) for s in STEPS)
    assert len(reopened.fronts(D1)) == 2


def test_append_mode_keeps_earlier_snapshots(tmp_path):
    url = str(tmp_path / "fronts.zarr")
    first = FrontStore.open(url, mode="w")
    _fill(first, D1)
    second = FrontStore.open(url, mode="a")
    _fill(second, D2)
    assert second.dates == [D1, D2]


def test_selecting_property_columns_keeps_the_join_key(store):
    """A column selection must not strip the key fronts() joins on."""
    _fill(store)
    joined = store.fronts(D1, property_columns=["gradb2_mean"])
    assert len(joined) == 2
    assert "gradb2_mean" in joined.columns


def test_selecting_geometry_columns_keeps_the_join_key(store):
    _fill(store)
    joined = store.fronts(D1, geometry_columns=["length_km"])
    assert "label" in joined.columns
    assert "gradb2_mean" in joined.columns


def test_dataset_honours_column_selection(store):
    _fill(store, D1)
    _fill(store, D2)
    ds = store.dataset(property_columns=["gradb2_mean"])
    assert len(ds) == 4
    assert set(ds["date"]) == {D1, D2}


# ---------------------------------------------------------------------------
#  The cross-front table
# ---------------------------------------------------------------------------

def test_cross_properties_round_trip(store):
    _fill(store)
    store.write_properties(D1, _properties(), cross_properties=_cross())
    assert store.has_cross_properties(D1)
    assert len(store.cross_properties(D1)) == 2


def test_a_snapshot_without_a_cross_radius_has_no_cross_table(store):
    _fill(store)
    assert store.has_cross_properties(D1) is False


def test_fronts_can_join_the_cross_table_under_a_prefix(store):
    _fill(store)
    store.write_properties(D1, _properties(), cross_properties=_cross())
    joined = store.fronts(D1, cross=True)
    assert "gradb2_mean" in joined.columns          # the front's own band
    assert "cross_gradb2_mean" in joined.columns    # its surroundings
    assert joined["gradb2_mean"].iloc[0] != joined["cross_gradb2_mean"].iloc[0]


def test_fronts_ignores_the_cross_table_unless_asked(store):
    _fill(store)
    store.write_properties(D1, _properties(), cross_properties=_cross())
    assert "cross_gradb2_mean" not in store.fronts(D1).columns


def test_rerunning_without_a_cross_radius_drops_the_stale_table(store):
    _fill(store)
    store.write_properties(D1, _properties(), cross_properties=_cross())
    store.write_properties(D1, _properties())       # radius turned back off
    assert store.has_cross_properties(D1) is False
