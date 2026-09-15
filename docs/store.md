# The front store

Everything a build produces is one zarr store. There are no sidecar files — no
NetCDF, no parquet, no metadata JSON. If you want a run's output, you open the
store.

```python
from front_finding.store import FrontStore

store = FrontStore.open("s3://dbof/.../fronts.zarr")   # or a local path
store.status()                    # what finished, per snapshot
store.fronts("20111204_000000")   # one row per front
store.dataset()                   # every snapshot, concatenated
```

The store is the published artifact, not an internal format. Nothing below is
private to the pipeline.

---

## Layout

One store per build, with a group per snapshot:

```
{products.root}/{build_version}/{pipeline}/fronts.zarr/
├── zarr.json                    build + source provenance (root attributes)
│
├── 20111204_000000/
│   ├── zarr.json                which steps have run here, and with what
│   ├── binary      (j, i) bool   front pixels                    ← find
│   ├── binary_unprocessed  (j, i) bool  pre-post-processing      ← find
│   ├── labels      (j, i) int32  connected components, 0 = none  ← group
│   ├── geometry/   one 1-D array per column                      ← group
│   ├── properties/ one 1-D array per column                      ← colocate
│   └── cross_properties/  same columns, the front's surroundings ← colocate
│
└── 20111206_180000/ …
```

Snapshot groups are named `YYYYMMDD_HHMMSS`.

**Rasters** are chunked at 1080 × 1080, which divides the LLC4320 rectangular
grid (12960 × 17280) exactly into 12 × 16 chunks. A window read touches only
the chunks it overlaps, so cropping one front out of the global grid costs
milliseconds and never materialises the whole array.

**Tables** are stored one array per column, so reading `gradb2_median` does not
pull the other forty-nine columns. Column order is recorded in the table
group's attributes — zarr lists arrays alphabetically, which would otherwise
silently reorder every table on read.

---

## Reading

### Discovery

```python
store.dates                       # ['20111204_000000', ...] sorted
store.status()                    # DataFrame: date × step
store.has(date, "colocate")       # bool
store.pending("group")            # dates where the step still needs to run
store.attrs                       # build + source provenance
store.step_attrs(date, "find")    # what that step recorded
```

`status()` gives one row per snapshot:

| date | find | group | colocate |
|---|---|---|---|
| 20111204_000000 | done | done | done |
| 20111206_180000 | done | partial | missing |

- **done** — the step finished and wrote its completion marker.
- **partial** — arrays are present but there is no marker. The step was
  interrupted. It is safe to re-run; see [Crash safety](#crash-safety).
- **missing** — nothing there.

### Rasters

```python
store.labels(date)                            # lazy zarr array — no data read
store.labels(date, window=(y0, y1, x0, x1))   # numpy, just that crop
store.binary(date, window=...)
store.binary_unprocessed(date, window=...)    # before any post-processing
```

`binary_unprocessed` is the threshold's own verdict, kept before sharpening,
thinning, cropping and spur removal narrow it. Reach for it when you need
candidate pixels the post-processing discarded — a front's full width, or what
a rejected component looked like. It is denser than `binary` and costs
proportionally more on disk. Stores built before it was kept do not have it,
so it raises KeyError there rather than returning an empty array.

Without a `window` you get the **lazy zarr array**, not a numpy copy — the
global grid is 224 million cells. Slice it, or pass a window.

To crop one front, take its bounding box from the geometry table:

```python
row = store.geometry(date).iloc[0]
crop = store.labels(date, window=(row.y0, row.y1, row.x0, row.x1))
mask = crop == row.label
```

### Tables

```python
store.geometry(date)                          # all columns
store.geometry(date, columns=["length_km"])   # just one array read
store.properties(date, columns=[...])
store.fronts(date)                            # geometry ⋈ properties
```

`fronts()` joins on `geometry.label == properties.flabel`, drops `flabel`, and
keys on `label`. If co-location has not run for that snapshot it returns
geometry alone, so a half-built store is still readable.

### The dataset

```python
store.dataset()                               # every snapshot, `date` column added
store.dataset(dates=[...], property_columns=[...])
```

Snapshots that have not been grouped are skipped. This concatenates in memory —
fine at ~117k fronts × 100 snapshots (≈11.7M rows), but it is not lazy.

---

## What the tables hold

### `geometry` — written by `group`, one row per front

| column | |
|---|---|
| `label` | integer front id; the key for everything else |
| `name` | stable string id, `{time}_{lat}_{lon}`, e.g. `20111204TT000000_78.4S_164.2W` |
| `time` | snapshot, ISO 8601 |
| `npix` | pixels in the front |
| `y0`, `y1`, `x0`, `x1` | bounding box on the global grid |
| `centroid_lat`, `centroid_lon` | centroid |
| `lat_min`, `lat_max`, `lon_min`, `lon_max` | extent |
| `length_km` | haversine length along the skeleton |
| `orientation` | degrees |
| `num_branches` | skeleton junctions |
| `mean_curvature`, `curvature_direction` | NaN when `skip_curvature` was set |

### `properties` — written by `colocate`, one row per front

`flabel`, `npix`, then one column per channel per statistic:

```
{channel}_mean  {channel}_std  {channel}_median  {channel}_p25  {channel}_p75  {channel}_p90
```

Which channels and which percentiles were used are recorded in
`store.step_attrs(date, "colocate")`. Values are sampled over each front
**dilated by `properties_dilation_radius` pixels**, so they describe a band
around the front rather than the line itself — check that attribute before
interpreting them.

When `dilate_only_to_front_pixels` was set, that band is further restricted to
pixels present in `binary_unprocessed`, so it follows the gradient ridge's real
width rather than a disc. The front's own pixels are never masked. Pixels the
mask drops fall into `cross_properties`, which stays unmasked.

### `cross_properties` — written by `colocate`, one row per front

The same columns as `properties`, describing a band **around** the front
instead of the front itself: everything within
`properties_dilation_radius + properties_cross_front_radius` pixels, minus the
pixels `properties` already sampled.

Unlike `properties`, this region is deliberately **not** a partition of the
grid. Neighbouring fronts and their bands fall inside it, and two fronts closer
than twice the outer radius share pixels. It answers "what is this front sitting
in", where `properties` answers "what is this front made of".

Present only when the run set a non-zero `properties_cross_front_radius`:

```python
store.has_cross_properties(date)
store.cross_properties(date, columns=["gradb2_mean"])
store.fronts(date, cross=True)     # joins it, columns prefixed `cross_`
```

`npix` there counts the band geometrically. Land inside it is dropped by
`nan_policy` when the statistics are taken, not by the count, so a coastal
front can report a large `npix` backed by few real values.

### Types

`write_table` narrows `float64 → float32` and `int64 → int32` on the way in.
The values are statistics of float32 fields, so the extra precision was noise
(measured worst relative error 2.9e-08), and no snapshot produces more fronts
than int32 holds. Label rasters are likewise int32.

---

## Provenance

Root attributes — what built this store:

```python
{'build_version': 'TEST01', 'pipeline': 'SURF', 'run_id': 'test01',
 'config_file': '/…/run_test_single_timestep.yaml',
 'source_bucket': 'dbof/', 'source_folder': 'test_globals_for_front_finding/',
 'finding_config': 'D', 'gradb2_channel': 'gradb2',
 'gradb2_subset': 'frontal_structure', 'dates': ['2011-12-04 00:00:00']}
```

`run_id` is the **source** dataset the fields came from. `build_version` and
`pipeline` are this build, and appear in the store's path. Products are
organised by the build that made them; the source is recorded here rather than
smuggled into filenames.

Per-snapshot attributes — what each step did:

```python
store.step_attrs(date, "find")
{'done': '2026-09-02T16:44:46+00:00', 'n_front_px': 3985035,
 'config': 'D', 'gradb2_channel': 'gradb2', 'gradb2_subset': 'frontal_structure'}

store.step_attrs(date, "colocate")
{'done': '…', 'n_fronts': 116982, 'channels': [...], 'percentiles': [25, 75, 90],
 'properties_dilation_radius': 1, 'properties_cross_front_radius': 0,
 'min_npix': 1, 'nan_policy': 'omit'}
```

---

## Crash safety

Zarr writes are not atomic, and a build can die halfway through a snapshot. So
arrays are written **first** and the step is marked done **last**.

**Never infer completion from an array existing. Ask `has()`.**

A step that was interrupted leaves arrays behind with no marker, which means:

- `has(date, step)` is `False`, so the step re-runs and overwrites
- `status()` shows `partial` rather than hiding it
- `pending(step)` returns that snapshot

Re-running a step deletes its arrays before rewriting, so a partial write
cannot leave stale columns or a stale array behind.

The unit of protection is **one step of one snapshot**. There is no
checkpointing *within* a step: if `find` dies nine minutes into a ten-minute
threshold, nothing is saved and it starts over. Across many snapshots that is
the right granularity; on a single long snapshot it means no partial credit.

---

## Writing

The pipeline's steps call these; you should not normally need them.

```python
store = FrontStore.open(url, mode="a")     # 'a' adds to an existing build
store.set_build_attrs(**provenance)

store.write_binary(date, arr, **info)                  # marks `find` done
store.write_group(date, labels, geometry_df, **info)   # marks `group` done
store.write_properties(date, properties_df, **info)    # marks `colocate` done
```

`write_raster` and `write_table` are the primitives beneath those and
deliberately mark nothing — only the step-level wrappers claim completion.

Open modes: `'r'` read-only, `'a'` read/write creating if absent, `'w'` create
fresh **discarding anything already there**.

---

## Publishing

`build-fronts --steps push` copies the store to S3, beside the source stores it
was built from:

```
s3://{bucket}/{folder}/{run_id}/{YYYYMMDD_HHMMSS}/frontal_structure.zarr   ← source
s3://{bucket}/{folder}/{run_id}/Fronts/{build_version}/{pipeline}/fronts.zarr  ← products
```

The store spans every snapshot, so it sits beside the date directories rather
than inside one, and carries the build version and pipeline in its key so two
builds of the same source dataset cannot overwrite each other.

Snapshots publish independently — pushing one date uploads that group plus the
store's root metadata, so a long run can publish as it goes. Chunks are
immutable once written and are skipped if already present; root metadata is
always refreshed, because it gains a key each time a snapshot is added.

---

## Porting code that read the old files

The previous layout wrote seven files per snapshot. Equivalents:

| was | now |
|---|---|
| `np.load(LLC4320_*_bfronts.npy)` | `store.binary(date)` |
| `np.load(labeled_fronts_global_*.npy)` | `store.labels(date)` |
| `np.load(..., mmap_mode='r')[y0:y1, x0:x1]` | `store.labels(date, window=(y0,y1,x0,x1))` |
| `pd.read_parquet(global_front_geometry_*)` | `store.geometry(date)` |
| `pd.read_parquet(front_properties_*)` | `store.properties(date)` |
| `pd.read_parquet(front_index_*)` | `store.geometry(date)` — it was a strict subset |
| `merge_geometry_colocation(g, p)` | `store.fronts(date)` |
| `json.load(metadata_*.json)` | `store.step_attrs(date, "group")` |
| `json.load(metadata_properties_*.json)` | `store.step_attrs(date, "colocate")` |
| `fronts_meta_*.meta` | `store.attrs` |

The `front_index` table is gone: its columns and values were a verified subset
of `geometry`.

Filename parsing is gone too. There is no `run_tag`, no `_bfronts` suffix, and
nothing to recover a date from — the snapshot is the group name and the source
`run_id` is a root attribute.
