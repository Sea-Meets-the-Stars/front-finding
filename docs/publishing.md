# Publishing a build to S3

The `push` step copies a finished store to the S3 dataset it was derived from.
Nothing is converted or repacked: the same zarr store, at a key beside the
source stores.

---

## Where it lands

The key is assembled in `front_finding.llc.publish.store_s3_prefix` from the
run config's **source** block. There is no separate destination config.

```
{bucket}/{folder}/{run_id}/Fronts/{build_version}/{pipeline}/fronts.zarr
```

| segment | comes from |
|---|---|
| `bucket` | `source.output.bucket` |
| `folder` | `source.output.folder` |
| `run_id` | `source.run.run_id`, or `--run-id` |
| `Fronts` | `publish.DEFAULT_SUBFOLDER`, not configurable from the CLI |
| `build_version` | `finding.build_version`, or `--build-version` |
| `pipeline` | `source.pipeline` |
| `fronts.zarr` | fixed |

Resolving `configs/run/small_fronts_dataset_00.yaml`:

```
s3://dbof/globals_for_cutouts/v2_2_01/Fronts/SMALL_DATASET/SURF/fronts.zarr
```

Reusing the source block is deliberate -- products land beside the stores they
were built from, so they cannot be written next to the wrong dataset. Nothing
in `products:` reaches S3; that block only decides where the store lives on
disk before the push.

`Fronts/` sits *beside* the snapshot directories rather than inside one,
because the store spans every snapshot:

```
s3://dbof/globals_for_cutouts/v2_2_01/
├── 20111204_000000/     frontal_structure.zarr, kinematic.zarr, …
├── 20111216_030000/     …                        ← the source stores
├── …
└── Fronts/
    └── SMALL_DATASET/           ← build_version
        └── SURF/                ← pipeline
            └── fronts.zarr
```

`build_version` and `pipeline` are in the key so two builds of the same source
dataset cannot overwrite each other.

---

## What ends up there

The three-snapshot SMALL_DATASET build, as published:

| group | objects | size |
|---|---|---|
| `properties/` | 2181 | 318.7 MB |
| `binary_unprocessed` | 498 | 35.2 MB |
| `labels` | 490 | 28.7 MB |
| `binary` | 490 | 20.5 MB |
| `geometry/` | 168 | 16.4 MB |
| `zarr.json` | 4 | — |
| | **3831** | **0.42 GB** |

Roughly 1,276 objects per snapshot. `properties/` dominates because it is
stored a column at a time -- 241 columns -- which is what makes reading a few
features cheap.

---

## How the upload is ordered

`push_run` publishes one snapshot at a time over a single S3 connection, and
each snapshot uploads the store's **root metadata first**, then the group.
A zarr group is unreadable without root metadata, so this ordering keeps a
store that is only partly pushed openable rather than broken.

Chunks are immutable once written, so an object that already exists is
skipped. Root metadata is rewritten as snapshots are added, so it always
re-uploads regardless of `clobber`.

---

## Reading it back

Identical to a local store apart from the endpoint:

```python
from front_finding.store import FrontStore

store = FrontStore.open(
    "s3://dbof/globals_for_cutouts/v2_2_01/Fronts/SMALL_DATASET/SURF/fronts.zarr",
    storage_options={"endpoint_url": "https://s3-west.nrp-nautilus.io"},
)

store.status()                   # what finished, per snapshot
store.fronts(store.dates[0])     # one row per front
store.labels(date, window=(y0, y1, x0, x1))
```

`storage_options` is passed to `zarr.open_group`, which hands it to fsspec's
s3fs. `endpoint_url` is required -- these stores are on Nautilus, not AWS.
Credentials come from the usual boto chain (`~/.aws/credentials`, or
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`); nothing goes in the URL.

Every method behaves as it does locally, but the network is the cost. Reading
all three snapshots whole -- 378,697 fronts -- takes about 6½ minutes from S3
against seconds from local disk. Column-at-a-time storage and chunked rasters
are what make partial reads worth doing: a handful of properties, or a window
of `labels`, costs a fraction of the whole store.

---

## A pushed store can report `done` with its arrays missing

This is the one place where reading from S3 differs from reading a local build,
and it is a consequence of the two orderings being opposite.

Locally, arrays are written **first** and the step's `done` marker **last**, so
a marker guarantees the arrays exist. `status()` and `has()` therefore read
markers, and `store.md` says plainly: never infer completion from an array
existing, ask `has()`.

Push reverses it -- metadata **first**, arrays after -- so a partly-pushed store
stays readable. The cost is that the marker can arrive without its arrays. A
push interrupted between the two produces a store that reports:

```
           date find group colocate
20111216_030000 done  done     done
```

and then fails on the first real read:

```
KeyError: 'geometry'
```

`status()` cannot flag this as `partial` either: `partial` means arrays present
with no marker, and here the marker is the part that made it.

So after a push, verify arrays rather than markers:

```python
EXPECTED = ("binary", "labels", "geometry", "properties")

for date in store.dates:
    absent = [name for name in EXPECTED if name not in store.root[date]]
    print(date, "complete" if not absent else f"MISSING {absent}")
```

Two arrays are conditional and are not in that list: `binary_unprocessed`
exists only when `save_unprocessed_binary` was set, and `cross_properties`
only when `properties_cross_front_radius > 0` -- ask `has_cross_properties`.

---

## Re-running

```bash
build-fronts --config configs/run/small_fronts_dataset_00.yaml --steps push
```

Safe to repeat: missing objects upload and existing chunks skip, so an
interrupted push is finished by running it again. `clobber.push: true`
re-uploads everything instead.

`push_run` and `push_timestamp` also take `dates`, `subfolder`, `run_id` and
`dry_run` when called from Python; the CLI passes only `clobber`.
