# front-finding

Ocean front detection and characterisation for the LLC4320 model.

Fronts are found by thresholding the squared buoyancy gradient (`gradb2`)
against a local percentile, sharpened onto the gradient ridge, thinned to
single-pixel skeletons, and de-spurred. Each connected front is then labelled,
measured geometrically, and co-located with the physical property fields that
surround it.

Global field generation is delegated to
[llc4320-native-grid-preprocessing](https://github.com/Sea-Meets-the-Stars/llc4320-native-grid-preprocessing)
(the `dbof` package); this repository finds, groups, co-locates and publishes.

## Install

```bash
pip install -e .
```

`dbof` is a hard dependency and is pulled from its git remote. When developing
both repositories together, install it editable first so it is not re-fetched:

```bash
pip install -e ../llc4320-native-grid-preprocessing
pip install -e .
```

Requires Python >= 3.12 and `$OS_OGCM` set to the root of the ocean-model data
tree; products are written under `$OS_OGCM/LLC/Fronts/`. On the compute box:

```bash
export OS_OGCM=/mnt/tank/Oceanography/data/OGCM
```

Export it *before* launching Jupyter — a running kernel does not inherit a
later `export`. The notebook sets it in its first cell if your shell did not.

## Usage

```bash
build-fronts --config configs/run/run_v5_100_timesteps.yaml --steps gradb2
```

| Step | Does |
|---|---|
| `gradb2` | Build the store that owns `gradb2` if needed, then export the channel |
| `find` | Threshold `gradb2` into a binary front map |
| `group` | Label the fronts and measure their geometric properties |
| `colocate` | Build the remaining subsets, export their channels, co-locate |
| `push` | Copy the front products back to S3 |

Steps may be combined (`--steps find,group`) and always execute in pipeline
order. `--steps all` is the default. `--run-id` and `--build-version` override
the config.

`gradb2`, `find` and `group` are self-contained: a binary front map costs one
subset and one NetCDF. `colocate` is the only step that needs the other fields.

## Configuration

A run config has two blocks. `source:` holds the preprocessing repo's (dbof)
keys describing where the global fields come from — pipeline, dataset `run_id`,
S3 location, subsets, dates. It is written back out verbatim whenever a zarr
store needs building, so it keeps dbof's own key names. `finding:` holds this
repo's knobs — which detection config, ice masking, percentiles.

See [configs/run/run_v5_100_timesteps.yaml](configs/run/run_v5_100_timesteps.yaml)
for a documented example, and `front_finding.buildconfig` for the schema.

Front-detection parameters (window, threshold, thinning, sharpening, spur
length) live separately in `src/front_finding/finding/configs/`, selected by
label via `finding.config` and loaded by `front_finding.finding.config`.

## Trying it

[configs/run/run_test_single_timestep.yaml](configs/run/run_test_single_timestep.yaml)
runs one snapshot end to end. It points at a new S3 folder, so nothing exists
there yet and the `gradb2` step triggers a dbof run — which is the point: it
exercises generation, not just the front stages.

```bash
pip install -e ".[notebook]"
jupyter notebook notebooks/test_run_single_timestep.ipynb
```

The notebook drives all four steps and then loads the output: front counts,
length and orientation distributions, a map of front centroids, a zoom on the
longest front, and the co-located per-front table. Or from the CLI:

```bash
build-fronts --config configs/run/run_test_single_timestep.yaml \
    --build-version TEST01 --steps gradb2,find,group
```

## Tests

```bash
pytest -q
```

147 tests, no network. The integration tests build a Fronts tree in a temp
directory and run the real find/group/colocate stages; everything else is
unit-level or pins the contract with dbof.

## Layout

```
src/front_finding/
  buildconfig.py   typed run configuration
  store.py         the zarr store every product is written to and read from
  cli/             build-fronts entry point
  finding/         detection: thresholding, sharpening, thinning, spur removal
  llc/             source-field reads, S3 publication
  properties/      labelling, geometry, co-location
configs/run/       run configurations
docs/store.md      the store's layout and reader API
notebooks/         worked examples against a real run
tests/             unit, contract and integration tests
```

## Outputs

Everything a build produces is **one zarr store** — no NetCDF, no parquet, no
metadata sidecars:

```
{products.root}/{build_version}/{pipeline}/fronts.zarr/
├── zarr.json                    build + source provenance
└── 20111204_000000/
    ├── binary      (j, i) bool   front pixels                    <- find
    ├── labels      (j, i) int32  connected components            <- group
    ├── geometry/   per-front shape: length, orientation, bbox    <- group
    └── properties/ per-front field statistics                    <- colocate
```

Read it from anywhere:

```python
from front_finding.store import FrontStore

store = FrontStore.open("output/TEST01/SURF/fronts.zarr")
store.status()                    # what finished, per snapshot
store.fronts("20111204_000000")   # one row per front
store.labels(date, window=(y0, y1, x0, x1))   # just that crop
store.dataset()                   # every snapshot, concatenated
```

Rasters are chunked so a window read never materialises the global grid, and
tables are stored one array per column so reading one property does not pull
the rest. A step is only "done" once it has written a completion marker, so an
interrupted run is re-runnable rather than silently half-claimed.

See [docs/store.md](docs/store.md) for the layout, the table columns, the
provenance attributes, and how to port code that read the old per-file outputs.
