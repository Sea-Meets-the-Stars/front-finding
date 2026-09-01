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
  cli/         build-fronts entry point
  finding/     detection: thresholding, sharpening, thinning, spur removal
  llc/         product paths, run metadata, S3 publication
  properties/  labelling, geometry, co-location
configs/run/     run configurations
notebooks/       worked examples against a real run
tests/           unit, contract and integration tests
```

## Outputs

Products are organised by the build that made them; filenames keep the source
`run_id`, so a file always names the dataset it came from:

```
$OS_OGCM/LLC/Fronts/{build_version}/{pipeline}/{date_prefix}/
    LLC4320_{timestamp}_{channel}_{run_id}.nc      exported field
    LLC4320_{timestamp}_{run_id}_bfronts.npy       binary front map
    labeled_fronts_global_*.npy                    label map
    front_index_*.parquet                          one row per front: label, name, bbox
    global_front_geometry_*.parquet                length, orientation, curvature, branches
    front_properties_*.parquet                     per-front property statistics
    fronts_meta_*.meta                              run descriptor
```
