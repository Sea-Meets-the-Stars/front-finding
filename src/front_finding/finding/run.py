""" Run front finding """
import os

from front_finding.finding import io as finding_io
from front_finding.finding import config as find_config
from front_finding.finding import algorithms as finding_algorithms
from front_finding.llc import source as llc_source


def find_gradb2_fronts(cfg, timestamp: str, config: str, version: str,
                gradb2_field: str, gradb2_subset: str,
                clobber: bool = False):
    """Find the fronts in a gradb2 field.

    The field is read straight from the S3 zarr store into memory.

    Args:
        cfg: The resolved run config (BuildJobConfig); locates the store.
        timestamp (str): Timestamp of the data to process.
        config (str): Front-finding config label (e.g. 'A').
        version (str): Run tag used in the output filename.
        gradb2_field (str): Fully-expanded gradb2 channel name.  For the DEPTH
            pipeline this carries a suffix, e.g. 'gradb2_sfc'.
        gradb2_subset (str): The subset that owns *gradb2_field*.
        clobber (bool, optional): Overwrite an existing map. Defaults to False.
    """

    # Check if the binary front field exists
    bfile = finding_io.binary_filename(timestamp, config, version)
    if os.path.isfile(bfile) and not clobber:
        print(f"Binary front field {bfile} exists and clobber is False. Returning")
        return

    # Read gradb2 from the store
    gradb2 = llc_source.read_channel(cfg, timestamp, gradb2_field, gradb2_subset,
                                     ice_mask=cfg.finding.ice_mask_find)
    print(f"Read {gradb2_field} with shape: {gradb2.shape}")


    # Load config
    print(f"Processing config: {config}")
    config_file = find_config.config_filename(config)
    cdict = find_config.load(config_file)

    # Binary parameters
    bparam = cdict['binary']
    bparam['n_workers'] = 10
    bparam['verbose'] = True

    # Do it
    fronts = finding_algorithms.fronts_from_gradb2(gradb2, **bparam)

    # Save em
    finding_io.save_binary_fronts(
        fronts, timestamp, config, version)

