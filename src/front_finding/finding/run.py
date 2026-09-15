""" Run front finding """

from front_finding.finding import config as find_config
from front_finding.finding import algorithms as finding_algorithms
from front_finding.llc import source as llc_source


def find_gradb2_fronts(cfg, store, timestamp: str, date: str, config: str,
                gradb2_field: str, gradb2_subset: str,
                clobber: bool = False):
    """Find the fronts in a gradb2 field.

    The field is read straight from the S3 zarr store into memory.

    Args:
        cfg: The resolved run config (BuildJobConfig); locates the source store.
        store: The FrontStore this build writes to.
        timestamp (str): Timestamp of the data to process.
        date (str): Snapshot group in the store, ``YYYYMMDD_HHMMSS``.
        config (str): Front-finding config label (e.g. 'A').
        gradb2_field (str): Fully-expanded gradb2 channel name.  For the DEPTH
            pipeline this carries a suffix, e.g. 'gradb2_sfc'.
        gradb2_subset (str): The subset that owns *gradb2_field*.
        clobber (bool, optional): Overwrite an existing map. Defaults to False.
    """

    if store.has(date, 'find') and not clobber:
        print(f"[{date}] fronts already found; pass clobber=True to redo")
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

    keep_raw = cfg.finding.save_unprocessed_binary
    result = finding_algorithms.fronts_from_gradb2(
        gradb2, return_unprocessed=keep_raw, **bparam)
    fronts, unprocessed = result if keep_raw else (result, None)

    store.write_binary(date, fronts, unprocessed=unprocessed,
                       config=config,
                       gradb2_channel=gradb2_field,
                       gradb2_subset=gradb2_subset)

