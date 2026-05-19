"""PLAsTiCC data loading and access utilities."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_CONFIG, PREPROCESS_CONFIG


def load_training_metadata(path=None):
    """Load training_set_metadata.csv.

    Parameters
    ----------
    path : str or Path, optional
        Path to the CSV. Defaults to DATA_CONFIG path.

    Returns
    -------
    pd.DataFrame
        Columns: object_id, ra, decl, gal_l, gal_b, ddf,
        hostgal_specz, hostgal_photoz, hostgal_photoz_err,
        distmod, mwebv, target
    """
    path = path or DATA_CONFIG["training_set_metadata"]
    return pd.read_csv(path)


def load_training_lightcurves(path=None):
    """Load training_set.csv.

    Parameters
    ----------
    path : str or Path, optional
        Path to the CSV. Defaults to DATA_CONFIG path.

    Returns
    -------
    pd.DataFrame
        Columns: object_id, mjd, passband, flux, flux_err, detected
    """
    path = path or DATA_CONFIG["training_set"]
    return pd.read_csv(path)


def load_test_metadata(path=None):
    """Load test_set_metadata.csv (same schema as training minus 'target').

    Parameters
    ----------
    path : str or Path, optional

    Returns
    -------
    pd.DataFrame
    """
    path = path or DATA_CONFIG["test_set_metadata"]
    return pd.read_csv(path)


def load_test_lightcurves_chunked(path=None, chunksize=5_000_000):
    """Yield chunks of test_set.csv (15 GB, cannot fit in memory).

    Parameters
    ----------
    path : str or Path, optional
    chunksize : int
        Rows per chunk.

    Yields
    ------
    pd.DataFrame
        Chunks of test light curve observations.
    """
    path = path or DATA_CONFIG["test_set"]
    for chunk in pd.read_csv(path, chunksize=chunksize):
        yield chunk


def get_object_lightcurve(object_id, lc_df):
    """Extract the full multi-band light curve for one object.

    Parameters
    ----------
    object_id : int
    lc_df : pd.DataFrame
        The full training_set or a chunk of it.

    Returns
    -------
    dict
        Keys are passband ints (0-5), values are dicts with
        'mjd', 'flux', 'flux_err', 'detected' arrays.
    """
    obj_data = lc_df[lc_df["object_id"] == object_id]
    result = {}
    for band in DATA_CONFIG["band_ids"]:
        band_data = obj_data[obj_data["passband"] == band]
        if len(band_data) == 0:
            continue
        result[band] = {
            "mjd": band_data["mjd"].values,
            "flux": band_data["flux"].values,
            "flux_err": band_data["flux_err"].values,
            "detected": band_data["detected"].values,
        }
    return result


def get_objects_by_class(metadata_df, target, n=None, random_state=42):
    """Get object IDs for a given class.

    Parameters
    ----------
    metadata_df : pd.DataFrame
    target : int
        PLAsTiCC class ID (e.g. 90 for SN Ia).
    n : int, optional
        Number of objects to return. If None, returns all.
    random_state : int

    Returns
    -------
    np.ndarray of object_ids
    """
    class_objects = metadata_df[metadata_df["target"] == target]["object_id"].values
    if n is not None and n < len(class_objects):
        rng = np.random.default_rng(random_state)
        class_objects = rng.choice(class_objects, size=n, replace=False)
    return class_objects


def validate_lightcurve(object_lc, min_detections=None):
    """Check whether a light curve has enough data points.

    Parameters
    ----------
    object_lc : dict
        Per-band light curve from get_object_lightcurve.
    min_detections : int, optional
        Defaults to PREPROCESS_CONFIG value.

    Returns
    -------
    bool
    """
    min_detections = min_detections or PREPROCESS_CONFIG["min_detections"]
    total = sum(len(v["mjd"]) for v in object_lc.values())
    return total >= min_detections
