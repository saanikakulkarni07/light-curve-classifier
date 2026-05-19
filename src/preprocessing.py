"""Light curve preprocessing: cleaning, normalization, padding for RNN input."""

import numpy as np
from config import DATA_CONFIG, PREPROCESS_CONFIG


def clip_flux_outliers(flux, flux_err, sigma=None):
    """Sigma-clip extreme flux values per object.

    Parameters
    ----------
    flux : np.ndarray
    flux_err : np.ndarray
    sigma : float, optional
        Defaults to PREPROCESS_CONFIG value.

    Returns
    -------
    np.ndarray, np.ndarray
        Clipped flux and flux_err (outliers replaced with NaN).
    """
    sigma = sigma or PREPROCESS_CONFIG["flux_clip_sigma"]
    median = np.nanmedian(flux)
    mad = np.nanmedian(np.abs(flux - median))
    std_est = 1.4826 * mad  # robust std estimate
    mask = np.abs(flux - median) > sigma * std_est
    clipped_flux = flux.copy()
    clipped_err = flux_err.copy()
    clipped_flux[mask] = np.nan
    clipped_err[mask] = np.nan
    return clipped_flux, clipped_err


def normalize_lightcurve(mjd, flux, flux_err):
    """Normalize a single-band light curve.

    - Time: shift to start at 0, divide by duration
    - Flux: divide by max(|flux|) so values are in [-1, 1]
    - Flux_err: scale by the same factor

    Parameters
    ----------
    mjd : np.ndarray
    flux : np.ndarray
    flux_err : np.ndarray

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray
        Normalized time, flux, flux_err.
    """
    # Time normalization
    t0 = mjd.min()
    duration = mjd.max() - t0
    norm_time = (mjd - t0) / duration if duration > 0 else mjd - t0

    # Flux normalization
    flux_scale = np.nanmax(np.abs(flux))
    if flux_scale == 0:
        flux_scale = 1.0
    norm_flux = flux / flux_scale
    norm_flux_err = flux_err / flux_scale

    return norm_time, norm_flux, norm_flux_err


def pad_sequence(sequence, max_len, pad_value=0.0):
    """Pad or truncate a 2D array to (max_len, n_features).

    Parameters
    ----------
    sequence : np.ndarray (T, n_features)
    max_len : int
    pad_value : float

    Returns
    -------
    np.ndarray (max_len, n_features)
    int
        Actual length before padding.
    """
    actual_len = min(len(sequence), max_len)
    n_features = sequence.shape[1]
    padded = np.full((max_len, n_features), pad_value, dtype=np.float32)
    padded[:actual_len] = sequence[:actual_len]
    return padded, actual_len


def prepare_multiband_rnn_input(object_lc, max_len=None):
    """Build RNN input for all bands concatenated chronologically.

    Each timestep is (delta_time, flux, flux_err, passband_onehot x6).
    All observations across bands are sorted by MJD.

    Parameters
    ----------
    object_lc : dict
        Per-band light curve from data.get_object_lightcurve.
    max_len : int, optional
        Defaults to PREPROCESS_CONFIG value.

    Returns
    -------
    np.ndarray (max_len, 9)
        Input sequence.
    int
        Actual sequence length before padding.
    """
    max_len = max_len or PREPROCESS_CONFIG["max_sequence_length"]
    n_bands = len(DATA_CONFIG["band_ids"])

    # Collect all observations across bands
    all_mjd, all_flux, all_flux_err, all_band = [], [], [], []
    for band, lc in object_lc.items():
        n_obs = len(lc["mjd"])
        all_mjd.append(lc["mjd"])
        all_flux.append(lc["flux"])
        all_flux_err.append(lc["flux_err"])
        all_band.append(np.full(n_obs, band, dtype=int))

    if not all_mjd:
        padded = np.zeros((max_len, 3 + n_bands), dtype=np.float32)
        return padded, 0

    all_mjd = np.concatenate(all_mjd)
    all_flux = np.concatenate(all_flux)
    all_flux_err = np.concatenate(all_flux_err)
    all_band = np.concatenate(all_band)

    # Sort by MJD
    sort_idx = np.argsort(all_mjd)
    all_mjd = all_mjd[sort_idx]
    all_flux = all_flux[sort_idx]
    all_flux_err = all_flux_err[sort_idx]
    all_band = all_band[sort_idx]

    # Normalize
    norm_time, norm_flux, norm_flux_err = normalize_lightcurve(
        all_mjd, all_flux, all_flux_err
    )

    # Build feature matrix: (delta_t, flux, flux_err, passband_onehot)
    n_obs = len(norm_time)
    features = np.zeros((n_obs, 3 + n_bands), dtype=np.float32)
    features[:, 0] = norm_time
    features[:, 1] = norm_flux
    features[:, 2] = norm_flux_err

    # One-hot encode passband
    for i, b in enumerate(all_band):
        features[i, 3 + b] = 1.0

    # Handle NaNs
    features = np.nan_to_num(features, nan=0.0)

    # Pad/truncate
    padded, actual_len = pad_sequence(features, max_len)
    return padded, actual_len


def batch_prepare_rnn(lc_df, metadata_df, max_len=None):
    """Prepare full training tensor from the light curve DataFrame.

    Parameters
    ----------
    lc_df : pd.DataFrame
        Full training light curve data.
    metadata_df : pd.DataFrame
        Training metadata with 'target' column.
    max_len : int, optional

    Returns
    -------
    np.ndarray (n_objects, max_len, 9)
        Input sequences.
    np.ndarray (n_objects,)
        Actual sequence lengths.
    np.ndarray (n_objects,)
        PLAsTiCC target class labels.
    np.ndarray (n_objects,)
        Object IDs.
    """
    from tqdm import tqdm
    from src.data import get_object_lightcurve, validate_lightcurve

    max_len = max_len or PREPROCESS_CONFIG["max_sequence_length"]
    n_bands = len(DATA_CONFIG["band_ids"])
    n_features = 3 + n_bands

    object_ids = metadata_df["object_id"].values
    targets = metadata_df["target"].values

    sequences = np.zeros((len(object_ids), max_len, n_features), dtype=np.float32)
    lengths = np.zeros(len(object_ids), dtype=np.int32)
    valid_mask = np.ones(len(object_ids), dtype=bool)

    for i, obj_id in enumerate(tqdm(object_ids, desc="Preparing RNN input")):
        object_lc = get_object_lightcurve(obj_id, lc_df)
        if not validate_lightcurve(object_lc):
            valid_mask[i] = False
            continue
        sequences[i], lengths[i] = prepare_multiband_rnn_input(object_lc, max_len)

    # Filter out invalid objects
    sequences = sequences[valid_mask]
    lengths = lengths[valid_mask]
    targets = targets[valid_mask]
    object_ids = object_ids[valid_mask]

    return sequences, lengths, targets, object_ids
