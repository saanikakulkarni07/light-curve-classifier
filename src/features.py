"""Hand-crafted light curve features for the Random Forest baseline."""

import numpy as np
from scipy.stats import skew, kurtosis
from astropy.timeseries import LombScargle
from config import DATA_CONFIG, FEATURE_CONFIG


def compute_lomb_scargle(mjd, flux, flux_err, min_period=None, max_period=None,
                         n_frequencies=None):
    """Compute Lomb-Scargle periodogram using astropy.timeseries.LombScargle.

    Parameters
    ----------
    mjd : np.ndarray
    flux : np.ndarray
    flux_err : np.ndarray
    min_period, max_period : float, optional (days)
    n_frequencies : int, optional

    Returns
    -------
    dict
        'best_period', 'best_power', 'fap', 'second_period', 'second_power',
        'third_period', 'third_power', 'period_ratio_21'
    """
    min_period = min_period or FEATURE_CONFIG["ls_min_period"]
    max_period = max_period or FEATURE_CONFIG["ls_max_period"]
    n_frequencies = n_frequencies or FEATURE_CONFIG["ls_n_frequencies"]

    result = {
        "best_period": np.nan, "best_power": np.nan, "fap": np.nan,
        "second_period": np.nan, "second_power": np.nan,
        "third_period": np.nan, "third_power": np.nan,
        "period_ratio_21": np.nan,
    }

    # Need at least 5 points for meaningful periodogram
    valid = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    if valid.sum() < 5:
        return result

    mjd_v, flux_v, err_v = mjd[valid], flux[valid], flux_err[valid]

    min_freq = 1.0 / max_period
    max_freq = 1.0 / min_period
    frequency = np.linspace(min_freq, max_freq, n_frequencies)

    ls = LombScargle(mjd_v, flux_v, err_v)
    power = ls.power(frequency)

    # Find top 3 peaks
    periods = 1.0 / frequency
    peak_indices = _find_peaks(power, n_peaks=FEATURE_CONFIG["ls_n_peaks"])

    if len(peak_indices) >= 1:
        result["best_period"] = periods[peak_indices[0]]
        result["best_power"] = power[peak_indices[0]]
        result["fap"] = ls.false_alarm_probability(power[peak_indices[0]])
    if len(peak_indices) >= 2:
        result["second_period"] = periods[peak_indices[1]]
        result["second_power"] = power[peak_indices[1]]
        result["period_ratio_21"] = result["second_period"] / result["best_period"]
    if len(peak_indices) >= 3:
        result["third_period"] = periods[peak_indices[2]]
        result["third_power"] = power[peak_indices[2]]

    return result


def _find_peaks(power, n_peaks=3):
    """Find the top n_peaks in a periodogram, separated by at least 5% in frequency."""
    indices = np.argsort(power)[::-1]
    selected = []
    for idx in indices:
        if len(selected) >= n_peaks:
            break
        # Ensure peaks are well-separated
        too_close = False
        for s in selected:
            if abs(idx - s) < len(power) * 0.05:
                too_close = True
                break
        if not too_close:
            selected.append(idx)
    return selected


def compute_statistical_features(flux, flux_err):
    """Basic statistical features of the flux time series.

    Parameters
    ----------
    flux : np.ndarray
    flux_err : np.ndarray

    Returns
    -------
    dict
    """
    valid = np.isfinite(flux)
    f = flux[valid] if valid.any() else np.array([0.0])
    e = flux_err[valid] if valid.any() else np.array([1.0])

    snr = np.abs(f) / np.clip(e, 1e-10, None)

    pcts = np.percentile(f, FEATURE_CONFIG["percentiles"]) if len(f) > 1 else np.zeros(5)

    result = {
        "mean_flux": np.mean(f),
        "std_flux": np.std(f),
        "skew_flux": float(skew(f)) if len(f) > 2 else 0.0,
        "kurtosis_flux": float(kurtosis(f)) if len(f) > 3 else 0.0,
        "median_flux": np.median(f),
        "iqr_flux": np.subtract(*np.percentile(f, [75, 25])) if len(f) > 1 else 0.0,
        "max_flux": np.max(f),
        "min_flux": np.min(f),
        "amplitude": np.max(f) - np.min(f),
        "mean_snr": np.mean(snr),
        "median_snr": np.median(snr),
        "n_observations": len(f),
    }

    # Fraction of points beyond 1 and 2 sigma
    mu, sigma = np.mean(f), np.std(f)
    if sigma > 0:
        result["beyond_1sigma"] = np.mean(np.abs(f - mu) > sigma)
        result["beyond_2sigma"] = np.mean(np.abs(f - mu) > 2 * sigma)
    else:
        result["beyond_1sigma"] = 0.0
        result["beyond_2sigma"] = 0.0

    for i, p in enumerate(FEATURE_CONFIG["percentiles"]):
        result[f"p{p}_flux"] = pcts[i]

    return result


def compute_temporal_features(mjd, flux):
    """Time-domain features.

    Parameters
    ----------
    mjd : np.ndarray
    flux : np.ndarray

    Returns
    -------
    dict
    """
    valid = np.isfinite(flux) & np.isfinite(mjd)
    t = mjd[valid] if valid.any() else np.array([0.0])
    f = flux[valid] if valid.any() else np.array([0.0])

    duration = t[-1] - t[0] if len(t) > 1 else 0.0

    # Rise and fall times relative to peak
    peak_idx = np.argmax(f)
    rise_time = t[peak_idx] - t[0] if peak_idx > 0 else 0.0
    fall_time = t[-1] - t[peak_idx] if peak_idx < len(t) - 1 else 0.0
    time_ratio = rise_time / duration if duration > 0 else 0.5

    # Rates
    peak_flux = f[peak_idx]
    base_flux = np.median(f)
    rise_rate = (peak_flux - f[0]) / rise_time if rise_time > 0 else 0.0
    fall_rate = (peak_flux - f[-1]) / fall_time if fall_time > 0 else 0.0

    # Max slope between consecutive points
    if len(t) > 1:
        dt = np.diff(t)
        df = np.diff(f)
        dt[dt == 0] = 1e-10
        slopes = np.abs(df / dt)
        max_slope = np.max(slopes)
        mean_cadence = np.mean(dt)
        cadence_std = np.std(dt)
    else:
        max_slope = 0.0
        mean_cadence = 0.0
        cadence_std = 0.0

    return {
        "duration": duration,
        "rise_time": rise_time,
        "fall_time": fall_time,
        "rise_rate": rise_rate,
        "fall_rate": fall_rate,
        "time_ratio": time_ratio,
        "max_slope": max_slope,
        "mean_cadence": mean_cadence,
        "cadence_std": cadence_std,
    }


def compute_color_features(object_lc, color_pairs=None):
    """Inter-band color features from contemporaneous observations.

    For each (band_a, band_b) pair, interpolate band_b flux onto
    band_a MJD grid and compute flux ratio features.

    Parameters
    ----------
    object_lc : dict
        Per-band light curve.
    color_pairs : list of (int, int), optional

    Returns
    -------
    dict
    """
    color_pairs = color_pairs or FEATURE_CONFIG["color_pairs"]
    result = {}

    for band_a, band_b in color_pairs:
        a_name = DATA_CONFIG["bands"][band_a]
        b_name = DATA_CONFIG["bands"][band_b]
        prefix = f"color_{a_name}_{b_name}"

        if band_a not in object_lc or band_b not in object_lc:
            result[f"{prefix}_mean"] = np.nan
            result[f"{prefix}_std"] = np.nan
            result[f"{prefix}_slope"] = np.nan
            continue

        lc_a = object_lc[band_a]
        lc_b = object_lc[band_b]

        # Interpolate band_b onto band_a's MJD grid
        flux_b_interp = np.interp(lc_a["mjd"], lc_b["mjd"], lc_b["flux"])

        # Flux ratio (avoid division by zero)
        safe_denom = np.clip(np.abs(flux_b_interp), 1e-10, None)
        ratio = lc_a["flux"] / safe_denom

        result[f"{prefix}_mean"] = np.nanmean(ratio)
        result[f"{prefix}_std"] = np.nanstd(ratio)

        # Color slope (change in ratio over time)
        if len(lc_a["mjd"]) > 1:
            dt = lc_a["mjd"][-1] - lc_a["mjd"][0]
            if dt > 0:
                result[f"{prefix}_slope"] = (ratio[-1] - ratio[0]) / dt
            else:
                result[f"{prefix}_slope"] = 0.0
        else:
            result[f"{prefix}_slope"] = 0.0

    return result


def _stetson_j(flux, flux_err):
    """Compute the Stetson J variability index.

    Measures correlated variability between consecutive observations.
    """
    if len(flux) < 3:
        return 0.0

    w_mean = _weighted_mean(flux, flux_err)
    residuals = (flux - w_mean) / np.clip(flux_err, 1e-10, None)

    # Consecutive pairs
    n = len(residuals) - 1
    products = residuals[:-1] * residuals[1:]
    signs = np.sign(products)
    stetson_j = np.sum(signs * np.sqrt(np.abs(products))) / n
    return float(stetson_j)


def _weighted_mean(flux, flux_err):
    """Inverse-variance weighted mean."""
    weights = 1.0 / np.clip(flux_err ** 2, 1e-20, None)
    return np.sum(flux * weights) / np.sum(weights)


def extract_per_band_features(mjd, flux, flux_err):
    """Extract all single-band features.

    Calls compute_statistical_features, compute_temporal_features,
    compute_lomb_scargle, and adds Stetson J.

    Parameters
    ----------
    mjd, flux, flux_err : np.ndarray

    Returns
    -------
    dict
    """
    features = {}
    features.update(compute_statistical_features(flux, flux_err))
    features.update(compute_temporal_features(mjd, flux))
    features.update(compute_lomb_scargle(mjd, flux, flux_err))
    features["stetson_j"] = _stetson_j(flux, flux_err)
    return features


def extract_all_features(object_id, object_lc, metadata_row):
    """Extract the full feature vector for one object.

    Runs per-band features for each of 6 bands (prefixed with band name),
    plus cross-band color features, plus metadata features.

    Parameters
    ----------
    object_id : int
    object_lc : dict
        Per-band light curves.
    metadata_row : pd.Series

    Returns
    -------
    dict
        Full feature vector (~130 features).
    """
    features = {"object_id": object_id}

    # Per-band features
    for band in DATA_CONFIG["band_ids"]:
        band_name = DATA_CONFIG["bands"][band]
        if band in object_lc and len(object_lc[band]["mjd"]) >= 3:
            lc = object_lc[band]
            band_feats = extract_per_band_features(
                lc["mjd"], lc["flux"], lc["flux_err"]
            )
            for k, v in band_feats.items():
                features[f"{band_name}_{k}"] = v
        else:
            # Fill with NaN for missing bands
            band_feats = extract_per_band_features(
                np.array([0.0]), np.array([0.0]), np.array([1.0])
            )
            for k, v in band_feats.items():
                features[f"{band_name}_{k}"] = np.nan

    # Cross-band color features
    features.update(compute_color_features(object_lc))

    # Metadata features
    for col in ["hostgal_photoz", "hostgal_photoz_err", "distmod", "mwebv", "ddf"]:
        features[col] = metadata_row.get(col, np.nan)

    return features
