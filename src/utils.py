"""Plotting, label encoding, and common helper utilities."""

import numpy as np
import matplotlib.pyplot as plt
from config import DATA_CONFIG, VIS_CONFIG


def encode_labels(targets):
    """Map PLAsTiCC target integers (6, 15, 16, ..., 95) to contiguous 0..13 indices.

    Parameters
    ----------
    targets : np.ndarray
        Array of PLAsTiCC class IDs.

    Returns
    -------
    np.ndarray
        Encoded labels (0 to n_classes-1).
    dict
        Mapping from original class ID to encoded index.
    dict
        Mapping from encoded index back to original class ID.
    """
    unique_classes = sorted(DATA_CONFIG["class_names"].keys())
    encode_map = {cls: i for i, cls in enumerate(unique_classes)}
    decode_map = {i: cls for cls, i in encode_map.items()}
    encoded = np.array([encode_map[t] for t in targets])
    return encoded, encode_map, decode_map


def decode_labels(encoded, decode_map):
    """Reverse of encode_labels.

    Parameters
    ----------
    encoded : np.ndarray
    decode_map : dict

    Returns
    -------
    np.ndarray
    """
    return np.array([decode_map[e] for e in encoded])


def get_class_name(target):
    """Get the human-readable name for a PLAsTiCC class ID."""
    return DATA_CONFIG["class_names"].get(target, f"Unknown ({target})")


def plot_lightcurve(object_lc, object_id=None, title=None, ax=None, show_errors=True):
    """Plot a multi-band light curve with LSST band colors.

    Parameters
    ----------
    object_lc : dict
        Per-band light curve from data.get_object_lightcurve.
    object_id : int, optional
    title : str, optional
    ax : matplotlib.axes.Axes, optional
    show_errors : bool
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=VIS_CONFIG["figure_size"], dpi=VIS_CONFIG["dpi"])

    for band, lc in sorted(object_lc.items()):
        band_name = DATA_CONFIG["bands"][band]
        color = VIS_CONFIG["band_colors"][band]
        if show_errors:
            ax.errorbar(lc["mjd"], lc["flux"], yerr=lc["flux_err"],
                        fmt="o", markersize=3, color=color, alpha=0.7,
                        label=band_name, capsize=0)
        else:
            ax.scatter(lc["mjd"], lc["flux"], s=10, color=color,
                       alpha=0.7, label=band_name)

    ax.set_xlabel("MJD")
    ax.set_ylabel("Flux")
    title = title or (f"Object {object_id}" if object_id else "Light Curve")
    ax.set_title(title)
    ax.legend(ncol=6, fontsize=8)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    return ax


def plot_lightcurve_grid(lc_df, metadata_df, object_ids, ncols=3):
    """Plot a grid of light curves, one per object.

    Parameters
    ----------
    lc_df : pd.DataFrame
    metadata_df : pd.DataFrame
    object_ids : array-like
    ncols : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    from src.data import get_object_lightcurve

    nrows = int(np.ceil(len(object_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             dpi=VIS_CONFIG["dpi"])
    axes = np.atleast_2d(axes)

    for i, obj_id in enumerate(object_ids):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        object_lc = get_object_lightcurve(obj_id, lc_df)
        meta = metadata_df[metadata_df["object_id"] == obj_id].iloc[0]
        class_name = get_class_name(meta["target"])
        plot_lightcurve(object_lc, object_id=obj_id,
                        title=f"{obj_id} ({class_name})", ax=ax, show_errors=False)

    # Hide empty subplots
    for i in range(len(object_ids), nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    return fig


def plot_training_curves(history, metric_name="loss"):
    """Plot train/val loss and metric curves.

    Parameters
    ----------
    history : dict
        Keys like 'train_loss', 'val_loss', 'val_f1_macro'.
    metric_name : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=VIS_CONFIG["dpi"])

    # Loss
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()

    # Metric
    if "val_f1_macro" in history:
        axes[1].plot(history["val_f1_macro"], label="Val Macro F1", color="green")
        axes[1].set_ylabel("Macro F1")
    elif "val_log_loss" in history:
        axes[1].plot(history["val_log_loss"], label="Val Log-Loss", color="red")
        axes[1].set_ylabel("Log-Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Validation Metric")
    axes[1].legend()

    plt.tight_layout()
    return fig


def plot_feature_importance(importances, feature_names, top_n=30, ax=None):
    """Horizontal bar chart of top feature importances.

    Parameters
    ----------
    importances : np.ndarray
    feature_names : list of str
    top_n : int
    ax : matplotlib.axes.Axes, optional
    """
    idx = np.argsort(importances)[-top_n:]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)), dpi=VIS_CONFIG["dpi"])
    ax.barh(range(len(idx)), importances[idx], color="#0072B2")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    return ax


def save_figure(fig, name):
    """Save a figure to the figures directory."""
    path = f"{VIS_CONFIG['figures_dir']}{name}.{VIS_CONFIG['save_format']}"
    fig.savefig(path, bbox_inches="tight", dpi=VIS_CONFIG["dpi"])
    print(f"Saved: {path}")
