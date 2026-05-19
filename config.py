"""Centralized configuration for light curve classifier pipeline."""

# --- Data Access ---
DATA_CONFIG = {
    "kaggle_url": "https://www.kaggle.com/competitions/PLAsTiCC-2018/data",
    "raw_dir": "data/raw/",
    "processed_dir": "data/processed/",
    "training_set_metadata": "data/raw/training_set_metadata.csv",
    "training_set": "data/raw/training_set.csv",
    "test_set_metadata": "data/raw/test_set_metadata.csv",
    "test_set": "data/raw/test_set.csv",
    "bands": {0: "u", 1: "g", 2: "r", 3: "i", 4: "z", 5: "y"},
    "band_ids": [0, 1, 2, 3, 4, 5],
    "n_classes": 14,
    "class_names": {
        6: "Microlensing",
        15: "TDE",
        16: "EB",
        42: "SN II",
        52: "SN Iax",
        62: "SN Ibc",
        64: "KN",
        65: "M-dwarf flare",
        67: "SN Ia-91bg",
        88: "AGN",
        90: "SN Ia",
        92: "RR Lyrae",
        95: "SLSN-I",
        53: "Mira",
    },
    "galactic_classes": {6, 16, 53, 65, 92},
    "extragalactic_classes": {15, 42, 52, 62, 64, 67, 88, 90, 95},
}

# --- Preprocessing ---
PREPROCESS_CONFIG = {
    "max_sequence_length": 256,
    "min_detections": 5,
    "flux_clip_sigma": 5.0,
    "normalize_flux": True,
    "time_normalize": True,
}

# --- Feature Engineering ---
FEATURE_CONFIG = {
    # Lomb-Scargle periodogram
    "ls_min_period": 0.1,              # days
    "ls_max_period": 500.0,            # days
    "ls_n_frequencies": 10000,
    "ls_n_peaks": 3,
    # Statistical features
    "percentiles": [5, 25, 50, 75, 95],
    # Color features: pairs of band IDs
    "color_pairs": [
        (0, 1),   # u-g
        (1, 2),   # g-r
        (2, 3),   # r-i
        (3, 4),   # i-z
        (4, 5),   # z-y
    ],
}

# --- Model ---
MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.15,
    "val_size": 0.15,
    # Random Forest baseline
    "rf_n_estimators": 500,
    "rf_max_depth": 30,
    "rf_class_weight": "balanced",
    # GRU / LSTM
    "rnn_type": "GRU",
    "rnn_input_size": 9,               # (delta_t, flux, flux_err, passband_onehot x6)
    "rnn_hidden_size": 128,
    "rnn_num_layers": 2,
    "rnn_dropout": 0.3,
    "rnn_bidirectional": True,
    "rnn_learning_rate": 1e-3,
    "rnn_batch_size": 256,
    "rnn_epochs": 80,
    "rnn_patience": 12,
    "rnn_weight_decay": 1e-4,
    "use_class_weights": True,
}

# --- Evaluation ---
EVAL_CONFIG = {
    "top_k_misclassified": 50,
}

# --- Visualization ---
VIS_CONFIG = {
    "figure_size": (10, 6),
    "dpi": 150,
    "save_format": "png",
    "figures_dir": "figures/",
    "band_colors": {
        0: "#56B4E9",   # u - sky blue
        1: "#009E73",   # g - green
        2: "#D55E00",   # r - orange
        3: "#CC79A7",   # i - pink
        4: "#0072B2",   # z - dark blue
        5: "#E69F00",   # y - yellow
    },
}
