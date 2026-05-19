"""Model definitions: Random Forest baseline and RNN for transient classification."""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config import MODEL_CONFIG, DATA_CONFIG


# ---------------------------------------------------------------------------
# Random Forest baseline
# ---------------------------------------------------------------------------

def build_rf_pipeline():
    """Build a scikit-learn pipeline: StandardScaler -> RandomForestClassifier."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=MODEL_CONFIG["rf_n_estimators"],
            max_depth=MODEL_CONFIG["rf_max_depth"],
            class_weight=MODEL_CONFIG["rf_class_weight"],
            random_state=MODEL_CONFIG["random_state"],
            n_jobs=-1,
        )),
    ])


# ---------------------------------------------------------------------------
# RNN (LSTM / GRU)
# ---------------------------------------------------------------------------

class LightCurveRNN(nn.Module):
    """Recurrent network for multi-class light curve classification.

    Architecture:
        - 2-layer bidirectional GRU (or LSTM)
        - Input: (batch, seq_len, input_size=9)
        - Packs sequences using actual lengths to ignore padding
        - Takes final hidden state (concat forward + backward)
        - FC head: hidden*2 -> 128 -> n_classes with dropout + BatchNorm

    Parameters
    ----------
    input_size : int
    hidden_size : int
    num_layers : int
    n_classes : int
    dropout : float
    bidirectional : bool
    rnn_type : str ("GRU" or "LSTM")
    """

    def __init__(self, input_size=None, hidden_size=None, num_layers=None,
                 n_classes=None, dropout=None, bidirectional=None, rnn_type=None):
        super().__init__()
        input_size = input_size or MODEL_CONFIG["rnn_input_size"]
        hidden_size = hidden_size or MODEL_CONFIG["rnn_hidden_size"]
        num_layers = num_layers or MODEL_CONFIG["rnn_num_layers"]
        n_classes = n_classes or DATA_CONFIG["n_classes"]
        dropout = dropout or MODEL_CONFIG["rnn_dropout"]
        bidirectional = bidirectional if bidirectional is not None else MODEL_CONFIG["rnn_bidirectional"]
        rnn_type = rnn_type or MODEL_CONFIG["rnn_type"]

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        rnn_cls = nn.GRU if rnn_type == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        fc_input = hidden_size * self.n_directions
        self.bn = nn.BatchNorm1d(fc_input)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(fc_input, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x, lengths):
        """Forward pass.

        Parameters
        ----------
        x : Tensor (batch, max_seq_len, input_size)
        lengths : Tensor (batch,) — actual sequence lengths

        Returns
        -------
        Tensor (batch, n_classes) — logits
        """
        # Sort by length (required by pack_padded_sequence)
        sorted_lengths, sort_idx = lengths.sort(descending=True)
        x_sorted = x[sort_idx]

        # Clamp lengths to at least 1
        sorted_lengths = sorted_lengths.clamp(min=1)

        packed = pack_padded_sequence(x_sorted, sorted_lengths.cpu(), batch_first=True)
        rnn_out, hidden = self.rnn(packed)

        # Extract final hidden state
        if isinstance(hidden, tuple):
            # LSTM returns (h_n, c_n)
            hidden = hidden[0]

        # hidden shape: (num_layers * n_directions, batch, hidden_size)
        # Take the last layer's forward and backward hidden states
        if self.bidirectional:
            h_forward = hidden[-2]   # last layer forward
            h_backward = hidden[-1]  # last layer backward
            final_hidden = torch.cat([h_forward, h_backward], dim=1)
        else:
            final_hidden = hidden[-1]

        # Unsort to restore original batch order
        _, unsort_idx = sort_idx.sort()
        final_hidden = final_hidden[unsort_idx]

        # FC head
        out = self.bn(final_hidden)
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# ---------------------------------------------------------------------------
# Dataset and DataLoader utilities
# ---------------------------------------------------------------------------

class LightCurveDataset(torch.utils.data.Dataset):
    """PyTorch dataset for light curve sequences.

    Parameters
    ----------
    sequences : np.ndarray (n_samples, max_len, n_features)
    lengths : np.ndarray (n_samples,)
    labels : np.ndarray (n_samples,) — integer class indices
    augment : bool
    """

    def __init__(self, sequences, lengths, labels, augment=False):
        self.sequences = torch.FloatTensor(sequences)
        self.lengths = torch.LongTensor(lengths)
        self.labels = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.sequences[idx]
        length = self.lengths[idx]
        y = self.labels[idx]

        if self.augment:
            # Add Gaussian noise scaled by flux_err (feature index 2)
            flux_err = x[:length, 2].clone()
            noise = torch.randn(length) * flux_err * 0.5
            x = x.clone()
            x[:length, 1] += noise  # add noise to flux

            # Randomly drop 10-20% of timesteps
            if length > 10:
                drop_frac = torch.empty(1).uniform_(0.1, 0.2).item()
                n_drop = int(length * drop_frac)
                drop_idx = torch.randperm(length)[:n_drop]
                x[drop_idx] = 0.0

        return x, length, y


def create_data_loaders(sequences, lengths, labels):
    """Create stratified train/val/test DataLoaders.

    Parameters
    ----------
    sequences : np.ndarray (n_samples, max_len, n_features)
    lengths : np.ndarray (n_samples,)
    labels : np.ndarray (n_samples,) — encoded integer labels

    Returns
    -------
    tuple of (train_loader, val_loader, test_loader)
    """
    from sklearn.model_selection import train_test_split

    rs = MODEL_CONFIG["random_state"]
    test_size = MODEL_CONFIG["test_size"]
    val_size = MODEL_CONFIG["val_size"]

    idx = np.arange(len(labels))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, stratify=labels, random_state=rs
    )
    relative_val = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=relative_val, stratify=labels[idx_trainval], random_state=rs
    )

    train_ds = LightCurveDataset(sequences[idx_train], lengths[idx_train],
                                 labels[idx_train], augment=True)
    val_ds = LightCurveDataset(sequences[idx_val], lengths[idx_val],
                               labels[idx_val], augment=False)
    test_ds = LightCurveDataset(sequences[idx_test], lengths[idx_test],
                                labels[idx_test], augment=False)

    bs = MODEL_CONFIG["rnn_batch_size"]
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False)

    return train_loader, val_loader, test_loader


def compute_class_weights(labels):
    """Compute inverse-frequency class weights for CrossEntropyLoss.

    Parameters
    ----------
    labels : np.ndarray of int

    Returns
    -------
    torch.Tensor (n_classes,)
    """
    classes, counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = DATA_CONFIG["n_classes"]
    weights = np.ones(n_classes)
    for cls, count in zip(classes, counts):
        weights[cls] = n_samples / (n_classes * count)
    return torch.FloatTensor(weights)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_rnn(model, train_loader, val_loader, class_weights=None, device="cpu"):
    """Train the RNN with early stopping on validation log-loss.

    Parameters
    ----------
    model : LightCurveRNN
    train_loader, val_loader : DataLoader
    class_weights : torch.Tensor, optional
    device : str

    Returns
    -------
    dict
        'train_loss', 'val_loss', 'val_f1_macro', 'val_log_loss', 'best_epoch'
    """
    from sklearn.metrics import f1_score

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MODEL_CONFIG["rnn_learning_rate"],
        weight_decay=MODEL_CONFIG["rnn_weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_f1_macro": [], "val_log_loss": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(MODEL_CONFIG["rnn_epochs"]):
        # Train
        model.train()
        train_losses = []
        for x_batch, len_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            len_batch = len_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch, len_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses, val_preds, val_probs, val_true = [], [], [], []
        with torch.no_grad():
            for x_batch, len_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                len_batch = len_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(x_batch, len_batch)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                val_probs.append(probs)
                val_preds.append(logits.argmax(dim=1).cpu().numpy())
                val_true.append(y_batch.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_probs = np.concatenate(val_probs)
        val_true = np.concatenate(val_true)

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_f1_macro"].append(val_f1)
        history["val_log_loss"].append(avg_val)

        scheduler.step(avg_val)

        print(
            f"Epoch {epoch+1}/{MODEL_CONFIG['rnn_epochs']} — "
            f"train_loss: {avg_train:.4f}, val_loss: {avg_val:.4f}, "
            f"val_f1: {val_f1:.4f}"
        )

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= MODEL_CONFIG["rnn_patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_epoch"] = int(np.argmin(history["val_loss"])) + 1
    return history
