"""CPU implementations of the feature-ablation LSTM and Attention-CLX backbone."""

from copy import deepcopy

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn


class CloseChangeLSTM(nn.Module):
    """One LSTM layer and a scalar head for the next close change."""

    def __init__(self, features):
        super().__init__()
        self.lstm = nn.LSTM(features, 24, batch_first=True)
        self.head = nn.Linear(24, 1)

    def forward(self, windows):
        _, (last_hidden, _) = self.lstm(windows)
        return self.head(last_hidden[-1])


class AttentionBackbone(nn.Module):
    """Conv1D, BiLSTM and per-hidden-channel temporal attention from Attention-CLX."""

    def __init__(self, features, steps):
        super().__init__()
        self.conv = nn.Conv1d(features, 64, kernel_size=1)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.time_weights = nn.Linear(steps, steps)
        self.head = nn.Linear(steps * 128, 1)

    def forward(self, windows):
        projected = self.dropout(self.conv(windows.transpose(1, 2)).relu())
        hidden, _ = self.lstm(projected.transpose(1, 2))
        hidden = self.dropout(hidden)
        weights = self.time_weights(hidden.transpose(1, 2)).softmax(dim=-1)
        attended = hidden * weights.transpose(1, 2)
        return self.head(attended.flatten(start_dim=1)).sigmoid()


def fit_forecast(name, windows, targets, epochs, seed):
    torch.manual_seed(seed)
    is_train = targets["split"].eq("train").to_numpy()
    is_validation = targets["split"].eq("validation").to_numpy()
    is_test = targets["split"].eq("test").to_numpy()

    feature_count = windows.shape[-1]
    _, steps, _ = windows.shape
    feature_scaler = StandardScaler().fit(windows[is_train].reshape(-1, feature_count))
    scaled = feature_scaler.transform(windows.reshape(-1, feature_count)).reshape(windows.shape)
    inputs = torch.from_numpy(scaled)
    actual = targets["actual_close"].to_numpy().reshape(-1, 1)
    previous = targets["last_close"].to_numpy().reshape(-1, 1)
    if name == "attention_backbone":
        target = actual
        offset = np.zeros_like(previous)
        target_scaler = MinMaxScaler()
        model = AttentionBackbone(feature_count, steps)
    else:
        target = actual - previous
        offset = previous
        target_scaler = StandardScaler()
        model = CloseChangeLSTM(feature_count)

    target_scaler.fit(target[is_train])
    labels = torch.from_numpy(target_scaler.transform(target[is_train]).astype(np.float32))
    train = inputs[is_train]
    validation = inputs[is_validation]
    test = inputs[is_test]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    def predict(features, mask):
        model.eval()
        with torch.inference_mode():
            output = model(features).numpy()
        return target_scaler.inverse_transform(output) + offset[mask]

    best = None
    best_mae = float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train))
        for batch in order.split(64):
            optimizer.zero_grad()
            loss = loss_fn(model(train[batch]), labels[batch])
            loss.backward()
            optimizer.step()

        validation_predictions = predict(validation, is_validation)
        mae = float(np.abs(actual[is_validation] - validation_predictions).mean())
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            best = deepcopy(model.state_dict())

    if best is None:
        raise RuntimeError(f"{name}: no finite validation checkpoint")
    model.load_state_dict(best)
    return predict(test, is_test).ravel(), {"validation_mae": best_mae, "epoch": best_epoch}
