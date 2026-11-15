from __future__ import annotations

import torch
import torch.nn as nn

try:  # torch >= 2.1
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:  # pragma: no cover
    from torch.nn.utils import weight_norm


class BiLSTMNet(nn.Module):
    """Sequence [B, L, F] -> scalar [B] (next-hour target, standardised)."""

    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.2, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        last = torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]
        return self.fc(last).squeeze(-1)


class Chomp1d(nn.Module):
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, : -self.chomp].contiguous() if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation, dropout, use_weight_norm=True):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        wn = weight_norm if use_weight_norm else (lambda m: m)
        self.net = nn.Sequential(
            wn(nn.Conv1d(c_in, c_out, kernel_size, padding=pad, dilation=dilation)),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            wn(nn.Conv1d(c_out, c_out, kernel_size, padding=pad, dilation=dilation)),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + self.down(x))


class TCNNet(nn.Module):
    """Causal dilated convolutions; [B, L, F] -> scalar [B]."""

    def __init__(self, input_dim, num_channels=(32, 64, 64), kernel_size=3, dropout=0.2,
                 use_weight_norm=True):
        super().__init__()
        layers, c_prev = [], input_dim
        for i, c in enumerate(num_channels):
            layers.append(TemporalBlock(c_prev, c, kernel_size, 2**i, dropout, use_weight_norm))
            c_prev = c
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(c_prev, 1)

    def forward(self, x):
        y = self.tcn(x.transpose(1, 2))
        return self.fc(y[:, :, -1]).squeeze(-1)
