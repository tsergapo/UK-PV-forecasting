from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..training.loop import train_network
from .base import Forecaster
from .networks import BiLSTMNet, TCNNet
from .seq_utils import Standardizer, build_design, rollout, valid_positions


class SequenceForecaster(Forecaster):
    """Autoregressive neural forecaster. Window row s = [y_{s-1}, covariates_s]; see seq_utils."""

    stochastic = True

    def __init__(
        self, target, exog_cols, network_factory, name, seed=0, seq_len=168, epochs=100,
        batch_size=64, lr=2e-3, weight_decay=1e-5, patience=6, val_fraction=0.15,
        refit_full=True, grad_clip=1.0, device=None, verbose=False,
    ):
        super().__init__(target, exog_cols)
        self.network_factory, self.name, self.seed = network_factory, name, seed
        self.seq_len, self.epochs, self.batch_size = seq_len, epochs, batch_size
        self.lr, self.weight_decay, self.patience = lr, weight_decay, patience
        self.val_fraction, self.refit_full, self.grad_clip = val_fraction, refit_full, grad_clip
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose
        self.net_ = None
        self.y_sc_: Standardizer | None = None
        self.x_sc_: Standardizer | None = None
        self.best_epoch_: int | None = None

    def required_context(self, horizon):
        return self.seq_len

    def _make_net(self, input_dim):
        torch.manual_seed(self.seed)
        return self.network_factory(input_dim).to(self.device)

    def _design(self, df):
        y = self.y_sc_.transform(df[self.target].to_numpy(float)[:, None])[:, 0]
        E = self.x_sc_.transform(df[self.exog_cols].to_numpy(float))
        return build_design(y, E)

    def fit(self, train):
        y = train[self.target].to_numpy(float)[:, None]
        E = train[self.exog_cols].to_numpy(float)
        self.y_sc_, self.x_sc_ = Standardizer.fit(y), Standardizer.fit(E)
        A, T = self._design(train)
        pos = valid_positions(A, T, self.seq_len)
        if len(pos) == 0:
            raise ValueError("No valid training windows (too many gaps?)")
        cut = int(len(A) * (1 - self.val_fraction))
        tr, va = pos[pos < cut], pos[pos >= cut]
        if len(tr) == 0 or len(va) == 0:
            tr, va = pos, None
        At = torch.as_tensor(A, device=self.device)
        Tt = torch.as_tensor(T, device=self.device)
        kw = dict(
            seq_len=self.seq_len, batch_size=self.batch_size, lr=self.lr,
            weight_decay=self.weight_decay, patience=self.patience, grad_clip=self.grad_clip,
            seed=self.seed, device=self.device, verbose=self.verbose, label=self.name,
        )
        net = self._make_net(A.shape[1])
        res = train_network(net, At, Tt, tr, va, epochs=self.epochs, **kw)
        self.best_epoch_ = res.best_epoch
        if self.refit_full and va is not None:
            net = self._make_net(A.shape[1])
            train_network(net, At, Tt, pos, None, epochs=max(res.best_epoch, 1), **kw)
        self.net_ = net.eval()
        return self

    def forecast_blocks(self, df, starts, horizon):
        A, _ = self._design(df)

        def predict_fn(win):
            with torch.no_grad():
                x = torch.as_tensor(win, dtype=torch.float32, device=self.device)
                return self.net_(x).cpu().numpy()

        scaled = rollout(predict_fn, A, starts, self.seq_len, horizon)
        return np.clip(self.y_sc_.inverse(scaled), 0.0, None)

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.net_.state_dict(), d / "model.pt")
        (d / "scalers.json").write_text(
            json.dumps({"y": self.y_sc_.to_dict(), "x": self.x_sc_.to_dict(),
                        "exog_cols": self.exog_cols, "best_epoch": self.best_epoch_})
        )


_TRAIN_KEYS = (
    "seq_len", "epochs", "batch_size", "lr", "weight_decay", "patience", "val_fraction",
    "refit_full", "grad_clip",
)


def _split(params, net_keys):
    net = {k: params[k] for k in net_keys if k in params}
    train = {k: params[k] for k in _TRAIN_KEYS if k in params}
    unknown = set(params) - set(net) - set(train)
    if unknown:
        raise ValueError(f"unknown model params: {sorted(unknown)}")
    return net, train


def make_lstm(target, exog_cols, seed=0, **params):
    net_kw, train_kw = _split(params, ("hidden_dim", "num_layers", "dropout", "bidirectional"))
    return SequenceForecaster(
        target, exog_cols, lambda d: BiLSTMNet(d, **net_kw), "lstm", seed=seed, **train_kw
    )


def make_tcn(target, exog_cols, seed=0, **params):
    net_kw, train_kw = _split(params, ("num_channels", "kernel_size", "dropout"))
    return SequenceForecaster(
        target, exog_cols, lambda d: TCNNet(d, **net_kw), "tcn", seed=seed, **train_kw
    )
