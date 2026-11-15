from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TrainResult:
    best_epoch: int
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    skipped_steps: int = 0


def iterate_batches(A, T, pos, seq_len, batch_size, shuffle, generator, device):
    """Gather windows on the fly from the continuous design matrix (no DataLoader needed)."""
    pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
    if shuffle:
        pos_t = pos_t[torch.randperm(len(pos_t), generator=generator).to(device)]
    offs = torch.arange(-seq_len + 1, 1, device=device)
    for i in range(0, len(pos_t), batch_size):
        pb = pos_t[i : i + batch_size]
        yield A[pb[:, None] + offs[None, :]], T[pb]


def train_network(
    net, A, T, train_pos, val_pos, *, seq_len, epochs, batch_size, lr, weight_decay,
    patience, grad_clip, seed, device, verbose=False, label="",
) -> TrainResult:
    """Adam + MSE with early stopping on `val_pos` (if given). Best weights are restored from
    a DEEP copy (the notebook's state_dict().copy() was shallow and kept the last weights)."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    use_val = val_pos is not None and len(val_pos) > 0
    best, best_epoch, best_state, bad = np.inf, 0, None, 0
    res = TrainResult(best_epoch=0)

    for ep in range(1, epochs + 1):
        net.train()
        tot, n, last_gn = 0.0, 0, float('nan')
        for xb, yb in iterate_batches(A, T, train_pos, seq_len, batch_size, True, gen, device):
            opt.zero_grad()
            loss = crit(net(xb), yb)
            if not torch.isfinite(loss):
                p_ok = all(bool(torch.isfinite(q).all()) for q in net.parameters())
                raise FloatingPointError(
                    f"[{label}] non-finite training loss at epoch {ep}. "
                    f"batch inputs finite: {bool(torch.isfinite(xb).all())} "
                    f"(max |x| = {float(xb.abs().max()):.1f}); targets finite: "
                    f"{bool(torch.isfinite(yb).all())}; network weights finite: {p_ok}; "
                    f"last gradient norm: {last_gn:.3g}; train loss so far: "
                    f"{[round(v, 4) for v in res.train_loss]}"
                )
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(
                net.parameters(), grad_clip if grad_clip else float("inf")))
            if not np.isfinite(gn):  # NaN/inf gradient from a finite loss: skip the update
                res.skipped_steps += 1
                if res.skipped_steps > 50:
                    raise FloatingPointError(
                        f"[{label}] {res.skipped_steps} optimizer steps had non-finite gradients "
                        f"(epoch {ep}); the network is numerically unstable"
                    )
                continue
            last_gn = gn
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        res.train_loss.append(tot / max(n, 1))

        if use_val:
            net.eval()
            vt, vn = 0.0, 0
            with torch.no_grad():
                for xb, yb in iterate_batches(A, T, val_pos, seq_len, 512, False, gen, device):
                    vt += crit(net(xb), yb).item() * len(yb)
                    vn += len(yb)
            vloss = vt / max(vn, 1)
            res.val_loss.append(vloss)
            if vloss < best - 1e-9:
                best, best_epoch, bad = vloss, ep, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
            if verbose:
                print(f"[{label}] epoch {ep:3d} train {res.train_loss[-1]:.4f} val {vloss:.4f}")
        else:
            best_epoch = ep

    if best_state is not None:
        net.load_state_dict(best_state)
    res.best_epoch = best_epoch
    if res.skipped_steps and verbose:
        print(f"[{label}] skipped {res.skipped_steps} steps with non-finite gradients")
    return res
