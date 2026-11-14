import numpy as np

from uk_pv.models.seq_utils import Standardizer, build_design, rollout, valid_positions


def _data(n=400, f=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    E = rng.normal(size=(n, f))
    return build_design(y, E)


def _predict(win):
    return np.tanh(0.5 * win[:, :, 0].mean(1) + 0.3 * win[:, -1, 0] + 0.2 * win[:, -1, 1:].sum(1))


def test_design_alignment():
    y = np.arange(5.0)
    A, T = build_design(y, np.arange(10.0).reshape(5, 2))
    assert np.isnan(A[0, 0]) and A[3, 0] == 2.0  # lag column = y_{s-1}
    assert T[3] == 3.0


def test_valid_positions_exclude_gaps():
    A, T = _data()
    A = A.copy()
    A[100, 1] = np.nan
    pos = valid_positions(A, T, seq_len=10)
    assert not any(100 - 0 <= t <= 109 for t in pos)  # windows containing row 100
    assert 110 in pos and 99 in pos


def test_rollout_matches_naive_loop():
    A, _ = _data()
    L, H = 12, 6
    starts = np.array([30, 77, 200])
    fast = rollout(_predict, A, starts, L, H, batch_size=2)
    for i, b in enumerate(starts):
        rows = A[b - L + 1 : b + H].copy()  # rows b-L+1 .. b+H-1
        preds = []
        for k in range(H):
            win = rows[k : k + L].copy()
            for j, p in enumerate(preds):          # lag entries of rows > b come from predictions
                rows[L + j, 0] = p
            win = rows[k : k + L]
            preds.append(float(_predict(win[None])[0]))
        assert np.allclose(fast[i], preds, atol=1e-6)


def test_rollout_does_not_peek_at_future_target():
    A, _ = _data()
    L, H, b = 12, 6, 100
    base = rollout(_predict, A, np.array([b]), L, H)
    A2 = A.copy()
    A2[b + 1 :, 0] = 123.0  # lag col of rows > b encodes y_b ... (future target)
    assert np.allclose(base, rollout(_predict, A2, np.array([b]), L, H))
    A3 = A.copy()
    A3[b, 0] = 5.0  # but y_{b-1} is legitimate history and must matter
    assert not np.allclose(base, rollout(_predict, A3, np.array([b]), L, H))


def test_standardizer_roundtrip_and_constant_column():
    x = np.column_stack([np.arange(10.0), np.ones(10)])
    s = Standardizer.fit(x)
    assert np.allclose(s.inverse(s.transform(x)), x)
    assert np.isfinite(s.transform(x)).all()
