import pandas as pd

from uk_pv.config import SplitConfig
from uk_pv.splits import boundary_of, make_folds
from synthetic import make_site


def test_folds_are_ordered_and_non_overlapping():
    df = make_site()
    cfg = SplitConfig(test_years=1, n_val_folds=1, fold_years=1)
    folds = make_folds(df.index, cfg)
    b = boundary_of(folds)
    assert b == b.normalize()
    val = [f for f in folds if f.kind == "val"]
    test = [f for f in folds if f.kind == "test"]
    assert val[-1].eval_end == b and test[0].eval_start == b
    for f in val:
        assert f.train_end == f.eval_start
    for f in test:
        assert f.train_end == b and f.eval_end <= df.index.max() + pd.Timedelta(hours=1)
    assert all(f.eval_start < f.eval_end for f in folds)


def test_too_little_data_raises():
    df = make_site(start="2022-01-01", end="2023-01-01")
    try:
        make_folds(df.index, SplitConfig(test_years=1, n_val_folds=2, fold_years=1))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
