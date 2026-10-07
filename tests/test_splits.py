"""Tests for the frozen episode-level train/test split."""

import pandas as pd

from netsense import config as cfg
from netsense.splits import TEST, TRAIN, split_episodes


def test_no_episode_is_in_both_halves(generated):
    split = split_episodes(generated, cfg.RANDOM_SEED)
    assert split.episode_id.is_unique
    assert set(split.split) == {TRAIN, TEST}


def test_each_type_and_cause_is_split_exactly_in_half(generated):
    split = split_episodes(generated, cfg.RANDOM_SEED)
    counts = split.groupby(["episode_type", "cause", "split"]).size().unstack()
    assert (counts[TRAIN] == counts[TEST]).all()
    assert counts.loc[(cfg.STABLE, cfg.NO_CAUSE), TRAIN] == 40
    for cause in cfg.CAUSES:
        assert counts.loc[(cfg.WORSENING, cause), TRAIN] == 30
        assert counts.loc[(cfg.RECOVERING, cause), TRAIN] == 15


def test_split_is_deterministic_and_ignores_row_order(generated):
    first = split_episodes(generated, cfg.RANDOM_SEED)
    shuffled = generated.sample(frac=1.0, random_state=1)
    pd.testing.assert_frame_equal(first, split_episodes(generated, cfg.RANDOM_SEED))
    pd.testing.assert_frame_equal(first, split_episodes(shuffled, cfg.RANDOM_SEED))
