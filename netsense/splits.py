"""Frozen episode-level train/test split, reused by Experiments 1-4.

Whole episodes go to either training or testing, so neighbouring minutes of one episode
never appear in both. Within each (episode_type, cause) group exactly half of the episodes
go to training, so both halves contain the same mix of episode types and causes.
"""

import numpy as np
import pandas as pd

from netsense import config as cfg

TRAIN, TEST = "train", "test"


def split_episodes(telemetry, seed):
    """Assign each episode of one seed's dataset to TRAIN or TEST.

    Returns a DataFrame with columns episode_id, episode_type, cause, split.
    The result depends only on the seed and the set of episodes, not on row order.
    """
    rng = np.random.default_rng(seed + cfg.SPLIT_SEED_OFFSET)
    episodes = (telemetry[["episode_id", "episode_type", "cause"]]
                .drop_duplicates().sort_values("episode_id").reset_index(drop=True))
    if episodes.episode_id.duplicated().any():
        raise ValueError("an episode_id has more than one episode_type/cause")

    episodes["split"] = TEST
    for _, group in episodes.groupby(["episode_type", "cause"], sort=True):
        shuffled = rng.permutation(group.index.to_numpy())
        episodes.loc[shuffled[: len(shuffled) // 2], "split"] = TRAIN
    return episodes


def training_rows(telemetry, seed):
    """Rows of one seed's telemetry that belong to training episodes."""
    split = split_episodes(telemetry, seed)
    train_ids = split.loc[split.split == TRAIN, "episode_id"]
    return telemetry[telemetry.episode_id.isin(train_ids)]
