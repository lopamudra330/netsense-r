import pytest

from netsense import config as cfg
from netsense.telemetry_generator import generate_dataset


@pytest.fixture(scope="session")
def generated():
    """Generated input for the default seed, shared by the validation tests."""
    telemetry, _ = generate_dataset(cfg.RANDOM_SEED)
    return telemetry
