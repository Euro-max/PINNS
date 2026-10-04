import pytest

from pinc.config import load_config
from pinc.tfsetup import setup


@pytest.fixture(scope="session", autouse=True)
def _tf_setup():
    setup(seed=0, dtype="float64")


@pytest.fixture(scope="session")
def cfg():
    return load_config()
