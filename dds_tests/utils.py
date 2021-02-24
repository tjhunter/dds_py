import dds
import logging
import pytest
import tempfile
import shutil
from pathlib import Path
import pkgutil
from io import UnsupportedOperation
from typing import Generator


_logger = logging.getLogger(__name__)


def _add_tests() -> None:
    """
    Adds programmatically all the sub-test files to the whitelist.
    """
    import dds_tests

    p: str = dds_tests.__path__  # type: ignore
    test_mods = [
        f"dds_tests.{m.name}"
        for m in pkgutil.iter_modules(p)
        if str(m.name).startswith("test_")
    ]
    _logger.info(str(test_mods))
    for tm in test_mods:
        dds.accept_module(tm)


_add_tests()


@pytest.fixture
def cleandir():
    tdir = Path(tempfile.mkdtemp(prefix="dds"))
    internal_dir = tdir.joinpath("internal_dir")
    data_dir = tdir.joinpath("data_dir")
    dds.set_store(
        "local",
        internal_dir=str(internal_dir),
        data_dir=str(data_dir),
        cache_objects=100,
    )
    _logger.debug(f"data dir: {tdir}")
    yield
    shutil.rmtree(str(tdir), ignore_errors=True)


# A standard path
spath = "/path"


class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1


def unreachable():
    # Will trigger a failure in the parsing
    async def _f():
        assert False

    return "0"


obj = UnsupportedOperation()
unreachable_method = obj.with_traceback
