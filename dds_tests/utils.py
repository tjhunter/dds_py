import dds
import logging
import pytest
import tempfile
import shutil
from pathlib import Path
import pkgutil

_logger = logging.getLogger(__name__)


def _add_tests():
    """
    Adds programmatically all the sub-test files to the whitelist.
    """
    import dds_tests
    test_mods = [f"dds_tests.{m.name}" for m in pkgutil.iter_modules(dds_tests.__path__) if str(m.name).startswith("test_")]
    _logger.info("XXX" + str(test_mods))
    for tm in test_mods:
        dds.whitelist_module(tm)

_add_tests()


@pytest.fixture
def cleandir():
    tdir = Path(tempfile.mkdtemp(prefix="dds"))
    internal_dir = tdir.joinpath("internal_dir")
    internal_dir.mkdir()
    data_dir = tdir.joinpath("data_dir")
    data_dir.mkdir()
    dds.set_store("local", internal_dir=str(internal_dir), data_dir=str(data_dir))
    _logger.debug(f"data dir: {tdir}")
    yield
    shutil.rmtree(str(tdir), ignore_errors=True)

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
