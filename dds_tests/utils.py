import dds
import logging
import pytest
import tempfile
import shutil
from pathlib import Path

dds.whitelist_module("dds_tests.test_refs")
dds.whitelist_module("dds_tests.test_basic")
dds.whitelist_module("dds_tests.test_sklearn")
_logger = logging.getLogger(__name__)


@pytest.fixture
def cleandir():
    tdir = Path(tempfile.mkdtemp(prefix="dds"))
    internal_dir = tdir.joinpath("internal_dir")
    internal_dir.mkdir()
    data_dir = tdir.joinpath("data_dir")
    data_dir.mkdir()
    dds.set_store("local",
                  internal_dir=str(internal_dir),
                  data_dir=str(data_dir))
    _logger.debug(f"data dir: {tdir}")
    yield
    shutil.rmtree(str(tdir), ignore_errors=True)


class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1


unreachable_var = 0

def unreachable():
    return str(unreachable_var)
