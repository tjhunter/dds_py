import dds
import logging
import pytest

dds.whitelist_module("dds_tests")

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(filename)s:%(lineno)s %(funcName)s %(message)s'
                    )
_logger = logging.getLogger(__name__)


@pytest.fixture
def cleandir():
    _logger.debug(f"Before ***")
    yield
    _logger.debug(f"After ***")


path_1 = "/path_1"


def f1(i: int) -> int: return i


def f2() -> str: return "A"


def f2_wrap(): return dds.keep(path_1, f2)


@pytest.mark.usefixtures("cleandir")
def test_1():
    assert dds.eval(f2) == "A"


@pytest.mark.usefixtures("cleandir")
def test_2():
    assert dds.eval(f2_wrap) == "A"
