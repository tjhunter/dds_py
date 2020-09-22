import dds
import pytest
from .utils import cleandir, Counter, spath, unreachable_method


def _fun():
    _ = unreachable_method(None)
    return "a"


def fun():
    return dds.keep(spath, _fun)


@pytest.mark.usefixtures("cleandir")
def test():
    assert dds.eval(fun) == "a"
