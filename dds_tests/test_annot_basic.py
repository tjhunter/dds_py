import dds
import pytest
from .utils import cleandir, Counter, spath

_ = cleandir

_c = Counter()

_p = "/path2"


@dds.dds_function("/p")
def f():
    _c.increment()
    return "a"


def f1():
    f()
    return f()


@pytest.mark.usefixtures("cleandir")
def test():
    assert f() == "a"
    assert _c.value == 1
    assert dds.eval(f) == "a"
    assert _c.value == 1
    assert f1() == "a"
    assert _c.value == 1
    assert dds.eval(f1) == "a"
    assert _c.value == 1
