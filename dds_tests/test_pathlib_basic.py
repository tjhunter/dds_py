import dds
import pytest
import pathlib
from .utils import cleandir, Counter, spath

_ = cleandir

_c = Counter()

_p = pathlib.Path("/path2")


def _fun():
    _c.increment()
    return "a"


def fun():
    return dds.keep(spath, _fun)


def f1():
    fun()
    fun()


def f():
    f1()
    fun()


@pytest.mark.usefixtures("cleandir")
def test():
    assert dds.eval(fun) == "a"
    assert _c.value == 1
    assert dds.keep(_p, fun) == "a"
    assert _c.value == 1
    dds.keep(_p, f)
    assert _c.value == 1
