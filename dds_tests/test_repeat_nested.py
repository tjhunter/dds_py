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


_p3 = pathlib.Path("/path3")


def f2():
    dds.keep(_p3, f1)


@pytest.mark.usefixtures("cleandir")
def test():
    fun()
    assert _c.value == 1
    dds.keep(_p, f)
    assert _c.value == 1
    dds.eval(f)
    assert _c.value == 1
    dds.eval(f2)
    assert _c.value == 1
