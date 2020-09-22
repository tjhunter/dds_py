import dds
import pytest
import pathlib
from .utils import cleandir, Counter, spath

_ = cleandir

_c = Counter()

_p = pathlib.Path("/path2")


def _fun(x=True):
    _c.increment()
    return "a" + str(x)


def fun1():
    return dds.keep(spath, _fun)


def fun2():
    return dds.keep(spath, _fun, True)


def fun3():
    return dds.keep(spath, _fun, False)


@pytest.mark.usefixtures("cleandir")
def test():
    fun1()
    assert _c.value == 1
    fun2()
    assert _c.value == 1
    fun3()
    assert _c.value == 2
