import dds
import pytest
from .utils import cleandir, Counter, spath

_ = cleandir

_c = Counter()

_p = "/path2"


def _fun():
    _c.increment()
    return "a"


def fun():
    return dds.keep("/path", _fun)


def f1():
    fun()
    fun()


def f():
    f1()
    fun()


@pytest.mark.usefixtures("cleandir")
def test():
    dds.eval(fun)
    assert _c.value == 1
