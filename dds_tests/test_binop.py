import dds
import pytest
from .utils import cleandir, Counter, spath

_ = cleandir

cst = 3

def z(): return 4

def _fun():
    x = (z() * 3).real
    return x


def fun():
    return dds.keep("/path", _fun) * 2


def f():
    fun() * 6


@pytest.mark.usefixtures("cleandir")
def test():
    fun()
    f()
    dds.eval(f)
