import dds
import pytest
from .utils import cleandir, Counter

_ = cleandir


def f(b):
    if b:
        return 1
    return f(True)


@dds.dds_function("/path")
def fun1():
    return f(False)


@pytest.mark.usefixtures("cleandir")
def test_rec():
    # Recursion not supported for now
    with pytest.raises(dds.structures.KSException):
        fun1()


_c = Counter()

xx = 5


def fun2():
    _c.increment()
    xx = 2
    return xx


@pytest.mark.usefixtures("cleandir")
def test_shadow():
    global xx
    assert dds.keep("/p", fun2) == 2
    assert _c.value == 1
    xx = 1
    assert dds.keep("/p", fun2) == 2
    assert _c.value == 1
