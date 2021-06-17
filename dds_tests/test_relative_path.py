import dds
import pytest
from .utils import cleandir, Counter, spath, unreachable_method

_c1 = Counter()
_c2 = Counter()
_c3 = Counter()

@dds.data_function("/p")
def fun2():
    _c3.increment()
    return 1


def fun1(x):
    fun2()
    _c1.increment()
    return x + 1

def fun1_1(y):
    fun2()
    _c2.increment()
    return dds.keep("./inner", fun1, y + 2)


def fun():
    x = 1
    dds.keep("/p1", fun1_1, x)
    x = 2
    dds.keep("/p2", fun1_1, x)


@pytest.mark.usefixtures("cleandir")
def test():
    dds.eval(fun)
    assert _c1.value == 2
    assert _c2.value == 2
    assert _c3.value == 1
    dds.eval(fun)
    assert _c1.value == 2
    assert _c2.value == 2
    assert _c3.value == 1
    assert dds.load("/p1.dir/inner") == 4
    assert dds.load("/p2.dir/inner") == 5
