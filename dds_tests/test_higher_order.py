import dds
import pytest
import pathlib
from .utils import cleandir, Counter, spath

_ = cleandir

_c = Counter()

x = 1


def g(i):
    _c.increment()
    return i + x


@dds.data_function("/f")
def f():
    return list(map(g, range(2)))


@pytest.mark.usefixtures("cleandir")
def test_gh133_1():
    global x
    assert f() == [1, 2]
    assert _c.value == 2
    assert f() == [1, 2]
    assert _c.value == 2
    x = 2
    assert f() == [2, 3]
    assert _c.value == 4
