import dds
import pytest
from .utils import cleandir, Counter

_ = cleandir

_c = Counter()

l = [1]


@dds.data_function("/p")
def f1():
    _c.increment()
    return len(l)


@pytest.mark.usefixtures("cleandir")
def test_gh140_list():
    global l
    _c.reset()
    f1()
    assert _c.value == 1
    l[0] = 2
    f1()
    assert _c.value == 2
    l[0] = 1
    f1()
    assert _c.value == 2


d = {0: 1}


@dds.data_function("/p")
def f2():
    _c.increment()
    return len(d)


@pytest.mark.usefixtures("cleandir")
def test_gh140_dict():
    _c.reset()
    f2()
    assert _c.value == 1
    d[0] = 2
    f2()
    assert _c.value == 2
    d[0] = 1
    f2()
    assert _c.value == 2
