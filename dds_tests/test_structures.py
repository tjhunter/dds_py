import dds
import pytest
from collections import OrderedDict
from .utils import cleandir, Counter
from .test_mod import structures

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


@pytest.mark.usefixtures("cleandir")
def test_gh140_list():
    structures.c.reset()
    structures.f1()
    assert structures.c.value == 1
    structures.l[0] = 2
    structures.f1()
    assert structures.c.value == 2
    structures.l[0] = 1
    structures.f1()
    assert structures.c.value == 2


od = OrderedDict({0: 1})


@dds.data_function("/p")
def f3():
    _c.increment()
    return len(od)


@pytest.mark.usefixtures("cleandir")
def test_gh140_ordereddict():
    _c.reset()
    f3()
    assert _c.value == 1
    od[0] = 2
    f3()
    assert _c.value == 2
    od[0] = 1
    f3()
    assert _c.value == 2
