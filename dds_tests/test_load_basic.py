import dds
import pytest
from .utils import cleandir, Counter

_ = cleandir


@dds.dds_function("/p")
def f():
    return "a"


def f1():
    f()
    return f()


@pytest.mark.usefixtures("cleandir")
def test_1():
    assert f() == "a"
    assert dds.load("/p") == "a"


_c2a = Counter()
_c2b = Counter()
v2a = 1
v2b = 1


@dds.dds_function("/p1")
def f2a():
    _c2a.increment()
    return "a" + str(v2a)


@dds.dds_function("/p2")
def f2b():
    _c2b.increment()
    return dds.load("/p1") + str(v2b)


def f2():
    f2a()
    f2b()
    return dds.load("/p2")


@pytest.mark.usefixtures("cleandir")
def test_exec_separately():
    _c2a.value = 0
    _c2b.value = 0
    f2a()
    assert f2b() == "a11"


@pytest.mark.usefixtures("cleandir")
def test_exec_then_eval():
    _c2a.value = 0
    _c2b.value = 0
    f2a()
    assert dds.eval(f2b) == "a11"


@pytest.mark.usefixtures("cleandir")
def test_no_unnecessary_exec():
    _c2a.value = 0
    _c2b.value = 0
    f2a()
    assert f2b() == "a11"
    assert f2b() == "a11"
    assert _c2b.value == 1


@pytest.mark.usefixtures("cleandir")
def test_no_unnecessary_exec_eval():
    _c2a.value = 0
    _c2b.value = 0
    f2a()
    assert dds.eval(f2b) == "a11"
    assert dds.eval(f2b) == "a11"
    assert _c2b.value == 1


@pytest.mark.usefixtures("cleandir")
def test_2_simple():
    _c2a.value = 0
    _c2b.value = 0
    assert f2() == "a11"
    assert _c2a.value == 1
    assert _c2b.value == 1
    assert f2() == "a11"
    assert _c2a.value == 1
    assert _c2b.value == 1


@pytest.mark.usefixtures("cleandir")
def test_2_simple_eval():
    _c2a.value = 0
    _c2b.value = 0
    assert f2() == "a11"
    assert _c2a.value == 1
    assert _c2b.value == 1
    assert dds.eval(f2) == "a11"
    assert _c2a.value == 1
    assert _c2b.value == 1


@pytest.mark.usefixtures("cleandir")
def test_3_changes():
    """
    Changes a path and expects the soft dependency to pick the update
    """
    global v2a, v2b
    _c2a.value = 0
    _c2b.value = 0
    assert f2() == "a11"
    assert _c2a.value == 1
    assert _c2b.value == 1
    v2a = 2
    assert f2() == "a21"
    assert _c2a.value == 2
    assert _c2b.value == 2
