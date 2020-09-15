import dds
import pytest
from .utils import cleandir, unreachable
from . import utils as u

path_1 = "/path_1"


def f1(i: int) -> int:
    return i


def f2() -> str:
    return "A"


def f2_wrap():
    return dds.keep(path_1, f2)


@pytest.mark.usefixtures("cleandir")
def test_1():
    assert dds.eval(f2) == "A"


@pytest.mark.usefixtures("cleandir")
def test_2():
    assert dds.eval(f2_wrap) == "A"


def fun_3(i):
    return "A" + str(i)


@pytest.mark.usefixtures("cleandir")
def test_3():
    assert dds.eval(fun_3, 3) == "A3"


def f4(i, j=0):
    return "A" + str(i) + str(j)


def f4_wrap():
    # f4(1)
    # f4(1, 2)
    f4(1, j=2)
    return f4(i=1, j=2)


@pytest.mark.usefixtures("cleandir")
def test_4():
    assert dds.eval(f4_wrap) == "A12"


def f5():
    return unreachable()


def f5_wrap():
    return dds.keep(path_1, f5)


@pytest.mark.usefixtures("cleandir")
def test_5():
    assert dds.eval(f5_wrap) == "0"
    u.unreachable_var = 1
    assert dds.eval(f5_wrap) == "0"
