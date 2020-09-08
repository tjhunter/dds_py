import dds
import pytest
from .utils import cleandir

path_1 = "/path_1"


def f1(i: int) -> int: return i


def f2() -> str: return "A"


def f2_wrap(): return dds.keep(path_1, f2)


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

