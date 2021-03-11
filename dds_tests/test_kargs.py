import dds
import pytest
from .utils import cleandir

_ = cleandir

path_1 = "/path_1"


def f1(x, y, z):
    return x + y + z


def f1_1():
    kargs = [1, 2, 3]
    return dds.keep("/p1", f1, *kargs)


def f1_2():
    kargs = [2]
    kwargs = [{"z": 3}]
    return f1(1, *kargs, **kwargs[0])


@pytest.mark.usefixtures("cleandir")
def test_kargs_1():
    assert dds.eval(f1_1) == 6
    assert dds.eval(f1_2) == 6


def f2(*args, **kwargs):
    return len(args) + len(kwargs)


def f2_1():
    kargs = [1, 2, 3]
    return dds.keep("/p1", f2, *kargs)


def f2_2():
    kargs = [2]
    kwargs = [{"z": 3}]
    return f2(1, *kargs, **kwargs[0])


@pytest.mark.usefixtures("cleandir")
def test_kargs_2():
    assert dds.eval(f2_1) == 3
    assert dds.eval(f2_2) == 3
