import dds
import pytest
import pathlib
from .utils import cleandir, Counter, spath

_ = cleandir

_c = Counter()

_p = pathlib.Path("/path2")


def _fun(x=True):
    _c.increment()
    return "a" + str(x)


def fun1():
    return dds.keep(spath, _fun)


def fun2():
    return dds.keep(spath, _fun, True)


def fun3():
    return dds.keep(spath, _fun, False)


def fun_outer1():
    return dds.eval(fun2)


@pytest.mark.usefixtures("cleandir")
def test():
    fun1()
    assert _c.value == 1
    fun2()
    assert _c.value == 1
    dds.eval(fun2)
    assert _c.value == 1
    fun3()
    assert _c.value == 2
    dds.eval(fun3)
    assert _c.value == 2


def fun_args(a, b):
    return a + b


def fun_args_outer():
    dds.keep(spath, fun_args, 1, 2)
    dds.keep(spath, fun_args, 1, b=2)
    dds.keep(spath, fun_args, a=1, b=2)
    return dds.keep(spath, fun_args, b=2, a=1)


@pytest.mark.usefixtures("cleandir")
def test_args():
    # TODO: these tests are just checking that the calls are processed
    # They should also check that the function is not retriggered again.
    assert dds.keep(spath, fun_args, 1, 2) == 3
    assert dds.keep(spath, fun_args, 1, b=2) == 3
    assert dds.keep(spath, fun_args, a=1, b=2) == 3
    assert dds.eval(fun_args_outer) == 3
    assert dds.eval(fun_args, 1, 2) == 3
    assert dds.eval(fun_args, a=1, b=2) == 3
