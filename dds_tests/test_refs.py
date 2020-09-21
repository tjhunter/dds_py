"""
Tests for the outer references
"""

import dds
import pytest
from .utils import cleandir, Counter
from io import UnsupportedOperation

_ = cleandir


p = "/path"
p2 = "/path2"


fun_1_obj = UnsupportedOperation()


def fun_1_f():
    _ = fun_1_obj.args
    return None


def fun_1_f1():
    dds.keep(p, fun_1_f)


@pytest.mark.usefixtures("cleandir")
def test_1():
    """ Unauthorized objects do not trigger errors """
    assert dds.eval(fun_1_f1) is None


fun_2_obj = 0
fun_2_counter = Counter()


def fun_2_f():
    _ = fun_2_obj
    fun_2_counter.increment()
    return None


def fun_2_f1():
    dds.keep(p, fun_2_f)


@pytest.mark.usefixtures("cleandir")
def test_2():
    """ Authorized objects are taken into account """
    global fun_2_obj
    assert dds.eval(fun_2_f1) is None
    fun_2_obj = 1
    assert dds.eval(fun_2_f1) is None
    assert fun_2_counter.value == 2, fun_2_counter.value


fun_3_obj = UnsupportedOperation("a")
fun_3_counter = Counter()


def fun_3_f():
    _ = fun_3_obj
    fun_3_counter.increment()


def fun_3_f1():
    dds.keep(p, fun_3_f)


@pytest.mark.usefixtures("cleandir")
def test_3():
    """ Unauthorized objects are not taken into account """
    global fun_3_obj
    assert dds.eval(fun_3_f1) is None
    fun_3_obj = UnsupportedOperation("b")
    assert dds.eval(fun_3_f1) is None
    assert fun_3_counter.value == 1, fun_3_counter.value


fun_4_obj = 0
fun_4_counter1 = Counter()
fun_4_counter2 = Counter()


def fun_4_f1():
    fun_4_counter1.increment()
    return None


def fun_4_f2():
    _ = fun_4_obj
    fun_4_counter2.increment()
    return None


def fun_4_f():
    dds.keep(p, fun_4_f1)
    dds.keep(p2, fun_4_f2)


@pytest.mark.usefixtures("cleandir")
def test_4():
    """ Authorized objects are taken into account """
    global fun_4_obj
    assert dds.eval(fun_4_f) is None
    assert fun_4_counter1.value == 1
    assert fun_4_counter2.value == 1
    fun_4_obj = 1
    assert dds.eval(fun_4_f) is None
    assert fun_4_counter1.value == 1
    assert fun_4_counter2.value == 2


fun_5_obj = 0
fun_5_counter1 = Counter()
fun_5_counter2 = Counter()


def fun_5_f1():
    _ = fun_5_obj
    fun_5_counter1.increment()
    return None


def fun_5_f2():
    fun_5_counter2.increment()
    return None


def fun_5_f():
    dds.keep(p, fun_5_f1)
    dds.keep(p2, fun_5_f2)


@pytest.mark.usefixtures("cleandir")
def test_5():
    """ Chained calls are reevaluated """
    global fun_5_obj
    assert dds.eval(fun_5_f) is None
    assert fun_5_counter1.value == 1
    assert fun_5_counter2.value == 1
    fun_5_obj = 1
    assert dds.eval(fun_5_f) is None
    assert fun_5_counter1.value == 2
    # This function comes later but has no argument -> no need to reevaluate
    assert fun_5_counter2.value == 1
