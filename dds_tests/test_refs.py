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
    assert fun_5_counter2.value == 2


"""
INFO     dds:__init__.py:80 Interaction tree:
INFO     dds:__init__.py:81 `- Fun <dds_tests.test_refs.fun_5_f> None <- d9cc78b7e6668b486708ea453301e6d726ff5ea9a94e3099f5f2c0c7fa156077
INFO     dds:__init__.py:81    |- dep: p -> <dds_tests.test_refs.p>: 379c9f23425a38698d164abeb339116b9295b8fa7ea8747a92d74fd7885beef0
INFO     dds:__init__.py:81    |- dep: p2 -> <dds_tests.test_refs.p2>: cd37cd9268924fd45046f0ea7eb21219d73bacc8627f2068aace46cf9fe6e169
INFO     dds:__init__.py:81    |- Fun <dds_tests.test_refs.fun_5_f1> /path <- d66d8e1ac71c6da994e5c4c9c16463487c032e538e150f19c30da38f21503e35
INFO     dds:__init__.py:81    |  `- dep: fun_5_obj -> <dds_tests.test_refs.fun_5_obj>: 5feceb66ffc86f38d952786c6d696c79c2dbc239dd4e91b46729d73a27fb57e9
INFO     dds:__init__.py:81    `- Fun <dds_tests.test_refs.fun_5_f2> /path2 <- 72a6daf748a1149e186e4f913cf163d00d7d28d55d7f58c6c7ff7ae8dfc8e2ed

INFO     dds:__init__.py:80 Interaction tree:
INFO     dds:__init__.py:81 `- Fun <dds_tests.test_refs.fun_5_f> None <- 6eae7252bbe4de6b37b31b23d430fc9921239a8c58e9ad046865acddf86cd6bc
INFO     dds:__init__.py:81    |- dep: p -> <dds_tests.test_refs.p>: 379c9f23425a38698d164abeb339116b9295b8fa7ea8747a92d74fd7885beef0
INFO     dds:__init__.py:81    |- dep: p2 -> <dds_tests.test_refs.p2>: cd37cd9268924fd45046f0ea7eb21219d73bacc8627f2068aace46cf9fe6e169
INFO     dds:__init__.py:81    |- Fun <dds_tests.test_refs.fun_5_f1> /path <- dbfc04108761fc60f596d8f5af50dd7082f3b5e7b40cb6bc7dac738d81a69a42
INFO     dds:__init__.py:81    |  `- dep: fun_5_obj -> <dds_tests.test_refs.fun_5_obj>: 6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b
INFO     dds:__init__.py:81    `- Fun <dds_tests.test_refs.fun_5_f2> /path2 <- 72a6daf748a1149e186e4f913cf163d00d7d28d55d7f58c6c7ff7ae8dfc8e2ed
"""
