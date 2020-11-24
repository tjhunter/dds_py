import dds
import pytest
from .utils import cleandir, Counter, spath
from dds_tests.unauthorized_mod import ShadowClass1, ShadowClass2

_ = cleandir

_c = Counter()


class Class1(object):
    def __init__(self):
        pass

    def method(self):
        _c.increment()
        return 1


def f():
    return Class1().method()


@pytest.mark.usefixtures("cleandir")
def test_simple():
    assert dds.keep("/path", f) == 1
    assert _c.value == 1
    assert dds.keep("/path", f) == 1
    assert _c.value == 1


def test_extern_class_f1():
    return ShadowClass1().method1()


@pytest.mark.usefixtures("cleandir")
def test_extern_class():
    assert dds.keep("/path", test_extern_class_f1) == 1


def test_extern_class_params_f1():
    return ShadowClass2("1", "2").method1()


@pytest.mark.usefixtures("cleandir")
def test_extern_class_params():
    assert dds.keep("/path", test_extern_class_params_f1) == 1
