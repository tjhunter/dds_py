import dds
import pytest
from .utils import cleandir, Counter

_ = cleandir

_c = Counter()


class Class1(object):
    @staticmethod
    def method1():
        _c.increment()
        return 1

    @staticmethod
    def method2():
        return Class1.method1() + 1

    @classmethod
    def test(cls): return 1

    def test2(self): return 4


def f():
    return Class1.method1()


@pytest.mark.usefixtures("cleandir")
def test_static_1():
    assert dds.keep("/path", f) == 1
    assert _c.value == 1
    assert dds.keep("/path", Class1.method1) == 1
    assert _c.value == 1
    assert dds.keep("/path", Class1.method2) == 2
    assert _c.value == 1
