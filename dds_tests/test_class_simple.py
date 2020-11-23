import dds
import pytest
from .utils import cleandir, Counter, spath

_ = cleandir

_c = Counter()

class_obj = 1


class Class1(object):
    def __init__(self):
        self._var = class_obj

    def method(self):
        _c.increment()
        return 1


def f():
    return Class1().method()


@pytest.mark.usefixtures("cleandir")
def test_simple():
    global class_obj
    class_obj = 1
    assert dds.keep("/path", f) == 1
    assert _c.value == 1
    assert dds.keep("/path", f) == 1
    assert _c.value == 1
