import dds
import pytest
from .utils import cleandir

_ = cleandir


@dds.data_function("/p")
def f1():
    def inner():
        return 3

    return inner


class C(object):
    def __call__(self):
        return 3


@dds.data_function("/p")
def f2():
    return C()


@pytest.mark.usefixtures("cleandir")
def test():
    # This call does not work because the inner function cannot
    # be serialized by python.
    with pytest.raises(AttributeError):
        assert f1().f()
    # An alternative is to explicitly define an object that can be
    # called.
    assert f2()() == 3
