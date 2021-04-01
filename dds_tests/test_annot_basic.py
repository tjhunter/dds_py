import dds
import pytest
from .utils import cleandir, Counter
from dds.structures import DDSException, DDSErrorCode

_ = cleandir

_c = Counter()

_p = "/path2"


@dds.data_function("/p")
def f():
    _c.increment()
    return "a"


def f1():
    f()
    return f()


@pytest.mark.usefixtures("cleandir")
def test():
    assert f() == "a"
    assert _c.value == 1
    assert dds.eval(f) == "a"
    assert _c.value == 1
    assert f1() == "a"
    assert _c.value == 1
    assert dds.eval(f1) == "a"
    assert _c.value == 1


@dds.data_function("/p")
def f2(x):
    return "a"


def f2_1():
    f2(3)


@pytest.mark.usefixtures("cleandir")
def test_args():
    with pytest.raises(DDSException) as e:
        f2(3)
    assert e.value.error_code == DDSErrorCode.ARG_IN_DATA_FUNCTION
    with pytest.raises(DDSException) as e:
        dds.eval(f2_1)
    assert e.value.error_code == DDSErrorCode.ARG_IN_DATA_FUNCTION
