from dds.structures import DDSException, DDSErrorCode
from dds import data_function, eval
import pytest
from .utils import cleandir
from dds_tests.unauthorized_mod.unauthorized import my_fun

_ = cleandir


@pytest.mark.usefixtures("cleandir")
def test_1():
    with pytest.raises(DDSException) as e:
        my_fun()
    assert "trieved, howe" in str(e.value), e


@data_function("/f")
def fun1():
    return 1


@data_function("/f/g")
def fun2():
    return 2


def f():
    fun1()
    fun2()


@pytest.mark.usefixtures("cleandir")
def test_2():
    with pytest.raises(DDSException) as e:
        eval(f)
    assert e.value.error_code == DDSErrorCode.OVERLAPPING_PATH
