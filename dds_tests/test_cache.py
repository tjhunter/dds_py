import dds
import pytest
from .utils import cleandir

_ = cleandir

path_1 = "/path_1"

my_var = 1


@dds.data_function("/p")
def f1() -> int:
    return my_var


@pytest.mark.usefixtures("cleandir")
def test_execute_when_var_changes():
    global my_var
    assert f1() == 1
    my_var = 2
    assert f1() == 2
