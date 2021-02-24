import dds
import pytest
from .utils import cleandir

_ = cleandir


@dds.data_function("/test")
def f() -> str:
    return ""


@pytest.mark.usefixtures("cleandir")
def test_typed_annotation():
    x: str = f()
    _ = x


def f2(x, y, z):
    return x + y + z


def f3():
    return 1


@pytest.mark.usefixtures("cleandir")
def test_keep_infer():
    assert dds.keep("/p", f2, 2, y=3, z=0) == 5
    assert dds.keep("/p3", f3) == 1
