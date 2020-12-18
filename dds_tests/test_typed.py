import dds
import pytest
from .utils import cleandir

_ = cleandir


@dds.dds_function("/test")
def f() -> str:
    return ""


@pytest.mark.usefixtures("cleandir")
def test_typed_annotation():
    x: str = f()
    _ = x
