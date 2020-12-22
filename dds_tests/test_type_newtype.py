import dds
import pytest
from .utils import cleandir, Counter, spath
from dds_tests.unauthorized_mod import ShadowClass1, ShadowClass2
from typing import NewType

_ = cleandir

X = NewType("X", str)


def f() -> X:
    return X("x")


@pytest.mark.usefixtures("cleandir")
def test_simple():
    assert dds.keep("/path", f) == "x"
