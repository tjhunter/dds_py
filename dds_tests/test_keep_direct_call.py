import dds
import pytest
from .utils import cleandir, Counter, spath

_ = cleandir

_c = Counter()


def fun1():
    _c.increment()
    return "a"

@pytest.mark.usefixtures("cleandir")
def test():
    assert dds.keep(spath, fun1) == "a"
    assert _c.value == 1
    assert dds.keep(spath, fun1) == "a"
    assert _c.value == 1
