import dds
import pytest
from .utils import cleandir, Counter, spath

_ = cleandir


def fun1():
    return dds.keep("/path", lambda: 1)


x = 1


def fun2():
    return dds.keep(
        "/path",
        lambda: (
            # comment
            x
            +
            # comment
            x
        ),
    )


@pytest.mark.usefixtures("cleandir")
def test():
    # Top-level calls work.
    assert fun1() == 1
    assert fun2() == 2
    # Not implemented yet
    # assert dds.eval(fun1) == 1
    # assert dds.eval(fun1) == 2
