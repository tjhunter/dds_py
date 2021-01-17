import dds
import pytest
from .utils import cleandir, Counter, spath, unreachable_method
from dds_tests.unauthorized_mod.functions import function1, function2

externa_fun = function1

@dds.data_function("/p")
def fun():
    return externa_fun()


@pytest.mark.usefixtures("cleandir")
def test():
    assert dds.eval(fun) == function1()
    print(locals())
    print(globals())
