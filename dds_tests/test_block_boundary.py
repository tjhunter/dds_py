import dds
import pytest
from .utils import cleandir, Counter

_ = cleandir

_var_function = {}


def _dummy():
    pass


def _eval_function(s, function_name, dct):
    global _var_function
    _var_function["key"] = None
    s2 = (
        s
        + f"""

global _var_function
_var_function["key"] = {function_name}
    """
    )
    d = {"_var_function": _var_function, "dds": dds}
    d.update(dct)
    exec(s2, d)

    f = _var_function["key"]
    f.__dds_source = s
    f.__module__ = _dummy.__module__
    globals()[function_name] = f
    return f


_c = Counter()


def f(i):
    _ = i
    _c.increment()
    return 1


subsequent_change_s1 = """
def function():
    i = 1
    _ = dds.keep("/p", f, i)
    # a
"""

subsequent_change_s2 = """
def function():
    i = 1
    _ = dds.keep("/p", f, i)
    # b
"""


@pytest.mark.usefixtures("cleandir")
def test_subsequent_change():
    fun = _eval_function(subsequent_change_s1, "function", {"f": f})
    dds.eval(fun)
    assert _c.value == 1
    fun = _eval_function(subsequent_change_s2, "function", {"f": f})
    dds.eval(fun)
    assert _c.value == 2


next_line_change_s1 = """
def function():
    i = 1
    _ = dds.keep("/p", f, i)
    
    # a
"""

next_line_change_s2 = """
def function():
    i = 1
    _ = dds.keep("/p", f, i)
    
    # b
"""


@pytest.mark.usefixtures("cleandir")
def test_next_line_change():
    _c.value = 0
    fun = _eval_function(next_line_change_s1, "function", {"f": f})
    dds.eval(fun)
    assert _c.value == 1
    fun = _eval_function(next_line_change_s2, "function", {"f": f})
    dds.eval(fun)
    assert _c.value == 1


subexpression_s1 = """
def function():
    i = 1
    _ = [
      dds.keep("/p", f, i)
      for x in range(3)
      if x < 10
    ]
"""

subexpression_s2 = """
def function():
    i = 1
    _ = [
      dds.keep("/p", f, i)
      for x in range(3)
      if x < 3  # change
    ]
"""


@pytest.mark.usefixtures("cleandir")
def test_subexpression():
    _c.value = 0
    fun = _eval_function(subexpression_s1, "function", {"f": f})
    dds.eval(fun)
    assert _c.value == 1
    fun = _eval_function(subexpression_s2, "function", {"f": f})
    dds.eval(fun)
    # TODO: should be 2
    assert _c.value == 1
