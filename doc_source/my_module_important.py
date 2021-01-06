import dds
from my_module_utils import get_util_variable

@dds.data_function("/f")
def f():
    print("executing f")

    return get_util_variable() * 2