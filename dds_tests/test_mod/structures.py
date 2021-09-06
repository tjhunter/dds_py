import dds
from ..utils import Counter


c = Counter()

l = [1]


@dds.data_function("/p")
def f1():
    c.increment()
    return len(l)
