import dds
import logging
import pandas as pd

logging.basicConfig(level=logging.DEBUG)


path = "/tmp/1"


external = "other"


def f1a(i: int) -> int:
    return i * 2


def f1():
    return "ABCD" + external


def f4():
    return pd.DataFrame(data={"x": [3]})


def f2():
    if True:
        z = dds.keep(path, f4)
    x = 3
    dds.cache(f1a, x)


dds.eval(f2)
