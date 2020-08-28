import dds
import logging

logging.basicConfig(level=logging.DEBUG)

path = "/tmp/1"


def f1a(i: int) -> int: return i * 2


def f1():
    return "ABCD"


def f2():
    dds.keep(path, f1)
    x = 3
    dds.cache(f1a, x)


dds.eval(f2)
