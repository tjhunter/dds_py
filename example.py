import karps_stone as ks
import logging
import pickle

logging.basicConfig(level=logging.DEBUG)

path = "/tmp/2"


def f1a(i: int) -> int: return i * 2


def f1():
    return 2


def f2():
    pickle.dumps({})
    ks.keep(path, f1)
    x = 3
    ks.cache(f1a, x)


ks.eval(f2)
