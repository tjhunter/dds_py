"""
Utilities related to structures
"""
import pathlib

from .structures import DDSPath, KSException

from typing import Union


class DDSPathUtils(object):

    @staticmethod
    def create(p: Union[str, pathlib.Path]) -> DDSPath:
        if isinstance(p, str):
            if not p or p[0] != "/":
                raise KSException(f"Provided path {p} is not absolute. All paths must be absolute")
            # TODO: more checks
            return DDSPath(p)
        if isinstance(p, pathlib.Path):
            if not p.is_absolute():
                raise KSException(f"Provided path {p} is not absolute. All paths must be absolute")
            return DDSPath(p.absolute().as_posix())
        raise NotImplementedError(f"Cannot make a path from object type {type(p)}: {p}")

