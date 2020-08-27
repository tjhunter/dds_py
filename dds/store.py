import logging
from typing import Any, Optional
import os
import pickle

from .structures import PyHash, Path, KSException

_logger = logging.getLogger(__name__)


# Model:
# /store/blobs/.../ -> the blobs
# /store/paths/... -> the metadata for each path
# /

class Store(object):

    def has_blob(self, key: PyHash) -> bool:
        pass

    def fetch_blob(self, key: PyHash) -> Optional[Any]:
        pass

    def store_blob(self, key: PyHash, blob: Any):
        pass

    def register(self, path: Path, key: PyHash):
        pass

    def get_path(self, path: Path) -> Optional[PyHash]:
        pass


class LocalFileStore(Store):

    def __init__(self, dir: str):
        self._root = dir
        if not os.path.isdir(dir):
            raise KSException(f"Path {dir} is not a directory")
        p_blobs = os.path.join(self._root, "blobs")
        if not os.path.exists(p_blobs):
            os.makedirs(p_blobs)
        p_paths = os.path.join(self._root, "paths")
        if not os.path.exists(p_paths):
            os.makedirs(p_paths)
        self._register = os.path.join(self._root, "register.pickle")
        if not os.path.exists(self._register):
            with open(self._register, "wb") as f:
                pickle.dump({}, f)

    def fetch_blob(self, key: PyHash) -> Any:
        p = os.path.join(self._root, "blobs", key)
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def store_blob(self, key: PyHash, blob: Any):
        p = os.path.join(self._root, "blobs", key)
        with open(p, "wb") as f:
            pickle.dump(blob, f)
        _logger.debug(f"Committed new blob in {key}")

    def register(self, path: Path, key: PyHash):
        with open(self._register, "rb") as f:
            dct = pickle.load(f)
        dct[path] = key
        with open(self._register, "wb") as f:
            pickle.dump(dct, f)
        _logger.debug(f"Committed {path} -> {key}")

    def get_path(self, path: Path) -> Optional[PyHash]:
        with open(self._register, "rb") as f:
            dct = pickle.load(f)
        if path in dct:
            return dct[path]
        return None

    def has_blob(self, key: PyHash) -> bool:
        return os.path.exists(os.path.join(self._root, "blobs", key))

