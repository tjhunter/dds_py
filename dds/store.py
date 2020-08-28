import logging
from typing import Any, Optional
import os
import pickle

from .structures import PyHash, Path, KSException

_logger = logging.getLogger(__name__)


# Model:
# /store/blobs/.../ -> the blobs
# /

class Store(object):

    def has_blob(self, key: PyHash) -> bool:
        pass

    def fetch_blob(self, key: PyHash) -> Optional[Any]:
        pass

    def store_blob(self, key: PyHash, blob: Any):
        """ idempotent
        """
        pass

    def register(self, path: Path, key: PyHash):
        """
        does not commit until sync_paths() is called: this path is available
        but is not publicly committed.
        """
        pass

    def get_path(self, path: Path) -> Optional[PyHash]:
        pass

    def sync_paths(self):
        """
        Exposes publicly all the changes that were not done on the public dataset.
        """
        pass
    # TODO: reset paths to start the store from scratch without losing data


class LocalFileStore(Store):

    def __init__(self, internal_dir: str, data_dir: str):
        self._root = internal_dir
        self._data_root = data_dir
        if not os.path.isdir(internal_dir):
            raise KSException(f"Path {internal_dir} is not a directory")
        if not os.path.isdir(data_dir):
            raise KSException(f"Path {data_dir} is not a directory")
        p_blobs = os.path.join(self._root, "blobs")
        if not os.path.exists(p_blobs):
            os.makedirs(p_blobs)
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
            if isinstance(blob, str):
                f.write(blob.encode(encoding="utf-8"))
            else:
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

    def sync_paths(self):
        with open(self._register, "rb") as f:
            dct = pickle.load(f)
        for (path, key) in dct.items():
            splits = [s.replace("/", "") for s in os.path.split(path)]
            loc_dir = os.path.join(self._data_root, *(splits[:-1]))
            loc = os.path.join(loc_dir, splits[-1])
            if not os.path.exists(loc_dir):
                _logger.debug(f"Creating dir {loc_dir}")
                os.makedirs(loc_dir)
            loc_blob = os.path.join(self._root, "blobs", key)
            if os.path.exists(loc) and os.path.realpath(loc) == loc_blob:
                _logger.debug(f"Link {loc} up to date")
            else:
                if os.path.exists(loc):
                    os.remove(loc)
                _logger.debug(f"Link {loc} -> {loc_blob}")
                os.symlink(loc_blob, loc)



