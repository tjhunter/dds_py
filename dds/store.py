# from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional
from collections import OrderedDict

from .codec import codec_registry
from .structures import PyHash, DDSPath, KSException, GenericLocation, ProtocolRef

_logger = logging.getLogger(__name__)


# Model:
# /store/blobs/.../ -> the blobs
# /


class Store(object):
    def has_blob(self, key: PyHash) -> bool:
        pass

    def fetch_blob(self, key: PyHash) -> Optional[Any]:
        pass

    def store_blob(
        self, key: PyHash, blob: Any, codec: Optional[ProtocolRef] = None
    ) -> None:
        """ idempotent
        """
        pass

    def sync_paths(self, paths: "OrderedDict[DDSPath, PyHash]") -> None:
        """
        Commits all the paths.
        """
        pass

    # TODO: reset paths to start the store from scratch without losing data


# TODO: add a notion of FileSystemType (Local, DBFS, S3)
# We need to have a matrix between FS types and object types


class LocalFileStore(Store):
    def __init__(self, internal_dir: str, data_dir: str, create_dirs: bool = True):
        self._root = internal_dir
        self._data_root = data_dir
        if not os.path.isdir(internal_dir):
            if create_dirs:
                _logger.debug(f"Creating dir {internal_dir}")
                os.makedirs(internal_dir)
            else:
                raise KSException(f"Path {internal_dir} is not a directory")
        if not os.path.isdir(data_dir):
            if create_dirs:
                _logger.debug(f"Creating dir {data_dir}")
                os.makedirs(data_dir)
            else:
                raise KSException(f"Path {data_dir} is not a directory")
        p_blobs = os.path.join(self._root, "blobs")
        if not os.path.exists(p_blobs):
            os.makedirs(p_blobs)

    def fetch_blob(self, key: PyHash) -> Any:
        p = os.path.join(self._root, "blobs", key)
        meta_p = os.path.join(self._root, "blobs", key + ".meta")
        if not os.path.exists(p) or not os.path.exists(meta_p):
            return None
        with open(meta_p, "rb") as f:
            ref = ProtocolRef(json.load(f)["protocol"])
        codec = codec_registry().get_codec(None, ref)
        return codec.deserialize_from(GenericLocation(p))

    def store_blob(
        self, key: PyHash, blob: Any, codec: Optional[ProtocolRef] = None
    ) -> None:
        protocol = codec_registry().get_codec(type(blob), codec)
        p = os.path.join(self._root, "blobs", key)
        protocol.serialize_into(blob, GenericLocation(p))
        meta_p = os.path.join(self._root, "blobs", key + ".meta")
        with open(meta_p, "wb") as f:
            f.write(json.dumps({"protocol": protocol.ref()}).encode("utf-8"))
        _logger.debug(f"Committed new blob in {key}")

    def has_blob(self, key: PyHash) -> bool:
        p = os.path.join(self._root, "blobs", key)
        return os.path.exists(p)

    def sync_paths(self, paths):
        for (path, key) in paths.items():
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
                _logger.info(f"Link {loc} -> {loc_blob}")
                os.symlink(loc_blob, loc)
