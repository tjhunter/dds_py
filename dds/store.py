import json
import logging
import os
import time
from pathlib import PurePath
from typing import Any, Optional, List, Union, Dict
from collections import OrderedDict

from .codec import codec_registry, CodecRegistry
from .structures import (
    PyHash,
    DDSPath,
    DDSException,
    GenericLocation,
    ProtocolRef,
    FileCodecProtocol,
    CodecProtocol,
    DDSErrorCode,
)
from .structures_utils import SupportedTypeUtils as STU

_logger = logging.getLogger(__name__)


# Model:
# /store/blobs/.../ -> the blobs
# /


class Store(object):
    def has_blob(self, key: PyHash) -> bool:
        pass

    def fetch_blob(self, key: PyHash) -> Optional[Any]:
        pass

    def store_blob(self, key: PyHash, blob: Any, codec: Optional[ProtocolRef]) -> None:
        """ idempotent
        """
        pass

    def sync_paths(self, paths: "OrderedDict[DDSPath, PyHash]") -> None:
        """
        Commits all the paths.
        """
        pass

    def fetch_paths(self, paths: List[DDSPath]) -> "OrderedDict[DDSPath, PyHash]":
        """
        Fetches a set of paths from the store. It is expected that all the paths are returned.
        """
        pass

    def codec_registry(self) -> CodecRegistry:
        """
        The registry of codecs associated to this instance of a store.

        It is not necessarily unique
        It may not be called for mutable operations during an evaluation. In that case, the behavior is not defined.
        """
        pass

    # TODO: reset paths to start the store from scratch without losing data


class NoOpStore(Store):
    """
    The store that never stores an object.

    This store is in practice of very limited value because it cannot store paths to an object either.
    As a result, dds.load() will not work correctly with this store.

    It is recommended to use this store only to debug specific issues for which DDS would be disabled
    altogether.
    """

    def has_blob(self, key: PyHash) -> bool:
        return False

    def fetch_blob(self, key: PyHash) -> Optional[Any]:
        raise DDSException(f"Blob {key} not store (NoOpStore)")

    def store_blob(self, key: PyHash, blob: Any, codec: Optional[ProtocolRef]) -> None:
        return

    def sync_paths(self, paths: "OrderedDict[DDSPath, PyHash]") -> None:
        """
        Commits all the paths.
        """
        return

    def fetch_paths(self, paths: List[DDSPath]) -> "OrderedDict[DDSPath, PyHash]":
        """
        Fetches a set of paths from the store. It is expected that all the paths are returned.
        """
        raise DDSException(f"Cannot fetch paths (the NoOpStore does not store paths)")

    def codec_registry(self) -> CodecRegistry:
        return codec_registry()


class MemoryStore(Store):
    """
    The store that stores all objects in memory, without saving them permanently in storage.
    It is an good example of how to implement a store that is fully functional.

    This store is useful when the following conditions are met:
    - there is limited value in storing objects beyond the lifetime of the process
    - some complex objects are not serializable
    - the objects are not too large in memory

    This store is not useful for most users, but is useful in debugging or testing context.
    """

    def __init__(self):
        self._cache: Dict[PyHash, Any] = {}
        self._paths: Dict[DDSPath, PyHash] = {}

    def has_blob(self, key: PyHash) -> bool:
        return key in self._cache

    def fetch_blob(self, key: PyHash) -> Optional[Any]:
        return self._cache.get(key)

    def store_blob(self, key: PyHash, blob: Any, codec: Optional[ProtocolRef]) -> None:
        if key in self._cache:
            _logger.warning(f"Overwriting key {key}")
        self._cache[key] = blob

    def sync_paths(self, paths: "OrderedDict[DDSPath, PyHash]") -> None:
        """
        Commits all the paths.
        """
        for (p, k) in paths.items():
            if p in self._paths:
                _logger.debug(f"Overwriting path: {p} -> {k}")
            else:
                _logger.debug(f"Registering path: {p} -> {k}")
            self._paths[p] = k

    def fetch_paths(self, paths: List[DDSPath]) -> "OrderedDict[DDSPath, PyHash]":
        """
        Fetches a set of paths from the store. It is expected that all the paths are returned.
        """
        missing_paths = [p for p in paths if p not in self._paths]
        if missing_paths:
            raise DDSException(f"Missing paths in store: {missing_paths}")
        return OrderedDict([(p, self._paths[p]) for p in paths])

    def codec_registry(self) -> CodecRegistry:
        # All the default content
        return codec_registry()


class LocalFileStore(Store):
    def __init__(self, internal_dir: str, data_dir: str, create_dirs: bool = True):
        self._root = internal_dir
        self._data_root = data_dir
        if not os.path.isdir(internal_dir):
            if create_dirs:
                _logger.debug(f"Creating dir {internal_dir}")
                os.makedirs(internal_dir)
            else:
                raise DDSException(
                    f"Path {internal_dir} is not a directory",
                    DDSErrorCode.STORE_PATH_NOT_FOUND,
                )
        if not os.path.isdir(data_dir):
            if create_dirs:
                _logger.debug(f"Creating dir {data_dir}")
                os.makedirs(data_dir)
            else:
                raise DDSException(
                    f"Path {data_dir} is not a directory",
                    DDSErrorCode.STORE_PATH_NOT_FOUND,
                )
        p_blobs = os.path.join(self._root, "blobs")
        if not os.path.exists(p_blobs):
            os.makedirs(p_blobs)

    def __repr__(self):
        return f"LocalFileStore(internal_dir={self._root} data_dir={self._data_root})"

    def fetch_blob(self, key: PyHash) -> Any:
        p = os.path.join(self._root, "blobs", key)
        meta_p = os.path.join(self._root, "blobs", key + ".meta")
        if not os.path.exists(p) or not os.path.exists(meta_p):
            return None
        with open(meta_p, "rb") as f:
            ref = ProtocolRef(json.load(f)["protocol"])
        codec = codec_registry().get_codec(None, ref)
        if isinstance(codec, CodecProtocol):
            return codec.deserialize_from(GenericLocation(p))
        elif isinstance(codec, FileCodecProtocol):
            # Directly deserializing from the final path
            return codec.deserialize_from(PurePath(p))

    def store_blob(
        self, key: PyHash, blob: Any, codec: Optional[ProtocolRef] = None
    ) -> None:
        protocol: Union[CodecProtocol, FileCodecProtocol] = codec_registry().get_codec(
            STU.from_type(type(blob)), codec
        )
        p = os.path.join(self._root, "blobs", key)
        if isinstance(protocol, CodecProtocol):
            protocol.serialize_into(blob, GenericLocation(p))
        elif isinstance(protocol, FileCodecProtocol):
            # This is the local file system, we can directly copy the file to its final destination
            protocol.serialize_into(blob, PurePath(p))
        else:
            raise DDSException(f"Wrong protocol type: {type(protocol)} {protocol}")
        meta_p = os.path.join(self._root, "blobs", key + ".meta")
        with open(meta_p, "wb") as f:
            f.write(
                json.dumps(
                    {
                        "protocol": protocol.ref(),
                        "timestamp_millis": current_timestamp(),
                    }
                ).encode("utf-8")
            )
        _logger.debug(f"Committed new blob in {key}")

    def has_blob(self, key: PyHash) -> bool:
        p = os.path.join(self._root, "blobs", key)
        return os.path.exists(p)

    def sync_paths(self, paths: "OrderedDict[DDSPath, PyHash]") -> None:
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

    def fetch_paths(self, paths: List[DDSPath]) -> "OrderedDict[DDSPath, PyHash]":
        res = OrderedDict()
        for path in paths:
            if path not in res:
                # Assemble the path
                splits = [s.replace("/", "") for s in os.path.split(path)]
                loc_dir = os.path.join(self._data_root, *(splits[:-1]))
                loc = os.path.join(loc_dir, splits[-1])
                if not os.path.exists(loc_dir):
                    _logger.debug(f"Dir {loc_dir} does not exist")
                    raise DDSException(
                        f"Requested to load path {path} but directory {loc_dir} does not exist"
                    )
                if not os.path.exists(loc):
                    raise DDSException(
                        f"Requested to load path {path} but path {loc} does not exist"
                    )
                rp = os.path.realpath(loc)
                # The key is the last element of the path
                key = PyHash(os.path.split(rp)[-1])
                res[path] = key
        return res

    def codec_registry(self) -> CodecRegistry:
        return codec_registry()


def current_timestamp() -> int:
    """ The current timestamp.

    Note: this timestamp is not secure because it depends on a reliable source
    of time on the client's machine. It is only used for limited precision
    operations nuch as garbage collection.
    """
    # TODO: use time.time_ns() when dropping support for python 3.6
    return int(round(time.time() * 1000))
