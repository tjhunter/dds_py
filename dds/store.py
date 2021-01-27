import json
import logging
import os
import time
from pathlib import PurePath
from typing import Any, Optional, List, Union
from collections import OrderedDict

from .codec import codec_registry, CodecRegistry
from .structures import (
    PyHash,
    DDSPath,
    KSException,
    GenericLocation,
    ProtocolRef,
    FileCodecProtocol,
    CodecProtocol,
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

    def fetch_paths(self, paths: List[DDSPath]) -> "OrderedDict[DDSPath, PyHash]":
        """
        Fetches a set of paths from the store. It is expected that all the paths are returned.
        """
        pass

    def codec_registry(self) -> CodecRegistry:
        """
        The registry of codecs associated to this instance of a store.

        It is not necessaily unique
        It may not be called for mutable operations during an evaluation. In that case, the behavior is not defined.
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
            raise KSException(f"{type(protocol)} {protocol}")
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
                    raise KSException(
                        f"Requested to load path {path} but directory {loc_dir} does not exist"
                    )
                if not os.path.exists(loc):
                    raise KSException(
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
