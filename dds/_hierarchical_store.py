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
from .store import Store, LocalFileStore
from ._lru_store import LRUCacheStore

_logger = logging.getLogger(__name__)


class HierarchicalStore(Store):
    """
    A store that uses a hierarchy of caches for optimal performance.

    This store assumes the following:
    1. A network store that may have slow retrievals
    2. A local cache on disk: large, alleviates network transfer, but still comes at the cost of serialization

    This should be completed with an in-memory LRU store (see create_hiearchical_store function below).

    Some data structures such as Spark dataframes are meant to be only stored in the network and are not stored
    locally.

    TODO: there is currently no cleanup strategy on the local store. It is assumed it can simply be wiped.

    Note: the paths are still stored in the remote store. The paths are not cached locally. This could cause
    coherency issues. Fetching and syncing paths will still incure a call to the remote store.
    """

    def __init__(self, remote: Store, local: LocalFileStore):
        self._remote = remote
        self._local = local

    def has_blob(self, key: PyHash) -> bool:
        # First check locally if the key is present, and then remotely.
        return self._local.has_blob(key) or self._remote.has_blob(key)

    def fetch_blob(self, key: PyHash) -> Optional[Any]:
        loc = self._local.fetch_blob(key)
        if loc is not None:
            # There is a slight ambiguity here as None might be a fully deserialized blob as well.
            return loc
        blob = self._remote.fetch_blob(key)
        # TODO: we do not fetch the meta information which contain the exact codec to follow.
        # In particular, this currently does not honor the preferred codec. This should not be an issue for
        # most types right now, but nevertheless is a bug.
        protocol = self.codec_registry().get_codec(STU.from_type(type(blob)), None)
        if isinstance(protocol, FileCodecProtocol):
            # Only in the case of a file protocol, we will attempt to keep it locally.
            # For instance, spark dataframes should not be kept locally.
            self._local.store_blob(key, blob, protocol.ref())
        return blob

    def store_blob(self, key: PyHash, blob: Any, codec: Optional[ProtocolRef]) -> None:
        # Always store to the remote blob without attempting for now to optimize to the local store.
        self._remote.store_blob(key, blob, codec)

    def sync_paths(self, paths: "OrderedDict[DDSPath, PyHash]") -> None:
        self._remote.sync_paths(paths)

    def fetch_paths(self, paths: List[DDSPath]) -> "OrderedDict[DDSPath, PyHash]":
        return self._remote.fetch_paths(paths)

    def codec_registry(self) -> CodecRegistry:
        return self._remote.codec_registry()


def create_hierarchical_store(
    remote_store: Store,
    internal_local_dir: str,
    memory_cache_size: int,
    create_local_dir: bool = True,
) -> Store:
    local_store = LocalFileStore(
        internal_local_dir,
        os.path.join(internal_local_dir, "_data_unused"),
        create_local_dir,
    )
    return LRUCacheStore(
        HierarchicalStore(remote_store, local_store), memory_cache_size
    )
