"""

"""

import logging
from typing import Any, Optional, OrderedDict
import os
from pathlib import Path
import json


import pyspark.sql
from pyspark.sql import DataFrame

from ..structures import CodecProtocol, ProtocolRef, GenericLocation
from ..structures import PyHash, DDSPath, KSException, GenericLocation
from ..codec import CodecRegistry
from ..store import Store

_logger = logging.getLogger(__name__)


class PySparkDatabricksCodec(CodecProtocol):
    def ref(self):
        return ProtocolRef("default.pyspark")

    # TODO: just use strings, it will be faster
    def handled_types(self):
        return [pyspark.sql.DataFrame]

    def serialize_into(self, blob: DataFrame, loc: GenericLocation):
        assert isinstance(blob, DataFrame), type(blob)
        blob.write.parquet(loc)
        _logger.debug(f"Committed dataframe to parquet: {loc}")

    def deserialize_from(self, loc: GenericLocation) -> DataFrame:
        session = pyspark.sql.SparkSession.getOrCreate()
        _logger.debug(f"Reading parquet from loc {loc} using session {session}")
        df = session.read.parquet(loc)
        _logger.debug(f"Done reading parquet from loc {loc}: {df}")
        return df


class StringDBFSCodec(CodecProtocol):
    def __init__(self, dbutils: Any):
        self._dbutils = dbutils

    def ref(self):
        return ProtocolRef("default.string")

    def handled_types(self):
        return [str]

    def serialize_into(self, blob: str, loc: GenericLocation):
        # try:
        self._dbutils.fs.put(loc, blob, overwrite=True)
        # except Exception as e:
        #     _logger.warning(f"Suppresing exception when writing blob to {loc}: {e}")

    def deserialize_from(self, loc: GenericLocation) -> str:
        return self._dbutils.fs.head(loc)


class DBFSStore(Store):
    def __init__(self, internal_dir: str, data_dir: str, dbutils: Any):
        internal_dir = Path(internal_dir)
        data_dir = Path(data_dir)
        _logger.debug(
            f"Created DBFSStore: internal_dir: {internal_dir} data_dir: {data_dir}"
        )
        self._internal_dir: Path = internal_dir
        self._data_dir: Path = data_dir
        self._dbutils = dbutils
        self._registry = CodecRegistry(
            [PySparkDatabricksCodec(), StringDBFSCodec(dbutils)]
        )

    def fetch_blob(self, key: PyHash) -> Optional[Any]:
        p = self._internal_dir.joinpath("blobs", key)
        meta = self._fetch_meta(key)
        if meta is None:
            return None
        ref = ProtocolRef(meta["protocol"])
        codec = self._registry.get_codec(None, ref)
        return codec.deserialize_from(GenericLocation(str(p)))

    def store_blob(self, key: PyHash, blob: Any, codec: Optional[ProtocolRef] = None):
        protocol = self._registry.get_codec(type(blob), codec)
        p = self._blob_path(key)
        protocol.serialize_into(blob, GenericLocation(str(p)))
        meta_p = self._internal_dir.joinpath("blobs", key + ".meta")
        try:
            meta = json.dumps({"protocol": protocol.ref()})
            self._put(meta_p, meta)
        except Exception as e:
            _logger.warning(f"Failed to write blob metadata to {meta_p}: {e}")
            raise e
        _logger.debug(f"Committed new blob in {key}")

    def has_blob(self, key: PyHash) -> bool:
        return self._fetch_meta(key) is not None

    def sync_paths(self, paths: OrderedDict[DDSPath, PyHash]):
        # This is a brute force approach that copies all the data and writes extra meta data.
        for (dds_p, key) in paths.items():
            # Look for the redirection file associated to this file
            # The paths are /.dds_links/path
            redir_p = Path("_dds_meta/").joinpath("./" + dds_p)
            redir_path = self._physical_path(redir_p)
            # Try to read the redirection information:
            _logger.debug(
                f"Attempting to read metadata for key {key}: {redir_path} {redir_p} {dds_p}"
            )
            meta: Optional[str] = None
            try:
                meta = self._head(redir_path)
            except Exception as e:
                _logger.debug(f"Could not read metadata for key {key}: {e}")
                meta = None
            if meta is not None:
                redir_key = json.loads(meta)["redirection_key"]
            else:
                redir_key = None
            if redir_key is None or redir_key != key:
                _logger.debug(
                    f"Path {dds_p} needs update (registered key {redir_key} != {key})"
                )
                blob_path = self._blob_path(key)
                obj_path = self._physical_path("./" + dds_p)
                _logger.debug(f"Copying {blob_path} -> {obj_path}")
                self._dbutils.fs.cp(str(blob_path), str(obj_path))
                _logger.debug(f"Linking new file {obj_path}")
                try:
                    meta = json.dumps({"redirection_key": key})
                    self._put(redir_path, meta)
                except Exception as e:
                    _logger.warning(
                        f"Failed to write blob metadata to {redir_path}: {e}"
                    )
                    raise e
            else:
                _logger.debug(f"Path {dds_p} is up to ddate (key {key})")

    def _blob_path(self, key: PyHash) -> Path:
        return self._internal_dir.joinpath("blobs", key)

    def _physical_path(self, dds_p: DDSPath) -> Path:
        return self._data_dir.joinpath(dds_p)

    def _fetch_meta(self, key: PyHash) -> Optional[Any]:
        meta_p = self._internal_dir.joinpath("blobs", key + ".meta")
        try:
            _logger.debug(f"Attempting to read metadata for key {key}: {meta_p}")
            meta: str = self._head(meta_p)
            _logger.debug(f"Attempting to read metadata for key {key}: meta = {meta}")
            return json.loads(meta)
        except Exception as e:
            _logger.debug(f"Could not read metadata for key {key}: {e}")
            return None

    def _head(self, p: Path) -> str:
        return self._dbutils.fs.head(str(p))

    def _put(self, p: Path, blob: str):
        return self._dbutils.fs.put(str(p), blob, overwrite=True)
