"""
Databricks-specific storage implementation. It is based on the dbutils object
which has to be provided at runtime.
"""
import base64
import json
import pickle
import logging
from pathlib import Path
from typing import Any, Optional, List, Type
from collections import OrderedDict

import pyspark.sql  # type:ignore
from pyspark.sql import DataFrame

from ..codec import CodecRegistry
from ..store import Store
from ..structures import CodecProtocol, ProtocolRef
from ..structures import PyHash, DDSPath, GenericLocation

_logger = logging.getLogger(__name__)


def _pprint_exception(e: Exception) -> str:
    return "".join(str(e).split("\n")[:3]).replace("\t", "")


class PySparkDatabricksCodec(CodecProtocol):
    def ref(self):
        return ProtocolRef("dbfs.pyspark")

    # TODO: just use strings, it will be faster
    def handled_types(self):
        return [pyspark.sql.DataFrame]

    def serialize_into(self, blob: DataFrame, loc: GenericLocation) -> None:
        assert isinstance(blob, DataFrame), type(blob)
        blob.write.parquet(loc)
        _logger.debug(f"Committed dataframe to parquet: {loc}")

    def deserialize_from(self, loc: GenericLocation) -> DataFrame:
        session = pyspark.sql.SparkSession.getActiveSession()
        _logger.debug(f"Reading parquet from loc {loc} using session {session}")
        df = session.read.parquet(loc)
        _logger.debug(f"Done reading parquet from loc {loc}: {df}")
        return df


class BytesDBFSCodec(CodecProtocol):
    def __init__(self, dbutils: Any, encode: bool = True):
        self._dbutils = dbutils
        self._encode = encode

    def ref(self):
        return ProtocolRef("dbfs.bytes")

    def handled_types(self):
        return [bytes]

    def serialize_into(self, blob: bytes, loc: GenericLocation) -> None:
        # The DBFS layer does not provide guarantees about respecting the encoding
        # For safety, the content is encoded first using base64
        if self._encode:
            blob_enc = base64.b64encode(blob) if self._encode else blob
            assert blob_enc.isascii()
            blob_str = blob_enc.decode("ascii")
        else:
            blob_str = blob.decode("utf-8")
        self._dbutils.fs.put(loc, blob_str, overwrite=True)

    def deserialize_from(self, loc: GenericLocation) -> bytes:
        blob_str: str = self._dbutils.fs.head(loc)  # type:ignore
        if self._encode:
            blob_enc = blob_str.encode("ascii")
            return base64.b64decode(blob_enc)
        return bytes(blob_str, encoding="utf-8")


class StringDBFSCodec(CodecProtocol):
    def __init__(self, dbutils: Any):
        self._codec = BytesDBFSCodec(dbutils, False)

    def ref(self) -> ProtocolRef:
        return ProtocolRef("dbfs.string")

    def handled_types(self) -> List[Type[Any]]:
        return [str]

    def serialize_into(self, blob: str, loc: GenericLocation) -> None:
        self._codec.serialize_into(blob.encode("utf-8"), loc)

    def deserialize_from(self, loc: GenericLocation) -> str:
        return self._codec.deserialize_from(loc).decode("utf-8")


class PickleDBFSCodec(CodecProtocol):
    def __init__(self, dbutils: Any):
        self._codec = BytesDBFSCodec(dbutils, True)

    def ref(self) -> ProtocolRef:
        return ProtocolRef("dbfs.pickle")

    def handled_types(self) -> List[Type[Any]]:
        return [object, type(None)]

    def serialize_into(self, blob: Any, loc: GenericLocation) -> None:

        self._codec.serialize_into(pickle.dumps(blob), loc)

    def deserialize_from(self, loc: GenericLocation) -> Any:
        return pickle.loads(self._codec.deserialize_from(loc))


class DBFSStore(Store):
    def __init__(self, internal_dir: str, data_dir: str, dbutils: Any):
        self._internal_dir: Path = Path(internal_dir)
        self._data_dir: Path = Path(data_dir)
        _logger.debug(
            f"Created DBFSStore: internal_dir: {self._internal_dir} data_dir: {self._data_dir}"
        )
        self._dbutils = dbutils
        self._registry = CodecRegistry(
            [
                PySparkDatabricksCodec(),
                StringDBFSCodec(dbutils),
                PickleDBFSCodec(dbutils),
                BytesDBFSCodec(dbutils, True),
            ]
        )

    def fetch_blob(self, key: PyHash) -> Optional[Any]:
        p = self._internal_dir.joinpath("blobs", key)
        meta = self._fetch_meta(key)
        if meta is None:
            return None
        ref = ProtocolRef(meta["protocol"])
        codec = self._registry.get_codec(None, ref)
        return codec.deserialize_from(GenericLocation(str(p)))

    def store_blob(
        self, key: PyHash, blob: Any, codec: Optional[ProtocolRef] = None
    ) -> None:
        protocol = self._registry.get_codec(type(blob), codec)
        p = self._blob_path(key)
        protocol.serialize_into(blob, GenericLocation(str(p)))
        meta_p = self._internal_dir.joinpath("blobs", key + ".meta")
        try:
            meta = json.dumps({"protocol": protocol.ref()})
            self._put(meta_p, meta)
        except Exception as e:
            _logger.warning(
                f"Failed to write blob metadata to {meta_p}: {_pprint_exception(e)}"
            )
            raise e
        _logger.debug(f"Committed new blob in {key}")

    def has_blob(self, key: PyHash) -> bool:
        return self._fetch_meta(key) is not None

    def sync_paths(self, paths: "OrderedDict[DDSPath, PyHash]") -> None:
        # This is a brute force approach that copies all the data and writes extra meta data.
        for (dds_p, key) in paths.items():
            # Look for the redirection file associated to this file
            # The paths are /_dds_meta/path
            redir_p = Path("_dds_meta/").joinpath("./" + dds_p)
            redir_path = self._physical_path(redir_p)
            # Try to read the redirection information:
            _logger.debug(
                f"Attempting to read metadata for key {key}: {redir_path} {redir_p} {dds_p}"
            )
            meta: Optional[str]
            try:
                meta = self._head(redir_path)
            except Exception as e:
                _logger.debug(
                    f"Could not read metadata for key {key}: {_pprint_exception(e)}"
                )
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
                obj_path = self._physical_path(Path("./" + dds_p))
                _logger.debug(f"Copying {blob_path} -> {obj_path}")
                self._dbutils.fs.cp(str(blob_path), str(obj_path), recurse=True)
                _logger.debug(f"Linking new file {obj_path}")
                try:
                    meta = json.dumps({"redirection_key": key})
                    self._put(redir_path, meta)
                except Exception as e:
                    _logger.warning(
                        f"Failed to write blob metadata to {redir_path}: {_pprint_exception(e)}"
                    )
                    raise e
            else:
                _logger.debug(f"Path {dds_p} is up to ddate (key {key})")

    def _blob_path(self, key: PyHash) -> Path:
        return self._internal_dir.joinpath("blobs", key)

    def _physical_path(self, dds_p: Path) -> Path:
        return self._data_dir.joinpath(dds_p)

    def _fetch_meta(self, key: PyHash) -> Optional[Any]:
        meta_p = self._internal_dir.joinpath("blobs", key + ".meta")
        try:
            _logger.debug(f"Attempting to read metadata for key {key}: {meta_p}")
            meta: str = self._head(meta_p)
            _logger.debug(f"Attempting to read metadata for key {key}: meta = {meta}")
            return json.loads(meta)
        except Exception as e:
            _logger.debug(
                f"Could not read metadata for key {key}: {_pprint_exception(e)}"
            )
            return None

    def _head(self, p: Path) -> str:
        return self._dbutils.fs.head(str(p))  # type:ignore

    def _put(self, p: Path, blob: str) -> Any:
        return self._dbutils.fs.put(str(p), blob, overwrite=True)
