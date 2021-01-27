"""
Databricks-specific storage implementation. It is based on the dbutils object
which has to be provided at runtime.
"""
import json
import logging
import tempfile
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from types import FunctionType
from typing import Any, Optional, List

from ..codec import CodecRegistry
from ..store import Store, current_timestamp
from ..structures import CodecProtocol, ProtocolRef, FileCodecProtocol, KSException
from ..structures import PyHash, DDSPath, GenericLocation, SupportedType as ST
from ..structures_utils import SupportedTypeUtils as STU

_logger = logging.getLogger(__name__)


def displayGraph(f: FunctionType) -> None:
    """
    Displays the graph of computation of a data function in a Databricks cell.

    Example:

    ```py
    @dds.data_function("/my_fun")
    def my_fun(): return 1

    displayGraph(my_fun)
    ```

    Arguments:
        f: any function that is supported by the [[dds.eval]] function.

    Limitations:
    - The browser must support SVG format. Some browser such as Chrome or Edge have limitations around this support.

    Recommended browser: Firefox.
    """
    name = str(id(f))
    from .._api import eval as dds_eval, _fetch_ipython_vars

    dds_eval(
        f,
        args=(),
        kwargs={},
        dds_export_graph=f"/tmp/graph_{name}.svg",
        dds_extra_debug=True,
        dds_stages=["analysis"],
    )
    _fetch_ipython_vars()["dbutils"].fs.cp(
        f"file:///tmp/graph_{name}.svg", f"/FileStore/plots/graph_{name}.svg"
    )
    _fetch_ipython_vars()["displayHTML"](
        f""" <img src="files/plots/graph_{name}.svg"> """
    )


class CommitType(str, Enum):
    """
    The types of commits that can be done with DBFS
    """

    NO_COMMIT = "no_commit"
    LINK_ONLY = "link_only"
    FULL = "full"


def _pprint_exception(e: Exception) -> str:
    return "".join(str(e).split("\n")[:3]).replace("\t", "")


class PySparkDatabricksCodec(CodecProtocol):
    def ref(self):
        return ProtocolRef("dbfs.pyspark")

    def handled_types(self):
        return [ST("pyspark.sql.DataFrame"), ST("pyspark.sql.dataframe.DataFrame")]

    def serialize_into(self, blob: Any, loc: GenericLocation) -> None:
        from pyspark.sql import DataFrame  # type: ignore

        assert isinstance(blob, DataFrame), type(blob)
        blob.write.parquet(loc)
        _logger.debug(f"Committed dataframe to parquet: {loc}")

    def deserialize_from(self, loc: GenericLocation) -> Any:
        import pyspark  # type: ignore

        session = pyspark.sql.SparkSession.getActiveSession()
        _logger.debug(f"Reading parquet from loc {loc} using session {session}")
        df = session.read.parquet(loc)
        _logger.debug(f"Done reading parquet from loc {loc}: {df}")
        return df


class DBFSStore(Store):
    def __init__(
        self, internal_dir: str, data_dir: str, dbutils: Any, commit_type: CommitType
    ):
        self._internal_dir: Path = Path(internal_dir)
        self._data_dir: Path = Path(data_dir)
        _logger.debug(
            f"Created DBFSStore: internal_dir: {self._internal_dir} data_dir: {self._data_dir}"
        )
        self._dbutils = dbutils
        self._commit_type = commit_type
        from .builtins import StringLocalFileCodec, PickleLocalFileCodec, BytesFileCodec
        from .pandas import PandasFileCodec

        slfc = StringLocalFileCodec()
        plfc = BytesFileCodec()
        bfc = PickleLocalFileCodec()

        self._registry = CodecRegistry(
            [PySparkDatabricksCodec()], [slfc, plfc, bfc, PandasFileCodec()],
        )
        # Deprecation hack
        # To ensure that older data already written can still be read, add the following compatibility routines:
        for (old_codec_ref, new_codec) in [
            ("dbfs.pickle", plfc),
            ("dbfs.string", slfc),
            ("dbfs.bytes", bfc),
        ]:
            self._registry._protocols[ProtocolRef(old_codec_ref)] = new_codec

    def fetch_blob(self, key: PyHash) -> Optional[Any]:
        p = self._blob_path(key)
        meta = self._fetch_meta(key)
        if meta is None:
            return None
        ref = ProtocolRef(meta["protocol"])
        codec = self._registry.get_codec(None, ref)
        if isinstance(codec, CodecProtocol):
            return codec.deserialize_from(GenericLocation(str(p)))
        elif isinstance(codec, FileCodecProtocol):
            # File codec protocol:
            # First copy the file locally and then deserialize the local file
            with tempfile.TemporaryDirectory() as td:
                lp = Path(td).joinpath("file")
                lp2 = f"file://{lp}"
                _logger.debug(f"Temporary copy from DBFS: {p} -> {lp2}")
                self._dbutils.fs.cp(str(p), str(lp2))
                return codec.deserialize_from(lp)
        else:
            raise KSException(f"{type(codec)} codec")

    def store_blob(
        self, key: PyHash, blob: Any, codec: Optional[ProtocolRef] = None
    ) -> None:
        protocol = self._registry.get_codec(STU.from_type(type(blob)), codec)
        _logger.debug(
            f"store_blob: {key} {type(blob)} {codec} {STU.from_type(type(blob))} -> protocol: {protocol}"
        )
        p = self._blob_path(key)
        if isinstance(protocol, CodecProtocol):
            protocol.serialize_into(blob, GenericLocation(str(p)))
        elif isinstance(protocol, FileCodecProtocol):
            # First put the blob into a temporary file and then push the blob to DBFS
            with tempfile.TemporaryDirectory() as td:
                lp = Path(td).joinpath("file")
                protocol.serialize_into(blob, lp)
                lp2 = f"file://{lp}"
                _logger.debug(f"Temporary copy from DBFS: {lp2} -> {p}")
                self._dbutils.fs.cp(lp2, str(p))
        else:
            raise KSException(f"{type(protocol)} codec")
        meta_p = self._blob_meta_path(key)
        try:
            meta = json.dumps(
                {"protocol": protocol.ref(), "timestamp_millis": current_timestamp()}
            )
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
        if self._commit_type == CommitType.NO_COMMIT:
            return
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
                if self._commit_type == CommitType.FULL:
                    _logger.debug(f"Copying {blob_path} -> {obj_path}")
                    # Optimization for the files saved with Spark: use spark to read and write.
                    # This can be much faster than using DBFS, which does a temporary copy on a local drive
                    blob_meta = json.loads(self._head(self._blob_meta_path(key)))
                    _logger.debug(f"sync_path: blob_meta: {blob_path}")
                    if blob_meta.get("protocol") == "dbfs.pyspark":
                        _logger.debug(f"Using pyspark to copy {blob_path}")
                        df = self.fetch_blob(key)
                        from pyspark.sql import DataFrame

                        assert isinstance(df, DataFrame), (type(df), key, blob_path)
                        self._dbutils.fs.rm(str(obj_path), recurse=True)
                        df.write.parquet(str(obj_path))
                    else:
                        self._dbutils.fs.cp(str(blob_path), str(obj_path), recurse=True)
                    _logger.debug(f"Done copying {blob_path} -> {obj_path}")
                else:
                    _logger.debug(f"Skip copy for {obj_path} (links-only commit)")
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
                _logger.debug(f"Path {dds_p} is up to date (key {key})")

    def fetch_paths(self, paths: List[DDSPath]) -> "OrderedDict[DDSPath, PyHash]":
        res = OrderedDict()
        # This is a brute force approach that copies all the data and writes extra meta data.
        for dds_p in paths:
            # TODO: this is the same code as sync_path, factorize
            # Look for the redirection file associated to this file
            # The paths are /_dds_meta/path
            redir_p = Path("_dds_meta/").joinpath("./" + dds_p)
            redir_path = self._physical_path(redir_p)
            # Try to read the redirection information:
            _logger.debug(
                f"Attempting to read metadata: {redir_path} {redir_p} {dds_p}"
            )
            meta: Optional[str]
            try:
                meta = self._head(redir_path)
            except Exception as e:
                _logger.debug(f"Could not read metadata: {_pprint_exception(e)}")
                raise e
            redir_key = json.loads(meta)["redirection_key"]
            res[dds_p] = PyHash(redir_key)
        return res

    def codec_registry(self) -> CodecRegistry:
        return self._registry

    def _blob_path(self, key: PyHash) -> Path:
        return self._internal_dir.joinpath("blobs", key)

    def _blob_meta_path(self, key: PyHash) -> Path:
        return self._internal_dir.joinpath("blobs", key + ".meta")

    def _physical_path(self, dds_p: Path) -> Path:
        return self._data_dir.joinpath(dds_p)

    def _fetch_meta(self, key: PyHash) -> Optional[Any]:
        meta_p = self._blob_meta_path(key)
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
