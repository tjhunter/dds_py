"""
The string protocol.
"""

import logging

# TODO: optimize the structure so that there is no need to load it until the
# last moment, if possible.

import pandas.core.frame

from ..structures import CodecProtocol, ProtocolRef, GenericLocation

_logger = logging.getLogger(__name__)


class PandasLocalCodec(CodecProtocol):

    def ref(self): return ProtocolRef("default.pandas_local")

    # TODO: just use strings, it will be faster
    def handled_types(self): return [pandas.DataFrame, pandas.core.frame.DataFrame]

    def serialize_into(self, blob: pandas.DataFrame, loc: GenericLocation):
        assert isinstance(blob, pandas.DataFrame)
        blob.to_parquet(loc)
        _logger.debug(f"Committed dataframe to parquet: {loc}")

    def deserialize_from(self, loc: GenericLocation) -> pandas.DataFrame:
        return pandas.read_parquet(loc)
