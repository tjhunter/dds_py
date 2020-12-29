"""
The pandas protocol.

It relies on pandas having a parquet driver installed (which may or may not be the case)
"""

import logging
from typing import Any

from ..structures import (
    CodecProtocol,
    ProtocolRef,
    GenericLocation,
    SupportedType as ST,
)

_logger = logging.getLogger(__name__)


class PandasLocalCodec(CodecProtocol):
    def ref(self):
        return ProtocolRef("default.pandas_local")

    def handled_types(self):
        return [ST("pandas.DataFrame"), ST("pandas.core.frame.DataFrame")]

    def serialize_into(self, blob: Any, loc: GenericLocation):
        import pandas

        assert isinstance(blob, pandas.DataFrame)
        blob.to_parquet(loc)
        _logger.debug(f"Committed dataframe to parquet: {loc}")

    def deserialize_from(self, loc: GenericLocation) -> "pandas.DataFrame":
        import pandas

        return pandas.read_parquet(loc)
