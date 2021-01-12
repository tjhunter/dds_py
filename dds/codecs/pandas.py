"""
The pandas protocol.

It relies on pandas having a parquet driver installed (which may or may not be the case)
"""

import logging
from pathlib import PurePath
from typing import Any

from ..structures import (
    ProtocolRef,
    SupportedType as ST,
    FileCodecProtocol,
)

_logger = logging.getLogger(__name__)


class PandasFileCodec(FileCodecProtocol):
    def ref(self):
        return ProtocolRef("local.pandas")

    def handled_types(self):
        return [ST("pandas.DataFrame"), ST("pandas.core.frame.DataFrame")]

    def serialize_into(self, blob: Any, loc: PurePath):
        import pandas

        assert isinstance(blob, pandas.DataFrame)
        blob.to_parquet(str(loc))
        _logger.debug(f"Committed dataframe to parquet: {loc}")

    def deserialize_from(self, loc: PurePath) -> "pandas.DataFrame":
        import pandas

        return pandas.read_parquet(str(loc))
