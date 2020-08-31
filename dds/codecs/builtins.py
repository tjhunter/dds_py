"""
The string protocol.
"""

from typing import Any

from ..structures import CodecProtocol, ProtocolRef, GenericLocation


class StringLocalCodec(CodecProtocol):

    def ref(self): return ProtocolRef("default.string")

    def handled_types(self): return [str]

    def serialize_into(self, blob: str, loc: GenericLocation):
        assert isinstance(blob, str)
        with open(loc, "wb") as f:
            f.write(blob.encode(encoding="utf-8"))

    def deserialize_from(self, loc: GenericLocation) -> str:
        with open(loc, "rb") as f:
            return f.read().decode("utf-8")


# TODO: generic python codec based on pickles
