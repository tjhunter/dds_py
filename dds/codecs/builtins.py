"""
The string protocol.
"""
import pickle
from typing import Any

from ..structures import (
    CodecProtocol,
    ProtocolRef,
    GenericLocation,
    CodecBackend,
)

Local = CodecBackend("Local")


class StringLocalCodec(CodecProtocol):
    def ref(self):
        return ProtocolRef("default.string")

    def handled_types(self):
        return [str]

    def serialize_into(self, blob: str, loc: GenericLocation) -> None:
        assert isinstance(blob, str)
        with open(loc, "wb") as f:
            f.write(blob.encode(encoding="utf-8"))

    def deserialize_from(self, loc: GenericLocation) -> str:
        with open(loc, "rb") as f:
            return f.read().decode("utf-8")


class PickleLocalCodec(CodecProtocol):
    def ref(self):
        return ProtocolRef("builtins.pickle")

    def handled_types(self):
        return [object, type(None)]

    def serialize_into(self, blob: Any, loc: GenericLocation) -> None:
        with open(loc, "wb") as f:
            pickle.dump(blob, f)

    def deserialize_from(self, loc: GenericLocation) -> Any:
        with open(loc, "rb") as f:
            return pickle.load(f)
