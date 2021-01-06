"""
The string protocol.
"""
import pickle
from pathlib import PurePath
from typing import Any, List

from ..structures import (
    FileCodecProtocol,
    ProtocolRef,
    CodecBackend,
    SupportedType,
)
from ..structures_utils import SupportedTypeUtils as STU

Local = CodecBackend("Local")


class StringLocalFileCodec(FileCodecProtocol):
    def ref(self):
        return ProtocolRef("local.string")

    def handled_types(self) -> List[SupportedType]:
        return [STU.from_type(str)]

    def serialize_into(self, blob: str, loc: PurePath) -> None:
        assert isinstance(blob, str), type(blob)
        with open(str(loc), "wb") as f:
            f.write(blob.encode(encoding="utf-8"))

    def deserialize_from(self, loc: PurePath) -> str:
        with open(str(loc), "rb") as f:
            return f.read().decode("utf-8")


class BytesFileCodec(FileCodecProtocol):
    def ref(self):
        return ProtocolRef("local.bytes")

    def handled_types(self) -> List[SupportedType]:
        return [STU.from_type(bytes), STU.from_type(bytearray)]

    def serialize_into(self, blob: Any, loc: PurePath) -> None:
        assert isinstance(blob, (bytes, bytearray)), type(blob)
        with open(str(loc), "wb") as f:
            f.write(blob)

    def deserialize_from(self, loc: PurePath) -> bytes:
        with open(str(loc), "rb") as f:
            return f.read()


class PickleLocalFileCodec(FileCodecProtocol):
    def ref(self):
        return ProtocolRef("local.pickle")

    def handled_types(self) -> List[SupportedType]:
        return [STU.from_type(type(None)), SupportedType("object")]

    def serialize_into(self, blob: Any, loc: PurePath) -> None:
        with open(str(loc), "wb") as f:
            pickle.dump(blob, f)

    def deserialize_from(self, loc: PurePath) -> Any:
        with open(str(loc), "rb") as f:
            return pickle.load(f)
