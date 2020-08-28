import logging
from typing import Any, Optional, Dict, NamedTuple, NewType, List, Type

from .structures import KSException

_logger = logging.getLogger(__name__)

# The name of a codec protocol.
ProtocolRef = NewType("ProtocolRef", str)

# A URI like wrapper to put stuff somewhere.
# The exact content and schema is determined by the store.
GenericLocation = NewType("GenericLocation", str)


class CodecProtocol(object):

    def ref(self) -> ProtocolRef:
        pass

    def handled_types(self) -> List[Type]:
        """ The list of types that this codec can handle """
        pass

    def serialize_into(self, blob: Any, loc: GenericLocation):
        """
        Simple in-memory serialization
        """
        pass

    def deserialize_from(self, loc: GenericLocation) -> Any:
        """ Simple in-memory deserialization """
        pass


class StringLocalCodec(CodecProtocol):

    def ref(self): return ProtocolRef("default.string")

    def handled_types(self): return [str]

    def serialize_into(self, blob: Any, loc: GenericLocation):
        assert isinstance(blob, str)
        with open(loc, "wb") as f:
            f.write(blob.encode(encoding="utf-8"))

    def deserialize_from(self, loc: GenericLocation) -> str:
        with open(loc, "rb") as f:
            return f.read().decode("utf-8")

# TODO: generic python codec based on pickles

class BlobMetaData(NamedTuple):
    protocol: ProtocolRef
    # TODO: creation date?


class CodecRegistry(object):

    def __init__(self, codecs: List[CodecProtocol]):
        self.codecs = list(codecs)
        self._handled_types: Dict[Type, CodecProtocol] = {}
        self._protocols: Dict[ProtocolRef, CodecProtocol] = {}
        for c in codecs:
            self.add_codec(c)

    def add_codec(self, codec: CodecProtocol):
        """ added codecs come on top """
        self.codecs.insert(0, codec)
        for t in codec.handled_types():
            self._handled_types[t] = codec
        self._protocols[codec.ref()] = codec

    # TODO: add the location too.
    def get_codec(self, obj_type: Optional[Type], ref: Optional[ProtocolRef]) -> CodecProtocol:
        if not obj_type and not ref:
            raise KSException("Missing both arguments")
        if ref:
            if ref not in self._protocols:
                raise KSException(f"Requested protocol {ref}, which is not registered")
            codec = self._protocols[ref]
        else:
            if obj_type not in self._handled_types:
                raise KSException(f"Requested protocol for type {obj_type}, which is not registered")
            codec = self._handled_types[obj_type]
        return codec


codec_registry: CodecRegistry = CodecRegistry([
    StringLocalCodec()
])

