import importlib.util
import logging
from typing import Optional, Dict, List, Type

from .codecs.builtins import StringLocalCodec
from .structures import KSException, CodecProtocol, ProtocolRef

_logger = logging.getLogger(__name__)


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


def _build_default_registry() -> CodecRegistry:
    codecs = [StringLocalCodec()]
    if importlib.util.find_spec("pandas") is not None:
        from .codecs.pandas import PandasLocalCodec
        _logger.info(f"Loading pandas codecs")
        codecs.append(PandasLocalCodec())
    else:
        _logger.debug(f"Cannot load pandas")
    return CodecRegistry(codecs)


_registry: Optional[CodecRegistry] = None


def codec_registry() -> CodecRegistry:
    global _registry
    if _registry is None:
        _registry = _build_default_registry()
    return _registry
