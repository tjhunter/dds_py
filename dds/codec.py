import importlib.util
import logging
from typing import Optional, Dict, List, Type, Any

from .codecs.builtins import StringLocalCodec, PickleLocalCodec
from .structures import KSException, CodecProtocol, ProtocolRef

_logger = logging.getLogger(__name__)


class CodecRegistry(object):
    def __init__(self, codecs: List[CodecProtocol]):
        self.codecs = list(codecs)
        self._handled_types: Dict[Type[Any], CodecProtocol] = {}
        self._protocols: Dict[ProtocolRef, CodecProtocol] = {}
        for c in codecs:
            self.add_codec(c)

    def add_codec(self, codec: CodecProtocol) -> None:
        """ added codecs come on top """
        self.codecs.insert(0, codec)
        for t in codec.handled_types():
            self._handled_types[t] = codec
        self._protocols[codec.ref()] = codec

    # TODO: add the location too.
    def get_codec(
        self, obj_type: Optional[Type[Any]], ref: Optional[ProtocolRef]
    ) -> CodecProtocol:
        # First the reference
        if ref:
            if ref not in self._protocols:
                raise KSException(f"Requested protocol {ref}, which is not registered")
            return self._protocols[ref]
        # Then the object type
        if obj_type:
            # Try to use object as backup
            cp = self._handled_types.get(obj_type) or self._handled_types.get(object)
            if cp is None:
                raise KSException(
                    f"Requested protocol for type {obj_type}, which is not registered"
                )
            return cp
        raise KSException(f"No protocol found")


def _build_default_registry() -> CodecRegistry:
    codecs: List[CodecProtocol] = []
    if importlib.util.find_spec("pandas") is not None:
        from .codecs.pandas import PandasLocalCodec

        _logger.info(f"Loading pandas codecs")
        codecs.append(PandasLocalCodec())
    else:
        _logger.debug(f"Cannot load pandas")
    # Put the default codecs at the bottom of the list:
    codecs.append(StringLocalCodec())
    codecs.append(PickleLocalCodec())
    return CodecRegistry(codecs)


_registry: Optional[CodecRegistry] = None


def codec_registry() -> CodecRegistry:
    global _registry
    if _registry is None:
        _registry = _build_default_registry()
    return _registry
