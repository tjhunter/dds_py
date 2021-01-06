import importlib.util
import logging
from typing import Optional, Dict, List, Union

from .structures import (
    KSException,
    CodecProtocol,
    ProtocolRef,
    SupportedType,
    FileCodecProtocol,
)
from .structures_utils import SupportedTypeUtils

_logger = logging.getLogger(__name__)


class CodecRegistry(object):
    """
    Invariants:
    - the codecs have precedence over the file codecs (they are more specialized)
    """

    def __init__(
        self, codecs: List[CodecProtocol], file_codecs: List[FileCodecProtocol]
    ):
        self.codecs = list(codecs)
        self.file_codecs = list(file_codecs)
        self._handled_types: Dict[
            SupportedType, Union[CodecProtocol, FileCodecProtocol]
        ] = {}
        self._protocols: Dict[ProtocolRef, Union[CodecProtocol, FileCodecProtocol]] = {}
        for c in list(codecs):
            self.add_codec(c)
        for fc in list(self.file_codecs):
            self.add_file_codec(fc)

    def add_codec(self, codec: CodecProtocol) -> None:
        """ added codecs come on top """
        self.codecs.insert(0, codec)
        for t in codec.handled_types():
            self._handled_types[t] = codec
        self._protocols[codec.ref()] = codec

    def add_file_codec(self, codec: FileCodecProtocol) -> None:
        """ added codecs come on top """
        self.file_codecs.insert(0, codec)
        for t in codec.handled_types():
            if t not in self._handled_types:
                self._handled_types[t] = codec
        if codec.ref() in self._protocols:
            _logger.warning(f"{codec.ref()} already in protocols, skipping {codec}")
        else:
            self._protocols[codec.ref()] = codec

    # TODO: add the location too.
    def get_codec(
        self, obj_type: Union[SupportedType, None], ref: Optional[ProtocolRef]
    ) -> Union[CodecProtocol, FileCodecProtocol]:
        # First the reference
        if ref:
            if ref not in self._protocols:
                raise KSException(f"Requested protocol {ref}, which is not registered")
            return self._protocols[ref]
        # Then the object type
        if obj_type is not None:
            # Try to use object as backup
            pref: SupportedType = obj_type
            cp = self._handled_types.get(pref) or self._handled_types.get(
                SupportedTypeUtils.from_type(object)
            )
            if cp is None:
                raise KSException(
                    f"Requested protocol for type {obj_type}, which is not registered"
                )
            return cp
        raise KSException(f"No protocol found")


def _build_default_registry() -> CodecRegistry:
    codecs: List[CodecProtocol] = []
    if importlib.util.find_spec("pandas") is not None:
        # TODO: this could be done without hardcoding pandas, simply by loading at the
        # name of the head module in the list of supported types.
        from .codecs.pandas import PandasLocalCodec

        _logger.info(f"Loading pandas codecs")
        codecs.append(PandasLocalCodec())
    else:
        _logger.debug(f"Cannot load pandas")
    # To prevent a circular import.
    from .codecs.builtins import (
        StringLocalFileCodec,
        PickleLocalFileCodec,
        BytesFileCodec,
    )

    return CodecRegistry(
        codecs, [StringLocalFileCodec(), BytesFileCodec(), PickleLocalFileCodec()]
    )


_registry: Optional[CodecRegistry] = None


def codec_registry() -> CodecRegistry:
    global _registry
    if _registry is None:
        _registry = _build_default_registry()
    return _registry
