import logging
from typing import TypeVar, Callable, Any, NewType, NamedTuple, Union, OrderedDict, FrozenSet, Optional, List, Type
from types import ModuleType, FunctionType
import inspect
import ast
import sys
from enum import Enum
import hashlib

from .structures import PyHash, Path, FunctionInteractions, KSException

_logger = logging.getLogger(__name__)


def introspect(f: Callable[[Any], Any]) -> FunctionInteractions:
    gctx = GlobalContext(whitelisted_packages=frozenset(["dds"]))
    return _introspect(f, [], None, gctx)


class Functions(str, Enum):
    Load = "load"
    Keep = "keep"
    Cache = "cache"
    Eval = "eval"


CanonicalPath = NewType("CanonicalPath", List[str])


class GlobalContext(NamedTuple):
    # The packages that are authorized for traversal
    whitelisted_packages: FrozenSet[str]


def _introspect(f: Callable[[Any], Any], args: List[Any], context_sig: Optional[PyHash], gctx: GlobalContext) -> FunctionInteractions:
    src = inspect.getsource(f)
    ast_src = ast.parse(src)
    body_lines = src.split("\n")
    ast_f = ast_src.body[0]
    fun_body_sig = _hash(src)
    fun_module = sys.modules[f.__module__]
    _logger.info(f"source: {src}")
    Test().visit(ast_f)
    # _logger.info(f"parsing:")
    fun_input_sig = context_sig if context_sig else _hash(args)
    fis = _inspect_fun(ast_f, gctx, fun_module, body_lines, fun_input_sig)
    # Find the outer interactions
    outputs = [tup for fi in fis for tup in fi.outputs]
    _logger.info(f"outputs: {outputs}")
    fun_return_sig = _hash([fun_body_sig, fun_input_sig])
    return FunctionInteractions(fun_body_sig, fun_return_sig,
                                fun_context_input_sig=context_sig,
                                outputs=outputs)


class Test(ast.NodeVisitor):
    def __init__(self):
        pass

    # def visit(self, node: ast.AST) -> Any:
    #     _logger.debug(f"node: {node}, {dir(node)}")
    #     self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        # _logger.debug(f"function: {node.name} {node.args} {node.body}")
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> Any:
        # _logger.debug(f"expr: {node}, {node.value}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        # _logger.debug(f"call: {node}, {_function_name(node.func)} {node.args} {node.keywords}")
        self.generic_visit(node)


def _function_name(node) -> List[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return _function_name(node.value) + [node.attr]
    assert False, (node, type(node))


def _inspect_fun(node: ast.FunctionDef,
                 gctx: GlobalContext,
                 mod: ModuleType,
                 function_body_lines: List[str],
                 function_args_hash: PyHash) -> List[FunctionInteractions]:
    _logger.debug(f"function: {node.name} {node.args} {node.body}")
    l: List[FunctionInteractions] = []
    for n in node.body:
        if isinstance(n, ast.Expr):
            function_body_hash = _hash(function_body_lines[:n.end_lineno])
            fi = _inspect_line(n, gctx, mod, function_body_hash, function_args_hash)
            if fi:
                l.append(fi)
    return l


def _inspect_line(ln,
                  gctx: GlobalContext,
                  mod: ModuleType,
                  function_body_hash: PyHash,
                  function_args_hash: PyHash) -> Optional[FunctionInteractions]:
    if isinstance(ln, ast.Expr):
        return _inspect_call(ln.value, gctx, mod, function_body_hash, function_args_hash)
    else:
        _logger.debug(f"Unknown expression type: {ln}")


def _inspect_call(node: ast.Call,
                  gctx: GlobalContext,
                  mod: ModuleType,
                  function_body_hash: PyHash,
                  function_args_hash: PyHash) -> Optional[FunctionInteractions]:
    fname = _function_name(node.func)
    obj_called = _retrieve_object(fname, mod, gctx, None)
    _logger.debug(f"{node.lineno} {node} {obj_called}")
    canon_path = _canonical_path(fname, mod)
    _logger.debug(f"canon_path: {canon_path}")

    if not _is_primary_function(canon_path):
        return None
    # Parse the arguments to find the function that is called
    if not node.args:
        raise KSException("Wrong number of args")
    if node.keywords:
        raise NotImplementedError(node)
    store_path: Optional[Path] = None
    f_called: str = ""
    if fname[-1] == Functions.Keep:
        store_path_symbol = node.args[0].id
        store_path = _retrieve_path(store_path_symbol, mod)
        f_called = node.args[1].id
    if fname[-1] == Functions.Cache:
        f_called = node.args[0].id
    if not f_called:
        raise NotImplementedError((fname[-1], node))

    # We just use the context in the function
    fun_called = _retrieve_object([f_called], mod, gctx, FunctionType)
    can_path = _canonical_path([f_called], mod)
    # _logger.debug(f"Introspecting function")
    context_sig = _hash([function_body_hash, function_args_hash, node.end_lineno])
    inner_intro = _introspect(fun_called, args=[], context_sig=context_sig, gctx=gctx)
    # _logger.debug(f"Introspecting function finished: {inner_intro}")
    _logger.debug(f"keep: {store_path} <- {can_path}: {inner_intro.fun_return_sig}")
    if store_path:
        inner_intro.outputs.append((store_path, inner_intro.fun_return_sig))
    return inner_intro


def _retrieve_object(path: List[str], mod: ModuleType, gctx: GlobalContext, expected_type: Optional[Type]) -> Union[None, FunctionType]:
    assert path
    fname = path[0]
    if fname not in mod.__dict__:
        raise KSException(f"Object {fname} not found in module {mod}. Choices are {mod.__dict__}")
    obj = mod.__dict__[fname]
    if len(path) >= 2:
        if not isinstance(obj, ModuleType):
            raise KSException(f"Object {fname} is not a module but of type {type(obj)}")
        if obj.__name__ not in gctx.whitelisted_packages:
            _logger.debug(f"Located module {obj.__name__}, skipped")
            return None
        return _retrieve_object(path[1:], obj, gctx, expected_type)
    if expected_type and not isinstance(obj, expected_type):
        _logger.debug(f"Object {fname} of type {type(obj)}, expected {expected_type}")
        # TODO: raise exception
        return None
    return obj


def _canonical_path(path: List[str], mod: ModuleType) -> CanonicalPath:
    """
    The canonical path of an entity in the python module hierarchy.
    """
    if not path:
        # Return the path of the module
        return CanonicalPath(mod.__name__.split("."))
    assert path
    fname = path[0]
    if fname not in mod.__dict__:
        raise KSException(f"Object {fname} not found in module {mod}. Choices are {mod.__dict__}")
    obj = mod.__dict__[fname]
    if isinstance(obj, ModuleType):
        # Look into this module to find the object:
        return _canonical_path(path[1:], obj)
    return CanonicalPath(_canonical_path([], mod) + [fname])


def _retrieve_path(fname: str, mod: ModuleType) -> Path:
    if fname not in mod.__dict__:
        raise KSException(f"Expected path {fname} not found in module {mod}. Choices are {mod.__dict__}")
    obj = mod.__dict__[fname]
    if not isinstance(obj, str):
        raise KSException(f"Object {fname} is not a function but of type {type(obj)}")
    return obj


def _is_primary_function(path: CanonicalPath) -> bool:
    # TODO use the proper definition of the module
    if len(path) == 2 and path[0] == "dds":
        if path[1] == Functions.Eval:
            raise KSException("invalid call to eval")
        return path[1] in (Functions.Keep, Functions.Load, Functions.Cache)
    return False


def _hash(x: Any) -> PyHash:
    def algo_str(s: str) -> PyHash:
        return PyHash(hashlib.sha256(s.encode()).hexdigest())
    if isinstance(x, str):
        return algo_str(x)
    if isinstance(x, int):
        return algo_str(str(x))
    if isinstance(x, list):
        return algo_str("|".join([_hash(y) for y in x]))
    raise NotImplementedError(str(type(x)))