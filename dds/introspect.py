import ast
import hashlib
import inspect
import logging
import abc
import sys
from enum import Enum
from types import ModuleType, FunctionType
import builtins
from functools import total_ordering
from typing import Tuple, Callable, Any, Dict, Set, Union, FrozenSet, Optional, \
    List, Type, NewType

from .structures import PyHash, DDSPath, FunctionInteractions, KSException

_logger = logging.getLogger(__name__)

Package = NewType("Package", str)


def introspect(f: Callable[[Any], Any]) -> FunctionInteractions:
    # TODO: exposed the whitelist
    gctx = GlobalContext(
        f.__module__,  # typing: ignore
        whitelisted_packages=_whitelisted_packages)
    return _introspect(f, [], None, gctx)


class Functions(str, Enum):
    Load = "load"
    Keep = "keep"
    Cache = "cache"
    Eval = "eval"


@total_ordering
class CanonicalPath(object):

    def __init__(self, p: List[str]):
        self._path = p

    def __hash__(self):
        return hash(tuple(self._path))

    def append(self, s: str) -> "CanonicalPath":
        return CanonicalPath(self._path + [s])

    def head(self) -> str:
        return self._path[0]

    def tail(self) -> "CanonicalPath":
        return CanonicalPath(self._path[1:])

    def get(self, i: int) -> str:
        return self._path[i]

    def __len__(self):
        return len(self._path)

    def __repr__(self):
        x = ".".join(self._path)
        return f"<{x}>"

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __ne__(self, other):
        return not (repr(self) == repr(other))

    def __lt__(self, other):
        return repr(self) < repr(other)


class GlobalContext(object):
    # The packages that are authorized for traversal

    def __init__(self,
                 start_module: ModuleType,
                 whitelisted_packages: Set[Package]):
        self.whitelisted_packages = whitelisted_packages
        self.start_module = start_module
        # Hashes of all the static objects
        self._hashes: Dict[CanonicalPath, PyHash] = {}

    def get_hash(self, path: CanonicalPath) -> PyHash:
        if path not in self._hashes:
            assert path, path
            if path.head() in sys.modules:
                start_mod = sys.modules[path.head()]
            else:
                assert path.head() == "__main__", path
                start_mod = self.start_module
            _logger.debug(f"get_hash: path={path} start_mod={start_mod}")
            obj = _retrieve_object(path.tail()._path, start_mod, self, None)
            hash = _hash(obj)
            _logger.debug(f"Cache hash: {path}: {hash}")
            self._hashes[path] = hash
        return self._hashes[path]

    def is_authorized_path(self, cp: CanonicalPath) -> bool:
        _logger.debug(f"is_authorized_path: {self.whitelisted_packages}")
        for idx in range(len(self.whitelisted_packages)):
            if ".".join(cp._path[:idx]) in self.whitelisted_packages:
                return True
        return False



def _introspect(f: Callable[[Any], Any], args: List[Any], context_sig: Optional[PyHash],
                gctx: GlobalContext) -> FunctionInteractions:
    _logger.debug(f"Starting _introspect: {f}")
    src = inspect.getsource(f)
    ast_src = ast.parse(src)
    body_lines = src.split("\n")
    ast_f = ast_src.body[0]
    fun_body_sig = _hash(src)
    fun_module = sys.modules[f.__module__]
    # _logger.info(f"source: {src}")
    # Test().visit(ast_f)
    fun_input_sig = context_sig if context_sig else _hash(args)
    (all_args_hash, fis) = _inspect_fun(ast_f, gctx, fun_module, body_lines, fun_input_sig)
    # Find the outer interactions
    outputs = [tup for fi in fis for tup in fi.outputs]
    _logger.info(f"outputs: {outputs}")
    fun_return_sig = _hash([fun_body_sig, all_args_hash])
    res = FunctionInteractions(fun_body_sig, fun_return_sig,
                                fun_context_input_sig=context_sig,
                                outputs=outputs)
    _logger.debug(f"End _introspect: {f}: {res}")
    return res


# class Test(ast.NodeVisitor):
#     def __init__(self):
#         pass
#
#     def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
#         # _logger.debug(f"function: {node.name} {node.args} {node.body}")
#         self.generic_visit(node)
#
#     def visit_Expr(self, node: ast.Expr) -> Any:
#         # _logger.debug(f"expr: {node}, {node.value}")
#         self.generic_visit(node)
#
#     def visit_Call(self, node: ast.Call) -> Any:
#         # _logger.debug(f"call: {node}, {_function_name(node.func)} {node.args} {node.keywords}")
#         self.generic_visit(node)


class IntroVisitor(ast.NodeVisitor):

    def __init__(self, start_mod: ModuleType,
                 gctx: GlobalContext, function_body_lines: List[str],
                 function_args_hash: PyHash):
        # TODO: start_mod is in the global context
        self._start_mod = start_mod
        self._gctx = gctx
        self._body_lines = function_body_lines
        self._args_hash = function_args_hash
        self.inters: List[FunctionInteractions] = []

    def visit_Call(self, node: ast.Call) -> Any:
        # _logger.debug(f"visit: {node} {dir(node)}")
        function_body_hash = _hash(self._body_lines[:node.lineno + 1])
        fi = _inspect_call(node, self._gctx, self._start_mod, function_body_hash, self._args_hash)
        if fi:
            self.inters.append(fi)
        self.generic_visit(node)


class ExternalVarsVisitor(ast.NodeVisitor):
    """
    Finds all the external variables of a function that should be hashed into the argument list.
    TODO: currently very crude, it does not look for assigned variables.
    """

    def __init__(self, start_mod: ModuleType, gctx: GlobalContext):
        self._start_mod = start_mod
        self._gctx = gctx
        self.vars: Set[CanonicalPath] = set()

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id not in self._start_mod.__dict__ or node.id in self.vars:
            return
        obj = self._start_mod.__dict__[node.id]
        if isinstance(obj, ModuleType) or isinstance(obj, Callable):
            # Modules and callables are tracked separately
            _logger.debug(f"visit name {node.id}: skipping (fun/mod)")
            return
        cpath = _canonical_path([node.id], self._start_mod)
        if self._gctx.is_authorized_path(cpath):
            _logger.debug(f"visit name {node.id}: authorized")
            self.vars.add(cpath)
        else:
            _logger.debug(f"Skipping unauthorized name: {node.id}: {cpath}")


def _function_name(node) -> List[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return _function_name(node.value) + [node.attr]
    assert False, (node, type(node))


def _mod_path(m: ModuleType) -> CanonicalPath:
    return CanonicalPath(m.__name__.split("."))


def _fun_path(f: Callable) -> CanonicalPath:
    mod = inspect.getmodule(f)
    return CanonicalPath(_mod_path(mod)._path + [f.__name__])


def _inspect_fun(node: ast.FunctionDef,
                 gctx: GlobalContext,
                 mod: ModuleType,
                 function_body_lines: List[str],
                 function_args_hash: PyHash) -> Tuple[PyHash, List[FunctionInteractions]]:
    _logger.debug(f"function: {node.name} {node.args} {node.body}")
    # Find all the outer dependencies to hash them
    vdeps = ExternalVarsVisitor(mod, gctx)
    vdeps.visit(node)
    ext_vars = sorted(list(vdeps.vars))
    _logger.debug(f"external: {ext_vars}")
    fun_args = []
    for ev in ext_vars:
        fun_args.append(ev)
        fun_args.append(gctx.get_hash(ev))
        # TODO: compute the hash of all the arguments, here, instead of in pieces around
    function_all_args_hash = _hash([function_args_hash, fun_args])
    # assert False, "add visitors"
    visitor = IntroVisitor(mod, gctx, function_body_lines, function_all_args_hash)
    visitor.visit(node)
    return (function_all_args_hash, visitor.inters)


def _inspect_call(node: ast.Call,
                  gctx: GlobalContext,
                  mod: ModuleType,
                  function_body_hash: PyHash,
                  function_args_hash: PyHash) -> Optional[FunctionInteractions]:
    fname = _function_name(node.func)
    _logger.debug(f"_inspect_call: fname: {fname}")
    # Do not try to parse the builtins
    if len(fname) == 1 and fname[0] in builtins.__dict__:
        _logger.debug(f"_inspect_call: skipping builtin")
        return None
    obj_called = _retrieve_object(fname, mod, gctx, None)
    if obj_called is None:
        # This is not an object to consider
        _logger.debug(f"_inspect_call: skipping")
        return None
    _logger.debug(f"_inspect_call: ln:{node.lineno} {node} {obj_called}")
    canon_path = _canonical_path(fname, mod)
    _logger.debug(f"_inspect_call: canon_path: {canon_path}")

    # Inspect in all cases since it can call primary functions deeper.
    # if not _is_primary_function(canon_path):
    #     _logger.debug(f"_inspect_call: {canon_path}: primary function")
    #     return None
    # # Parse the arguments to find the function that is called
    # if not node.args:
    #     raise KSException("Wrong number of args")
    if node.keywords:
        raise NotImplementedError(node)
    store_path: Optional[DDSPath] = None
    f_called: str = ""
    if fname[-1] == Functions.Keep:
        if len(node.args) < 2:
            raise KSException(f"Wrong number of args: expected 2+, got {node.args}")
        store_path_symbol = node.args[0].id
        _logger.debug(f"_inspect_call: Keep: store_path_symbol: {store_path_symbol} {type(store_path_symbol)}")
        store_path = _retrieve_path(store_path_symbol, mod)
        f_called = node.args[1].id
    elif fname[-1] == Functions.Cache:
        if len(node.args) < 2:
            raise KSException(f"Wrong number of args: expected 2+, got {node.args}")
        f_called = node.args[0].id
    else:
        f_called = fname[-1]
    # if not f_called:
    #     raise NotImplementedError((fname[-1], node))

    # We just use the context in the function
    fun_called = _retrieve_object([f_called], mod, gctx, FunctionType)
    _logger.debug(f"_inspect_call: fun_called: {fun_called}")
    can_path = _canonical_path([f_called], mod)
    # _logger.debug(f"Introspecting function")
    context_sig = _hash([function_body_hash, function_args_hash, node.lineno])
    inner_intro = _introspect(fun_called, args=[], context_sig=context_sig, gctx=gctx)
    # _logger.debug(f"Introspecting function finished: {inner_intro}")
    _logger.debug(f"_inspect_call: keep: {store_path} <- {can_path}: {inner_intro.fun_return_sig}")
    if store_path:
        inner_intro.outputs.append((store_path, inner_intro.fun_return_sig))
    return inner_intro


def _retrieve_object(
        path: List[str],
        mod: ModuleType,
        gctx: GlobalContext,
        expected_type: Optional[Type]) -> Union[None, FunctionType]:
    assert path
    fname = path[0]
    _logger.debug(f"_retrieve_object: {path} {mod}")
    if fname not in mod.__dict__:
        # If the name is not in scope, it is assumed to be defined in the function body -> skipped
        # (it is included with the code lines)
        # TODO: confirm this choice and use locals() against that case
        return None
        # raise KSException(f"Object {fname} not found in module {mod}. Choices are {mod.__dict__}")
    obj = mod.__dict__[fname]
    if len(path) >= 2:
        if not isinstance(obj, ModuleType):
            raise KSException(f"Object {fname} is not a module but of type {type(obj)}")
        return _retrieve_object(path[1:], obj, gctx, expected_type)
    # Check the real module of the object, if available (such as for functions)
    obj_mod = inspect.getmodule(obj)
    if obj_mod is not None:
        obj_mod_path = _mod_path(obj_mod)
        if not gctx.is_authorized_path(obj_mod_path):
            _logger.debug(f"Actual module {obj_mod_path} for obj {obj} is unauthorized")
            return None
        else:
            _logger.debug(f"Actual module {obj_mod_path} for obj {obj}: authorized")
    if expected_type and not isinstance(obj, expected_type):
        _logger.debug(f"Object {fname} of type {type(obj)}, expected {expected_type}")
        # TODO: raise exception
        return None
    # Drop if this object is not to be considered:
    if isinstance(obj, (FunctionType, abc.ABCMeta)):
        fun_mod = inspect.getmodule(obj)
        p = _mod_path(fun_mod)
        _logger.debug(f"_retrieve_object: {path} -> {obj}: {p}")
        if not gctx.is_authorized_path(p):
            _logger.debug(f"_retrieve_object: dropping unauthorized function {path} -> {obj}: {fun_mod.__name__}")
            return None
        else:
            _logger.debug(f"_retrieve_object: authorized function {path} -> {obj}: {fun_mod.__name__}")
    else:
        _logger.debug(f"_retrieve_object: not checking: {obj} {type(obj)}")
    return obj


def _canonical_path(path: List[str], mod: ModuleType) -> CanonicalPath:
    """
    The canonical path of an entity in the python module hierarchy.
    """
    if not path:
        # Return the path of the module
        return _mod_path(mod)
    assert path
    fname = path[0]
    if fname not in mod.__dict__:
        raise KSException(f"Object {fname} not found in module {mod}. Choices are {mod.__dict__}")
    obj = mod.__dict__[fname]
    if isinstance(obj, ModuleType):
        # Look into this module to find the object:
        return _canonical_path(path[1:], obj)
    assert len(path) == 1, path
    ref_module = inspect.getmodule(obj)
    if ref_module == mod or ref_module is None:
        return _canonical_path([], mod).append(fname)
    else:
        _logger.debug(f"Redirection: {path} {mod} {ref_module}")
        return _canonical_path(path, ref_module)


def _retrieve_path(fname: str, mod: ModuleType) -> DDSPath:
    if fname not in mod.__dict__:
        raise KSException(f"Expected path {fname} not found in module {mod}. Choices are {mod.__dict__}")
    obj = mod.__dict__[fname]
    if not isinstance(obj, str):
        raise KSException(f"Object {fname} is not a function but of type {type(obj)}")
    return DDSPath(obj)


def _is_primary_function(path: CanonicalPath) -> bool:
    # TODO use the proper definition of the module
    if len(path) == 2 and path.head() == "dds":
        if path.get(1) == Functions.Eval:
            raise KSException("invalid call to eval")
        return path.get(1) in (Functions.Keep, Functions.Load, Functions.Cache)
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
    if isinstance(x, CanonicalPath):
        return algo_str(repr(x))
    raise NotImplementedError(str(type(x)))


_whitelisted_packages: Set[Package] = {Package("dds"), Package("__main__")}


def whitelist_module(module: Union[str, ModuleType]):
    global _whitelisted_packages
    if isinstance(module, ModuleType):
        module = module.__name__
    assert isinstance(module, str), (module, type(module))
    _whitelisted_packages.add(Package(module))
