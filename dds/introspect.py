import ast
import hashlib
import importlib
import inspect
import logging
import abc
import sys
from collections import OrderedDict
from enum import Enum
from types import ModuleType, FunctionType
import builtins
from functools import total_ordering
from typing import Tuple, Callable, Any, Dict, Set, Union, FrozenSet, Optional, \
    List, Type, NewType, NamedTuple, OrderedDict as OrderedDictType

from .structures import PyHash, DDSPath, FunctionInteractions, KSException

_logger = logging.getLogger(__name__)

Package = NewType("Package", str)


def introspect(f: Callable[[Any], Any], start_globals: Dict[str, Any]) -> FunctionInteractions:
    # TODO: exposed the whitelist
    # TODO: add the arguments of the function
    gctx = GlobalContext(
        f.__module__,  # typing: ignore
        whitelisted_packages=_whitelisted_packages,
        start_globals=start_globals)
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
                 whitelisted_packages: Set[Package],
                 start_globals: Dict[str, Any]):
        self.whitelisted_packages = whitelisted_packages
        self.start_module = start_module
        self.start_globals = start_globals
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


class FunctionArgContext(NamedTuple):
    # The keys of the arguments that are known at call time
    named_args: OrderedDictType[str, Optional[PyHash]]
    # The key of the environment when calling the function
    inner_call_key: Optional[PyHash]


def _introspect(f: Callable[[Any], Any], args: List[Any], context_sig: Optional[PyHash],
                gctx: GlobalContext) -> FunctionInteractions:
    # TODO: remove args for now?
    arg_sig = inspect.signature(f)
    _logger.debug(f"Starting _introspect: {f}: arg_sig={arg_sig}")
    assert not args, ("Not implemented", args)
    fun_arg_context = FunctionArgContext(
        # TODO: bind the arguments to their slots
        named_args=OrderedDict(list((n, None) for n in arg_sig.parameters.keys())),
        inner_call_key=context_sig)
    src = inspect.getsource(f)
    ast_src = ast.parse(src)
    body_lines = src.split("\n")
    ast_f = ast_src.body[0]
    fun_body_sig = _hash(src)
    fun_module = inspect.getmodule(f)

    fun_input_sig = context_sig if context_sig else _hash(args)
    (all_args_hash, fis) = _inspect_fun(ast_f, gctx, fun_module, body_lines, fun_input_sig, fun_arg_context)
    # Find the outer interactions
    outputs = [tup for fi in fis for tup in fi.outputs]
    # _logger.info(f"_introspect: outputs: {outputs}")
    fun_return_sig = _hash([fun_body_sig, all_args_hash])
    res = FunctionInteractions(fun_body_sig, fun_return_sig,
                                fun_context_input_sig=context_sig,
                                outputs=outputs)
    _logger.debug(f"End _introspect: {f}: {res}")
    return res


class IntroVisitor(ast.NodeVisitor):

    def __init__(self,
                 start_mod: ModuleType,
                 gctx: GlobalContext,
                 function_body_lines: List[str],
                 function_args_hash: PyHash,
                 function_var_names: List[str]):
        # TODO: start_mod is in the global context
        self._start_mod = start_mod
        self._gctx = gctx
        self._function_var_names = set(function_var_names)
        self._body_lines = function_body_lines
        self._args_hash = function_args_hash
        self.inters: List[FunctionInteractions] = []

    def visit_Call(self, node: ast.Call) -> Any:
        # _logger.debug(f"visit: {node} {dir(node)}")
        function_body_hash = _hash(self._body_lines[:node.lineno + 1])
        fi = _inspect_call(node, self._gctx, self._start_mod, function_body_hash, self._args_hash, self._function_var_names)
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
        # If this an object that
        if not _is_authorized_type(type(obj), self._gctx):
            _logger.debug(f"Type {type(obj)} of object {node.id} not authorized")
            return
        cpath = _canonical_path([node.id], self._start_mod, self._gctx)
        if self._gctx.is_authorized_path(cpath):
            _logger.debug(f"visit name {node.id}: authorized")
            self.vars.add(cpath)
        else:
            _logger.debug(f"Skipping unauthorized name: {node.id}: {cpath}")
            pass


class LocalVarsVisitor(ast.NodeVisitor):
    """
    A brute-force attempt to find all the variables defined in the scope of a module.
    """

    def __init__(self, existing_vars: List[str]):
        self.vars: Set[str] = set(existing_vars)

    def visit_Name(self, node: ast.Name) -> Any:
        # _logger.debug(f"visit_vars: {node.id} {node.ctx}")
        if isinstance(node.ctx, ast.Store):
            self.vars.add(node.id)
        self.generic_visit(node)


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


def _is_authorized_type(tpe: Type, gctx: GlobalContext) -> bool:
    """
    True if the type is defined within the whitelisted hierarchy
    """
    if tpe is None:
        return True
    if tpe in (int, float, str, bytes):
        return True
    if issubclass(tpe, object):
        mod = inspect.getmodule(tpe)
        if mod is None:
            _logger.debug(f"_is_authorized_type: type {tpe} has no module")
            return False
        mod_path = _mod_path(mod)
        if gctx.is_authorized_path(mod_path):
            msg = f"Type {tpe} ({mod_path}) is authorized: not implemented"
            _logger.warning(msg)
            raise NotImplementedError(msg)
        return False
    else:
        msg = f"Type {tpe} is not implemented"
        _logger.warning(msg)
        raise NotImplementedError(msg)


def _inspect_fun(node: ast.FunctionDef,
                 gctx: GlobalContext,
                 mod: ModuleType,
                 function_body_lines: List[str],
                 function_args_hash: PyHash,
                 arg_ctx: FunctionArgContext) -> Tuple[PyHash, List[FunctionInteractions]]:
    # _logger.debug(f"function: {node.name} {node.args} {node.body}")
    # Find all the outer dependencies to hash them
    vdeps = ExternalVarsVisitor(mod, gctx)
    vdeps.visit(node)
    ext_vars = sorted(list(vdeps.vars))
    # _logger.debug(f"external: {ext_vars}")
    lvars_v = LocalVarsVisitor(list(arg_ctx.named_args.keys()))
    lvars_v.visit(node)
    lvars = sorted(list(lvars_v.vars))
    _logger.debug(f"local vars: {lvars}")
    fun_args = []
    for ev in ext_vars:
        fun_args.append(ev)
        fun_args.append(gctx.get_hash(ev))
        # TODO: compute the hash of all the arguments, here, instead of in pieces around
    function_all_args_hash = _hash([function_args_hash, fun_args])
    visitor = IntroVisitor(mod, gctx, function_body_lines, function_all_args_hash, lvars)
    visitor.visit(node)
    return function_all_args_hash, visitor.inters


def _inspect_call(node: ast.Call,
                  gctx: GlobalContext,
                  mod: ModuleType,
                  function_body_hash: PyHash,
                  function_args_hash: PyHash,
                  var_names: Set[str]) -> Optional[FunctionInteractions]:
    fname = _function_name(node.func)
    _logger.debug(f"fname: {fname}")
    # Do not try to parse the builtins
    if len(fname) == 1 and fname[0] in builtins.__dict__:
        _logger.debug(f"skipping builtin")
        return None
    if fname[0] in var_names:
        _logger.debug(f"skipping local var")
        return None
    # The object that is being used to make the call.
    obj_called = _retrieve_object(fname, mod, gctx, None)
    if obj_called is None:
        # This is not an object to consider
        _logger.debug(f"skipping none")
        return None
    _logger.debug(f"ln:{node.lineno} {node} {obj_called}")

    if node.keywords:
        raise NotImplementedError(node)
    store_path: Optional[DDSPath] = None
    f_called: str
    if fname[-1] == Functions.Keep:
        if len(node.args) < 2:
            raise KSException(f"Wrong number of args: expected 2+, got {node.args}")
        store_path_symbol: str = node.args[0].id
        _logger.debug(f"Keep: store_path_symbol: {store_path_symbol} {type(store_path_symbol)}")
        store_path = _retrieve_path(store_path_symbol, mod, gctx)
        f_called = node.args[1].id
    elif fname[-1] == Functions.Cache:
        if len(node.args) < 2:
            raise KSException(f"Wrong number of args: expected 2+, got {node.args}")
        f_called = node.args[0].id
    else:
        f_called = fname[-1]

    # We just use the context in the function
    fun_called = _retrieve_object([f_called], mod, gctx, FunctionType)
    can_path = _fun_path(fun_called)
    _logger.debug(f"fun_called: {fun_called} : {can_path}")
    # _logger.debug(f"Introspecting function")
    context_sig = _hash([function_body_hash, function_args_hash, node.lineno])
    inner_intro = _introspect(fun_called, args=[], context_sig=context_sig, gctx=gctx)
    # _logger.debug(f"Introspecting function finished: {inner_intro}")
    _logger.debug(f"keep: {store_path} <- {can_path}: {inner_intro.fun_return_sig}")
    if store_path:
        inner_intro.outputs.append((store_path, inner_intro.fun_return_sig))
    return inner_intro


def _retrieve_object(
        path: List[str],
        mod: ModuleType,
        gctx: GlobalContext,
        expected_type: Optional[Type]) -> Optional[Any]:
    """
    Returns an object, recursively traversing the hierarchy.

    Only returns objects that are authorized, None otherwise.
    Throws error if the object does not exist
    """
    assert path
    fname = path[0]
    if fname not in mod.__dict__:
        # In some cases (old versions of jupyter) the module is not listed
        # -> try to load it from the root
        _logger.debug(f"Could not find {fname} in {mod}, attempting a direct load")
        try:
            loaded_mod = importlib.import_module(fname)
        except ModuleNotFoundError:
            loaded_mod = None
        if loaded_mod is None:
            _logger.debug(f"Could not load name {fname}, looking into the globals")
            if fname in gctx.start_globals:
                _logger.debug(f"Found {fname} in start_globals")
                return gctx.start_globals[fname]
            else:
                _logger.debug(f"{fname} not found in start_globals")
                return None
        return _retrieve_object_rec(path[1:], loaded_mod, gctx, expected_type)
    else:
        return _retrieve_object_rec(path, mod, gctx, expected_type)


def _retrieve_object_rec(
        path: List[str],
        mod: ModuleType,
        gctx: GlobalContext,
        expected_type: Optional[Type]) -> Optional[Any]:
    assert path
    fname = path[0]
    _logger.debug(f"_retrieve_object_rec: {path} {mod}")
    if fname not in mod.__dict__:
        # If the name is not in scope, it is assumed to be defined in the function body -> skipped
        # (it is included with the code lines)
        # TODO: confirm this choice and use locals() against that case
        return None
        # raise KSException(f"Object {fname} not found in module {mod}. Choices are {mod.__dict__}")
    obj = mod.__dict__[fname]
    if len(path) >= 2:
        if isinstance(obj, ModuleType):
            return _retrieve_object(path[1:], obj, gctx, expected_type)
        if _is_authorized_type(type(obj), gctx):
            raise NotImplementedError(f"Object {fname} of type {type(obj)} is authorized")
        _logger.debug(f"Object {fname} of type {type(obj)} is authorized, skipping path {path}")
        return None
    # Check the real module of the object, if available (such as for functions)
    obj_mod = inspect.getmodule(obj)
    if obj_mod is not None:
        obj_mod_path = _mod_path(obj_mod)
        if not gctx.is_authorized_path(obj_mod_path):
            _logger.debug(f"Actual module {obj_mod_path} for obj {obj} is not authorized")
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
        _logger.debug(f"{path} -> {obj}: {p}")
        if not gctx.is_authorized_path(p):
            _logger.debug(f"dropping unauthorized function {path} -> {obj}: {fun_mod.__name__}")
            return None
        else:
            _logger.debug(f"authorized function {path} -> {obj}: {fun_mod.__name__}")
    else:
        _logger.debug(f"not checking: {obj} {type(obj)}")
    return obj


def _canonical_path(path: List[str], mod: ModuleType, gctx: GlobalContext) -> CanonicalPath:
    if not path:
        # Return the path of the module
        return _mod_path(mod)
    assert path
    fname = path[0]
    if fname not in mod.__dict__:
        _logger.debug(f"Path {path} not found in {mod} -> attempting direct load")
        try:
            loaded_mod = importlib.import_module(fname)
        except ModuleNotFoundError:
            loaded_mod = None
        if loaded_mod is None:
            _logger.debug(f"Could not load name {fname} from modules")
            if fname not in gctx.start_globals:
                raise KSException(f"Object {fname} not found in module {mod}. Choices are {mod.__dict__.keys()}")
            else:
                _logger.debug(f"Found {fname} in start_globals")
                loaded_mod = gctx.start_globals[fname]
                if not isinstance(loaded_mod, ModuleType) and len(path) >= 2:
                    # This is a sub variable, not accepted for now.
                    raise KSException(f"Object {fname} of type {type(loaded_mod)} not accepted for path {path}")
        return _canonical_path_rec(path[1:], loaded_mod, gctx)
    else:
        return _canonical_path_rec(path, mod, gctx)


def _canonical_path_rec(path: List[str], mod: ModuleType, gctx: GlobalContext) -> CanonicalPath:
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
        return _canonical_path(path[1:], obj, gctx)
    assert len(path) == 1, path
    ref_module = inspect.getmodule(obj)
    if ref_module == mod or ref_module is None:
        return _canonical_path([], mod, gctx).append(fname)
    else:
        _logger.debug(f"Redirection: {path} {mod} {ref_module}")
        return _canonical_path(path, ref_module, gctx)


def _retrieve_path(fname: str, mod: ModuleType, gctx: GlobalContext) -> DDSPath:
    if fname not in mod.__dict__:
        # Look in the global context to see it is defined there (happens with some REPLs like Databricks)
        if fname not in gctx.start_globals:
            raise KSException(f"Expected path {fname} not found in module {mod} or globals. Choices are {mod.__dict__.keys()} or globals {gctx.start_globals.keys()}")
        else:
            obj = gctx.start_globals[fname]
    else:
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
