import abc
import ast
import importlib
import pathlib
import inspect
import logging
from collections import OrderedDict
from enum import Enum
from pathlib import PurePosixPath
from types import ModuleType, FunctionType
from typing import Tuple, Callable, Any, Dict, Set, Union, Optional, List, Type, NewType

from .fun_args import dds_hash as _hash, FunctionArgContext, get_arg_ctx_ast
from .structures import (
    PyHash,
    DDSPath,
    FunctionInteractions,
    KSException,
    CanonicalPath,
    ExternalDep,
    LocalDepPath,
)
from ._print_ast import pformat
from .structures_utils import DDSPathUtils, LocalDepPathUtils

_logger = logging.getLogger(__name__)

Package = NewType("Package", str)

# The name of a local function var
LocalVar = NewType("LocalVar", str)


def introspect(
    f: Callable[[Any], Any], start_globals: Dict[str, Any], arg_ctx: FunctionArgContext
) -> FunctionInteractions:
    # TODO: exposed the whitelist
    # TODO: add the arguments of the function
    gctx = GlobalContext(
        f.__module__,  # typing: ignore
        whitelisted_packages=_whitelisted_packages,
        start_globals=start_globals,
    )
    return _introspect(f, arg_ctx, gctx)


class Functions(str, Enum):
    Load = "load"
    Keep = "keep"
    Cache = "cache"
    Eval = "eval"


class GlobalContext(object):
    # The packages that are authorized for traversal

    def __init__(
        self,
        start_module: ModuleType,
        whitelisted_packages: Set[Package],
        start_globals: Dict[str, Any],
    ):
        self.whitelisted_packages = whitelisted_packages
        self.start_module = start_module
        self.start_globals = start_globals
        # Hashes of all the static objects
        self._hashes: Dict[CanonicalPath, PyHash] = {}

    def get_hash(self, path: CanonicalPath, obj: Any) -> PyHash:
        if path not in self._hashes:
            key = _hash(obj)
            _logger.debug(f"Cache key: {path}: {type(obj)} {key}")
            self._hashes[path] = key
            return key
        return self._hashes[path]

    def is_authorized_path(self, cp: CanonicalPath) -> bool:
        _logger.debug(f"is_authorized_path: {self.whitelisted_packages}")
        for idx in range(len(self.whitelisted_packages)):
            if ".".join(cp._path[:idx]) in self.whitelisted_packages:
                return True
        return False


def _introspect(
    f: Callable[[Any], Any], arg_ctx: FunctionArgContext, gctx: GlobalContext,
) -> FunctionInteractions:
    # TODO: remove args for now?
    arg_sig = inspect.signature(f)
    src = inspect.getsource(f)
    _logger.debug(f"Starting _introspect: {f}: arg_sig={arg_sig} src={src}")
    ast_src = ast.parse(src)
    body_lines = src.split("\n")
    ast_f = ast_src.body[0]
    fun_module = inspect.getmodule(f)

    fun_path = _fun_path(f)

    fis = InspectFunction.inspect_fun(
        ast_f, gctx, fun_module, body_lines, arg_ctx, fun_path
    )
    _logger.debug(f"End _introspect: {f}: {fis}")
    return fis


class IntroVisitor(ast.NodeVisitor):
    def __init__(
        self,
        start_mod: ModuleType,
        gctx: GlobalContext,
        function_body_lines: List[str],
        function_args_hash: PyHash,
        function_var_names: Set[str],
    ):
        # TODO: start_mod is in the global context
        self._start_mod = start_mod
        self._gctx = gctx
        self._function_var_names = set(function_var_names)
        self._body_lines = function_body_lines
        self._args_hash = function_args_hash
        self.inters: List[FunctionInteractions] = []

    def visit_Call(self, node: ast.Call) -> Any:
        # _logger.debug(f"visit: {node} {dir(node)}")
        function_body_hash = _hash(self._body_lines[: node.lineno + 1])
        # The list of all the previous interactions.
        # This enforces the concept that the current call depends on previous calls.
        function_inters_sig = _hash([fi.fun_return_sig for fi in self.inters])
        fi = InspectFunction.inspect_call(
            node,
            self._gctx,
            self._start_mod,
            function_body_hash,
            self._args_hash,
            function_inters_sig,
            self._function_var_names,
        )
        if fi is not None:
            self.inters.append(fi)
        self.generic_visit(node)


class ExternalVarsVisitor(ast.NodeVisitor):
    """
    Finds all the external variables of a function that should be hashed into the argument list.
    TODO: currently very crude, it does not look for assigned variables.
    """

    def __init__(
        self, start_mod: ModuleType, gctx: GlobalContext, local_vars: Set[LocalVar]
    ):
        self._start_mod = start_mod
        self._gctx = gctx
        self._local_vars = local_vars
        # TODO: rename to deps
        self.vars: Dict[LocalDepPath, ExternalDep] = {}

    def visit_Name(self, node: ast.Name) -> Any:
        local_dep_path = LocalDepPath(PurePosixPath(node.id))
        _logger.debug(
            f"ExternalVarsVisitor:visit_Name: id: {node.id} local_dep_path:{local_dep_path}"
        )
        if not isinstance(node.ctx, ast.Load):
            _logger.debug(
                f"ExternalVarsVisitor:visit_Name: id: {node.id} skipping ctx: {node.ctx}"
            )
            return
        # If it is a var that is already part of the function, do not introspect
        if len(local_dep_path.parts) == 1:
            v = str(local_dep_path)
            if v in self._local_vars:
                _logger.debug(
                    f"ExternalVarsVisitor:visit_Name: id: {node.id} skipping, in vars"
                )
                return
        if local_dep_path in self.vars:
            return
        # TODO: this will fail in submodule
        # if str(local_dep_path) not in self._start_mod.__dict__ or str(local_dep_path) not in self._gctx.start_globals:
        #     _logger.debug(
        #         f"ExternalVarsVisitor:visit_Name: local_dep_path {local_dep_path} "
        #         f"not found in module {self._start_mod}: \n{self._start_mod.__dict__.keys()} \nor in start_globals: {self._gctx.start_globals}"
        #     )
        #     return
        res = ObjectRetrieval.retrieve_object(
            local_dep_path, self._start_mod, self._gctx
        )
        if res is None:
            # Nothing to do, it is not interesting.
            _logger.debug(f"visit_Name: {local_dep_path}: skipping (unauthorized)")
            return
        (obj, path) = res
        if isinstance(obj, Callable):
            # Modules and callables are tracked separately
            _logger.debug(f"visit name {local_dep_path}: skipping (fun)")
            return
        if isinstance(obj, ModuleType):
            # Modules and callables are tracked separately
            # TODO: this is not accurate, as a variable could be called in a submodule
            _logger.debug(f"visit name {local_dep_path}: skipping (module)")
            return
        sig = self._gctx.get_hash(path, obj)
        self.vars[local_dep_path] = ExternalDep(
            local_path=local_dep_path, path=path, sig=sig
        )


class LocalVarsVisitor(ast.NodeVisitor):
    """
    A brute-force attempt to find all the variables defined in the scope of a module.
    """

    def __init__(self, existing_vars: List[str]):
        self.vars: Set[str] = set(existing_vars)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        # TODO: make the error message more explicit
        # TODO: this is used in tests. It would be better to have something more robust.
        raise NotImplementedError(f"You cannot use async function with dds")

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
    if isinstance(node, ast.Call):
        return _function_name(node.func)
    if isinstance(node, (ast.Constant, ast.NameConstant)):
        s = str(node.value)
        s = s[:4]
        return [f"Str({s}...)"]
    if isinstance(node, ast.Str):
        s = str(node.s)
        s = s[:4]
        return [f"Str({s}...)"]
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
    if tpe in (int, float, str, bytes, PurePosixPath, Callable, ModuleType):
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


class InspectFunction(object):
    @classmethod
    def get_local_vars(
        cls, node: ast.FunctionDef, arg_ctx: FunctionArgContext
    ) -> List[LocalVar]:
        lvars_v = LocalVarsVisitor(list(arg_ctx.named_args.keys()))
        lvars_v.visit(node)
        lvars = sorted(list(lvars_v.vars))
        _logger.debug(f"local vars: {lvars}")
        return [LocalVar(s) for s in lvars]

    @classmethod
    def get_external_deps(
        cls,
        node: ast.FunctionDef,
        mod: ModuleType,
        gctx: GlobalContext,
        vars: Set[LocalVar],
    ) -> List[ExternalDep]:
        vdeps = ExternalVarsVisitor(mod, gctx, vars)
        vdeps.visit(node)
        return sorted(vdeps.vars.values(), key=lambda ed: ed.local_path)

    @classmethod
    def inspect_fun(
        cls,
        node: ast.FunctionDef,
        gctx: GlobalContext,
        mod: ModuleType,
        function_body_lines: List[str],
        arg_ctx: FunctionArgContext,
        fun_path: CanonicalPath,
    ) -> FunctionInteractions:
        local_vars = set(cls.get_local_vars(node, arg_ctx))
        _logger.debug(f"inspect_fun: local_vars: {local_vars}")
        vdeps = ExternalVarsVisitor(mod, gctx, local_vars)
        vdeps.visit(node)
        ext_deps = sorted(vdeps.vars.values(), key=lambda ed: ed.local_path)
        _logger.debug(f"inspect_fun: ext_deps: {ext_deps}")
        arg_keys = FunctionArgContext.relevant_keys(arg_ctx)
        sig_list: List[Any] = ([(ed.local_path, ed.sig) for ed in ext_deps] + arg_keys)
        input_sig = _hash(sig_list)
        calls_v = IntroVisitor(mod, gctx, function_body_lines, input_sig, local_vars)
        calls_v.visit(node)
        body_sig = _hash(function_body_lines)
        _logger.debug(f"inspect_fun: interactions: {calls_v.inters}")
        return_sig = _hash(
            [input_sig, body_sig] + [i.fun_return_sig for i in calls_v.inters]
        )
        return FunctionInteractions(
            arg_input=arg_ctx,
            fun_body_sig=body_sig,
            fun_return_sig=return_sig,
            external_deps=ext_deps,
            parsed_body=calls_v.inters,
            store_path=None,
            fun_path=fun_path,
        )

    @classmethod
    def inspect_call(
        cls,
        node: ast.Call,
        gctx: GlobalContext,
        mod: ModuleType,
        function_body_hash: PyHash,
        function_args_hash: PyHash,
        function_inter_hash: PyHash,
        var_names: Set[str],
    ) -> Optional[FunctionInteractions]:
        _logger.debug(f"Inspect call:\n {pformat(node)}")

        local_path = LocalDepPath(PurePosixPath("/".join(_function_name(node.func))))
        _logger.debug(f"inspect_call: local_path: {local_path}")
        if str(local_path) in var_names:
            _logger.debug(
                f"inspect_call: local_path: {local_path} is rejected (in vars)"
            )
            return
        z = ObjectRetrieval.retrieve_object(local_path, mod, gctx)
        if z is None:
            _logger.debug(f"inspect_call: local_path: {local_path} is rejected")
            return
        caller_fun, caller_fun_path = z
        if not isinstance(caller_fun, Callable):
            raise NotImplementedError(
                f"Expected callable for {caller_fun_path}, got {type(caller_fun)}"
            )

        # Check if this is a call we should do something about.
        if caller_fun_path == CanonicalPath(["dds", "keep"]):
            # Call to the keep function:
            # - bring the path
            # - bring the callee
            # - parse the arguments
            # - introspect the callee
            if len(node.args) < 2:
                raise KSException(f"Wrong number of args: expected 2+, got {node.args}")
            store_path = cls._retrieve_store_path(node.args[0], mod, gctx)
            called_path_symbol = node.args[1].id
            called_local_path = LocalDepPath(PurePosixPath(called_path_symbol))
            called_z = ObjectRetrieval.retrieve_object(called_local_path, mod, gctx)
            if not called_z:
                # Not sure what to do yet in this case.
                raise NotImplementedError(
                    f"Invalid called_z: {called_local_path} {mod}"
                )
            called_fun, call_fun_path = called_z
            context_sig = _hash(
                [function_body_hash, function_args_hash, function_inter_hash]
            )
            # TODO: deal with the arguments here
            if node.keywords:
                raise NotImplementedError(
                    (_function_name(node.func), node, node.keywords)
                )
            # For now, accept the constant arguments. This is enough for some basic objects.
            arg_ctx = FunctionArgContext(
                named_args=get_arg_ctx_ast(called_fun, node.args[2:]),
                inner_call_key=context_sig,
            )
            inner_intro = _introspect(called_fun, arg_ctx, gctx=gctx)
            inner_intro = inner_intro._replace(store_path=store_path)
            return inner_intro
        if caller_fun_path == CanonicalPath(["dds", "eval"]):
            raise NotImplementedError("eval")
        if caller_fun_path == CanonicalPath(["dds", "load"]):
            raise NotImplementedError("load")

        # Normal function call.
        # Just introspect the function call.
        # TODO: deal with the arguments here
        context_sig = _hash(
            [function_body_hash, function_args_hash, function_inter_hash]
        )
        arg_ctx = FunctionArgContext(
            named_args=OrderedDict([]), inner_call_key=context_sig
        )
        return _introspect(caller_fun, arg_ctx, gctx=gctx)

    @classmethod
    def _retrieve_store_path(
        cls, local_path_node: ast.AST, mod: ModuleType, gctx: GlobalContext
    ) -> DDSPath:
        called_path_symbol: str
        if isinstance(local_path_node, ast.Constant):
            # Just a string, directly access it.
            return DDSPathUtils.create(local_path_node.value)
        elif isinstance(local_path_node, ast.Str):
            # Just a string, directly access it.
            return DDSPathUtils.create(local_path_node.s)
        elif isinstance(local_path_node, ast.Name):
            store_path_symbol = local_path_node.id
        else:
            raise NotImplementedError(
                f"{type(local_path_node)} {pformat(local_path_node)}"
            )
        _logger.debug(
            f"Keep: store_path_symbol: {store_path_symbol} {type(store_path_symbol)}"
        )
        store_path_local_path = LocalDepPath(PurePosixPath(store_path_symbol))
        # Retrieve the store path value and the called function
        store_z = ObjectRetrieval.retrieve_object(store_path_local_path, mod, gctx)
        if not store_z:
            # Not sure what to do yet in this case.
            raise NotImplementedError(f"Invalid store_z: {store_path_local_path} {mod}")
        store_path, _ = store_z
        return DDSPathUtils.create(store_path)


class ObjectRetrieval(object):
    @classmethod
    def retrieve_object(
        cls, local_path: LocalDepPath, context_mod: ModuleType, gctx: GlobalContext
    ) -> Optional[Tuple[Any, CanonicalPath]]:
        """Retrieves the object and also provides the canonical path of the object"""
        assert len(local_path.parts), local_path
        fname = local_path.parts[0]
        sub_path = LocalDepPathUtils.tail(local_path)
        if fname not in context_mod.__dict__:
            # In some cases (old versions of jupyter) the module is not listed
            # -> try to load it from the root
            _logger.debug(
                f"Could not find {fname} in {context_mod}, attempting a direct load"
            )
            try:
                loaded_mod = importlib.import_module(fname)
            except ModuleNotFoundError:
                loaded_mod = None
            if loaded_mod is None:
                _logger.debug(f"Could not load name {fname}, looking into the globals")
                if fname in gctx.start_globals:
                    _logger.debug(f"Found {fname} in start_globals")
                    obj = gctx.start_globals[fname]
                    if isinstance(obj, ModuleType) and not LocalDepPathUtils.empty(sub_path):
                        # Referring to function from an imported module.
                        # Redirect the search to the module
                        _logger.debug(f"{fname} is module {obj}, checking for {sub_path}")
                        return cls.retrieve_object(sub_path, obj, gctx)
                    obj_path = CanonicalPath(
                        ["__global__"] + [str(x) for x in local_path.parts]
                    )

                    if _is_authorized_type(type(obj), gctx) or isinstance(
                        obj,
                        (
                            Callable,
                            ModuleType,
                            pathlib.PosixPath,
                            pathlib.PurePosixPath,
                        ),
                    ):
                        _logger.debug(
                            f"Object[start_globals] {fname} ({type(obj)}) of path {obj_path} is authorized,"
                        )
                        return obj, obj_path
                    else:
                        _logger.debug(
                            f"Object[start_globals] {fname} of type {type(obj)} is noft authorized (type), dropping path {obj_path}"
                        )
                        return None
                else:
                    _logger.debug(f"{fname} not found in start_globals")
                    return None
            return cls._retrieve_object_rec(sub_path, loaded_mod, gctx)
        else:
            return cls._retrieve_object_rec(local_path, context_mod, gctx)

    @classmethod
    def _retrieve_object_rec(
        cls, local_path: LocalDepPath, context_mod: ModuleType, gctx: GlobalContext
    ) -> Optional[Tuple[Any, CanonicalPath]]:
        _logger.debug(f"_retrieve_object_rec: {local_path} {context_mod}")
        if not local_path.parts:
            # The final position. It is the given module, if authorized.
            obj_mod_path = _mod_path(context_mod)
            if not gctx.is_authorized_path(obj_mod_path):
                _logger.debug(
                    f"_retrieve_object_rec: Actual module {obj_mod_path} for obj {context_mod} is not authorized"
                )
                return None
            else:
                _logger.debug(
                    f"_retrieve_object_rec: Actual module {obj_mod_path} for obj {context_mod}: authorized"
                )
            return context_mod, obj_mod_path
        # At least one more path to explore
        fname = local_path.parts[0]
        tail_path = LocalDepPath(PurePosixPath("/".join(local_path.parts[1:])))
        if fname not in context_mod.__dict__:
            # It should be in the context module, this was assumed to be taken care of
            raise NotImplementedError(
                f"_retrieve_object_rec: Object {fname} not found in module {context_mod}."
                f"  {local_path} {context_mod.__dict__}"
            )
        obj = context_mod.__dict__[fname]

        if not tail_path.parts:
            # Final path.
            # If it is a module, continue recursion
            if isinstance(obj, ModuleType):
                return cls._retrieve_object_rec(tail_path, obj, gctx)
            # Special treatement for objects that may be defined in other modules but are redirected in this one.
            if isinstance(obj, Callable):
                mod_obj = inspect.getmodule(obj)
                if mod_obj is None:
                    _logger.debug(
                        f"_retrieve_object_rec: cannot infer definition module: path: {local_path} mod: {context_mod} "
                    )
                    return None
                if mod_obj is not context_mod:
                    _logger.debug(
                        f"_retrieve_object_rec: {context_mod} is not definition module, redirecting to {mod_obj}"
                    )
                    return cls._retrieve_object_rec(local_path, mod_obj, gctx)
            obj_mod_path = _mod_path(context_mod)
            obj_path = obj_mod_path.append(fname)
            if gctx.is_authorized_path(obj_path):
                # TODO: simplify the authorized types
                if _is_authorized_type(type(obj), gctx) or isinstance(
                    obj,
                    (Callable, ModuleType, pathlib.PosixPath, pathlib.PurePosixPath),
                ):
                    _logger.debug(
                        f"Object {fname} ({type(obj)}) of path {obj_path} is authorized,"
                    )
                    return obj, obj_path
                else:
                    _logger.debug(
                        f"Object {fname} of type {type(obj)} is not authorized (type), dropping path {obj_path}"
                    )
            else:
                _logger.debug(
                    f"Object {fname} of type {type(obj)} and path {obj_path} is not authorized (path)"
                )
                return None

        # More to explore
        # If it is a module, continue recursion
        if isinstance(obj, ModuleType):
            return cls._retrieve_object_rec(tail_path, obj, gctx)

        # The rest is not authorized for now.
        msg = f"Failed to consider object type {type(obj)} at path {local_path} context_mod: {context_mod}"
        _logger.debug(msg)
        return None


def _retrieve_object(
    path: List[str], mod: ModuleType, gctx: GlobalContext, expected_type: Optional[Type]
) -> Optional[Any]:
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
    path: List[str], mod: ModuleType, gctx: GlobalContext, expected_type: Optional[Type]
) -> Optional[Any]:
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
            raise NotImplementedError(
                f"Object {fname} of type {type(obj)} is authorized"
            )
        _logger.debug(
            f"Object {fname} of type {type(obj)} is authorized, skipping path {path}"
        )
        return None
    # Check the real module of the object, if available (such as for functions)
    obj_mod = inspect.getmodule(obj)
    if obj_mod is not None:
        obj_mod_path = _mod_path(obj_mod)
        if not gctx.is_authorized_path(obj_mod_path):
            _logger.debug(
                f"Actual module {obj_mod_path} for obj {obj} is not authorized"
            )
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
            _logger.debug(
                f"dropping unauthorized function {path} -> {obj}: {fun_mod.__name__}"
            )
            return None
        else:
            _logger.debug(f"authorized function {path} -> {obj}: {fun_mod.__name__}")
    else:
        _logger.debug(f"not checking: {obj} {type(obj)}")
    return obj


def _canonical_path(
    path: List[str], mod: ModuleType, gctx: GlobalContext
) -> CanonicalPath:
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
                raise KSException(
                    f"Object {fname} not found in module {mod}. Choices are {mod.__dict__.keys()}"
                )
            else:
                _logger.debug(f"Found {fname} in start_globals")
                loaded_mod = gctx.start_globals[fname]
                if not isinstance(loaded_mod, ModuleType) and len(path) >= 2:
                    # This is a sub variable, not accepted for now.
                    raise KSException(
                        f"Object {fname} of type {type(loaded_mod)} not accepted for path {path}"
                    )
        return _canonical_path_rec(path[1:], loaded_mod, gctx)
    else:
        return _canonical_path_rec(path, mod, gctx)


def _canonical_path_rec(
    path: List[str], mod: ModuleType, gctx: GlobalContext
) -> CanonicalPath:
    """
    The canonical path of an entity in the python module hierarchy.
    """
    if not path:
        # Return the path of the module
        return _mod_path(mod)
    assert path
    fname = path[0]
    if fname not in mod.__dict__:
        raise KSException(
            f"Object {fname} not found in module {mod}. Choices are {mod.__dict__}"
        )
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
            raise KSException(
                f"Expected path {fname} not found in module {mod} or globals. Choices are {mod.__dict__.keys()} or globals {gctx.start_globals.keys()}"
            )
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


_whitelisted_packages: Set[Package] = {Package("dds"), Package("__main__")}


def whitelist_module(module: Union[str, ModuleType]) -> None:
    global _whitelisted_packages
    if isinstance(module, ModuleType):
        module = module.__name__
    assert isinstance(module, str), (module, type(module))
    _whitelisted_packages.add(Package(module))
