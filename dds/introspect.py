import ast
import importlib
import pathlib
import inspect
import logging
from collections import OrderedDict
from enum import Enum
from pathlib import PurePosixPath
from types import ModuleType, FunctionType
from typing import (
    Tuple,
    Callable,
    Any,
    Dict,
    Set,
    Union,
    Optional,
    List,
    Type,
    NewType,
    Sequence,
)

from .fun_args import dds_hash as _hash, get_arg_ctx_ast
from .structures import (
    PyHash,
    FunctionArgContext,
    DDSPath,
    FunctionInteractions,
    KSException,
    CanonicalPath,
    ExternalDep,
    LocalDepPath,
    FunctionArgContextHash,
)
from ._print_ast import pformat
from .structures_utils import DDSPathUtils, LocalDepPathUtils
from ._lambda_funs import is_lambda, inspect_lambda_condition

_logger = logging.getLogger(__name__)

Package = NewType("Package", str)

# The name of a local function var
LocalVar = NewType("LocalVar", str)


def introspect(
    f: Callable[[Any], Any], start_globals: Dict[str, Any], arg_ctx: FunctionArgContext
) -> FunctionInteractions:
    # TODO: exposed the whitelist
    # TODO: add the arguments of the function
    gctx = EvalMainContext(
        f.__module__,  # type: ignore
        whitelisted_packages=_whitelisted_packages,
        start_globals=start_globals,
    )
    fun: FunctionType = f  # type: ignore
    return _introspect(fun, arg_ctx, gctx, call_stack=[])


class Functions(str, Enum):
    Load = "load"
    Keep = "keep"
    Cache = "cache"
    Eval = "eval"


PythonId = NewType("PythonId", int)


class GlobalContext(object):
    """
    The global context holds caches about from past evaluations.
    """

    def __init__(self):
        self.cached_fun_calls: Dict[
            Tuple[CanonicalPath, FunctionArgContextHash], List[CanonicalPath]
        ] = {}
        # The cached interactions
        self.cached_fun_interactions: Dict[
            Tuple[
                CanonicalPath,
                FunctionArgContextHash,
                Tuple[Tuple[CanonicalPath, PythonId], ...],
            ],
            FunctionInteractions,
        ] = {}


# Set this to None to disable the global context.
# TODO: expose as an option
_global_context: Optional[GlobalContext] = GlobalContext()  # type: ignore


class EvalMainContext(object):
    """
    The shared information across a single run.
    """

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
        self.cached_fun_interactions: Dict[
            Tuple[CanonicalPath, FunctionArgContextHash], FunctionInteractions
        ] = dict()
        self.cached_objects: Dict[
            Tuple[LocalDepPath, CanonicalPath], Optional[Tuple[Any, CanonicalPath]]
        ] = dict()

    def get_hash(self, path: CanonicalPath, obj: Any) -> PyHash:
        if path not in self._hashes:
            key = _hash(obj)
            _logger.debug(f"Cache key: %s: %s %s", path, type(obj), key)
            self._hashes[path] = key
            return key
        return self._hashes[path]

    def is_authorized_path(self, cp: CanonicalPath) -> bool:
        for idx in range(len(self.whitelisted_packages)):
            if ".".join(cp._path[:idx]) in self.whitelisted_packages:
                return True
        return False


def _all_paths(fis: FunctionInteractions) -> Set[CanonicalPath]:
    res: Set[CanonicalPath] = {fis.fun_path}
    for fis0 in fis.parsed_body:
        res.update(_all_paths(fis0))
    return res


def _introspect(
    f: FunctionType,
    arg_ctx: FunctionArgContext,
    gctx: EvalMainContext,
    call_stack: List[CanonicalPath],
) -> FunctionInteractions:
    # Check if the function has already been evaluated.
    fun_path = _fun_path(f)
    arg_ctx_hash = FunctionArgContext.as_hashable(arg_ctx)
    # In most cases, lambda functions will change id's each time. Skipping for now.
    if (
        not is_lambda(f)
        and _global_context is not None
        and (fun_path, arg_ctx_hash) in _global_context.cached_fun_calls
    ):
        dep_paths = _global_context.cached_fun_calls[(fun_path, arg_ctx_hash)]
        _logger.debug(
            f"{fun_path} in cache, evaluating if {len(dep_paths)} python objects have changed"
        )
        ids: List[Tuple[CanonicalPath, PythonId]] = []
        for dep_path in dep_paths:
            obj = ObjectRetrieval.retrieve_object_global(dep_path, gctx)
            ids.append((dep_path, PythonId(id(obj))))
        tup = tuple(ids)
        if (fun_path, arg_ctx_hash, tup) in _global_context.cached_fun_interactions:
            _logger.debug(f"{fun_path} in interaction cache, skipping analysis")
            return _global_context.cached_fun_interactions[
                (fun_path, arg_ctx_hash, tup)
            ]
        else:
            _logger.debug(f"{fun_path} not in interaction cache, objects have changed")

    fun_module = inspect.getmodule(f)
    if fun_module is None:
        raise KSException(f"Could not find module: f; {f} module: {fun_module}")
    # _logger.debug(f"_introspect: {f}: fun_path={fun_path} fun_module={fun_module}")
    ast_f: Union[ast.Lambda, ast.FunctionDef]
    if is_lambda(f):
        # _logger.debug(f"_introspect: is_lambda: {f}")
        src = inspect.getsource(f)
        h = _hash(src)
        # Have a stable name for the lambda function
        fun_path = CanonicalPath(fun_path._path[:-1] + [fun_path._path[-1] + h])
        fis_key = (fun_path, arg_ctx_hash)
        fis_ = gctx.cached_fun_interactions.get(fis_key)
        if fis_ is not None:
            return fis_
        # Not seen before, continue.
        _logger.debug(f"_introspect: is_lambda: fun_path={fun_path} src={src}")
        ast_f = inspect_lambda_condition(f)
        assert isinstance(ast_f, ast.Lambda), type(ast_f)
        # _logger.debug(f"_introspect: is_lambda: {ast_f}")
    else:
        fis_key = (fun_path, arg_ctx_hash)
        fis_ = gctx.cached_fun_interactions.get(fis_key)
        if fis_ is not None:
            return fis_
        src = inspect.getsource(f)
        # _logger.debug(f"Starting _introspect: {f}: src={src}")
        ast_src = ast.parse(src)
        ast_f = ast_src.body[0]  # type: ignore
        assert isinstance(ast_f, ast.FunctionDef), type(ast_f)
        # _logger.debug(f"_introspect ast_src:\n {pformat(ast_f)}")
    body_lines = src.split("\n")

    fis = InspectFunction.inspect_fun(
        ast_f, gctx, fun_module, body_lines, arg_ctx, fun_path, call_stack
    )
    # Cache the function interactions
    gctx.cached_fun_interactions[fis_key] = fis
    # cache the function interactions in the global context
    if not is_lambda(f) and _global_context is not None:
        dep_paths = sorted(_all_paths(fis))
        _global_context.cached_fun_calls[(fun_path, arg_ctx_hash)] = dep_paths
        # Find the id of each corresponding object
        obj_ids: List[Tuple[CanonicalPath, PythonId]] = []
        for dep_path in dep_paths:
            obj = ObjectRetrieval.retrieve_object_global(dep_path, gctx)
            obj_ids.append((dep_path, PythonId(id(obj))))
        tup = tuple(obj_ids)
        _global_context.cached_fun_interactions[(fun_path, arg_ctx_hash, tup)] = fis
    return fis


class IntroVisitor(ast.NodeVisitor):
    def __init__(
        self,
        start_mod: ModuleType,
        gctx: EvalMainContext,
        function_body_lines: List[str],
        function_args_hash: PyHash,
        function_var_names: Set[LocalVar],
        call_stack: List[CanonicalPath],
    ):
        # TODO: start_mod is in the global context
        self._start_mod = start_mod
        self._gctx = gctx
        self._function_var_names = set(function_var_names)
        self._body_lines = function_body_lines
        self._args_hash = function_args_hash
        self._call_stack = call_stack
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
            self._call_stack,
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
        self, start_mod: ModuleType, gctx: EvalMainContext, local_vars: Set[LocalVar]
    ):
        self._start_mod = start_mod
        self._gctx = gctx
        self._local_vars = local_vars
        # TODO: rename to deps
        self.vars: Dict[LocalDepPath, ExternalDep] = {}
        # All the dependencies that are encountered but do not lead to an external dep.
        self._rejected_paths: Set[LocalDepPath] = set()

    def visit_Name(self, node: ast.Name) -> Any:
        local_dep_path = LocalDepPath(PurePosixPath(node.id))
        _logger.debug(
            "ExternalVarsVisitor:visit_Name: id: %s local_dep_path:%s",
            node.id,
            local_dep_path,
        )
        if not isinstance(node.ctx, ast.Load):
            _logger.debug(
                "ExternalVarsVisitor:visit_Name: id: %s skipping ctx: %s",
                node.id,
                node.ctx,
            )
            return
        # If it is a var that is already part of the function, do not introspect
        if len(local_dep_path.parts) == 1:
            v = str(local_dep_path)
            if v in self._local_vars:
                _logger.debug(
                    "ExternalVarsVisitor:visit_Name: id: %s skipping, in vars", node.id
                )
                return
        if local_dep_path in self.vars or local_dep_path in self._rejected_paths:
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
            _logger.debug("visit_Name: %s: skipping (unauthorized)", local_dep_path)
            self._rejected_paths.add(local_dep_path)
            return
        (obj, path) = res
        if isinstance(obj, FunctionType):
            # Modules and callables are tracked separately
            _logger.debug(f"visit name %s: skipping (fun)", local_dep_path)
            self._rejected_paths.add(local_dep_path)
            return
        if isinstance(obj, ModuleType):
            # Modules and callables are tracked separately
            # TODO: this is not accurate, as a variable could be called in a submodule
            _logger.debug(f"visit name %s: skipping (module)", local_dep_path)
            self._rejected_paths.add(local_dep_path)
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


def _function_name(node: ast.AST) -> List[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.BinOp):
        return [type(node).__name__.split(".")[-1]]
    if isinstance(node, ast.Attribute):
        return _function_name(node.value) + [node.attr]
    if isinstance(node, ast.Call):
        return _function_name(node.func)
    if isinstance(node, (ast.Constant, ast.NameConstant)):
        s = str(node.value)
        s = s[:4]
        return [f"Str({s}...)"]
    # TODO: these names are just here to make sure that a valid function name is returned
    # They should not be returned: this name will never be found amongst the functions.
    if isinstance(node, ast.Str):
        s = str(node.s)
        s = s[:4]
        return [f"Str({s}...)"]
    types = [
        ast.Subscript,
        ast.Compare,
        ast.UnaryOp,
        ast.BoolOp,
        ast.IfExp,
        ast.Subscript,
        ast.Index,
        ast.Slice,
        ast.ExtSlice,
    ]
    for t in types:
        if isinstance(node, t):
            return [t.__name__]
    _logger.error(
        f"Cannot understand nodes of type {type(node)}. Syntax tree: {pformat(node)}"
    )
    assert False, (node, type(node))


def _mod_path(m: ModuleType) -> CanonicalPath:
    return CanonicalPath(m.__name__.split("."))


def _fun_path(f: FunctionType) -> CanonicalPath:
    mod = inspect.getmodule(f)
    if mod is None:
        raise KSException(f"Function {f} has no module")
    return CanonicalPath(_mod_path(mod)._path + [f.__name__])


def _is_authorized_type(tpe: Type[Any], gctx: EvalMainContext) -> bool:
    """
    True if the type is defined within the whitelisted hierarchy
    """
    if tpe is None:
        return True
    if tpe in (int, float, str, bytes, PurePosixPath, FunctionType, ModuleType):
        return True
    if issubclass(tpe, object):
        mod = inspect.getmodule(tpe)
        if mod is None:
            _logger.debug(f"_is_authorized_type: type %s has no module", tpe)
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
        cls, body: Sequence[ast.AST], arg_ctx: FunctionArgContext
    ) -> List[LocalVar]:
        lvars_v = LocalVarsVisitor(list(arg_ctx.named_args.keys()))
        for node in body:
            lvars_v.visit(node)
        lvars = sorted(list(lvars_v.vars))
        _logger.debug(f"local vars: %s", lvars)
        return [LocalVar(s) for s in lvars]

    @classmethod
    def get_external_deps(
        cls,
        node: ast.FunctionDef,
        mod: ModuleType,
        gctx: EvalMainContext,
        vars: Set[LocalVar],
    ) -> List[ExternalDep]:
        vdeps = ExternalVarsVisitor(mod, gctx, vars)
        vdeps.visit(node)
        return sorted(vdeps.vars.values(), key=lambda ed: ed.local_path)

    @classmethod
    def inspect_fun(
        cls,
        node: Union[ast.FunctionDef, ast.Lambda],
        gctx: EvalMainContext,
        mod: ModuleType,
        function_body_lines: List[str],
        arg_ctx: FunctionArgContext,
        fun_path: CanonicalPath,
        call_stack: List[CanonicalPath],
    ) -> FunctionInteractions:
        body: Sequence[ast.AST]
        if isinstance(node, ast.FunctionDef):
            body = node.body
        elif isinstance(node, ast.Lambda):
            body = [node.body]
        else:
            raise KSException(f"unknown ast node {type(node)}")
        local_vars = set(cls.get_local_vars(body, arg_ctx))
        _logger.debug(f"inspect_fun: %s local_vars: %s", fun_path, local_vars)
        vdeps = ExternalVarsVisitor(mod, gctx, local_vars)
        for n in body:
            vdeps.visit(n)
        ext_deps = sorted(vdeps.vars.values(), key=lambda ed: ed.local_path)
        _logger.debug(f"inspect_fun: ext_deps: %s", ext_deps)
        arg_keys = FunctionArgContext.relevant_keys(arg_ctx)
        sig_list: List[Any] = ([(ed.local_path, ed.sig) for ed in ext_deps] + arg_keys)  # type: ignore
        input_sig = _hash(sig_list)
        calls_v = IntroVisitor(
            mod, gctx, function_body_lines, input_sig, local_vars, call_stack
        )
        for n in body:
            calls_v.visit(n)
        body_sig = _hash(function_body_lines)
        return_sig = _hash(
            [input_sig, body_sig] + [i.fun_return_sig for i in calls_v.inters]
        )

        # Look at the annotations to see if there is a reference to a dds_function
        if isinstance(node, ast.FunctionDef):
            store_path = cls._path_annotation(node, mod, gctx)
        else:
            store_path = None
        _logger.debug(f"inspect_fun: path from annotation: %s", store_path)

        return FunctionInteractions(
            arg_input=arg_ctx,
            fun_body_sig=body_sig,
            fun_return_sig=return_sig,
            external_deps=ext_deps,
            parsed_body=calls_v.inters,
            store_path=store_path,
            fun_path=fun_path,
        )

    @classmethod
    def _path_annotation(
        cls, node: ast.FunctionDef, mod: ModuleType, gctx: EvalMainContext
    ) -> Optional[DDSPath]:
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call):
                local_path = LocalDepPath(
                    PurePosixPath("/".join(_function_name(dec.func)))
                )
                _logger.debug(f"_path_annotation: local_path: %s", local_path)
                z = ObjectRetrieval.retrieve_object(local_path, mod, gctx)
                if z is None:
                    _logger.debug(
                        f"_path_annotation: local_path: %s is rejected", local_path
                    )
                    return None
                caller_fun, caller_fun_path = z
                _logger.debug(f"_path_annotation: caller_fun_path: %s", caller_fun_path)
                if caller_fun_path == CanonicalPath(
                    ["dds", "_annotations", "dds_function"]
                ):
                    if len(dec.args) != 1:
                        raise KSException(
                            f"Wrong number of arguments for decorator: {pformat(dec)}"
                        )
                    store_path = cls._retrieve_store_path(dec.args[0], mod, gctx)
                    return store_path
        return None

    @classmethod
    def inspect_call(
        cls,
        node: ast.Call,
        gctx: EvalMainContext,
        mod: ModuleType,
        function_body_hash: PyHash,
        function_args_hash: PyHash,
        function_inter_hash: PyHash,
        var_names: Set[LocalVar],
        call_stack: List[CanonicalPath],
    ) -> Optional[FunctionInteractions]:
        # _logger.debug(f"Inspect call:\n %s", pformat(node))

        local_path = LocalDepPath(PurePosixPath("/".join(_function_name(node.func))))
        # _logger.debug(f"inspect_call: local_path: %s", local_path)
        if str(local_path) in var_names:
            # _logger.debug(
            #     f"inspect_call: local_path: %s is rejected (in vars)", local_path
            # )
            return None
        z = ObjectRetrieval.retrieve_object(local_path, mod, gctx)
        if z is None:
            # _logger.debug(f"inspect_call: local_path: %s is rejected", local_path)
            return None
        caller_fun, caller_fun_path = z
        if not isinstance(caller_fun, FunctionType):
            raise NotImplementedError(
                f"Expected FunctionType for {caller_fun_path}, got {type(caller_fun)}"
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
            called_path_ast = node.args[1]
            if isinstance(called_path_ast, ast.Name):
                called_path_symbol = node.args[1].id  # type: ignore
            else:
                raise NotImplementedError(
                    f"Cannot use nested callables of type {called_path_ast}"
                )
            called_local_path = LocalDepPath(PurePosixPath(called_path_symbol))
            called_z = ObjectRetrieval.retrieve_object(called_local_path, mod, gctx)
            if not called_z:
                # Not sure what to do yet in this case.
                raise NotImplementedError(
                    f"Invalid called_z: {called_local_path} {mod}"
                )
            called_fun, call_fun_path = called_z
            if call_fun_path in call_stack:
                raise KSException(
                    f"Detected circular function calls or (co-)recursive calls."
                    f"This is currently not supported. Change your code to split the "
                    f"recursive section into a separate function. "
                    f"Function: {call_fun_path}"
                    f"Call stack: {' '.join([str(p) for p in call_stack])}"
                )
            context_sig = _hash(
                [function_body_hash, function_args_hash, function_inter_hash]
            )
            new_call_stack = call_stack + [call_fun_path]
            # TODO: deal with the arguments here
            if node.keywords:
                raise NotImplementedError(
                    (_function_name(node.func), node, node.keywords)
                )
            # For now, accept the constant arguments. This is enough for some basic objects.
            arg_ctx = FunctionArgContext(
                named_args=get_arg_ctx_ast(called_fun, node.args[2:]),  # type: ignore
                inner_call_key=context_sig,
            )
            inner_intro = _introspect(called_fun, arg_ctx, gctx, new_call_stack)
            inner_intro = inner_intro._replace(store_path=store_path)
            return inner_intro
        if caller_fun_path == CanonicalPath(["dds", "eval"]):
            raise NotImplementedError("eval")
        if caller_fun_path == CanonicalPath(["dds", "load"]):
            raise NotImplementedError("load")

        if caller_fun_path in call_stack:
            raise KSException(
                f"Detected circular function calls or (co-)recursive calls."
                f"This is currently not supported. Change your code to split the "
                f"recursive section into a separate function. "
                f"Function: {caller_fun_path}"
                f"Call stack: {' '.join([str(p) for p in call_stack])}"
            )

        # Normal function call.
        # Just introspect the function call.
        # TODO: deal with the arguments here
        context_sig = _hash(
            [function_body_hash, function_args_hash, function_inter_hash]
        )
        # For now, do not look carefully at the arguments, just parse the arguments of
        # the functions.
        # TODO: add more arguments if we can parse constant arguments
        arg_ctx = FunctionArgContext(
            named_args=get_arg_ctx_ast(caller_fun, []), inner_call_key=context_sig,
        )
        new_call_stack = call_stack + [caller_fun_path]
        return _introspect(caller_fun, arg_ctx, gctx, new_call_stack)

    @classmethod
    def _retrieve_store_path(
        cls, local_path_node: ast.AST, mod: ModuleType, gctx: EvalMainContext
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
            f"Keep: store_path_symbol: %s %s",
            store_path_symbol,
            type(store_path_symbol),
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
        cls, local_path: LocalDepPath, context_mod: ModuleType, gctx: EvalMainContext
    ) -> Optional[Tuple[Any, CanonicalPath]]:
        """Retrieves the object and also provides the canonical path of the object"""
        assert len(local_path.parts), local_path
        mod_path = _mod_path(context_mod)
        obj_key = (local_path, mod_path)
        if obj_key in gctx.cached_objects:
            return gctx.cached_objects[obj_key]

        fname = local_path.parts[0]
        sub_path = LocalDepPathUtils.tail(local_path)
        if fname not in context_mod.__dict__:
            # In some cases (old versions of jupyter) the module is not listed
            # -> try to load it from the root
            _logger.debug(
                f"Could not find {fname} in {context_mod}, attempting a direct load"
            )
            loaded_mod: Optional[ModuleType]
            try:
                loaded_mod = importlib.import_module(fname)
            except ModuleNotFoundError:
                loaded_mod = None
            if loaded_mod is None:
                # Looking into the globals (only if the scope is currently __main__ or __global__)
                mod_path = _mod_path(context_mod)
                if mod_path.get(0) not in ("__main__", "__global__"):
                    _logger.debug(
                        f"Could not load name %s and not in global context (%s), skipping ",
                        fname,
                        mod_path,
                    )
                    return None
                else:
                    _logger.debug(
                        f"Could not load name %s, looking into the globals (mod_path: %s, %s)",
                        fname,
                        mod_path,
                        mod_path.get(0),
                    )
                _logger.debug(f"Could not load name {fname}, looking into the globals")
                if fname in gctx.start_globals:
                    _logger.debug(f"Found {fname} in start_globals")
                    obj = gctx.start_globals[fname]
                    if isinstance(obj, ModuleType) and not LocalDepPathUtils.empty(
                        sub_path
                    ):
                        # Referring to function from an imported module.
                        # Redirect the search to the module
                        _logger.debug(
                            f"{fname} is module {obj}, checking for {sub_path}"
                        )
                        res = cls.retrieve_object(sub_path, obj, gctx)
                        gctx.cached_objects[obj_key] = res
                        return res
                    if isinstance(obj, ModuleType):
                        # Fully resolve the name of the module:
                        obj_path = _mod_path(obj)
                    elif isinstance(obj, FunctionType):
                        obj_path = _fun_path(obj)
                    else:
                        obj_path = CanonicalPath(
                            ["__global__"] + [str(x) for x in local_path.parts]
                        )
                    if not gctx.is_authorized_path(obj_path):
                        _logger.debug(
                            f"Object[start_globals] {fname} of type {type(obj)} is not authorized (path),"
                            f" dropping path {obj_path}"
                        )
                        gctx.cached_objects[obj_key] = None
                        return None

                    if _is_authorized_type(type(obj), gctx) or isinstance(
                        obj,
                        (
                            FunctionType,
                            ModuleType,
                            pathlib.PosixPath,
                            pathlib.PurePosixPath,
                            str,
                        ),
                    ):
                        _logger.debug(
                            f"Object[start_globals] {fname} ({type(obj)}) of path {obj_path} is authorized,"
                        )
                        res = obj, obj_path
                        gctx.cached_objects[obj_key] = res
                        return res
                    else:
                        _logger.debug(
                            f"Object[start_globals] {fname} of type {type(obj)} is noft authorized (type), dropping path {obj_path}"
                        )
                        gctx.cached_objects[obj_key] = None
                        return None
                else:
                    _logger.debug(f"{fname} not found in start_globals")
                    gctx.cached_objects[obj_key] = None
                    return None
            res = cls._retrieve_object_rec(sub_path, loaded_mod, gctx)
            gctx.cached_objects[obj_key] = res
            return res
        else:
            res = cls._retrieve_object_rec(local_path, context_mod, gctx)
            gctx.cached_objects[obj_key] = res
            return res

    @classmethod
    def retrieve_object_global(
        cls, path: CanonicalPath, gctx: EvalMainContext
    ) -> Optional[Any]:
        # The head is always assumed to be a module for now
        mod_name = path.head()
        obj_key = (LocalDepPath(PurePosixPath("")), path)
        if obj_key in gctx.cached_objects:
            return gctx.cached_objects[obj_key]

        mod = importlib.import_module(mod_name)
        if mod is None:
            raise KSException(
                f"Cannot process path {path}: module {mod_name} cannot be loaded"
            )
        sub_path = path.tail()
        dep_path = LocalDepPath(PurePosixPath("/".join(sub_path._path)))
        _logger.debug(f"Calling retrieve_object on {dep_path}, {mod}")
        z = cls.retrieve_object(dep_path, mod, gctx)
        if z is None:
            raise KSException(
                f"Cannot process path {path}: object cannot be retrieved. dep_path: {dep_path} module: {mod}"
            )
        obj, _ = z
        gctx.cached_objects[obj_key] = (obj, path)
        return obj

    @classmethod
    def _retrieve_object_rec(
        cls, local_path: LocalDepPath, context_mod: ModuleType, gctx: EvalMainContext
    ) -> Optional[Tuple[Any, CanonicalPath]]:
        # _logger.debug(f"_retrieve_object_rec: {local_path} {context_mod}")
        if not local_path.parts:
            # The final position. It is the given module, if authorized.
            obj_mod_path = _mod_path(context_mod)
            if not gctx.is_authorized_path(obj_mod_path):
                # _logger.debug(
                #     f"_retrieve_object_rec: Actual module {obj_mod_path} for obj {context_mod} is not authorized"
                # )
                return None
            else:
                # _logger.debug(
                #     f"_retrieve_object_rec: Actual module {obj_mod_path} for obj {context_mod}: authorized"
                # )
                pass
            return context_mod, obj_mod_path
        # At least one more path to explore
        fname = local_path.parts[0]
        tail_path = LocalDepPathUtils.tail(local_path)
        if fname not in context_mod.__dict__:
            # It should be in the context module, this was assumed to be taken care of
            raise NotImplementedError(
                f"_retrieve_object_rec: Object {fname} not found in module {context_mod}."
                f"  {local_path} {context_mod.__dict__}"
            )
        obj = context_mod.__dict__[fname]

        if LocalDepPathUtils.empty(tail_path):
            # Final path.
            # If it is a module, continue recursion
            if isinstance(obj, ModuleType):
                return cls._retrieve_object_rec(tail_path, obj, gctx)
            # Special treatement for objects that may be defined in other modules but are redirected in this one.
            if isinstance(obj, FunctionType):
                mod_obj = inspect.getmodule(obj)
                if mod_obj is None:
                    # _logger.debug(
                    #     f"_retrieve_object_rec: cannot infer definition module: path: {local_path} mod: {context_mod} "
                    # )
                    return None
                if mod_obj is not context_mod:
                    # _logger.debug(
                    #     f"_retrieve_object_rec: {context_mod} is not definition module, redirecting to {mod_obj}"
                    # )
                    return cls._retrieve_object_rec(local_path, mod_obj, gctx)
            obj_mod_path = _mod_path(context_mod)
            obj_path = obj_mod_path.append(fname)
            if gctx.is_authorized_path(obj_path):
                # TODO: simplify the authorized types
                if _is_authorized_type(type(obj), gctx) or isinstance(
                    obj,
                    (
                        FunctionType,
                        ModuleType,
                        pathlib.PosixPath,
                        pathlib.PurePosixPath,
                    ),
                ):
                    # _logger.debug(
                    #     f"_retrieve_object_rec: Object {fname} ({type(obj)}) of path {obj_path} is authorized,"
                    # )
                    return obj, obj_path
                else:
                    # _logger.debug(
                    #     f"_retrieve_object_rec: Object {fname} of type {type(obj)} is not authorized (type), dropping path {obj_path}"
                    # )
                    pass
            else:
                # _logger.debug(
                #     f"_retrieve_object_rec: Object {fname} of type {type(obj)} and path {obj_path} is not authorized (path)"
                # )
                return None

        # _logger.debug(
        #     f"_retrieve_object_rec: non-terminal fname={fname} obj: {type(obj)} tail_path: {tail_path} {isinstance(obj, FunctionType)}"
        # )
        # More to explore
        # If it is a module, continue recursion
        if isinstance(obj, ModuleType):
            return cls._retrieve_object_rec(tail_path, obj, gctx)

        # Some objects like types are also collables

        # We still have a path but we have reached a callable.
        # In this case, determine if the function is allowed. If this is the case, stop here.
        # (the rest of the path is method calls)
        if isinstance(obj, FunctionType):
            obj_mod_path = _mod_path(context_mod)
            obj_path = obj_mod_path.append(fname)
            if gctx.is_authorized_path(obj_path):
                return obj, obj_path

        # The rest is not authorized for now.
        # msg = f"Failed to consider object type {type(obj)} at path {local_path} context_mod: {context_mod}"
        # _logger.debug(msg)
        return None


_whitelisted_packages: Set[Package] = {
    Package("dds"),
    Package("__main__"),
    Package("__global__"),
}


def whitelist_module(module: Union[str, ModuleType]) -> None:
    global _whitelisted_packages
    if isinstance(module, ModuleType):
        module = module.__name__
    assert isinstance(module, str), (module, type(module))
    _whitelisted_packages.add(Package(module))
