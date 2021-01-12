import ast
import inspect
import logging
from enum import Enum
from pathlib import PurePosixPath
from types import ModuleType, FunctionType
from typing import (
    Tuple,
    Callable,
    cast,
    Any,
    Dict,
    Set,
    Union,
    Optional,
    List,
    NewType,
    Sequence,
)

from ._eval_ctx import EvalMainContext, Package
from ._global_ctx import _global_context, PythonId
from ._lambda_funs import is_lambda, inspect_lambda_condition
from ._print_ast import pformat
from ._retrieve_objects import ObjectRetrieval, function_path
from .fun_args import dds_hash as dds_hash, get_arg_ctx_ast
from .structures import (
    PyHash,
    FunctionArgContext,
    DDSPath,
    FunctionInteractions,
    KSException,
    CanonicalPath,
    ExternalDep,
    LocalDepPath,
)
from .structures_utils import DDSPathUtils

_logger = logging.getLogger(__name__)


# The name of a local function var
LocalVar = NewType("LocalVar", str)


def introspect(
    f: Callable[[Any], Any], eval_ctx: EvalMainContext, arg_ctx: FunctionArgContext
) -> FunctionInteractions:
    # TODO: exposed the whitelist
    # TODO: add the arguments of the function
    fun: FunctionType = cast(FunctionType, f)
    return _introspect(fun, arg_ctx, eval_ctx, call_stack=[])


class Functions(str, Enum):
    Load = "load"
    Keep = "keep"
    Eval = "eval"


def _all_paths(fis: FunctionInteractions) -> Set[CanonicalPath]:
    res: Set[CanonicalPath] = {fis.fun_path}
    for fis0 in fis.parsed_body:
        res.update(_all_paths(fis0))
    return res


def _introspect_class(
    c: type,
    arg_ctx: FunctionArgContext,
    gctx: EvalMainContext,
    call_stack: List[CanonicalPath],
) -> FunctionInteractions:
    # Check if the function has already been evaluated.
    fun_path = function_path(c)
    arg_ctx_hash = FunctionArgContext.as_hashable(arg_ctx)

    # TODO: add to the global interactions cache

    fun_module = inspect.getmodule(c)
    if fun_module is None:
        raise KSException(f"Could not find module: class:{c} module: {fun_module}")
    # _logger.debug(f"_introspect: {f}: fun_path={fun_path} fun_module={fun_module}")
    fis_key = (fun_path, arg_ctx_hash)
    fis_ = gctx.cached_fun_interactions.get(fis_key)
    if fis_ is not None:
        return fis_
    src = inspect.getsource(c)
    # _logger.debug(f"Starting _introspect_class: {c}: src={src}")
    ast_src = ast.parse(src)
    ast_f: ast.ClassDef = ast_src.body[0]  # type: ignore
    assert isinstance(ast_f, ast.ClassDef), type(ast_f)
    _logger.debug(f"_introspect ast_src:\n {pformat(ast_f)}")
    body_lines = src.split("\n")

    # For each of the functions in the body, look for interactions.
    fis = InspectFunction.inspect_class(
        ast_f, gctx, fun_module, body_lines, arg_ctx, fun_path, call_stack
    )
    # Cache the function interactions
    gctx.cached_fun_interactions[fis_key] = fis
    # cache the function interactions in the global context
    if _global_context is not None:
        dep_paths = sorted(_all_paths(fis))
        _global_context.cached_fun_calls[(fun_path, arg_ctx_hash)] = dep_paths
        # Find the id of each corresponding object
        obj_ids: List[Tuple[CanonicalPath, PythonId]] = []
        for dep_path in dep_paths:
            obj = ObjectRetrieval.retrieve_object_global(dep_path, gctx)
            obj_ids.append((dep_path, PythonId(id(obj))))
        # tup = tuple(obj_ids)
        # _logger.debug(f"cached_fun_interactions: {(fun_path, arg_ctx_hash, tup)}")
        # _global_context.cached_fun_interactions[(fun_path, arg_ctx_hash, tup)] = fis
    return fis


def _introspect(
    obj: Union[FunctionType, type],
    arg_ctx: FunctionArgContext,
    gctx: EvalMainContext,
    call_stack: List[CanonicalPath],
) -> FunctionInteractions:
    if isinstance(obj, FunctionType):
        return _introspect_fun(obj, arg_ctx, gctx, call_stack)
    if isinstance(obj, type):
        return _introspect_class(obj, arg_ctx, gctx, call_stack)
    raise KSException(
        f"Expected function or class, got object of type {type(obj)} instead: {obj}"
    )


def _introspect_fun(
    f: FunctionType,
    arg_ctx: FunctionArgContext,
    gctx: EvalMainContext,
    call_stack: List[CanonicalPath],
) -> FunctionInteractions:
    # Check if the function has already been evaluated.
    fun_path = function_path(f)
    arg_ctx_hash = FunctionArgContext.as_hashable(arg_ctx)
    # In most cases, lambda functions will change id's each time. Skipping for now.
    if (
        not is_lambda(f)
        and _global_context is not None
        and (fun_path, arg_ctx_hash) in _global_context.cached_fun_calls
    ):
        dep_paths = _global_context.cached_fun_calls[(fun_path, arg_ctx_hash)]
        # _logger.debug(
        #     f"{fun_path} in cache, evaluating if {len(dep_paths)} python objects have changed"
        # )
        ids: List[Tuple[CanonicalPath, PythonId]] = []
        for dep_path in dep_paths:
            obj = ObjectRetrieval.retrieve_object_global(dep_path, gctx)
            ids.append((dep_path, PythonId(id(obj))))
        tup = tuple(ids)
        if (fun_path, arg_ctx_hash, tup) in _global_context.cached_fun_interactions:
            # _logger.debug(
            #     f"{fun_path} in interaction cache, skipping analysis: {(fun_path, arg_ctx_hash, tup)}"
            # )
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
        h = dds_hash(src)
        # Have a stable name for the lambda function
        fun_path = CanonicalPath(fun_path._path[:-1] + [fun_path._path[-1] + h])
        fis_key = (fun_path, arg_ctx_hash)
        fis_ = gctx.cached_fun_interactions.get(fis_key)
        if fis_ is not None:
            return fis_
        # Not seen before, continue.
        # _logger.debug(f"_introspect: is_lambda: fun_path={fun_path} src={src}")
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
    # Register the path as a potential link to dependencies
    if fis.store_path:
        gctx.resolved_references[fis.store_path] = fis.fun_return_sig
    # cache the function interactions in the global context
    if not is_lambda(f) and _global_context is not None:
        dep_paths = sorted(_all_paths(fis))
        _global_context.cached_fun_calls[(fun_path, arg_ctx_hash)] = dep_paths
        # Find the id of each corresponding object
        obj_ids: List[Tuple[CanonicalPath, PythonId]] = []
        for dep_path in dep_paths:
            obj = ObjectRetrieval.retrieve_object_global(dep_path, gctx)
            obj_ids.append((dep_path, PythonId(id(obj))))
        # tup = tuple(obj_ids)
        # _logger.debug(f"cached_fun_interactions: {(fun_path, arg_ctx_hash, tup)}")
        # _global_context.cached_fun_interactions[(fun_path, arg_ctx_hash, tup)] = fis
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
        self.load_paths: List[DDSPath] = []

    def visit_Call(self, node: ast.Call) -> Any:
        # _logger.debug(f"visit: {node} {dir(node)} {pformat(node)}")
        function_body_hash = dds_hash(self._body_lines[: node.lineno + 1])
        # The list of all the previous interactions.
        # This enforces the concept that the current call depends on previous calls.
        function_inters_sig = dds_hash([fi.fun_return_sig for fi in self.inters])
        # Check the call for dds calls or sub_calls.
        fi_or_p = InspectFunction.inspect_call(
            node,
            self._gctx,
            self._start_mod,
            function_body_hash,
            self._args_hash,
            function_inters_sig,
            self._function_var_names,
            self._call_stack,
        )
        if fi_or_p is not None and isinstance(fi_or_p, FunctionInteractions):
            self.inters.append(fi_or_p)
        # str is the underlying type of a DDSPath
        if fi_or_p is not None and isinstance(fi_or_p, str):
            self.load_paths.append(fi_or_p)
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
        # _logger.debug(
        #     "ExternalVarsVisitor:visit_Name: id: %s local_dep_path:%s",
        #     node.id,
        #     local_dep_path,
        # )
        if not isinstance(node.ctx, ast.Load):
            # _logger.debug(
            #     "ExternalVarsVisitor:visit_Name: id: %s skipping ctx: %s",
            #     node.id,
            #     node.ctx,
            # )
            return
        # If it is a var that is already part of the function, do not introspect
        if len(local_dep_path.parts) == 1:
            v = str(local_dep_path)
            if v in self._local_vars:
                # _logger.debug(
                #     "ExternalVarsVisitor:visit_Name: id: %s skipping, in vars", node.id
                # )
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
            # _logger.debug("visit_Name: %s: skipping (unauthorized)", local_dep_path)
            self._rejected_paths.add(local_dep_path)
            return
        (obj, path) = res
        if isinstance(obj, FunctionType):
            # Modules and callables are tracked separately
            # _logger.debug(f"visit name %s: skipping (fun)", local_dep_path)
            self._rejected_paths.add(local_dep_path)
            return
        if isinstance(obj, ModuleType):
            # Modules and callables are tracked separately
            # TODO: this is not accurate, as a variable could be called in a submodule
            # _logger.debug(f"visit name %s: skipping (module)", local_dep_path)
            self._rejected_paths.add(local_dep_path)
            return
        if inspect.isclass(obj):
            # Classes are tracked separately
            # _logger.debug(f"visit name %s: skipping (class)", local_dep_path)
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


class InspectFunction(object):
    @classmethod
    def get_local_vars(
        cls, body: Sequence[ast.AST], arg_ctx: FunctionArgContext
    ) -> List[LocalVar]:
        lvars_v = LocalVarsVisitor(list(arg_ctx.named_args.keys()))
        for node in body:
            lvars_v.visit(node)
        lvars = sorted(list(lvars_v.vars))
        # _logger.debug(f"local vars: %s", lvars)
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
    def inspect_class(
        cls,
        node: ast.ClassDef,
        gctx: EvalMainContext,
        mod: ModuleType,
        class_body_lines: List[str],
        arg_ctx: FunctionArgContext,
        fun_path: CanonicalPath,
        call_stack: List[CanonicalPath],
    ) -> FunctionInteractions:
        # Look into the base classes first.
        # TODO: take into account the base classes

        # All the body is considered as a single big function for the purpose of
        # code structure: the function interactions are built for each element,
        # but the code lines are provided from the top of the function.

        method_fis: List[FunctionInteractions] = []
        for elem in node.body:
            if isinstance(elem, ast.FunctionDef):
                # Parsing the function call
                fis_ = cls.inspect_fun(
                    elem, gctx, mod, class_body_lines, arg_ctx, fun_path, call_stack
                )
                if fis_ is not None:
                    method_fis.append(fis_)
                    # _logger.debug(f"inspect_class: {fis_}")

        body_sig = dds_hash(class_body_lines)
        # All the sub-dependencies are handled with method introspections
        return_sig = dds_hash([body_sig] + [i.fun_return_sig for i in method_fis])

        return FunctionInteractions(
            arg_input=arg_ctx,
            fun_body_sig=body_sig,
            fun_return_sig=return_sig,
            external_deps=[],
            parsed_body=method_fis,
            store_path=None,  # No store path can be associated by default to a class
            fun_path=fun_path,
            indirect_deps=[],
        )

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
        # _logger.debug(f"inspect_fun: %s local_vars: %s", fun_path, local_vars)
        vdeps = ExternalVarsVisitor(mod, gctx, local_vars)
        for n in body:
            vdeps.visit(n)
        ext_deps = sorted(vdeps.vars.values(), key=lambda ed: ed.local_path)
        # _logger.debug(f"inspect_fun: ext_deps: %s", ext_deps)
        arg_keys = FunctionArgContext.relevant_keys(arg_ctx)
        sig_list: List[Any] = ([(ed.local_path, ed.sig) for ed in ext_deps] + arg_keys)  # type: ignore
        input_sig = dds_hash(sig_list)
        calls_v = IntroVisitor(
            mod, gctx, function_body_lines, input_sig, local_vars, call_stack
        )
        for n in body:
            calls_v.visit(n)
        body_sig = dds_hash(function_body_lines)
        # Remove duplicates but keep the order in the list of paths:
        indirect_deps = _no_dups(calls_v.load_paths)

        def fetch(dep: DDSPath) -> PyHash:
            key = gctx.resolved_references.get(dep)
            assert (
                key is not None
            ), f"Missing dep {dep} for {fun_path}: {call_stack} {gctx.resolved_references}"
            return key

        indirect_deps_sigs = [fetch(dep) for dep in indirect_deps]

        return_sig = dds_hash(
            [input_sig, body_sig]
            + [i.fun_return_sig for i in calls_v.inters]
            + indirect_deps_sigs
        )

        # Look at the annotations to see if there is a reference to a data_function
        if isinstance(node, ast.FunctionDef):
            store_path = cls._path_annotation(node, mod, gctx)
        else:
            store_path = None
        # _logger.debug(f"inspect_fun: path from annotation: %s", store_path)

        return FunctionInteractions(
            arg_input=arg_ctx,
            fun_body_sig=body_sig,
            fun_return_sig=return_sig,
            external_deps=ext_deps,
            parsed_body=calls_v.inters,
            store_path=store_path,
            fun_path=fun_path,
            indirect_deps=indirect_deps,
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
                # _logger.debug(f"_path_annotation: local_path: %s", local_path)
                z = ObjectRetrieval.retrieve_object(local_path, mod, gctx)
                if z is None:
                    # _logger.debug(
                    #     f"_path_annotation: local_path: %s is rejected", local_path
                    # )
                    return None
                caller_fun, caller_fun_path = z
                # _logger.debug(f"_path_annotation: caller_fun_path: %s", caller_fun_path)
                if caller_fun_path == CanonicalPath(
                    ["dds", "_annotations", "dds_function"]
                ) or caller_fun_path == CanonicalPath(
                    ["dds", "_annotations", "data_function"]
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
    ) -> Union[FunctionInteractions, DDSPath, None]:
        # _logger.debug(f"Inspect call:\n %s", pformat(node))

        local_path = LocalDepPath(PurePosixPath("/".join(_function_name(node.func))))
        # _logger.debug(f"inspect_call: local_path: %s", local_path)
        # We may do sub-method calls on an object -> filter out based on the name of the object
        if str(local_path.parts[0]) in var_names:
            # _logger.debug(
            #     f"inspect_call: local_path: %s is rejected (head in vars)", local_path
            # )
            return None

        # _logger.debug(f"inspect_call:local_path:{local_path} mod:{mod}\n %s", pformat(node))
        z = ObjectRetrieval.retrieve_object(local_path, mod, gctx)
        # _logger.debug(f"inspect_call:local_path:{local_path} mod:{mod} z:{z}")
        if z is None:
            # _logger.debug(f"inspect_call: local_path: %s is rejected", local_path)
            return None
        caller_fun, caller_fun_path = z
        if not isinstance(caller_fun, FunctionType) and not inspect.isclass(caller_fun):
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
            if function_inter_hash is None:
                raise KSException("not implemented: function_inter_hash")
            context_sig = dds_hash(
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
        if caller_fun_path == CanonicalPath(["dds", "load"]):
            # Evaluation call: get the argument and returns the function interaction for this call.
            if len(node.args) != 1:
                raise KSException(f"Wrong number of args: expected 1, got {node.args}")
            store_path = cls._retrieve_store_path(node.args[0], mod, gctx)
            _logger.debug(f"inspect_call:eval: store_path: {store_path}")
            return store_path

        if caller_fun_path == CanonicalPath(["dds", "eval"]):
            raise NotImplementedError("eval")

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
        context_sig = dds_hash(
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
        # _logger.debug(
        #     f"Keep: store_path_symbol: %s %s",
        #     store_path_symbol,
        #     type(store_path_symbol),
        # )
        store_path_local_path = LocalDepPath(PurePosixPath(store_path_symbol))
        # Retrieve the store path value and the called function
        store_z = ObjectRetrieval.retrieve_object(store_path_local_path, mod, gctx)
        if not store_z:
            # Not sure what to do yet in this case.
            raise NotImplementedError(f"Invalid store_z: {store_path_local_path} {mod}")
        store_path, _ = store_z
        return DDSPathUtils.create(store_path)


def _no_dups(paths: List[DDSPath]) -> List[DDSPath]:
    s = set()
    res = []
    for p in paths:
        if p not in s:
            s.add(p)
            res.append(p)
    return res


_accepted_packages: Set[Package] = {
    Package("dds"),
    Package("__main__"),
    Package("__global__"),
}


def accept_module(module: Union[str, ModuleType]) -> None:
    global _accepted_packages
    if isinstance(module, ModuleType):
        module = module.__name__
    assert isinstance(module, str), (module, type(module))
    _accepted_packages.add(Package(module))
