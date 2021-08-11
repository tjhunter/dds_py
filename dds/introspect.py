import keyword
import builtins
import ast
import inspect
from collections import OrderedDict
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

from ._eval_ctx import (
    EvalMainContext,
    Package,
    ExternalObject,
    AuthorizedObject,
    ObjectRetrievalType,
)
from ._global_ctx import _global_context, PythonId
from ._lambda_funs import is_lambda, inspect_lambda_condition
from ._print_ast import pformat
from ._retrieve_objects import ObjectRetrieval, function_path
from .fun_args import dds_hash_commut, HashKey as HK, dds_hash, get_arg_ctx_ast
from .structures import (
    PyHash,
    FunctionArgContext,
    DDSPath,
    FunctionInteractions,
    DDSException,
    CanonicalPath,
    ExternalDep,
    LocalDepPath,
    DDSErrorCode,
)
from .structures_utils import DDSPathUtils, CanonicalPathUtils

_logger = logging.getLogger(__name__)
_hash_key_body_sig = HK("body_sig")
_hash_key_fun_body = HK("function_body_hash")
_hash_key_fun_input = HK("function_input_hash")
_hash_key_fun_inter = HK("function_inter_hash")


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


# All the builtins that should not be looked at.
# See https://stackoverflow.com/questions/50112359/how-to-get-the-list-of-all-built-in-functions-in-python
python_builtin_names: Set[str] = set(
    [name for name, _ in vars(builtins).items()] + list(keyword.kwlist)
)


def get_assign_targets(node: Any) -> List[LocalVar]:
    """
    Returns the name of assignment targets from expressions.
    """
    # The type of node should be Union[ast.Assign, ast.Tuple, ast.Attribute, ast.Name, ast.Call]
    if isinstance(node, ast.Name):
        return [LocalVar(node.id)]
    if isinstance(node, ast.Assign):
        return [lv for target in node.targets for lv in get_assign_targets(target)]
    if isinstance(node, ast.Tuple):
        return [lv for elt in node.elts for lv in get_assign_targets(elt)]
    if isinstance(node, ast.Attribute):
        # Just get the top id, not the full paths
        return get_assign_targets(node.value)
    if isinstance(node, ast.Call):
        return get_assign_targets(node.func)
    _logger.warning(
        "Expected assignment object to be of type Tuple or Name in AST, got %s: %s",
        type(node),
        node,
    )
    return []
    # TODO: configurable behaviour about such issues. It is indicative of an issue in most cases.
    # raise DDSException(
    #     f"Expected assignment object to be of type Tuple or Name in AST, got {type(node)}: {pformat(node)}",
    #     error_code=DDSErrorCode.UNKNOWN_AST_NODE,
    # )


def _fis_to_siglist(fis: List[FunctionInteractions]) -> List[Tuple[HK, PyHash]]:
    # Including an index in the case of the function interactions.
    # There may be multiple function calls to the same function, and the current
    # hashing algorithm does not allow duplicates of the same elements (they get xor'd out).
    # The penalty to pay for indexing is that adding or removing function calls may
    # introduce a reindexing and hence a recomputation of a few hashes.
    # Since this is limited to a single function, this is not considered to have a large impact.
    return [(HK(f"fun_dep_{idx}"), i.fun_return_sig) for (idx, i) in enumerate(fis)]


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
        raise DDSException(
            f"Could not find module: class:{c} module: {fun_module}",
            DDSErrorCode.MODULE_NOT_FOUND,
        )
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
    raise DDSException(
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
        raise DDSException(
            f"Could not find module: f; {f} module: {fun_module}",
            DDSErrorCode.MODULE_NOT_FOUND,
        )
    # _logger.debug(f"_introspect: {f}: fun_path={fun_path} fun_module={fun_module}")
    ast_f: Union[ast.Lambda, ast.FunctionDef]
    if is_lambda(f):
        # _logger.debug(f"_introspect: is_lambda: {f}")
        src = inspect.getsource(f)
        h = dds_hash(src)
        # Have a stable name for the lambda function
        fun_path = CanonicalPath(
            fun_path._path.parent.joinpath(fun_path._path.stem + h)
        )
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
        function_input_sig: PyHash,
        function_var_names: Set[LocalVar],
        call_stack: List[CanonicalPath],
        fun_path: CanonicalPath,
    ):
        # TODO: start_mod is in the global context
        current_fun_name = LocalVar(CanonicalPathUtils.last(fun_path))
        self._start_mod = start_mod
        self._gctx = gctx
        self._function_var_names = set(function_var_names)
        self._body_lines = function_body_lines
        self._input_sig = function_input_sig
        self._call_stack = call_stack
        self._store_names: Set[LocalVar] = {current_fun_name}
        self.inters: List[FunctionInteractions] = []
        self.load_paths: List[DDSPath] = []

    def visit_Call(self, node: ast.Call) -> Any:
        # _logger.debug(f"visit: {node} {dir(node)} {pformat(node)}")
        # This is a bit brute-force (not working for multi-line function calls)
        # but it should be good enough in practice for most cases.
        # TODO: refine it based of the nested parse tree?
        function_body_hash = dds_hash(self._body_lines[: node.lineno + 1])
        # The list of all the previous interactions.
        # This enforces the concept that the current call depends on previous calls.
        function_inters_sig: Optional[PyHash] = (
            dds_hash_commut(_fis_to_siglist(self.inters))
        )
        # Check the call for dds calls or sub_calls.
        fi_or_p = InspectFunction.inspect_call(
            node,
            self._gctx,
            self._start_mod,
            function_body_hash,
            self._input_sig,
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

    def visit_Assign(self, node: ast.Assign) -> Any:
        targets = get_assign_targets(node)
        if targets:
            self._store_names.update(targets)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        # Look at names of variables that are names imported in the context of the function (in the module) but that are
        # not builtins.
        # This neglects the case of shadowing within the function: if the function has a variable that has the same name
        # as another function, then a mismatch will happen.
        if (
            node.id in self._start_mod.__dict__
            and node.id not in python_builtin_names
            and LocalVar(node.id) not in self._function_var_names
            and LocalVar(node.id) not in self._store_names
        ):
            # Quick check that it is indeed a function or a module:
            # TODO: add a test for modules
            obj = self._start_mod.__dict__[node.id]
            self._store_names.add(LocalVar(node.id))
            # Just handling functions, not modules.
            # Handling modules is more complicated (requires tracing the full call) and it can be easily worked around
            # by directly importing the function.
            if isinstance(obj, (FunctionType,)):
                # Building a fake AST node to handle functions called without arguments. They may not
                _logger.debug(f"visit_name: {node} {pformat(node)} {self._store_names}")
                # No arg given
                call_node = ast.Call(
                    func=node, args=[], keywords=[], starargs=None, kwargs=None
                )
                # This is a bit brute-force (not working for multi-line function calls)
                # but it should be good enough in practice for most cases.
                # TODO: refine it based of the nested parse tree?
                function_body_hash = dds_hash(self._body_lines[: node.lineno + 1])
                # The list of all the previous interactions.
                # This enforces the concept that the current call depends on previous calls.
                function_inters_sig: Optional[PyHash] = (
                    dds_hash_commut(_fis_to_siglist(self.inters))
                )
                # Check the call for dds calls or sub_calls.
                fi_or_p = InspectFunction.inspect_call(
                    call_node,
                    self._gctx,
                    self._start_mod,
                    function_body_hash,
                    self._input_sig,
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

    def visit_Name(self, node: ast.Name, debug: bool = False) -> Any:
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
        res: ObjectRetrievalType = ObjectRetrieval.retrieve_object(
            local_dep_path, self._start_mod, self._gctx
        )
        if debug:
            _logger.debug(f"visit_Name: {local_dep_path} {self._start_mod} -> {res}")
        if res is None:
            # Nothing to do, it is not interesting.
            if debug:
                _logger.debug("visit_Name: %s: skipping (unauthorized)", local_dep_path)
            self._rejected_paths.add(local_dep_path)
            return
        elif isinstance(res, ExternalObject):
            # External object. Should be tracked at name level
            self.vars[local_dep_path] = ExternalDep(
                local_path=local_dep_path, path=res.resolved_path, sig=None
            )
        else:
            assert isinstance(res, AuthorizedObject)
            (obj, path) = (res.object_val, res.resolved_path)
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

    def __init__(self, existing_vars: List[str], fun_path: CanonicalPath):
        self.vars: Set[str] = set(existing_vars)
        self._fun_path = fun_path

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        raise DDSException(
            f"Function {self._fun_path} rejected by DDS because it includes a call to an "
            f"async function. You cannot use async function with DDS",
            DDSErrorCode.CONSTRUCT_NOT_SUPPORTED,
        )

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
        cls,
        body: Sequence[ast.AST],
        arg_ctx: FunctionArgContext,
        fun_path: CanonicalPath,
    ) -> List[LocalVar]:
        lvars_v = LocalVarsVisitor(list(arg_ctx.named_args.keys()), fun_path)
        for node in body:
            lvars_v.visit(node)
        lvars = sorted(list(lvars_v.vars))
        # _logger.debug(f"local vars: %s", lvars)
        return [LocalVar(s) for s in lvars]

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

        return_sig = dds_hash_commut(
            [(_hash_key_body_sig, body_sig)] + _fis_to_siglist(method_fis)
        )
        assert return_sig is not None

        return FunctionInteractions(
            arg_input=arg_ctx,
            fun_body_sig=body_sig,
            fun_return_sig=return_sig,
            # The dependencies are for now all in the function bodies
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
        debug: bool = False,
    ) -> FunctionInteractions:
        body: Sequence[ast.AST]
        if isinstance(node, ast.FunctionDef):
            body = node.body
        elif isinstance(node, ast.Lambda):
            body = [node.body]
        else:
            raise DDSException(f"unknown ast node {type(node)}")
        local_vars = set(cls.get_local_vars(body, arg_ctx, fun_path))
        # _logger.debug(f"inspect_fun: %s local_vars: %s", fun_path, local_vars)
        vdeps = ExternalVarsVisitor(mod, gctx, local_vars)
        for n in body:
            vdeps.visit(n)
        ext_deps = sorted(vdeps.vars.values(), key=lambda ed: ed.local_path)
        if debug:
            _logger.debug(f"inspect_fun: ext_deps: %s", ext_deps)

        # The variables that are hashable: authorized variables outside of the function
        sig_variables: List[Tuple[LocalDepPath, PyHash]] = [
            (ed.local_path, ed.sig) for ed in ext_deps if ed.sig is not None
        ]
        sig_variables_distinct: Dict[LocalDepPath, PyHash] = dict(sig_variables)
        # External dependencies
        ext_deps_vars: Dict[LocalDepPath, CanonicalPath] = dict(
            [(ed.local_path, ed.path) for ed in ext_deps if ed.sig is None]
        )
        input_sig = _build_return_sig(
            # The body signature will depend on the exact location of the function calls
            body_sig=None,
            arg_ctx=arg_ctx,
            # TODO: does it also need the indirect deps here?
            indirect_deps={},
            # Sub function interactions are processed one at
            # a time inside the function.
            sub_fis=[],
            ext_deps=ext_deps_vars,
            ext_vars=sig_variables_distinct,
        ) or dds_hash([])
        calls_v = IntroVisitor(
            mod, gctx, function_body_lines, input_sig, local_vars, call_stack, fun_path
        )
        for n in body:
            calls_v.visit(n)
        body_sig = dds_hash(function_body_lines)
        # Remove duplicates but keep the order in the list of paths:
        indirect_dep = _no_dups(calls_v.load_paths)

        def fetch(dep: DDSPath) -> PyHash:
            key = gctx.resolved_references.get(dep)
            assert (
                key is not None
            ), f"Missing dep {dep} for {fun_path}: {call_stack} {gctx.resolved_references}"
            return key

        indirect_deps_sigs = dict([(dep, fetch(dep)) for dep in indirect_dep])

        # Look at the annotations to see if there is a reference to a data_function
        if isinstance(node, ast.FunctionDef):
            store_path = cls._path_annotation(node, mod, gctx)
        else:
            store_path = None
        # _logger.debug(f"inspect_fun: path from annotation: %s", store_path)
        return_sig = _build_return_sig(
            body_sig=body_sig,
            arg_ctx=arg_ctx,
            indirect_deps=indirect_deps_sigs,
            sub_fis=calls_v.inters,
            ext_deps=ext_deps_vars,
            ext_vars=sig_variables_distinct,
        )
        assert return_sig is not None

        return FunctionInteractions(
            arg_input=arg_ctx,
            fun_body_sig=body_sig,
            fun_return_sig=return_sig,
            external_deps=ext_deps,
            parsed_body=calls_v.inters,
            store_path=store_path,
            fun_path=fun_path,
            indirect_deps=indirect_dep,
        )

    @classmethod
    def _path_annotation(
        cls,
        node: ast.FunctionDef,
        mod: ModuleType,
        gctx: EvalMainContext,
        debug: bool = False,
    ) -> Optional[DDSPath]:
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call):
                local_path = LocalDepPath(
                    PurePosixPath("/".join(_function_name(dec.func)))
                )
                # _logger.debug(f"_path_annotation: local_path: %s", local_path)
                z: ObjectRetrievalType = ObjectRetrieval.retrieve_object(
                    local_path, mod, gctx
                )
                if debug:
                    _logger.debug(f"z: {z}")
                if z is None or isinstance(z, ExternalObject):
                    if debug:
                        _logger.debug(
                            f"_path_annotation: local_path: %s is rejected", local_path
                        )
                    return None
                assert isinstance(z, AuthorizedObject)
                caller_fun, caller_fun_path = (z.object_val, z.resolved_path)
                # _logger.debug(f"_path_annotation: caller_fun_path: %s", caller_fun_path)
                if caller_fun_path == CanonicalPathUtils.from_list(
                    ["dds", "_annotations", "dds_function"]
                ) or caller_fun_path == CanonicalPathUtils.from_list(
                    ["dds", "_annotations", "data_function"]
                ):
                    if len(dec.args) != 1:
                        raise DDSException(
                            f"Wrong number of arguments for decorator: {pformat(dec)}"
                        )
                    store_path = cls._retrieve_store_path(
                        dec.args[0], mod, gctx, local_path
                    )
                    return store_path
        return None

    @classmethod
    def inspect_call(
        cls,
        node: ast.Call,
        gctx: EvalMainContext,
        mod: ModuleType,
        function_body_hash: PyHash,
        function_input_sig: PyHash,
        function_inter_hash: Optional[PyHash],
        var_names: Set[LocalVar],
        call_stack: List[CanonicalPath],
        debug: bool = False,
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
        z: ObjectRetrievalType = ObjectRetrieval.retrieve_object(local_path, mod, gctx)
        if debug:
            _logger.debug(f"inspect_call:local_path:{local_path} mod:{mod} z:{z}")
        if z is None or isinstance(z, ExternalObject):
            if debug:
                _logger.debug(f"inspect_call: local_path: %s is rejected", local_path)
            return None
        assert isinstance(z, AuthorizedObject)
        caller_fun, caller_fun_path = (z.object_val, z.resolved_path)
        if not isinstance(caller_fun, FunctionType) and not inspect.isclass(caller_fun):
            raise DDSException(
                f"Expected FunctionType or class for {caller_fun_path}, got {type(caller_fun)}",
                DDSErrorCode.UNSUPPORTED_CALLABLE_TYPE,
            )

        # Check if this is a call we should do something about.

        if caller_fun_path in call_stack:
            # Recursive calls are not supported currently.
            raise DDSException(
                f"Detected circular function calls or (co-)recursive calls."
                f"This is currently not supported. Change your code to split the "
                f"recursive section into a separate function. "
                f"Function: {caller_fun_path}"
                f"Call stack: {' '.join([str(p) for p in call_stack])}",
                DDSErrorCode.CIRCULAR_CALL,
            )

        elif caller_fun_path == CanonicalPathUtils.from_list(["dds", "load"]):
            # Evaluation call: get the argument and returns the function interaction for this call.
            if len(node.args) != 1:
                raise DDSException(f"Wrong number of args: expected 1, got {node.args}")
            store_path = cls._retrieve_store_path(node.args[0], mod, gctx, local_path)
            _logger.debug(f"inspect_call:eval: store_path: {store_path}")
            return store_path

        elif caller_fun_path == CanonicalPathUtils.from_list(["dds", "eval"]):
            raise DDSException(
                f"Cannot process {local_path}: this function is calling dds.eval, which"
                f" is not allowed inside other eval calls. Suggestion: remove the "
                f"call to dds.eval inside {local_path}",
                DDSErrorCode.EVAL_IN_EVAL,
            )

        # 2 cases left: normal call or kept call (normal call wrapped in ddd.keep())

        # The context signature that will be needed for these calls.
        context_sig = dds_hash_commut(
            [
                (_hash_key_body_sig, function_body_hash),
                (_hash_key_fun_input, function_input_sig),
            ]
            + (
                [(_hash_key_fun_inter, function_inter_hash)]
                if function_inter_hash is not None
                else []
            )
        )

        if caller_fun_path == CanonicalPathUtils.from_list(["dds", "keep"]):
            # Call to the keep function:
            # - bring the path
            # - bring the callee
            # - parse the arguments
            # - introspect the callee
            if len(node.args) < 2:
                raise DDSException(
                    f"Wrong number of args: expected 2+, got {node.args}"
                )
            store_path = cls._retrieve_store_path(node.args[0], mod, gctx, local_path)
            called_path_ast = node.args[1]
            if isinstance(called_path_ast, ast.Name):
                called_path_symbol = node.args[1].id  # type: ignore
            else:
                raise DDSException(
                    f"Introspection of {local_path} failed: cannot use nested callables of"
                    f" type {called_path_ast}. Only "
                    f"regular function names are allowed for now. Suggestion: if you are "
                    f"using a complex callable such as a method, wrap it inside a top-level "
                    f"function.",
                    DDSErrorCode.UNSUPPORTED_CALLABLE_TYPE,
                )
            called_local_path = LocalDepPath(PurePosixPath(called_path_symbol))
            called_z: ObjectRetrievalType = ObjectRetrieval.retrieve_object(
                called_local_path, mod, gctx
            )
            if debug:
                _logger.debug(f"called_z: {called_z}")
            if not called_z or isinstance(called_z, ExternalObject):
                # Not sure what to do yet in this case.
                raise DDSException(
                    f"Introspection of {local_path} failed: cannot access called function"
                    f" {called_local_path}. The function {called_local_path} was expected "
                    f"to be found in module {mod}, but could not be retrieved. The usual reason is"
                    f"that that this object is not a regular top-level function. "
                    f"Suggestion: ensure that this function is a top-level function.",
                    DDSErrorCode.UNSUPPORTED_CALLABLE_TYPE,
                )
            assert isinstance(called_z, AuthorizedObject)
            called_fun, call_fun_path = called_z.object_val, called_z.resolved_path
            if call_fun_path in call_stack:
                raise DDSException(
                    f"Detected circular function calls or (co-)recursive calls."
                    f"This is currently not supported. Change your code to split the "
                    f"recursive section into a separate function. "
                    f"Function: {call_fun_path}"
                    f"Call stack: {' '.join([str(p) for p in call_stack])}",
                    DDSErrorCode.CIRCULAR_CALL,
                )
            new_call_stack = call_stack + [call_fun_path]
            # TODO: this is an approximation as all the arguments may be keyworded.
            # This assumes that only the function's normal arguments are going to be keyworded.
            kwargs = OrderedDict([(n.arg, n.value) for n in node.keywords])
            # For now, accept the constant arguments. This is enough for some basic objects.
            arg_ctx = FunctionArgContext(
                named_args=get_arg_ctx_ast(called_fun, node.args[2:], kwargs),  # type: ignore
                inner_call_key=context_sig,
            )
            inner_intro = _introspect(called_fun, arg_ctx, gctx, new_call_stack)
            inner_intro = inner_intro._replace(store_path=store_path)
            return inner_intro

        # Normal function call.
        # Just introspect the function call.
        # For now, do not look carefully at the arguments, just parse the arguments of
        # the functions.
        # TODO: add more arguments if we can parse constant arguments
        arg_ctx = FunctionArgContext(
            named_args=get_arg_ctx_ast(caller_fun, [], OrderedDict()),
            inner_call_key=context_sig,
        )
        new_call_stack = call_stack + [caller_fun_path]
        return _introspect(caller_fun, arg_ctx, gctx, new_call_stack)

    @classmethod
    def _retrieve_store_path(
        cls,
        local_path_node: ast.AST,
        mod: ModuleType,
        gctx: EvalMainContext,
        local_path: LocalDepPath,
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
            raise DDSException(
                f"Invalid path type: {type(local_path_node)} encountered in {local_path} (module {mod}). "
                f"Suggestion: the path to a node can only be a string, a Path object or a "
                f"variable name that points to a string or Path object. "
                f"See the documentation for more details."
                f"Full parse tree: {pformat(local_path_node)}",
                DDSErrorCode.STORE_PATH_NOT_SUPPORTED,
            )
        # _logger.debug(
        #     f"Keep: store_path_symbol: %s %s",
        #     store_path_symbol,
        #     type(store_path_symbol),
        # )
        store_path_local_path = LocalDepPath(PurePosixPath(store_path_symbol))
        # Retrieve the store path value and the called function
        store_z: ObjectRetrievalType = ObjectRetrieval.retrieve_object(
            store_path_local_path, mod, gctx
        )
        if not store_z or isinstance(store_z, ExternalObject):
            # Not sure what to do yet in this case.
            raise DDSException(
                f"Invalid path {store_path_local_path} encountered in {local_path} (module {mod}). "
                f"Suggestion: the path to a node can only be a string, a Path object or a "
                f"variable name that points to a string or Path object. "
                f"See the documentation for more details."
                f"Full parse tree: {pformat(local_path_node)}",
                DDSErrorCode.STORE_PATH_NOT_SUPPORTED,
            )
        store_path = store_z.object_val
        return DDSPathUtils.create(store_path)


def _no_dups(paths: List[DDSPath]) -> List[DDSPath]:
    """
    Removes duplicate while keeping the order.
    """
    s = set()
    res = []
    for p in paths:
        if p not in s:
            s.add(p)
            res.append(p)
    return res


def _build_return_sig(
    body_sig: Optional[PyHash],
    arg_ctx: FunctionArgContext,
    indirect_deps: Dict[DDSPath, PyHash],
    sub_fis: List[FunctionInteractions],
    ext_deps: Dict[LocalDepPath, CanonicalPath],
    ext_vars: Dict[LocalDepPath, PyHash],
) -> Optional[PyHash]:
    """
    Builds a key from the content of a function.

    This function builds an cryptographically secure hash of the content of
    a function by combining the following elements of the function:
    - body_sig -> the body signature (hashing the text of the body)
    - the map of the indirect dependencies:
        dep_{path} -> hash
    - the list of all sub-function calls
    - the input signature of the
    - the external dependencies:
        map of symbol name -> fully qualified path in the module hierarchy
    - the external variables:
        map of symbol name -> hash of the content of the function
    - the arguments of the function:
        map of arg_name -> signature of the input

    These elements are combined using the commutative hash function.
    """
    body: List[Tuple[HK, PyHash]] = [] if body_sig is None else [
        (_hash_key_body_sig, body_sig)
    ]
    arg: List[Tuple[HK, PyHash]]
    if any(sig is None for sig in arg_ctx.named_args.values()):
        assert arg_ctx.inner_call_key is not None, f"{arg_ctx} {body_sig}"
        arg = [(HK("arg_context"), arg_ctx.inner_call_key)]
    else:
        arg = [
            (HK(f"arg_{name}"), cast(PyHash, sig))
            for (name, sig) in arg_ctx.named_args.items()
        ]
    all_pairs: List[Tuple[HK, PyHash]] = (
        body
        + arg
        + [(HK(f"dep_{dep}"), sig_) for (dep, sig_) in indirect_deps.items()]
        + _fis_to_siglist(sub_fis)
        + [
            (HK(f"ext_dep_{local_path}"), dds_hash(sig))
            for (local_path, sig) in ext_deps.items()
        ]
        + [
            (HK(f"ext_variable_{local_path}"), sig)
            for (local_path, sig) in ext_vars.items()
        ]
    )
    if not all_pairs:
        return None
    return dds_hash_commut(all_pairs)


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
