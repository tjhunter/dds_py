import ast
import inspect
import logging
from collections import OrderedDict
from pathlib import PurePosixPath
from types import ModuleType, FunctionType
from typing import (
    cast,
    Callable,
    Any,
    Set,
    Union,
    List,
    Sequence,
)

from ._eval_ctx import (
    EvalMainContext,
    ExternalObject,
    AuthorizedObject,
    ObjectRetrievalType,
)
from ._lambda_funs import is_lambda, inspect_lambda_condition
from ._print_ast import pformat
from ._retrieve_objects import ObjectRetrieval, function_path
from .fun_args import dds_hash, get_arg_list
from .introspect import (
    InspectFunction,
    ExternalVarsVisitor,
    LocalVar,
    _function_name,
    get_assign_targets,
    python_builtin_names,
)
from .structures import (
    FunctionArgContext,
    DDSPath,
    DDSException,
    CanonicalPath,
    LocalDepPath,
    FunctionIndirectInteractions,
    DDSErrorCode,
)
from .structures_utils import CanonicalPathUtils as CPU

_logger = logging.getLogger(__name__)


def introspect_indirect(
    f: Callable[[Any], Any], eval_ctx: EvalMainContext
) -> FunctionIndirectInteractions:
    fun: FunctionType = cast(FunctionType, f)
    return _introspect(fun, eval_ctx, call_stack=[])


def _introspect(
    obj: Union[FunctionType, type],
    gctx: EvalMainContext,
    call_stack: List[CanonicalPath],
) -> FunctionIndirectInteractions:
    if isinstance(obj, FunctionType):
        return _introspect_fun(obj, gctx, call_stack)
    if isinstance(obj, type):
        return _introspect_class(obj, gctx, call_stack)
    raise DDSException(
        f"Expected function or class, got object of type {type(obj)} instead: {obj}"
    )


def _introspect_class(
    c: type, gctx: EvalMainContext, call_stack: List[CanonicalPath],
) -> FunctionIndirectInteractions:
    # Check if the function has already been evaluated.
    fun_path = function_path(c)

    # TODO: add to the global interactions cache

    fun_module = inspect.getmodule(c)
    if fun_module is None:
        msg = (
            f"Could not find the module for class {c}. "
            f"This usually happens when the class is defined at run time, or when the class"
            f"is moved between modules. Suggestion: rewrite your code to use functions instead,"
            f"or do not accept the module for this class (see dds.accept_module)."
        )
        raise DDSException(msg, DDSErrorCode.MODULE_NOT_FOUND)
    # _logger.debug(f"_introspect: {f}: fun_path={fun_path} fun_module={fun_module}")
    fiis_ = gctx.cached_indirect_interactions.get(fun_path)
    if fiis_ is not None:
        return fiis_
    src = inspect.getsource(c)
    # _logger.debug(f"Starting _introspect_class: {c}: src={src}")
    ast_src = ast.parse(src)
    ast_f: ast.ClassDef = ast_src.body[0]  # type: ignore
    assert isinstance(ast_f, ast.ClassDef), type(ast_f)
    _logger.debug(f"_introspect ast_src:\n {pformat(ast_f)}")

    # For each of the functions in the body, look for interactions.
    fiis = InspectFunctionIndirect.inspect_class(
        ast_f, gctx, fun_module, fun_path, call_stack
    )
    # Cache the function interactions
    gctx.cached_indirect_interactions[fun_path] = fiis
    return fiis


def _introspect_fun(
    f: FunctionType, gctx: EvalMainContext, call_stack: List[CanonicalPath],
) -> FunctionIndirectInteractions:
    # Check if the function has already been evaluated.
    fun_path = function_path(f)

    fun_module = inspect.getmodule(f)
    if fun_module is None:
        msg = (
            f"Could not find the module for function {f} located at {fun_path}. "
            f"This usually happens when the function is defined at run time, or when the function"
            f"is moved between modules. Suggestion: rewrite your code to use functions instead, "
            f"or do not accept the module for this class (see dds.accept_module)."
        )
        raise DDSException(msg, DDSErrorCode.MODULE_NOT_FOUND)
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
        fiis_ = gctx.cached_indirect_interactions.get(fun_path)
        if fiis_ is not None:
            return fiis_
        # Not seen before, continue.
        # _logger.debug(f"_introspect: is_lambda: fun_path={fun_path} src={src}")
        ast_f = inspect_lambda_condition(f)
        assert isinstance(ast_f, ast.Lambda), type(ast_f)
        # _logger.debug(f"_introspect: is_lambda: {ast_f}")
    else:
        fiis_ = gctx.cached_indirect_interactions.get(fun_path)
        if fiis_ is not None:
            return fiis_
        src = inspect.getsource(f)
        # _logger.debug(f"Starting _introspect: {f}: src={src}")
        ast_src = ast.parse(src)
        ast_f = ast_src.body[0]  # type: ignore
        assert isinstance(ast_f, ast.FunctionDef), type(ast_f)
        # _logger.debug(f"_introspect ast_src:\n {pformat(ast_f)}")

    # The names of the arguments, which are considered as variable names.
    arg_names = [LocalVar(v) for v in get_arg_list(f)]
    fiis = InspectFunctionIndirect.inspect_fun(
        ast_f, gctx, fun_module, fun_path, arg_names, call_stack
    )
    # Cache the function interactions
    gctx.cached_indirect_interactions[fun_path] = fiis
    return fiis


class InspectFunctionIndirect(object):
    @classmethod
    def inspect_fun(
        cls,
        node: Union[ast.FunctionDef, ast.Lambda],
        gctx: EvalMainContext,
        mod: ModuleType,
        fun_path: CanonicalPath,
        arg_names: List[LocalVar],
        call_stack: List[CanonicalPath],
    ) -> FunctionIndirectInteractions:
        body: Sequence[ast.AST]
        if isinstance(node, ast.FunctionDef):
            body = node.body
        elif isinstance(node, ast.Lambda):
            body = [node.body]
        else:
            raise DDSException(
                f"unknown ast node {type(node)}", DDSErrorCode.UNKNOWN_AST_NODE
            )
        dummy_arg_ctx = FunctionArgContext(OrderedDict(), None)
        local_vars = set(
            InspectFunction.get_local_vars(body, dummy_arg_ctx, fun_path) + arg_names
        )
        # _logger.debug(f"inspect_fun: %s local_vars: %s", fun_path, local_vars)
        vdeps = ExternalVarsVisitor(mod, gctx, local_vars)
        for n in body:
            vdeps.visit(n)
        # _logger.debug(f"inspect_fun: %s ExternalVarsVisitor: %s", fun_path, vdeps.vars)
        calls_v = IntroVisitorIndirect(mod, gctx, local_vars, call_stack, fun_path)
        for n in body:
            calls_v.visit(n)

        # Look at the annotations to see if there is a reference to a data_function
        if isinstance(node, ast.FunctionDef):
            store_path = InspectFunction._path_annotation(node, mod, gctx)
        else:
            store_path = None
        # _logger.debug(f"inspect_fun: path from annotation: %s", store_path)

        return FunctionIndirectInteractions(
            store_path=store_path, fun_path=fun_path, indirect_deps=calls_v.results,
        )

    @classmethod
    def inspect_class(
        cls,
        node: ast.ClassDef,
        gctx: EvalMainContext,
        mod: ModuleType,
        fun_path: CanonicalPath,
        call_stack: List[CanonicalPath],
    ) -> FunctionIndirectInteractions:
        # Look into the base classes first.
        # TODO: take into account the base classes

        # All the body is considered as a single big function for the purpose of
        # code structure: the function interactions are built for each element,
        # but the code lines are provided from the top of the function.

        method_fis: List[FunctionIndirectInteractions] = []
        for elem in node.body:
            if isinstance(elem, ast.FunctionDef):
                # Parsing the function call
                # TODO: this does not include the names of the args. Class parsing will break if args have the same names as packages.
                fis_ = cls.inspect_fun(elem, gctx, mod, fun_path, [], call_stack)
                if fis_ is not None:
                    method_fis.append(fis_)
                    # _logger.debug(f"inspect_class: {fis_}")

        return FunctionIndirectInteractions(
            store_path=None,  # No store path can be associated by default to a class
            fun_path=fun_path,
            indirect_deps=method_fis,
        )

    @classmethod
    def inspect_call(
        cls,
        node: ast.Call,
        gctx: EvalMainContext,
        mod: ModuleType,
        var_names: Set[LocalVar],
        call_stack: List[CanonicalPath],
    ) -> Union[FunctionIndirectInteractions, DDSPath, None]:
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
        # _logger.debug(f"inspect_call:local_path:{local_path} mod:{mod} z:{z}")
        if z is None or isinstance(z, ExternalObject):
            # _logger.debug(f"inspect_call: local_path: %s is rejected", local_path)
            return None
        assert isinstance(z, AuthorizedObject)
        caller_fun, caller_fun_path = z.object_val, z.resolved_path
        if not isinstance(caller_fun, FunctionType) and not inspect.isclass(caller_fun):
            raise DDSException(
                f"Expected FunctionType or class for {caller_fun_path}, got {type(caller_fun)}",
                DDSErrorCode.UNSUPPORTED_CALLABLE_TYPE,
            )

        # Check if this is a call we should do something about.
        if caller_fun_path == CPU.from_list(["dds", "keep"]):
            # Call to the keep function:
            # - bring the path
            # - bring the callee
            # - parse the arguments
            # - introspect the callee
            if len(node.args) < 2:
                raise DDSException(
                    f"Wrong number of args: expected 2+, got {node.args}"
                )
            store_path = InspectFunction._retrieve_store_path(
                node.args[0], mod, gctx, local_path
            )
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
            # For now, accept the constant arguments. This is enough for some basic objects.
            inner_intro = _introspect(called_fun, gctx, new_call_stack)
            inner_intro = inner_intro._replace(store_path=store_path)
            return inner_intro
        if caller_fun_path == CPU.from_list(["dds", "load"]):
            # Evaluation call: get the argument and returns the function interaction for this call.
            if len(node.args) != 1:
                raise DDSException(f"Wrong number of args: expected 1, got {node.args}")
            store_path = InspectFunction._retrieve_store_path(
                node.args[0], mod, gctx, local_path
            )
            _logger.debug(f"inspect_call:eval: store_path: {store_path}")
            return store_path

        if caller_fun_path == CPU.from_list(["dds", "eval"]):
            raise DDSException(
                f"Cannot process {local_path}: this function is calling dds.eval, which"
                f" is not allowed inside other eval calls. Suggestion: remove the "
                f"call to dds.eval inside {local_path}",
                DDSErrorCode.EVAL_IN_EVAL,
            )

        if caller_fun_path in call_stack:
            raise DDSException(
                f"Detected circular function calls or (co-)recursive calls."
                f"This is currently not supported. Change your code to split the "
                f"recursive section into a separate function. "
                f"Function: {caller_fun_path}"
                f"Call stack: {' '.join([str(p) for p in call_stack])}",
                DDSErrorCode.CIRCULAR_CALL,
            )
        # Normal function call.
        new_call_stack = call_stack + [caller_fun_path]
        return _introspect(caller_fun, gctx, new_call_stack)


class IntroVisitorIndirect(ast.NodeVisitor):
    def __init__(
        self,
        start_mod: ModuleType,
        gctx: EvalMainContext,
        function_var_names: Set[LocalVar],
        call_stack: List[CanonicalPath],
        fun_path: CanonicalPath,
    ):
        current_fun_name = LocalVar(CPU.last(fun_path))
        # TODO: start_mod is in the global context
        self._start_mod = start_mod
        self._gctx = gctx
        self._function_var_names = set(function_var_names)
        self._store_names: Set[LocalVar] = {current_fun_name}
        self._call_stack = call_stack
        # All the calls to a load and subsequent function calls, ordered
        self.results: List[Union[FunctionIndirectInteractions, DDSPath]] = []

    def visit_Call(self, node: ast.Call) -> Any:
        # _logger.debug(f"visit: {node} {dir(node)} {pformat(node)}")
        # The list of all the previous interactions.
        # Check the call for dds calls or sub_calls.
        fi_or_p = InspectFunctionIndirect.inspect_call(
            node,
            self._gctx,
            self._start_mod,
            self._function_var_names,
            self._call_stack,
        )
        if fi_or_p is not None:
            self.results.append(fi_or_p)
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
                # _logger.debug(f"visit_name: {node} {pformat(node)} {self._store_names}")
                # No arg given
                call_node = ast.Call(
                    func=node, args=[], keywords=[], starargs=None, kwargs=None
                )
                fi_or_p = InspectFunctionIndirect.inspect_call(
                    call_node,
                    self._gctx,
                    self._start_mod,
                    self._function_var_names,
                    self._call_stack,
                )
                if fi_or_p is not None:
                    self.results.append(fi_or_p)

        self.generic_visit(node)
