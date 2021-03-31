"""
This file is concerned with the extraction of objects given a path.
"""
import importlib
import inspect
import logging
import pathlib
import typing
from pathlib import PurePosixPath
from types import ModuleType, FunctionType
from typing import (
    Any,
    Union,
    Optional,
    Type,
)

from ._eval_ctx import (
    EvalMainContext,
    ObjectRetrievalType,
    ExternalObject,
    AuthorizedObject,
)
from .structures import DDSException, CanonicalPath, LocalDepPath, DDSErrorCode
from .structures_utils import LocalDepPathUtils, CanonicalPathUtils

_logger = logging.getLogger(__name__)


def _mod_path(m: ModuleType) -> CanonicalPath:
    return CanonicalPathUtils.from_list(m.__name__.split("."))


def function_path(f: Union[type, FunctionType]) -> CanonicalPath:
    mod = inspect.getmodule(f)
    if mod is None:
        raise DDSException(
            f"Function {f} has no module. DDS is expecting that the function"
            f" {f} be associated with a module. However, the interpreter could "
            f"not find a module associated to this function. This can happen "
            f"when the function is defined at runtime. Suggestion: define "
            f"this function in a Python module.",
            DDSErrorCode.FUNCTION_NO_MODULE,
        )
    return CanonicalPath(_mod_path(mod)._path.joinpath(f.__name__))


def _is_authorized_type(tpe: Type[Any], gctx: EvalMainContext) -> bool:
    """
    True if the type is defined within the whitelisted hierarchy

    Note: the hierarchy is currently only concerned with modules, not with any sub-object.
    """
    if tpe is None:
        return True
    if tpe in (int, float, str, bytes, PurePosixPath, FunctionType, ModuleType):
        return True
    if issubclass(tpe, object):
        mod = inspect.getmodule(tpe)
        if mod is None:
            # _logger.debug(f"_is_authorized_type: type %s has no module", tpe)
            return False
        mod_path = _mod_path(mod)
        if gctx.is_authorized_path(mod_path):
            msg = (
                f"Type {tpe} ({mod_path}) is authorized. This is currently not implemented."
                f" Suggestion: use a built-in type instead."
            )
            _logger.warning(msg)
            raise DDSException(msg, DDSErrorCode.AUTHORIZED_TYPE_NOT_UNDERSTOOD)
        return False
    else:
        msg = f"Type {tpe} is not implemented"
        _logger.warning(msg)
        raise DDSException(msg, DDSErrorCode.AUTHORIZED_TYPE_NOT_UNDERSTOOD)


class ObjectRetrieval(object):
    @classmethod
    def retrieve_object(
        cls,
        local_path: LocalDepPath,
        context_mod: ModuleType,
        gctx: EvalMainContext,
        debug: bool = False,
    ) -> ObjectRetrievalType:
        """Retrieves the object and also provides the canonical path of the object"""
        assert len(local_path.parts), local_path
        mod_path = _mod_path(context_mod)
        obj_key = (local_path, mod_path)

        if obj_key in gctx.cached_objects:
            # _logger.debug(f"retrieve_object: found in cache: obj_key: {obj_key}")
            return gctx.cached_objects[obj_key]
        # _logger.debug(f"retrieve_object: not found in cache: obj_key: {obj_key}")

        fname = local_path.parts[0]
        sub_path = LocalDepPathUtils.tail(local_path)
        if fname not in context_mod.__dict__:
            # In some cases (old versions of jupyter) the module is not listed
            # -> try to load it from the root
            # _logger.debug(
            #     f"Could not find {fname} in {context_mod}, attempting a direct load"
            # )
            # TODO: is this worth supporting?
            loaded_mod: Optional[ModuleType]
            try:
                loaded_mod = importlib.import_module(fname)
            except ModuleNotFoundError:
                loaded_mod = None
            if loaded_mod is None:
                # Looking into the globals (only if the scope is currently __main__ or __global__)
                mod_path = _mod_path(context_mod)
                if CanonicalPathUtils.head(mod_path) not in ("__main__", "__global__"):
                    # _logger.debug(
                    #     f"Could not load name %s and not in global context (%s), skipping ",
                    #     fname,
                    #     mod_path,
                    # )
                    return None
                else:
                    # _logger.debug(
                    #     f"Could not load name %s, looking into the globals (mod_path: %s, %s)",
                    #     fname,
                    #     mod_path,
                    #     mod_path.get(0),
                    # )
                    pass
                # _logger.debug(f"Could not load name {fname}, looking into the globals")
                if fname in gctx.start_globals:
                    # _logger.debug(f"Found {fname} in start_globals")
                    obj = gctx.start_globals[fname]
                    if isinstance(obj, ModuleType) and not LocalDepPathUtils.empty(
                        sub_path
                    ):
                        # Referring to function from an imported module.
                        # Redirect the search to the module
                        # _logger.debug(
                        #     f"{fname} is module {obj}, checking for {sub_path}"
                        # )
                        res = cls.retrieve_object(sub_path, obj, gctx)
                        gctx.cached_objects[obj_key] = res
                        return res
                    if isinstance(obj, ModuleType):
                        # Fully resolve the name of the module:
                        obj_path = _mod_path(obj)
                    elif isinstance(obj, FunctionType):
                        obj_path = function_path(obj)
                    else:
                        obj_path = CanonicalPathUtils.from_list(
                            ["__global__"] + [str(x) for x in local_path.parts]
                        )
                    if not gctx.is_authorized_path(obj_path):
                        # _logger.debug(
                        #     f"Object[start_globals] {fname} of type {type(obj)} is not authorized (path),"
                        #     f" dropping path {obj_path}"
                        # )
                        res = ExternalObject(obj_path)
                        gctx.cached_objects[obj_key] = res
                        return res

                    # TODO: why do we need another function check?
                    # TODO: why strings / paths here?
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
                        # _logger.debug(
                        #     f"Object[start_globals] {fname} ({type(obj)}) of path {obj_path} is authorized,"
                        # )
                        res = AuthorizedObject(obj, obj_path)
                        gctx.cached_objects[obj_key] = res
                        return res
                    else:
                        # _logger.debug(
                        #     f"Object[start_globals] {fname} of type {type(obj)} is noft authorized (type), dropping path {obj_path}"
                        # )
                        res = ExternalObject(obj_path)
                        gctx.cached_objects[obj_key] = res
                        return res
                else:
                    # _logger.debug(f"{fname} not found in start_globals")
                    gctx.cached_objects[obj_key] = None
                    return None
            res = cls._retrieve_object_rec(sub_path, loaded_mod, gctx)
            gctx.cached_objects[obj_key] = res
            return res
        else:
            res = cls._retrieve_object_rec(local_path, context_mod, gctx)
            if debug:
                _logger.debug(
                    f"retrieve_object: _retrieve_object_rec: local_path:{local_path} context_mod:{context_mod} res:{res}"
                )
            gctx.cached_objects[obj_key] = res
            return res

    @classmethod
    def retrieve_object_global(
        cls, path: CanonicalPath, gctx: EvalMainContext
    ) -> Optional[Any]:
        """
        Retrieves an object given its global path.
        """
        # The head is always assumed to be a module for now
        mod_name = CanonicalPathUtils.head(path)
        obj_key = (LocalDepPath(PurePosixPath("")), path)
        if obj_key in gctx.cached_objects:
            return gctx.cached_objects[obj_key]

        mod = importlib.import_module(mod_name)
        if mod is None:
            raise DDSException(
                f"Cannot process path {path}: module {mod_name} cannot be loaded. "
                f"DDS is attempting to load an object in the path {path} and is expecting"
                f"to find it in the module {mod_name}. However, "
                f"Python is indicating that {mod_name} is not a module that can be loaded. "
                f"Suggestions: ensure that {mod_name} is a Python module.",
                DDSErrorCode.MODULE_NOT_FOUND,
            )
        sub_path = CanonicalPathUtils.tail(path)
        dep_path = LocalDepPath(sub_path._path)
        # _logger.debug(f"Calling retrieve_object on {dep_path}, {mod}")
        z = cls.retrieve_object(dep_path, mod, gctx)
        if z is None or isinstance(z, ExternalObject):
            raise DDSException(
                f"Cannot load path {path}: this object cannot be retrieved, however "
                f"the module '{mod_name}' exists. The typical cause of the issue is "
                f"that the module {mod_name} has not been whitelisted for use by DDS. Use the "
                f"function 'dds.accept_module' to whitelist {mod_name} or one of its "
                f"submodules."
                f" dep_path: {dep_path} module: {mod}",
                DDSErrorCode.MODULE_NOT_FOUND,
            )
        elif isinstance(z, AuthorizedObject):
            obj = z.object_val
            gctx.cached_objects[obj_key] = AuthorizedObject(obj, path)
            return obj
        else:
            assert False

    @classmethod
    def _retrieve_object_rec(
        cls,
        local_path: LocalDepPath,
        context_mod: ModuleType,
        gctx: EvalMainContext,
        debug: bool = False,
    ) -> ObjectRetrievalType:
        if debug:
            _logger.debug(f"_retrieve_object_rec: {local_path} {context_mod}")
        if not local_path.parts:
            # The final position. It is the given module, if authorized.
            obj_mod_path = _mod_path(context_mod)
            if not gctx.is_authorized_path(obj_mod_path):
                if debug:
                    _logger.debug(
                        f"_retrieve_object_rec: Actual module {obj_mod_path} for obj {context_mod} is not authorized"
                    )
                return ExternalObject(
                    CanonicalPathUtils.append(obj_mod_path, local_path)
                )
            else:
                # _logger.debug(
                #     f"_retrieve_object_rec: Actual module {obj_mod_path} for obj {context_mod}: authorized"
                # )
                pass
            return AuthorizedObject(context_mod, obj_mod_path)
        # At least one more path to explore
        fname = local_path.parts[0]
        tail_path = LocalDepPathUtils.tail(local_path)
        if fname not in context_mod.__dict__:
            mod_keys = sorted(context_mod.__dict__.keys())
            # It should be in the context module, this was assumed to be taken care of
            raise DDSException(
                f"_retrieve_object_rec: Object {fname} not found in module {context_mod}."
                f"DDS attempted to load object with path {local_path} inside the module "
                f"{context_mod}. However, no object called {fname} is present in that module. "
                f"The other keys of that module are: {mod_keys}.",
                DDSErrorCode.OBJECT_PATH_NOT_FOUND,
            )
        obj = context_mod.__dict__[fname]
        if debug:
            _logger.debug(
                f"_retrieve_object_rec: {local_path} {context_mod} {type(obj)} {obj}"
            )

        if LocalDepPathUtils.empty(tail_path):
            # Final path.
            # If it is a module, continue recursion
            if isinstance(obj, ModuleType):
                return cls._retrieve_object_rec(tail_path, obj, gctx)
            # Special treatment for objects that may be defined in other modules but are redirected in this one.
            if isinstance(obj, (FunctionType, type)):
                mod_obj = inspect.getmodule(obj)
                if mod_obj is None:
                    # _logger.debug(
                    #     f"_retrieve_object_rec: cannot infer definition module: path: {local_path} mod: {context_mod} "
                    # )
                    return None
                if mod_obj in [typing]:
                    # The redirection module is a python system module like the typing module.
                    # Do not attempt to introspect further, this goes into python implementation details and is not
                    # authorized.
                    # Note: it skips the definition of the newtype along the way, but this is considered a corner case.
                    return None
                if mod_obj is not context_mod:
                    # Reimporting from another module.
                    # Get the true name of the function.
                    # The function may have been renamed in the code and this name is not the one from
                    # the original module -> check the original name of the function:
                    local_path_mod = LocalDepPath(PurePosixPath(obj.__name__)).joinpath(
                        LocalDepPathUtils.tail(local_path)
                    )
                    if debug:
                        _logger.debug(
                            f"_retrieve_object_rec: {context_mod} is not definition module, redirecting to {local_path_mod} ; {mod_obj}"
                        )
                    return cls._retrieve_object_rec(local_path_mod, mod_obj, gctx)
            obj_mod_path = _mod_path(context_mod)
            obj_path = CanonicalPathUtils.append(obj_mod_path, fname)
            if gctx.is_authorized_path(obj_path):
                # TODO: simplify the authorized types
                if (
                    _is_authorized_type(type(obj), gctx)
                    or isinstance(
                        obj,
                        (
                            FunctionType,
                            ModuleType,
                            pathlib.PosixPath,
                            pathlib.PurePosixPath,
                        ),
                    )
                    or inspect.isclass(obj)
                ):
                    if debug:
                        _logger.debug(
                            f"_retrieve_object_rec: Object {fname} ({type(obj)}) of path {obj_path} is authorized,"
                        )
                    return AuthorizedObject(obj, obj_path)
                else:
                    if debug:
                        _logger.debug(
                            f"_retrieve_object_rec: Object {fname} of type {type(obj)} is not authorized (type), dropping path {obj_path}"
                        )
                    return ExternalObject(obj_path)
            else:
                if debug:
                    _logger.debug(
                        f"_retrieve_object_rec: Object {fname} of type {type(obj)} and path {obj_path} is not authorized (path)"
                    )
                return ExternalObject(obj_path)

        if debug:
            _logger.debug(
                f"_retrieve_object_rec: non-terminal fname={fname} obj: {type(obj)} tail_path: {tail_path} {isinstance(obj, FunctionType)} {isinstance(obj, ModuleType)} {isinstance(obj, type)}"
            )
        # More to explore
        # If it is a module, continue recursion
        if isinstance(obj, ModuleType):
            return cls._retrieve_object_rec(tail_path, obj, gctx)

        # Some objects like types are also callables

        # We still have a path but we have reached a callable.
        # In this case, determine if the function is allowed. If this is the case, stop here.
        # (the rest of the path is method calls)
        if isinstance(obj, FunctionType):
            obj_mod_path = _mod_path(context_mod)
            obj_path = CanonicalPathUtils.append(obj_mod_path, fname)
            if gctx.is_authorized_path(obj_path):
                return AuthorizedObject(obj, obj_path)
            else:
                return ExternalObject(obj_path)

        # We still have a path but we have reached a class.
        # In this case, determine if the class is allowed. If this is the case, stop here.
        # (the rest of the path is method calls)
        if isinstance(obj, type):
            obj_mod_path = function_path(obj)
            # obj_mod_path = _mod_path(context_mod)
            obj_path = CanonicalPathUtils.append(obj_mod_path, fname)
            # _logger.debug(f"_retrieve_object_rec:(type) whitelisted_packages: {obj_path}->{gctx.is_authorized_path(obj_path)} {gctx.whitelisted_packages}")
            if gctx.is_authorized_path(obj_path):
                return AuthorizedObject(obj, obj_path)
            else:
                return ExternalObject(obj_path)

        # The rest is not authorized for now.
        # msg = f"Failed to consider object type {type(obj)} at path {local_path} context_mod: {context_mod}"
        # _logger.debug(msg)
        return None
