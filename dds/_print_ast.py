# Copied from:
# https://raw.githubusercontent.com/asottile/astpretty/master/astpretty.py

import ast
import contextlib
from typing import Any
from typing import Generator
from typing import Tuple
from typing import Type
from typing import Union

# if TYPE_CHECKING:
#     from typed_ast import ast27
#     from typed_ast import ast3
#     ASTType = Union[ast.AST, ast27.AST, ast3.AST]

AST: Tuple[Type[Any], ...] = (ast.AST,)
ASTType = ast.AST
expr_context: Tuple[Type[Any], ...] = (ast.expr_context,)
# try:  # pragma: no cover (with typed-ast)
#     from typed_ast import ast27
#     from typed_ast import ast3
# except ImportError:  # pragma: no cover (without typed-ast)
#     typed_support = False
# else:  # pragma: no cover (with typed-ast)
#     AST += (ast27.AST, ast3.AST)
#     expr_context += (ast27.expr_context, ast3.expr_context)
#     typed_support = True


def _is_sub_node(node: Any) -> bool:
    return isinstance(node, AST) and not isinstance(node, expr_context)


def _is_leaf(node: "ASTType") -> bool:
    for field in node._fields:
        attr = getattr(node, field)
        if _is_sub_node(attr):
            return False
        elif isinstance(attr, (list, tuple)):
            for val in attr:
                if _is_sub_node(val):
                    return False
    else:
        return True


def _fields(n: "ASTType", show_offsets: bool = True) -> Tuple[str, ...]:
    if show_offsets:
        return n._attributes + n._fields
    else:
        return n._fields


def _leaf(node: "ASTType", show_offsets: bool = True) -> str:
    if isinstance(node, AST):
        return "{}({})".format(
            type(node).__name__,
            ", ".join(
                "{}={}".format(
                    field,
                    _leaf(getattr(node, field), show_offsets=show_offsets),
                )
                for field in _fields(node, show_offsets=show_offsets)
            ),
        )
    elif isinstance(node, list):
        return "[{}]".format(
            ", ".join(_leaf(x, show_offsets=show_offsets) for x in node),
        )
    else:
        return repr(node)


def pformat(
    node: Union["ASTType", None, str],
    indent: str = "    ",
    show_offsets: bool = True,
    _indent: int = 0,
) -> str:
    if node is None:
        return repr(node)
    elif isinstance(node, str):  # pragma: no cover (ast27 typed-ast args)
        return repr(node)
    elif _is_leaf(node):
        return _leaf(node, show_offsets=show_offsets)
    else:

        class state:
            indent = _indent

        @contextlib.contextmanager
        def indented() -> Generator[None, None, None]:
            state.indent += 1
            yield
            state.indent -= 1

        def indentstr() -> str:
            return state.indent * indent

        def _pformat(el: Union["ASTType", None, str], _indent: int = 0) -> str:
            return pformat(
                el,
                indent=indent,
                show_offsets=show_offsets,
                _indent=_indent,
            )

        out = type(node).__name__ + "(\n"
        with indented():
            for field in _fields(node, show_offsets=show_offsets):
                attr = getattr(node, field)
                if attr == []:
                    representation = "[]"
                elif (
                    isinstance(attr, list)
                    and len(attr) == 1
                    and isinstance(attr[0], AST)
                    and _is_leaf(attr[0])
                ):
                    representation = f"[{_pformat(attr[0])}]"
                elif isinstance(attr, list):
                    representation = "[\n"
                    with indented():
                        for el in attr:
                            representation += "{}{},\n".format(
                                indentstr(),
                                _pformat(el, state.indent),
                            )
                    representation += indentstr() + "]"
                elif isinstance(attr, AST):
                    representation = _pformat(attr, state.indent)
                else:
                    representation = repr(attr)
                out += f"{indentstr()}{field}={representation},\n"
        out += indentstr() + ")"
        return out
