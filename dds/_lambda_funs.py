"""
Parsing of the lambda functions
"""

from typing import Iterable, Tuple, Optional, Callable, List, Dict, Any

import ast
import asttokens
import inspect
import logging
from ._print_ast import pformat
from .structures import KSException

_logger = logging.getLogger(__name__)


def _walk_with_parent(node: ast.AST) -> Iterable[Tuple[ast.AST, Optional[ast.AST]]]:
    """Walk the abstract syntax tree by (node, parent)."""
    stack = [(node, None)]  # type: List[Tuple[ast.AST, Optional[ast.AST]]]
    while stack:
        node, parent = stack.pop()

        for child in ast.iter_child_nodes(node):
            stack.append((child, node))

        yield node, parent


def is_lambda(fun: Callable[..., Any]) -> bool:
    """
    Check whether the condition is a lambda function.

    :param fun: condition function of a contract
    :return: True if condition is defined as lambda function
    """
    return fun.__name__ == "<lambda>"


def inspect_lambda_condition(fun: Callable[..., Any]) -> ast.Lambda:
    """
    Parse the file in which condition resides and figure out
    the corresponding lambda AST node.

    :param fun: condition lambda function
    :return:
        inspected lambda function, or None if the condition
        is not a lambda function
    """
    assert is_lambda(fun), fun

    # Parse the whole file and find the AST node of the condition lambda.
    # This is necessary, since condition.__code__ gives us only a line number
    # which is too vague to find the lambda node.
    lines, condition_lineno = inspect.findsource(fun)
    _logger.debug(f"_parse_lambda: {lines}")

    atok = asttokens.ASTTokens("".join(lines), parse=True)

    parent_of = dict()  # type: Dict[ast.AST, Optional[ast.AST]]
    for node, parent in _walk_with_parent(atok.tree):
        parent_of[node] = parent

    # node of the decorator
    call_node = None  # type: Optional[ast.Call]

    for node in ast.walk(atok.tree):
        if isinstance(node, ast.Lambda) and node.lineno - 1 == condition_lineno:
            # Go up all the way to the decorator
            ancestor = parent_of[node]

            while ancestor is not None and not isinstance(ancestor, ast.Call):
                ancestor = parent_of[ancestor]

            call_node = ancestor
            break
    _logger.debug(f"_parse_lambda: call_node: {pformat(call_node)}")

    for node in ast.walk(call_node):
        if isinstance(node, ast.Lambda):
            return node
    raise KSException(f"Could not parse lambda {pformat(call_node)}")
    #
    # lambda_node = None  # type: Optional[ast.Lambda]
    # if len(call_node.args) > 0:
    #     lambda_node = call_node.args[0]
    #
    # elif len(call_node.keywords) > 0:
    #     for keyword in call_node.keywords:
    #         if keyword.arg == "condition":
    #             lambda_node = keyword.value
    #             break
    # else:
    #     raise AssertionError()
    #
    # return lambda_node
