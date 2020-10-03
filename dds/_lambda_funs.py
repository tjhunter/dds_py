"""
Parsing of the lambda functions
code inspired by:
http://xion.io/post/code/python-get-lambda-code.html
"""

from typing import Iterable, Tuple, Optional, Callable, List, Dict, Any

import ast
import asttokens  # type: ignore
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

    parent_of: Dict[ast.AST, Optional[ast.AST]] = dict()
    for node, parent in _walk_with_parent(atok.tree):
        parent_of[node] = parent

    # node of the decorator
    call_node: Optional[ast.Call] = None

    for node in ast.walk(atok.tree):
        if isinstance(node, ast.Lambda) and node.lineno - 1 == condition_lineno:
            # Go up all the way to the decorator
            ancestor = parent_of[node]

            while ancestor is not None and not isinstance(ancestor, ast.Call):
                ancestor = parent_of[ancestor]

            if ancestor is not None and isinstance(ancestor, ast.Call):
                call_node = ancestor
                break
    _logger.debug(f"_parse_lambda: call_node: {pformat(call_node)}")

    if call_node is None:
        raise KSException(f"Could not find call node {pformat(call_node)}")

    for node in ast.walk(call_node):
        if isinstance(node, ast.Lambda):
            return node
    raise KSException(f"Could not parse lambda {pformat(call_node)}")
