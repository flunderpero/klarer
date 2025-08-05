from __future__ import annotations

import textwrap
from typing import Any, Callable

from . import ast, error, parser, token
from .span import Span

id_ = 0


def next_id() -> int:
    global id_  # noqa: PLW0603
    id_ += 1
    return id_


def errors_str(errors: list[error.Error]) -> str:
    return "\n".join(f"{x} at {x.stacktrace}" for x in errors)


def ignore_span() -> Span:
    span = Span("test.kl", "", 0, 0)
    span.__eq__ = lambda other: True  # noqa: ARG005
    return span


# Parse the given code and set id to 0 and `span` to `ignore_span` on every node.
def parse(code: str) -> ast.Module:
    global id_  # noqa: PLW0603
    id_ = 0
    code = strip(code)
    tokens, errors = token.tokenize(token.Input("test.kl", code.strip()))
    assert errors_str(errors) == ""
    module, errors = parser.parse(parser.Input(tokens, next_id))
    assert errors_str(errors) == ""

    def adapt(node: ast.Node, _parent: ast.Node | None) -> ast.Node:
        ast.walk(node, adapt)
        node.id = 0
        node.span = ignore_span()
        return node

    ast.walk(module, adapt)
    return module


def parse_first(code: str) -> ast.Node:
    module = parse(code)
    assert len(module.nodes) == 1, f"Expected 1 node, got {module.nodes}"
    return module.nodes[0]


def strip(s: str) -> str:
    return textwrap.dedent(s).strip() + "\n"


def node(kind: Callable, **kwargs: Any) -> ast.Node:
    defaults: Any = {"id": 0, "span": ignore_span()}
    match kind:
        case ast.IntLit:
            defaults.update({"bits": 64, "signed": True})
        case ast.Assign:
            defaults.update({"mut": False})
        case ast.FunDef:
            defaults.update({"namespace": None})
        case ast.Name:
            defaults.update({"kind": "ident"})
    return kind(**defaults | kwargs)
