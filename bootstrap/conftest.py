from __future__ import annotations

import contextlib
import os
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, cast

from . import ast, compiler, error, gogen, ir, parser, token, types
from .span import Span

ast.nid_enabled = False

id_ = 0


def next_id() -> int:
    global id_  # noqa: PLW0603
    id_ += 1
    return id_


def errors_str(errors: list[error.Error]) -> str:
    return "\n".join(f"{x} at {x.stacktrace}" for x in errors)


def parse(code: str) -> tuple[ast.Module, list[error.Error], list[str]]:
    global id_  # noqa: PLW0603
    id_ = 0
    code = stripln(code)
    tokens, errors = token.tokenize(token.Input("test.kl", code.strip()))
    assert errors_str(errors) == ""
    module, errors = parser.parse(parser.Input(tokens, next_id))
    return module, errors, [x.short_message() for x in errors]


def parse_first(code: str) -> ast.Node:
    module, _, errors = parse(code)
    assert errors == [], f"Expected no errors, got {errors}"
    assert len(module.nodes) == 1, f"Expected 1 node, got {module.nodes}"
    return module.nodes[0]


def stripln(s: str) -> str:
    return textwrap.dedent(s).strip() + "\n"


def strip(s: str) -> str:
    return textwrap.dedent(s).strip()


def node(kind: Callable, **kwargs: Any) -> ast.Node:
    defaults: Any = {"id": 0, "span": Span("test.kl", "", 0, 0)}
    match kind:
        case ast.IntLit:
            defaults.update({"bits": 64, "signed": True})
        case ast.FunDef:
            defaults.update({"namespace": None})
        case ast.If:
            defaults.update({"else_block": None})
        case ast.Name:
            defaults.update({"kind": "ident"})
        case ast.ShapeAlias:
            defaults.update({"behaviours": []})
        case ast.ProductShape:
            defaults.update({"behaviours": [], "composites": []})
        case ast.ShapeLit:
            defaults.update({"shape_ref": None, "behaviours": [], "composites": []})
        case ast.ShapeLit:
            defaults.update({"behaviours": []})
        case ast.SumShape:
            defaults.update({"behaviours": []})
    return kind(**defaults | kwargs)


def shape(kind: Callable, **kwargs: Any) -> types.Shape:
    defaults: Any = {"span": Span("test.kl", "", 0, 0)}
    match kind:
        case types.PrimitiveShape:
            defaults.update({"name": None, "behaviours": types.Behaviours(())})
        case types.ProductShape:
            defaults.update({"name": None, "attrs": (), "behaviours": types.Behaviours(())})
        case types.FunShape:
            defaults.update({"name": None, "builtin": False, "namespace": None})
    return kind(**defaults | kwargs)


def empty_shape() -> types.Shape:
    return shape(types.ProductShape)


def typecheck(code: str) -> TypeChecker:
    module, _, errors = parse(code)
    assert errors == [], f"Expected no errors, got {errors}"
    result = types.typecheck(module)
    assert errors_str(result.errors) == ""
    return TypeChecker(module, result.type_env)


def typecheck_err(code: str) -> tuple[list[error.Error], list[str]]:
    module, _, errors = parse(code)
    assert errors == [], f"Expected no errors, got {errors}"
    result = types.typecheck(module)
    return result.errors, [x.short_message() for x in result.errors]


def generate_ir(code: str) -> ir.IR:
    module, _, errors = parse(code)
    assert errors == [], f"Expected no errors, got {errors}"
    result = types.typecheck(module)
    assert errors_str(result.errors) == ""
    return ir.generate_ir(result.fun_specs)


def codegen(code: str) -> str:
    module, _, errors = parse(code)
    assert errors == [], f"Expected no errors, got {errors}"
    result = types.typecheck(module)
    assert errors_str(result.errors) == ""
    ir_ = ir.generate_ir(result.fun_specs)
    return gogen.gogen(ir_)


def compile_and_run(code: str, *, debug: str = "") -> compiler.RunStep:
    outfile = tempfile.gettempdir() + "/test"
    code = strip(code)
    compilation = compiler.compile(token.Input("test.kl", code), str(outfile))
    run: compiler.RunStep | None = None
    try:
        for step in compilation:
            match step:
                case compiler.TokenStep():
                    if debug == "token":
                        print(step)
                    assert errors_str(step.errors) == ""
                case compiler.ParseStep():
                    if debug == "parse":
                        print(step)
                    assert errors_str(step.errors) == ""
                case compiler.TypecheckStep():
                    if debug == "typecheck":
                        print(step)
                    assert errors_str(step.errors) == ""
                case compiler.AbortStep():
                    raise AssertionError(f"Unexpected abort step: {step}")
                case compiler.IRStep():
                    if debug == "ir":
                        print(step)
                case compiler.CodeGenStep():
                    if debug == "codegen":
                        print(step)
                case compiler.CompileStep():
                    assert step.stderr == ""
                    assert step.returncode == 0
                case compiler.RunStep():
                    run = step
                case _:
                    raise AssertionError(f"Unexpected step: {step}")
    finally:
        with contextlib.suppress(BaseException):
            os.remove(outfile)
            os.remove(outfile + ".go")
    assert run is not None
    return run


def compile_and_run_success(code: str, *, debug: str = "") -> str:
    run = compile_and_run(code, debug=debug)
    assert run.stderr == ""
    assert run.returncode == 0
    return run.stdout


@dataclass
class TypeChecker:
    module: ast.Module
    type_env: types.TypeEnv

    def at[T: types.Shape](self, line: int, col: int, node_typ: type[ast.Node], typ: type[T] | None = None) -> T:
        node = self.node_at(line, col, node_typ)
        if typ is not None:
            assert isinstance(self.type_env.get(node), typ), f"Expected {typ}, got {type(self.type_env.get(node))}"
        return cast(T, self.type_env.get(node))

    def node_at(self, line: int, col: int, typ: type[ast.Node]) -> ast.Node:
        res: ast.Node | None = None

        def visit(node: ast.Node, _parent: ast.Node | None) -> ast.Node:
            nonlocal res
            start = node.span.start_line_col()
            if start[0] == line and start[1] >= col and isinstance(node, typ):  # noqa: SIM102
                if res is None or start[0] < res.span.start_line_col()[0]:
                    res = node
            ast.walk(node, visit)
            return node

        visit(self.module, None)
        assert res is not None, f"No node found at {line}:{col}"
        return res

    def debug(self) -> str:
        lines = []

        def visit(node: ast.Node, _parent: ast.Node | None) -> ast.Node:
            if not isinstance(node, (ast.Module, ast.FunDef)):
                try:
                    typ = self.type_env.node_shapes[node.id]
                    lines.append(
                        "\n".join(x.rstrip() for x in node.span.formatted_lines(0, enclosing_empty_lines=False))
                    )
                    lines.append(f"{typ.__class__.__name__:10}: {typ}")
                    lines.append("")
                except KeyError:
                    print("Node has no type - fix that!", node)
            ast.walk(node, visit)
            return node

        visit(self.module, None)
        return "\n".join(lines)
