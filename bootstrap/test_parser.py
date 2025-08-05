from __future__ import annotations

from . import ast
from .conftest import node, parse_first

int_type = node(ast.NominalType, name="Int")
str_type = node(ast.NominalType, name="Str")
bool_type = node(ast.NominalType, name="Bool")


def test_literals() -> None:
    assert parse_first("42") == node(ast.IntLit, value=42)
    assert parse_first("true") == node(ast.BoolLit, value=True)
    assert parse_first("false") == node(ast.BoolLit, value=False)
    assert parse_first('"hello"') == node(ast.StrLit, value="hello")
    assert parse_first("'c'") == node(ast.CharLit, value="c")


def test_assign() -> None:
    assert parse_first("x = 42") == node(
        ast.Assign,
        target=node(ast.Name, name="x"),
        value=node(ast.IntLit, value=42),
    )


def test_fun_def() -> None:
    assert parse_first("foo = fun(a, b) do end") == node(
        ast.FunDef, name="foo", params=["a", "b"], body=node(ast.Block, nodes=[])
    )


def test_fun_shapes() -> None:
    assert parse_shape_decl("Foo = fun(a Int, b Str) Bool", "Foo") == node(
        ast.FunShape,
        params=[
            node(ast.Attr, name="a", shape=int_type),
            node(ast.Attr, name="b", shape=str_type),
        ],
        result=bool_type,
    )


def test_product_shapes() -> None:
    assert parse_shape_decl("Foo = {a Int}", "Foo") == node(
        ast.ProductShape,
        attrs=[node(ast.Attr, name="a", shape=int_type)],
    )


def test_sum_shapes() -> None:
    assert parse_shape_decl("Foo = Int | Str", "Foo") == node(ast.SumShape, variants=[int_type, str_type])


def test_complex_shapes() -> None:
    assert parse_shape_decl("Foo = {a Int, b Str} | Int | {c {d Bool}}", "Foo") == node(
        ast.SumShape,
        variants=[
            node(
                ast.ProductShape,
                attrs=[
                    node(ast.Attr, name="a", shape=int_type),
                    node(ast.Attr, name="b", shape=str_type),
                ],
            ),
            int_type,
            node(
                ast.ProductShape,
                attrs=[
                    node(
                        ast.Attr,
                        name="c",
                        shape=node(
                            ast.ProductShape,
                            attrs=[node(ast.Attr, name="d", shape=bool_type)],
                        ),
                    )
                ],
            ),
        ],
    )


def test_block() -> None:
    assert parse_first("do end") == node(ast.Block, nodes=[])
    assert parse_first("do 42 end") == node(ast.Block, nodes=[node(ast.IntLit, value=42)])


def parse_shape_decl(code: str, expected_name: str) -> ast.Shape:
    decl = parse_first(code)
    assert isinstance(decl, ast.ShapeDecl)
    assert decl.name == expected_name
    return decl.shape
