from __future__ import annotations

from . import ast
from .conftest import node, parse_first

int_type = node(ast.ShapeRef, name="Int")
str_type = node(ast.ShapeRef, name="Str")
bool_type = node(ast.ShapeRef, name="Bool")


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
    assert parse_first("foo = fun(a, b): end") == node(
        ast.FunDef,
        name="foo",
        params=[node(ast.FunParam, name="a"), node(ast.FunParam, name="b")],
        body=node(ast.Block, nodes=[]),
    )


def test_fun_def_with_behaviour_ns() -> None:
    assert parse_first("@Foo.bar = fun(a, b): end") == node(
        ast.FunDef,
        name="bar",
        namespace="Foo",
        params=[node(ast.FunParam, name="a"), node(ast.FunParam, name="b")],
        body=node(ast.Block, nodes=[]),
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


def test_shape_refs() -> None:
    assert parse_shape_decl("Foo = Bar", "Foo") == node(ast.ShapeRef, name="Bar")


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


def test_product_shape_composition() -> None:
    assert parse_shape_decl("Foo = {a Int} + Bar", "Foo") == node(
        ast.ProductShapeComp,
        lhs=node(ast.ProductShape, attrs=[node(ast.Attr, name="a", shape=int_type)]),
        rhs=node(ast.ShapeRef, name="Bar"),
    )


def test_product_shape_composition_preceeds_sum_shape() -> None:
    assert parse_shape_decl("Foo = {a Int} + Bar | Baz", "Foo") == node(
        ast.SumShape,
        variants=[
            node(
                ast.ProductShapeComp,
                lhs=node(ast.ProductShape, attrs=[node(ast.Attr, name="a", shape=int_type)]),
                rhs=node(ast.ShapeRef, name="Bar"),
            ),
            node(ast.ShapeRef, name="Baz"),
        ],
    )


def test_shape_behaviour() -> None:
    decl = parse_first("Foo = Bar | Baz + @Foo + @Baz")
    assert isinstance(decl, ast.ShapeDecl)
    assert decl.behaviours == ["Foo", "Baz"]
    assert decl.shape == node(ast.SumShape, variants=[node(ast.ShapeRef, name="Bar"), node(ast.ShapeRef, name="Baz")])


def test_shape_literal() -> None:
    assert parse_first("{a = 42}") == node(
        ast.ShapeLit, attrs=[node(ast.ShapeLitAttr, name="a", value=node(ast.IntLit, value=42))]
    )
    assert parse_first("Foo{a = 42}") == node(
        ast.ShapeLit,
        shape_ref=node(ast.ShapeRef, name="Foo"),
        attrs=[node(ast.ShapeLitAttr, name="a", value=node(ast.IntLit, value=42))],
    )


def test_shape_literal_with_behaviour() -> None:
    assert parse_first("{a = 42} + @Foo") == node(
        ast.ShapeLit,
        behaviours=["Foo"],
        attrs=[node(ast.ShapeLitAttr, name="a", value=node(ast.IntLit, value=42))],
    )


def test_nested_shape_literal() -> None:
    assert parse_first("{a = {b = 42}}") == node(
        ast.ShapeLit,
        attrs=[
            node(
                ast.ShapeLitAttr,
                name="a",
                value=node(
                    ast.ShapeLit,
                    attrs=[node(ast.ShapeLitAttr, name="b", value=node(ast.IntLit, value=42))],
                ),
            )
        ],
    )


def test_nested_shape_literal_with_shape_ref() -> None:
    assert parse_first("Foo{a = Bar{b = 42}}") == node(
        ast.ShapeLit,
        shape_ref=node(ast.ShapeRef, name="Foo"),
        attrs=[
            node(
                ast.ShapeLitAttr,
                name="a",
                value=node(
                    ast.ShapeLit,
                    shape_ref=node(ast.ShapeRef, name="Bar"),
                    attrs=[node(ast.ShapeLitAttr, name="b", value=node(ast.IntLit, value=42))],
                ),
            )
        ],
    )


def test_member() -> None:
    assert parse_first("foo.bar") == node(ast.Member, target=node(ast.Name, name="foo"), name="bar")


def test_if() -> None:
    assert parse_first("if case true: end") == node(
        ast.If, arms=[node(ast.IfArm, cond=node(ast.BoolLit, value=True), block=node(ast.Block, nodes=[]))]
    )
    assert parse_first(
        """
        if
            case true:
                1
            case false:
                2
            else:
                3
        end
        """
    ) == node(
        ast.If,
        arms=[
            node(
                ast.IfArm,
                cond=node(ast.BoolLit, value=True),
                block=node(ast.Block, nodes=[node(ast.IntLit, value=1)]),
            ),
            node(
                ast.IfArm,
                cond=node(ast.BoolLit, value=False),
                block=node(ast.Block, nodes=[node(ast.IntLit, value=2)]),
            ),
        ],
        else_block=node(ast.Block, nodes=[node(ast.IntLit, value=3)]),
    )


def test_call() -> None:
    assert parse_first("foo()") == node(ast.Call, callee=node(ast.Name, name="foo"), args=[])
    assert parse_first("foo(42)") == node(
        ast.Call, callee=node(ast.Name, name="foo"), args=[node(ast.IntLit, value=42)]
    )


def test_block() -> None:
    assert parse_first(": end") == node(ast.Block, nodes=[])
    assert parse_first(": 42 end") == node(ast.Block, nodes=[node(ast.IntLit, value=42)])


def test_behaviour_fun_def() -> None:
    assert parse_first("@Foo.bar = fun(foo): end") == node(
        ast.FunDef,
        name="bar",
        namespace="Foo",
        params=[node(ast.FunParam, name="foo")],
        body=node(ast.Block, nodes=[]),
    )


def parse_shape_decl(code: str, expected_name: str) -> ast.Shape:
    decl = parse_first(code)
    assert isinstance(decl, ast.ShapeDecl)
    assert decl.name == expected_name
    return decl.shape
