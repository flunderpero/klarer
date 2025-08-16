from __future__ import annotations

from . import ast
from .conftest import node, parse, parse_first

int_shape = node(ast.ShapeRef, name="Int")
str_shape = node(ast.ShapeRef, name="Str")
bool_shape = node(ast.ShapeRef, name="Bool")
unit_shape = node(ast.UnitShape)


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


def test_complex_fun_def() -> None:
    assert parse_first("foo = fun(a Str, b {value Str} + @Value) {value Str} + @Value: end") == node(
        ast.FunDef,
        name="foo",
        params=[
            node(ast.FunParam, name="a", shape=str_shape),
            node(
                ast.FunParam,
                name="b",
                shape=node(
                    ast.ProductShape,
                    attrs=[node(ast.Attr, name="value", shape=str_shape)],
                    behaviours=[node(ast.Behaviour, name="Value")],
                ),
            ),
        ],
        result=node(
            ast.ProductShape,
            attrs=[node(ast.Attr, name="value", shape=str_shape)],
            behaviours=[node(ast.Behaviour, name="Value")],
        ),
        body=node(ast.Block, nodes=[]),
    )


def test_fun_shapes() -> None:
    assert parse_shape_alias("Foo = fun(a Int, b Str) Bool", "Foo") == node(
        ast.FunShape,
        params=[
            node(ast.Attr, name="a", shape=int_shape),
            node(ast.Attr, name="b", shape=str_shape),
        ],
        result=bool_shape,
    )


def test_shape_alias() -> None:
    assert parse_shape_alias("Foo = Bar", "Foo") == node(ast.ShapeRef, name="Bar")


def test_product_shapes() -> None:
    assert parse_shape_alias("Foo = {a Int}", "Foo") == node(
        ast.ProductShape, attrs=[node(ast.Attr, name="a", shape=int_shape)]
    )


def test_sum_shapes() -> None:
    assert parse_shape_alias("Foo = Int | Str", "Foo") == node(ast.SumShape, variants=[int_shape, str_shape])


def test_complex_shapes() -> None:
    assert parse_shape_alias("Foo = {a Int, b Str} | Int | {c {d Bool} + @Bar}", "Foo") == node(
        ast.SumShape,
        variants=[
            node(
                ast.ProductShape,
                attrs=[node(ast.Attr, name="a", shape=int_shape), node(ast.Attr, name="b", shape=str_shape)],
            ),
            int_shape,
            node(
                ast.ProductShape,
                attrs=[
                    node(
                        ast.Attr,
                        name="c",
                        shape=node(
                            ast.ProductShape,
                            attrs=[node(ast.Attr, name="d", shape=bool_shape)],
                            behaviours=[node(ast.Behaviour, name="Bar")],
                        ),
                    )
                ],
            ),
        ],
    )


def test_behaviour_has_to_come_after_composition() -> None:
    _, _, errors = parse("""
        Foo = {name Str} + @Value + {value {}}
    """)
    assert errors == ["Expected `behaviour identifier`, got `{`", "Unexpected token `}`"]


def test_shape_product_literal_basics() -> None:
    assert parse_first("{a = 42}") == node(
        ast.ShapeLit, attrs=[node(ast.ShapeLitAttr, name="a", value=node(ast.IntLit, value=42))]
    )


def test_shape_product_literal_with_shape_ref() -> None:
    assert parse_first("Foo{a = 42}") == node(
        ast.ShapeLit,
        shape_ref=node(ast.ShapeRef, name="Foo"),
        attrs=[node(ast.ShapeLitAttr, name="a", value=node(ast.IntLit, value=42))],
    )


def test_shape_literal_with_behaviour() -> None:
    assert parse_first("{a = 42} + @Foo") == node(
        ast.ShapeLit,
        attrs=[node(ast.ShapeLitAttr, name="a", value=node(ast.IntLit, value=42))],
        behaviours=[node(ast.Behaviour, name="Foo")],
    )


def test_composite_product_shape_literal() -> None:
    assert parse_first("{a = {b = 42} + Foo{c = 137} + @Bar}") == node(
        ast.ShapeLit,
        attrs=[
            node(
                ast.ShapeLitAttr,
                name="a",
                value=node(
                    ast.ShapeLit,
                    attrs=[node(ast.ShapeLitAttr, name="b", value=node(ast.IntLit, value=42))],
                    composites=[
                        node(
                            ast.ShapeLit,
                            attrs=[node(ast.ShapeLitAttr, name="c", value=node(ast.IntLit, value=137))],
                            shape_ref=node(ast.ShapeRef, name="Foo"),
                        )
                    ],
                    behaviours=[node(ast.Behaviour, name="Bar")],
                ),
            )
        ],
    )


def test_shape_literal_behaviour_has_to_come_after_composition() -> None:
    _, _, errors = parse("""
        Foo = {name Str} + @Value + {value {}}
    """)
    assert errors == ["Expected `behaviour identifier`, got `{`", "Unexpected token `}`"]


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
    assert str(parse_first("@Foo.bar = fun(foo Str): end")) == str(
        node(
            ast.FunDef,
            name="bar",
            namespace="Foo",
            params=[node(ast.FunParam, name="foo", shape=str_shape)],
            result=unit_shape,
            body=node(ast.Block, nodes=[]),
        )
    )


def parse_shape_alias(code: str, expected_name: str) -> ast.Shape:
    alias = parse_first(code)
    assert isinstance(alias, ast.ShapeAlias)
    assert alias.name == expected_name
    return alias.shape
