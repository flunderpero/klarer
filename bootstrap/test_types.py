from __future__ import annotations

import pytest

from . import ast, types
from .conftest import shape, typecheck, typecheck_err


def test_literals() -> None:
    tc = typecheck("""
        42
        true
        'c'
        "hello"
    """)
    assert tc.at(1, 1, ast.IntLit) == types.Int
    assert tc.at(2, 1, ast.BoolLit) == types.Bool
    assert tc.at(3, 1, ast.CharLit) == types.Char
    assert tc.at(4, 1, ast.StrLit) == types.Str


def test_block() -> None:
    tc = typecheck("""
       :
            "hello"
            42
        end
    """)
    assert tc.at(1, 1, ast.Block) == types.Int


def test_assign() -> None:
    tc = typecheck("""
        a = 42
        a
    """)
    assert tc.at(1, 1, ast.Assign) == types.Unit
    assert tc.at(2, 1, ast.Name) == types.Int


@pytest.mark.skip
def test_assign_shape_literal_must_conform() -> None:
    # Missing attribute.
    _, errors = typecheck_err("""
        Person = {name Str, age Int}
        mut foo = Person{name = "Peter", age = 42}
        foo = {age = 24}
    """)
    assert errors == ["`{age Int}` is not the same shape as `{name Str, age Int}`"]

    # Wrong type.
    _, errors = typecheck_err("""
        mut foo = {field = 42}
        foo = {field = "Peter"}
    """)
    assert errors == ["`{field Str}` is not the same shape as `{field Int}`"]

    # More attributes.
    _, errors = typecheck_err("""
        mut foo = {name = "Peter", age = 42}
        foo = {name = "Paul", age = 24, profession = "Nerd"}
    """)
    assert errors == ["`{name Str, age Int, profession Str}` is not the same shape as `{name Str, age Int}`"]

    # Nested shapes.
    _, errors = typecheck_err("""
        mut foo = {value = {pass = "Peter", age = 42}}
        foo.value = {pass = "Paul"}
    """)
    assert errors == ["`{pass Str}` is not the same shape as `{pass Str, age Int}`"]


def test_fun_basics() -> None:
    tc = typecheck(""" f = fun(): 42 end """)
    assert tc.at(1, 1, ast.FunDef) == shape(types.FunShape, name="f", params=(), result=types.Int)
    tc = typecheck(""" main = fun(): end """)
    assert tc.at(1, 1, ast.FunDef) == shape(types.FunShape, name="main", params=(), result=types.Unit)


def test_call_without_params() -> None:
    tc = typecheck("""
        f = fun(): 42 end
        f()
    """)
    assert tc.at(2, 1, ast.Call) == types.Int


def test_call_specialization() -> None:
    tc = typecheck("""
        f = fun(a Int | Bool | Str): a end
        f(42)
        f(true)
        f("hello")
    """)
    assert tc.at(2, 1, ast.Name) == shape(
        types.FunShape, name="f", params=(types.Attr("a", types.Int),), result=types.Int
    )
    assert tc.at(2, 1, ast.Call) == types.Int

    assert tc.at(3, 1, ast.Name) == shape(
        types.FunShape, name="f", params=(types.Attr("a", types.Bool),), result=types.Bool
    )
    assert tc.at(3, 1, ast.Call) == types.Bool

    assert tc.at(4, 1, ast.Name) == shape(
        types.FunShape, name="f", params=(types.Attr("a", types.Str),), result=types.Str
    )
    assert tc.at(4, 1, ast.Call) == types.Str


def test_simple_product_shape() -> None:
    tc = typecheck("""
        Person = {name Str, age Int}
    """)
    assert tc.at(1, 1, ast.ProductShape) == shape(
        types.ProductShape,
        attrs=(types.Attr("name", types.Str), types.Attr("age", types.Int)),
    )


def test_shape_literal_basics() -> None:
    tc = typecheck("""
        Person = {name Str, age Int}

        main = fun():
            p = Person{name = "John", age = 42}
        end
    """)
    assert tc.at(1, 1, ast.ProductShape) == shape(
        types.ProductShape,
        attrs=(types.Attr("name", types.Str), types.Attr("age", types.Int)),
    )
    assert str(tc.at(4, 1, ast.ShapeLit)) == str(
        shape(
            types.ProductShape,
            name="Person",
            attrs=(types.Attr("name", types.Str), types.Attr("age", types.Int)),
        )
    )


def test_shape_literal_must_conform_to_shape_alias() -> None:
    # Missing attribute.
    _, errors = typecheck_err("""
        Person = {name Str, age Int}
        foo = Person{age = 42}
    """)
    assert errors == ["`{age Int}` does not conform to shape `Person`"]

    # Wrong type.
    _, errors = typecheck_err("""
        Value = {value Str}
        v = Value{value = 42}
    """)
    assert errors == ["`{value Int}` does not conform to shape `Value`"]


def test_shape_literal_can_conform_to_shape_alias() -> None:
    tc = typecheck("""
        Value = {value {}}
        v = Value{value = {name = "Peter"}}
        v
    """)
    assert tc.at(3, 1, ast.Name) == shape(
        types.ProductShape,
        name="Value",
        attrs=(types.Attr("value", shape(types.ProductShape, attrs=(types.Attr("name", types.Str),))),),
    )


def test_read_member() -> None:
    tc = typecheck("""
        foo = {name = "Peter", age = 42}
        foo.name
    """)
    assert tc.at(2, 1, ast.Member) == types.Str


def test_behaviour() -> None:
    tc = typecheck("""
        @Value.print_value = fun(v {value Str}):
            print(v.value)
        end

        v = {value = "PASS"} + @Value
        v.print_value()
    """)
    assert tc.at(1, 1, ast.FunDef) == shape(
        types.FunShape,
        name="print_value",
        namespace="Value",
        params=(
            types.Attr(
                "v",
                shape(
                    types.ProductShape,
                    attrs=(types.Attr("value", shape(types.PrimitiveShape, name="Str")),),
                ),
            ),
        ),
        result=types.Unit,
    )
    assert str(tc.at(6, 1, ast.Member)) == str(
        shape(
            types.FunShape,
            name="print_value",
            namespace="Value",
            params=(
                types.Attr(
                    "v",
                    shape(
                        types.ProductShape,
                        attrs=(types.Attr("value", types.Str),),
                        behaviours=("@Value",),
                    ),
                ),
            ),
            result=types.Unit,
        )
    )
