from __future__ import annotations

import pytest

from . import ast, types
from .conftest import empty_shape, typ, typecheck, typecheck_err


def test_literals() -> None:
    tc = typecheck("""
        42
        true
        'c'
        "hello"
    """)
    assert tc.type_at(1, 1, ast.IntLit) == types.IntShape
    assert tc.type_at(2, 1, ast.BoolLit) == types.BoolShape
    assert tc.type_at(3, 1, ast.CharLit) == types.CharShape
    assert tc.type_at(4, 1, ast.StrLit) == types.StrShape


def test_block() -> None:
    tc = typecheck("""
       :
            "hello"
            42
        end
    """)
    assert tc.type_at(1, 1, ast.Block) == types.IntShape


def test_assign() -> None:
    tc = typecheck("""
        a = 42
        a
    """)
    assert tc.type_at(1, 1, ast.Assign) == types.UnitShape
    assert tc.type_at(2, 1, ast.Name) == types.IntShape


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


def test_fun() -> None:
    tc = typecheck(""" f = fun(): 42 end """)
    assert tc.type_at(1, 1, ast.FunDef) == typ(types.FunShape, name="f", params=(), result=types.IntShape)
    tc = typecheck(""" main = fun(): end """)
    assert tc.type_at(1, 1, ast.FunDef) == typ(types.FunShape, name="main", params=(), result=types.UnitShape)


def test_fun_infer_from_member() -> None:
    tc = typecheck(
        """
            f = fun(a):
                a.value
                a
            end """
    )
    assert tc.type_at(1, 1, ast.FunDef) == typ(
        types.FunShape,
        name="f",
        params=(
            types.Attr(
                "a",
                typ(
                    types.ProductShape,
                    attrs=(types.Attr("value", empty_shape()),),
                ),
            ),
        ),
        result=typ(
            types.ProductShape,
            attrs=(
                types.Attr(
                    "value",
                    empty_shape(),
                ),
            ),
        ),
    )


def test_fun_infer_from_binop() -> None:
    tc = typecheck(""" f = fun(a): a == 42 end """)
    assert tc.type_at(1, 1, ast.FunDef) == typ(
        types.FunShape,
        name="f",
        params=(types.Attr("a", types.IntShape),),
        result=types.BoolShape,
    )


def test_fun_infer_from_accessing_member_of_shape() -> None:
    tc = typecheck("""
        f = fun(a):
            a.value.nested
        end
    """)
    assert tc.type_at(1, 1, ast.FunDef) == typ(
        types.FunShape,
        name="f",
        params=(
            types.Attr(
                "a",
                typ(
                    types.ProductShape,
                    attrs=(types.Attr("value", typ(types.ProductShape, attrs=(types.Attr("nested", empty_shape()),))),),
                ),
            ),
        ),
        result=empty_shape(),
    )


def test_fun_infer_from_assigning_shape_attr() -> None:
    tc = typecheck("""
        f = fun(a):
            b = a.value
            b.nested
        end
    """)
    assert tc.type_at(1, 1, ast.FunDef) == typ(
        types.FunShape,
        name="f",
        params=(
            types.Attr(
                "a",
                typ(
                    types.ProductShape,
                    attrs=(types.Attr("value", typ(types.ProductShape, attrs=(types.Attr("nested", empty_shape()),))),),
                ),
            ),
        ),
        result=empty_shape(),
    )


def test_fun_infer_from_beign_passed_to_fun() -> None:
    tc = typecheck("""
        f = fun(a):
            a.value
        end

        g = fun(a):
            f(a)
        end
    """)
    assert tc.type_at(1, 1, ast.FunDef) == typ(
        types.FunShape,
        name="f",
        params=(types.Attr("a", typ(types.ProductShape, attrs=(types.Attr("value", empty_shape()),))),),
        result=empty_shape(),
    )
    assert tc.type_at(5, 1, ast.FunDef) == typ(
        types.FunShape,
        name="g",
        params=(types.Attr("a", typ(types.ProductShape, attrs=(types.Attr("value", empty_shape()),))),),
        result=empty_shape(),
    )


def test_fun_infer_from_being_called() -> None:
    tc = typecheck("""
        f = fun(g):
            g(42, "hello")
        end
    """)
    assert str(tc.type_at(1, 1, ast.FunDef)) == str(
        typ(
            types.FunShape,
            name="f",
            params=(
                types.Attr(
                    "g",
                    typ(
                        types.FunShape,
                        name="g",
                        params=(types.Attr("$0", types.IntShape), types.Attr("$1", types.StrShape)),
                        result=empty_shape(),
                    ),
                ),
            ),
            result=empty_shape(),
        )
    )


def test_call_without_params() -> None:
    tc = typecheck("""
        f = fun(): 42 end
        f()
    """)
    assert tc.type_at(2, 1, ast.Call) == types.IntShape


def test_call_specialization() -> None:
    tc = typecheck("""
        f = fun(a): a end
        f(42)
        f(true)
        f("hello")
    """)
    assert tc.type_at(2, 1, ast.Name) == typ(
        types.FunShape, name="f", params=(types.Attr("a", types.IntShape),), result=types.IntShape
    )
    assert tc.type_at(2, 1, ast.Call) == types.IntShape

    assert tc.type_at(3, 1, ast.Name) == typ(
        types.FunShape, name="f", params=(types.Attr("a", types.BoolShape),), result=types.BoolShape
    )
    assert tc.type_at(3, 1, ast.Call) == types.BoolShape

    assert tc.type_at(4, 1, ast.Name) == typ(
        types.FunShape, name="f", params=(types.Attr("a", types.StrShape),), result=types.StrShape
    )
    assert tc.type_at(4, 1, ast.Call) == types.StrShape


def test_simple_product_shape() -> None:
    tc = typecheck("""
        Person = {name Str, age Int}
    """)
    assert tc.type_at(1, 1, ast.ProductShape) == typ(
        types.ProductShape,
        attrs=(types.Attr("name", types.StrShape), types.Attr("age", types.IntShape)),
    )


def test_shape_literal() -> None:
    tc = typecheck("""
        Person = {name Str, age Int}

        main = fun():
            p = Person{name = "John", age = 42}
        end
    """)
    assert tc.type_at(1, 1, ast.ProductShape) == typ(
        types.ProductShape,
        attrs=(types.Attr("name", types.StrShape), types.Attr("age", types.IntShape)),
    )
    assert str(tc.type_at(4, 1, ast.ShapeLit)) == str(
        typ(
            types.ProductShape,
            attrs=(types.Attr("name", types.StrShape), types.Attr("age", types.IntShape)),
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
    assert tc.type_at(3, 1, ast.Name) == typ(
        types.ProductShape,
        attrs=(
            types.Attr("value", typ(types.ProductShape, attrs=(types.Attr("name", types.StrShape),), behaviours=())),
        ),
        behaviours=(),
    )


def test_read_member() -> None:
    tc = typecheck("""
        foo = {name = "Peter", age = 42}
        foo.name
    """)
    assert tc.type_at(2, 1, ast.Member) == types.StrShape


def test_behaviour() -> None:
    tc = typecheck("""
        @Value.print_value = fun(v):
            print(v.value)
        end

        v = {value = "PASS"} + @Value
        v.print_value()
    """)
    assert tc.type_at(1, 1, ast.FunDef) == typ(
        types.FunShape,
        name="print_value",
        namespace="Value",
        params=(types.Attr("v", typ(types.ProductShape, attrs=(types.Attr("value", empty_shape()),), behaviours=())),),
        result=types.UnitShape,
    )
    assert str(tc.type_at(6, 1, ast.Member)) == str(
        typ(
            types.FunShape,
            name="print_value",
            namespace="Value",
            params=(
                types.Attr(
                    "v",
                    typ(
                        types.ProductShape,
                        attrs=(types.Attr("value", types.StrShape),),
                        behaviours=("@Value",),
                    ),
                ),
            ),
            result=types.UnitShape,
        )
    )
