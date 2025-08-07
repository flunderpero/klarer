from __future__ import annotations

from . import ast, types
from .conftest import typ, typecheck, typecheck_err


def test_literals() -> None:
    tc = typecheck("""
        42
        true
        'c'
        "hello"
    """)
    assert tc.type_at(1, 1, ast.IntLit) == types.IntTyp
    assert tc.type_at(2, 1, ast.BoolLit) == types.BoolTyp
    assert tc.type_at(3, 1, ast.CharLit) == types.CharTyp
    assert tc.type_at(4, 1, ast.StrLit) == types.StrTyp


def test_block() -> None:
    tc = typecheck("""
        do
            "hello"
            42
        end
    """)
    assert tc.type_at(1, 1, ast.Block) == types.IntTyp


def test_assign() -> None:
    tc = typecheck("""
        a = 42
        a
    """)
    assert tc.type_at(1, 1, ast.Assign) == types.UnitTyp
    assert tc.type_at(2, 1, ast.Name) == types.IntTyp


def test_assign_shape_literal_must_conform() -> None:
    # Missing attribute.
    _, errors = typecheck_err("""
        Person = {name Str, age Int}
        mut foo = Person{name = "Peter", age = 42}
        foo = {age = 24}
    """)
    assert errors == ["`{age Int}` is not assignable to `{name Str, age Int}`"]

    # Wrong type.
    _, errors = typecheck_err("""
        mut foo = {field = 42}
        foo = {field = "Peter"}
    """)
    assert errors == ["`{field Str}` is not assignable to `{field Int}`"]

    # More attributes.
    _, errors = typecheck_err("""
        mut foo = {name = "Peter", age = 42}
        foo = {name = "Paul", age = 24, profession = "Nerd"}
    """)
    assert errors == ["`{name Str, age Int, profession Str}` is not assignable to `{name Str, age Int}`"]

    # Nested shapes.
    _, errors = typecheck_err("""
        mut foo = {value = {pass = "Peter", age = 42}}
        foo.value = {pass = "Paul"}
    """)
    assert errors == ["`{pass Str}` is not assignable to `{pass Str, age Int}`"]


def test_fun() -> None:
    tc = typecheck(""" f = fun() do 42 end """)
    assert tc.type_at(1, 1, ast.FunDef) == types.Typ(
        types.Fun("f", (), types.IntTyp, types.builtin_span, builtin=False)
    )
    tc = typecheck(""" main = fun() do end """)
    assert tc.type_at(1, 1, ast.FunDef) == types.Typ(
        types.Fun("main", (), types.UnitTyp, types.builtin_span, builtin=False)
    )


def test_fun_infer_from_binop() -> None:
    tc = typecheck(""" f = fun(a) do a == 42 end """)
    assert tc.type_at(1, 1, ast.FunDef) == types.Typ(
        types.Fun("f", (types.Attr("a", types.IntTyp),), types.BoolTyp, types.builtin_span, builtin=False)
    )


def test_call_without_params() -> None:
    tc = typecheck("""
        f = fun() do 42 end
        f()
    """)
    assert tc.type_at(2, 1, ast.Call) == types.IntTyp


def test_call_specialization() -> None:
    tc = typecheck("""
        f = fun(a) do a end
        f(42)
        f(true)
        f("hello")
    """)
    assert tc.type_at(2, 1, ast.Name) == typ(
        types.Fun, name="f", params=(types.Attr("a", types.IntTyp),), result=types.IntTyp
    )
    assert tc.type_at(2, 1, ast.Call) == types.IntTyp

    assert tc.type_at(3, 1, ast.Name) == typ(
        types.Fun, name="f", params=(types.Attr("a", types.BoolTyp),), result=types.BoolTyp
    )
    assert tc.type_at(3, 1, ast.Call) == types.BoolTyp

    assert tc.type_at(4, 1, ast.Name) == typ(
        types.Fun, name="f", params=(types.Attr("a", types.StrTyp),), result=types.StrTyp
    )
    assert tc.type_at(4, 1, ast.Call) == types.StrTyp


def test_simple_product_shape() -> None:
    tc = typecheck("""
        Person = {name Str, age Int}
    """)
    assert tc.type_at(1, 1, ast.ProductShape) == typ(
        types.Shape,
        attrs=(types.Attr("name", types.StrTyp), types.Attr("age", types.IntTyp)),
    )


def test_shape_literal() -> None:
    tc = typecheck("""
        Person = {name Str, age Int}

        main = fun() do
            p = Person{name = "John", age = 42}
        end
    """)
    assert tc.type_at(1, 1, ast.ProductShape) == typ(
        types.Shape,
        attrs=(types.Attr("name", types.StrTyp), types.Attr("age", types.IntTyp)),
    )
    assert str(tc.type_at(4, 1, ast.ShapeLit)) == str(
        typ(
            types.Shape,
            attrs=(types.Attr("name", types.StrTyp), types.Attr("age", types.IntTyp)),
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


def test_shape_literal_can_subsume_shape_alias() -> None:
    tc = typecheck("""
        Value = {value {}}
        v = Value{value = {name = "Peter"}}
        v
    """)
    assert tc.type_at(3, 1, ast.Name) == typ(
        types.Shape,
        attrs=(
            types.Attr(
                "value", typ(types.Shape, attrs=(types.Attr("name", types.StrTyp),), variants=(), behaviours=())
            ),
        ),
        variants=(),
        behaviours=(),
    )


def test_read_member() -> None:
    tc = typecheck("""
        foo = {name = "Peter", age = 42}
        foo.name
    """)
    assert tc.type_at(2, 1, ast.Member) == types.StrTyp


def test_write_member() -> None:
    typecheck("""
        foo = {name = "Peter", age = 42}
        foo.name = "John"
    """)

    _, errors = typecheck_err("""
        foo = {name = "Peter", age = 42}
        foo.name = 42
    """)
    assert errors == ["`Int` is not assignable to `Str`"]
