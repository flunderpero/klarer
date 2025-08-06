from __future__ import annotations

from . import ast, types
from .conftest import typecheck


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


def test_fun() -> None:
    tc = typecheck(""" f = fun() do 42 end """)
    assert tc.type_at(1, 1, ast.FunDef) == types.Typ(
        types.Fun("f", [], types.IntTyp, types.builtin_span, builtin=False)
    )
    tc = typecheck(""" main = fun() do end """)
    assert tc.type_at(1, 1, ast.FunDef) == types.Typ(
        types.Fun("main", [], types.UnitTyp, types.builtin_span, builtin=False)
    )


def test_fun_infer_from_binop() -> None:
    tc = typecheck(""" f = fun(a) do a == 42 end """)
    assert tc.type_at(1, 1, ast.FunDef) == types.Typ(
        types.Fun("f", [types.Attr("a", types.IntTyp)], types.BoolTyp, types.builtin_span, builtin=False)
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
    assert tc.type_at(2, 1, ast.Call) == types.IntTyp
    assert tc.type_at(3, 1, ast.Call) == types.BoolTyp
    assert tc.type_at(4, 1, ast.Call) == types.StrTyp
