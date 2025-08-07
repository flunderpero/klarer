from __future__ import annotations

from .conftest import generate_ir, strip


def test_literals() -> None:
    result = generate_ir("""
    main = fun() do
        42
        true
        a = 'c'
    end
    """)
    assert str(result) == strip("""
        declare main() none:
        block_1:
            _1 = 42
            _2 = 1
            _3 = 99
            ret none _none
    """)


def test_call_simple() -> None:
    result = generate_ir("""
    main = fun() do
        print("hello")
    end
    """)
    assert str(result) == strip("""
        s0 = "hello"

        declare main() none:
        block_1:
            call none print, Str s0
            ret none _none
    """)


def test_call_multiple() -> None:
    result = generate_ir("""

    foo = fun() do "hello" end

    main = fun() do
        print(foo())
    end

    """)
    assert str(result) == strip("""
        s0 = "hello"

        declare foo_Str() Str:
        block_1:
            ret Str s0

        declare main() none:
        block_1:
            _1 = call Str foo_Str
            call none print, Str _1
            ret none _none
    """)


def test_simple_product_shape_literal() -> None:
    result = generate_ir("""
        main = fun() do
            p = {name = "John", age = 42}
        end
    """)
    assert str(result) == strip("""
        s0 = "John"

        name_Str_age_Int{Str, I64}

        declare main() none:
        block_1:
            _1 = 42
            _2 = alloc name_Str_age_Int{Str, I64}, Str s0, I64 _1
            ret none _none

    """)
