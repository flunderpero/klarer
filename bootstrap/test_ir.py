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

        age_Int_name_Str{I64, Str}

        declare main() none:
        block_1:
            _1 = 42
            _2 = alloc age_Int_name_Str{I64, Str}, I64 _1, Str s0
            ret none _none

    """)


def test_read_member() -> None:
    result = generate_ir("""
        main = fun() do
            foo = {name = "Peter", age = 42}
            print(foo.name)
        end
    """)
    assert str(result) == strip("""
        s0 = "Peter"

        age_Int_name_Str{I64, Str}

        declare main() none:
        block_1:
            _1 = 42
            _2 = alloc age_Int_name_Str{I64, Str}, I64 _1, Str s0
            _3 = getptr age_Int_name_Str{I64, Str} _2, *Str, 1
            _4 = load *Str _3
            call none print, Str _4
            ret none _none
    """)


def test_write_member() -> None:
    result = generate_ir("""
        main = fun() do
            foo = {name = "Peter", age = 42}
            foo.name = "John"
        end
    """)
    assert str(result) == strip("""
        s0 = "Peter"
        s1 = "John"

        age_Int_name_Str{I64, Str}

        declare main() none:
        block_1:
            _1 = 42
            _2 = alloc age_Int_name_Str{I64, Str}, I64 _1, Str s0
            _3 = getptr age_Int_name_Str{I64, Str} _2, *Str, 1
            store Str s1, _3
            ret none _none
    """)
