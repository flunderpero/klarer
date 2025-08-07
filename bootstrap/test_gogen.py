from __future__ import annotations

from .conftest import codegen, strip


def test_happy_path() -> None:
    r = codegen("""
        main = fun() do
            print("Hello, world!")
        end
    """)
    assert r == strip("""
        package main

        var s0 = "Hello, world!"

        func main() {
            Print(s0)
        }
    """)


def test_simple_product_shape_literal() -> None:
    r = codegen("""
        main = fun() do
            p = {name = "John", age = 42}
        end
    """)
    assert r == strip("""
        package main

        var s0 = "John"

        type name_Str_age_Int struct{_0 string; _1 int}

        func main() {
            _1 := 42
            _2 := name_Str_age_Int{_0: s0, _1: _1}
        }
    """)
