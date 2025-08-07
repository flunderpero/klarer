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
            _2 := &name_Str_age_Int{_0: s0, _1: _1}
        }
    """)


def test_read_member() -> None:
    r = codegen("""
        main = fun() do
            foo = {name = "Peter", age = 42}
            print(foo.name)
        end
    """)
    assert r == strip("""
        package main

        var s0 = "Peter"

        type name_Str_age_Int struct{_0 string; _1 int}

        func main() {
            _1 := 42
            _2 := &name_Str_age_Int{_0: s0, _1: _1}
            _4 := _2._0
            Print(_4)
        }
    """)


def test_write_member() -> None:
    r = codegen("""
        main = fun() do
            foo = {name = "FAIL", age = 42}
            foo.name = "PASS"
            print(foo.name)
        end
    """)
    assert r == strip("""
        package main

        var s0 = "FAIL"
        var s1 = "PASS"

        type name_Str_age_Int struct{_0 string; _1 int}

        func main() {
            _1 := 42
            _2 := &name_Str_age_Int{_0: s0, _1: _1}
            _2._0 = s1
            _5 := _2._0
            Print(_5)
        }
    """)
