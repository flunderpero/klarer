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
