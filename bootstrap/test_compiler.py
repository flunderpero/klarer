from __future__ import annotations

from .conftest import compile_and_run_success, stripln


def test_happy_path() -> None:
    stdout = compile_and_run_success("""

        main = fun() do
            print("PASS")
        end

    """)
    assert stdout == stripln("""
        PASS
    """)


def test_call() -> None:
    stdout = compile_and_run_success("""

        what_to_print = fun() do
            "PASS"
        end

        main = fun() do
            print(what_to_print())
        end

    """)
    assert stdout == stripln("""
        PASS
    """)


def test_read_member_of_simple_shape_literal() -> None:
    stdout = compile_and_run_success("""
        main = fun() do
            foo = {name = "PASS", age = 42}
            print(foo.name)
        end
    """)
    assert stdout == stripln("""
        PASS
    """)


def test_read_member_of_nested_shape_literal() -> None:
    stdout = compile_and_run_success(
        """
        main = fun() do
            foo = {value = {pass = "PASS", age = 42}}
            print(foo.value.pass)
        end
    """
    )
    assert stdout == stripln("""
        PASS
    """)


def test_write_member_of_simple_shape_literal() -> None:
    stdout = compile_and_run_success(
        """
        main = fun() do
            foo = {name = "FAIL", age = 42}
            foo.name = "PASS"
            print(foo.name)
        end
    """
    )
    assert stdout == stripln("""
        PASS
    """)


def test_write_member_of_nested_shape_literal() -> None:
    stdout = compile_and_run_success(
        """
        main = fun() do
            foo = {value = {pass = "FAIL", age = 42}}
            foo.value.pass = "PASS"
            print(foo.value.pass)
        end
    """
    )
    assert stdout == stripln("""
        PASS
    """)
