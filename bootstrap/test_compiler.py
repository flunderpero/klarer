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


def test_mut_assign_shapes_with_different_attribute_order() -> None:
    stdout = compile_and_run_success("""
        main = fun() do
            mut x = {name = "PASS1", age = 42}
            print(x.name)
            x = {age = 24, name = "PASS2"}
            print(x.name)
        end
    """)
    assert stdout == stripln("""
        PASS1
        PASS2
    """)


def test_shape_literal_can_subsume_shape_alias() -> None:
    stdout = compile_and_run_success("""
        Value = {value {}}

        main = fun() do
            v = Value{value = {status = "PASS"}}
            print(v.value.status)
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
