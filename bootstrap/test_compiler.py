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


def test_simple_product_shape_literal() -> None:
    stdout = compile_and_run_success(
        """
        f = fun (a) do end

        main = fun() do
            p = {name = "John", age = 42}
            f(p)
            print("PASS")
        end
    """
    )
    # todo: do something with `p` in the code above
    # todo: the `f` function above is only there to prevent unused variable warnings
    #       in Go.
    assert stdout == stripln("""
        PASS
    """)


def test_nested_product_shape_literal() -> None:
    stdout = compile_and_run_success(
        """
        f = fun (a) do end

        main = fun() do
            p = {name = "John", age = {years = 42}}
            f(p)
            print("PASS")
        end
    """
    )
    # todo: do something with `p` in the code above
    # todo: the `f` function above is only there to prevent unused variable warnings
    #       in Go.
    assert stdout == stripln("""
        PASS
    """)
