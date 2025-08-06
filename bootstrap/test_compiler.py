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
