from __future__ import annotations

from . import token


def test_token() -> None:
    tokens, errors = token.tokenize(
        token.Input(
            "test.kl",
            """
    -- This is a comment

    List = {elements []{}, index Int}

    @List.new = fun(): end

    42 "hello" 'c' true false

    x = 42
    """,
        )
    )
    assert errors == []
    assert [(t.kind, t.value) for t in tokens] == [
        (token.Kind.comment, "-- This is a comment"),
        (token.Kind.type_ident, "List"),
        (token.Kind.eq, None),
        (token.Kind.curly_left, None),
        (token.Kind.ident, "elements"),
        (token.Kind.braket_left, None),
        (token.Kind.braket_right, None),
        (token.Kind.curly_left, None),
        (token.Kind.curly_right, None),
        (token.Kind.comma, None),
        (token.Kind.ident, "index"),
        (token.Kind.type_ident, "Int"),
        (token.Kind.curly_right, None),
        (token.Kind.behaviour_ns, "List"),
        (token.Kind.dot, None),
        (token.Kind.ident, "new"),
        (token.Kind.eq, None),
        (token.Kind.fun, None),
        (token.Kind.paren_left, None),
        (token.Kind.paren_right, None),
        (token.Kind.colon, None),
        (token.Kind.end, None),
        (token.Kind.int_lit, "42"),
        (token.Kind.str_lit, "hello"),
        (token.Kind.char_lit, "c"),
        (token.Kind.true, None),
        (token.Kind.false, None),
        (token.Kind.ident, "x"),
        (token.Kind.eq, None),
        (token.Kind.int_lit, "42"),
        (token.Kind.eof, None),
    ]
