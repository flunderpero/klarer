from __future__ import annotations

from dataclasses import dataclass
from traceback import format_stack
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .span import Span


@dataclass(unsafe_hash=True)
class SimpleError:
    span: Span
    message: str
    stacktrace: str

    def __str__(self) -> str:
        code = "\n".join(self.span.formatted_lines())
        return f"{self.span}: {self.message}\n{code}"

    def short_message(self) -> str:
        return self.message


@dataclass(unsafe_hash=True)
class DuplicateError:
    name: str
    span: Span
    defined_here: Span
    stacktrace: str

    def __str__(self) -> str:
        code = "\n".join(self.span.formatted_lines())
        defined_here_code = "\n".join(self.defined_here.formatted_lines())
        return f"{self.span}: Duplicate `{self.name}` at:\n{code}\nis already defined here:\n{defined_here_code}"

    def short_message(self) -> str:
        return f"Duplicate `{self.name}`"


@dataclass(unsafe_hash=True)
class WithDefinitionError:
    span: Span
    message: str
    defined_here: Span
    stacktrace: str

    def __str__(self) -> str:
        code = "\n".join(self.span.formatted_lines())
        s = f"{self.span}: {self.message}\n{code}"
        if self.defined_here.start == 0 and self.defined_here.end == 0:
            return s
        defined_here_code = "\n".join(self.defined_here.formatted_lines())
        return s + f"\nDefined here:\n{defined_here_code}"

    def short_message(self) -> str:
        return self.message


@dataclass(unsafe_hash=True)
class CascadedError:
    span: Span
    cause: Error
    stacktrace: str

    def __str__(self) -> str:
        return f"{self.span}: {self.cause}"

    def short_message(self) -> str:
        return f"Cascaded error: {self.cause.short_message()}"


Error = SimpleError | WithDefinitionError | DuplicateError | CascadedError


def _stack() -> str:
    return "".join(x for x in format_stack()[:-2] if "bootstrap" in x)


def unknown_token(span: Span, token: str) -> Error:
    return SimpleError(span, f"Unknown token `{token}`", _stack())


def unterminated_str_lit(span: Span, *, eof: bool) -> Error:
    return SimpleError(
        span,
        "Unexpected end of file while parsing string literal" if eof else "Unterminated string literal",
        _stack(),
    )


def unterminated_char_lit(span: Span, *, eof: bool) -> Error:
    return SimpleError(
        span,
        "Unexpected end of file while parsing char literal" if eof else "Unterminated char literal",
        _stack(),
    )


def unexpected_token(span: Span, got: str, *expected: str) -> Error:
    if not expected:
        return SimpleError(span, f"Unexpected token `{got}`", _stack())
    expected_names = ", ".join(f"`{x}`" for x in expected)
    prefix = "Expected one of " if len(expected) > 1 else "Expected "
    return SimpleError(span, f"{prefix}{expected_names}, got `{got}`", _stack())


def if_condition_must_not_contain_assigment(span: Span) -> Error:
    return SimpleError(span, "If condition must not contain assignment", _stack())


def expected_ident(expr: str, span: Span) -> Error:
    return SimpleError(span, f"Expected an identifier (lowercase), got `{expr}`", _stack())


def duplicate_declaration(name: str, span: Span, defined_here: Span) -> Error:
    return DuplicateError(name, span, defined_here, _stack())


def undefined_name(name: str, span: Span) -> Error:
    return SimpleError(span, f"Undefined name `{name}`", _stack())


def no_member(name: str, target: str, span: Span, target_defined_here: Span) -> Error:
    return WithDefinitionError(span, f"No member `{name}` in type `{target}`", target_defined_here, _stack())


def unexpected_shape(expected: str, got: str, span: Span) -> Error:
    return SimpleError(span, f"Expected {expected}, got {got}", _stack())


def cascaded_error(cause: Error, span: Span) -> Error:
    return CascadedError(span, cause, _stack())


def does_not_subsume(it: str, to: str, span: Span) -> Error:
    return SimpleError(span, f"`{it}` does not conform to shape `{to}`", _stack())


def is_not_same(it: str, as_: str, span: Span) -> Error:
    return SimpleError(span, f"`{it}` is not the same shape as `{as_}`", _stack())


def not_callable(span: Span, defined_here: Span) -> Error:
    return WithDefinitionError(span, "Only functions and structs can be called", defined_here, _stack())


def invalid_main(span: Span) -> Error:
    # todo: How to specify the unit type?
    return SimpleError(span, "`main` must conform to the signature `main() -> None`", _stack())
