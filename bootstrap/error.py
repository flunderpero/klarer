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
    cause: Error | None
    stacktrace: str

    def __str__(self) -> str:
        code = "\n".join(self.span.formatted_lines())
        s = f"{self.span}: {self.message}\n{code}"
        if self.defined_here.start == 0 and self.defined_here.end == 0:
            return s
        defined_here_code = "\n".join(self.defined_here.formatted_lines())
        s += f"\nDefined here:\n{defined_here_code}"
        if self.cause:
            s += f"\n    Caused by: {str(self.cause).replace('\n', '\n    ')}"
        return s

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
    return WithDefinitionError(span, f"No member `{name}` in type `{target}`", target_defined_here, None, _stack())


def unexpected_shape(expected: str, got: str, span: Span) -> Error:
    return SimpleError(span, f"Expected {expected}, got {got}", _stack())


def cascaded_error(cause: Error, span: Span) -> Error:
    return CascadedError(span, cause, _stack())


def does_not_conform_to(it: str, to: str, span: Span, defined_here: Span, cause: Error | None) -> Error:
    return WithDefinitionError(span, f"`{it}` does not conform to shape `{to}`", defined_here, cause, _stack())


def is_not_same(it: str, as_: str, span: Span) -> Error:
    return SimpleError(span, f"`{it}` is not the same shape as `{as_}`", _stack())


def not_callable(span: Span, defined_here: Span) -> Error:
    return WithDefinitionError(span, "Only functions and structs can be called", defined_here, None, _stack())


def invalid_main(span: Span) -> Error:
    # todo: How to specify the unit type?
    return SimpleError(span, "`main` must conform to the signature `main() -> None`", _stack())


def cannot_add_method_to_interface_behaviour(behaviour: str, fun: str, span: Span) -> Error:
    return SimpleError(span, f"Cannot add method `{fun}` to interface behaviour `{behaviour}`.", _stack())


def cannot_add_interface_method_to_non_interface_behaviour(behaviour: str, fun: str, span: Span) -> Error:
    return SimpleError(span, f"Cannot add interface method `{fun}` to non-interface behaviour `{behaviour}`", _stack())


def shape_is_not_a_variant(it: str, sum_shape: str, span: Span, defined_here: Span) -> Error:
    return WithDefinitionError(span, f"`{it}` is not a variant of `{sum_shape}`", defined_here, None, _stack())


def field_not_found(name: str, span: Span, defined_here: Span) -> Error:
    return WithDefinitionError(span, f"Field `{name}` not found", defined_here, None, _stack())


def variant_not_found(variant: str, span: Span, defined_here: Span) -> Error:
    return WithDefinitionError(span, f"Variant `{variant}` not found", defined_here, None, _stack())


def function_result_does_not_conform(self_fun: str, other_fun: str, self_span: Span, other_span: Span) -> Error:
    return WithDefinitionError(
        self_span, f"Function result `{self_fun}` does not conform to `{other_fun}`", other_span, None, _stack()
    )


def wrong_number_of_parameters(self_fun: str, other_fun: str, self_span: Span, other_span: Span) -> Error:
    return WithDefinitionError(
        self_span,
        f"Wrong number of parameters for function `{self_fun}` compared to `{other_fun}`",
        other_span,
        None,
        _stack(),
    )


def failed_to_specialize(specialized: str, base: str, self_span: Span, other_span: Span, error: Error) -> Error:
    return WithDefinitionError(
        self_span,
        f"`{base}` cannot be called as `{specialized}`",
        other_span,
        error,
        _stack(),
    )
