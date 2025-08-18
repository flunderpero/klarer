from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Literal

from . import ast, error
from .span import Span, log

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass(eq=True, frozen=True)
class Behaviour:
    name: str
    funs: tuple[FunShape, ...]
    interface: bool

    def __str__(self) -> str:
        interface = "(interface)" if self.interface else ""
        return self.name + interface

    def mangled_name(self) -> str:
        return self.name

    def fun(self, name: str) -> FunShape | None:
        for fun in self.funs:
            if fun.name == name:
                return fun
        return None


@dataclass(eq=True, frozen=True)
class Behaviours:
    behaviours: tuple[str, ...]
    scope: Scope = field(compare=False, hash=False, repr=False)

    # def not_conforms_to(self, other: Behaviours) -> error.Error | None:
    #     """`self` conforms to `other` if it has at least all the behaviour _function_ of `other`."""
    #     self_funs = self.functions()
    #     other_funs = other.functions()
    #     for other_fun in other_funs:
    #         # Find a function with the same name in `self_funs`.
    #         self_fun = next((x for x in self_funs if x.name == other_fun.name), None)
    #         if not self_fun:
    #             return error.behaviour_method_not_found(str(other_fun), other_fun.span)
    #         if not self_fun.not_conforms_to(other_fun):
    #             return error.behaviour_method_does_not_conform(
    #                 str(self_fun), str(other_fun), self_fun.span, other_fun.span
    #             )
    #     return None

    def fun(self, name: str) -> FunShape | None:
        for behaviour_name in reversed(self.behaviours):
            behaviour_binding = self.scope.lookup(behaviour_name)
            assert behaviour_binding is not None, f"Behaviour {behaviour_name} not found"
            assert isinstance(behaviour_binding.value, Behaviour)
            fun = behaviour_binding.value.fun(name)
            if fun:
                return fun
        return None

    def merge(self, other: Behaviours) -> Behaviours:
        behaviours = list(self.behaviours)
        for behaviour in other.behaviours:
            if behaviour not in behaviours:
                behaviours.append(behaviour)
        scope = self.scope if other.scope.is_child_of(self.scope) else other.scope
        return Behaviours(tuple(behaviours), scope)

    def __str__(self) -> str:
        return " + ".join(str(x) for x in self.behaviours)


@dataclass(eq=True, frozen=True)
class PrimitiveShape:
    name: Literal["Bool", "Char", "Int", "Str"]
    behaviours: Behaviours
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return self.name

    def mangled_name(self) -> str:
        return self.name

    def not_conforms_to(self, other: Shape) -> error.Error | None:
        # For now, all primitives only conform to themselves, a sum shape, or the empty shape.
        # Later on, when we have different sized integers, I8 will conform to I16, etc.
        if self == other:
            return None
        if isinstance(other, ProductShape) and other.is_empty():
            return None
        if not isinstance(other, SumShape):
            return error.does_not_conform_to(str(self), str(other), self.span, other.span, None)
        if not any(not self.not_conforms_to(x) for x in other.variants):
            return error.shape_is_not_a_variant(str(self), str(other), self.span, other.span)
        return None


@dataclass(eq=True, frozen=True)
class UnitShape:
    behaviours: Behaviours
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return "<unit>"

    def mangled_name(self) -> str:
        return "_unit_"

    def not_conforms_to(self, other: Shape) -> error.Error | None:
        if not isinstance(other, UnitShape):
            return error.unexpected_shape("<unit>", str(other), other.span)
        return None


@dataclass(frozen=True)
class Field:
    name: str
    scope: Scope = field(compare=False, hash=False, repr=False)
    shape_: Shape | None
    shape_name: str

    @staticmethod
    def with_shape(name: str, shape: Shape) -> Field:
        return Field(name, Scope.empty(), shape, name)

    @staticmethod
    def with_shape_name(name: str, shape_name: str, scope: Scope) -> Field:
        return Field(name, scope, None, shape_name)

    @property
    def shape(self) -> Shape:
        if self.shape_ is not None:
            return self.shape_
        binding = self.scope.lookup(self.shape_name)
        assert binding is not None, f"ShapeRef bindings should always be found: {self.shape_name}"
        assert isinstance(binding.value, Shape), f"Expected Shape, got {binding.value}"
        return binding.value

    def __str__(self) -> str:
        return self.name + " " + str(self.shape)

    def mangled_name(self) -> str:
        return self.name + "_" + self.shape.mangled_name()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Field) and self.name == other.name and self.shape == other.shape

    def __hash__(self) -> int:
        return hash((self.name, self.shape))


def same_tuple(a: tuple[Any, ...], b: tuple[Any, ...]) -> bool:
    return sorted_tuple(a) == sorted_tuple(b)


def sorted_tuple(a: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(sorted(a, key=lambda x: x.mangled_name()))


@dataclass(eq=True, frozen=True)
class ListShape:
    inner: Shape
    behaviours: Behaviours
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return f"[{self.inner}]"

    def mangled_name(self) -> str:
        return f"List_of_{self.inner.mangled_name()}"

    def not_conforms_to(self, other: Shape) -> error.Error | None:
        if not isinstance(other, ListShape):
            return error.unexpected_shape("a list", str(other), other.span)
        return self.inner.not_conforms_to(other.inner)


@dataclass(eq=True, frozen=True)
class ProductShape:
    name: str | None
    fields: tuple[Field, ...]
    behaviours: Behaviours
    span: Span = field(compare=False, hash=False, repr=False)

    @staticmethod
    def empty(span: Span, scope: Scope) -> ProductShape:
        return ProductShape(None, (), Behaviours((), scope), span)

    def is_named(self) -> bool:
        return self.name is not None

    def field(self, name: str) -> Field | None:
        return next((x for x in self.fields if x.name == name), None)

    def __str__(self) -> str:
        name = f"{self.name}" if self.name else ""
        fields = ", ".join(str(x) for x in self.fields)
        return f"{name}{{{fields}}}"

    @property
    def fields_sorted(self) -> tuple[Field, ...]:
        return tuple(sorted(self.fields, key=lambda x: x.name))

    def mangled_name(self) -> str:
        name = [x.mangled_name() for x in sorted_tuple(self.fields)]
        if self.name:
            return self.name + "_" + "_".join(name)
        return "_".join(name)

    def is_empty(self) -> bool:
        return not self.fields

    def not_conforms_to(self, other: Shape) -> error.Error | None:
        """A product shape conforms the other shape if it has at least all the
        fields of the other shape conform.

        The empty shape `{}` conforms any other shape.

        Examples:
        - {name Str, age Int} conforms to {name Str}
        - {} conforms any shape, function, or primitive

        If `other` is a sum shape, then the product shape conforms to the sum shape
        if any of the variants conform to the product shape.

        """
        if isinstance(other, SumShape) and not any(x.not_conforms_to(other) for x in other.variants):
            return error.shape_is_not_a_variant(str(other), str(self), other.span, self.span)

        if not isinstance(other, ProductShape):
            return error.unexpected_shape("a product shape", str(other), other.span)

        # All fields of `other` must be present in `self`.
        for other_field in other.fields:
            # Find a field with the same name in `self.fields`.
            self_field = next((x for x in self.fields if x.name == other_field.name), None)
            if not self_field:
                return error.field_not_found(other_field.name, self.span, other_field.shape.span)
            if err := self_field.shape.not_conforms_to(other_field.shape):
                return err
        return None


@dataclass(eq=True, frozen=True, repr=False)
class SumShape:
    name: str | None
    variants: tuple[Shape, ...]
    behaviours: Behaviours
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        variants = " | ".join(str(x) for x in self.variants)
        name = f" {self.name}" if self.name else ""
        return f"{name} {variants}"

    def mangled_name(self) -> str:
        name = [x.mangled_name() for x in sorted_tuple(self.variants)]
        if self.name:
            return self.name + "_" + "_".join(name)
        return "_".join(name)

    def not_conforms_to(self, other: Shape) -> error.Error | None:
        """A sum shape conforms the other shape if it has at least all the
        variants of the other shape conform.

        A sum shape also conforms to the empty shape `{}`.
        """
        if isinstance(other, ProductShape) and other.is_empty():
            return None

        if not isinstance(other, SumShape):
            return error.unexpected_shape("a sum shape", str(other), other.span)

        # All variants of `other` must be present in `self`.
        for other_variant in other.variants:
            if not any(self_variant.not_conforms_to(other_variant) for self_variant in self.variants):
                return error.variant_not_found(str(other_variant), self.span, other_variant.span)
        return None


@dataclass(eq=True, frozen=True)
class Param:
    name: str
    shape: Shape

    def __str__(self) -> str:
        return self.name + " " + str(self.shape)

    def mangled_name(self) -> str:
        return self.name + "_" + self.shape.mangled_name()


@dataclass(eq=True, frozen=True)
class FunShape:
    name: str | None
    params: tuple[Param, ...]
    result: Shape
    behaviour: str | None
    span: Span = field(compare=False, hash=False, repr=False)
    builtin: bool

    @property
    def is_named(self) -> bool:
        return self.name is not None

    def __str__(self) -> str:
        params = ", ".join(str(x) for x in self.params)
        name = f" {self.name}" if self.name else ""
        name = f" {self.behaviour}.{name[1:]}" if self.behaviour else name
        return f"fun{name}({params}) -> {self.result}"

    def mangled_name(self) -> str:
        if self.builtin:
            assert self.name is not None
            name = self.name
            if self.behaviour:
                name = self.behaviour[1:] + "__" + name
            return name
        params = [x.mangled_name() for x in [*self.params, self.result]]
        name = ""
        if self.name:
            name = self.name + "__"
        if self.behaviour:
            name = self.behaviour[1:] + "__" + name
        return name + "_".join(params)

    def not_conforms_to(self, other: Shape) -> error.Error | None:
        """A function conforms the empty shape or another function if all
        its parameters and result conform the other function's parameters and result.

        Examples:
        - fun(a {name Str}) conforms to fun(a {})

        """
        if isinstance(other, ProductShape) and other.is_empty():
            return None
        if not isinstance(other, FunShape):
            return error.unexpected_shape("a function", str(other), other.span)
        if self.result.not_conforms_to(other.result):
            return error.function_result_does_not_conform(
                str(self.result), str(other.result), self.result.span, other.result.span
            )
        if len(self.params) != len(other.params):
            return error.wrong_number_of_parameters(str(self), str(other), self.span, other.span)
        for self_param, other_param in zip(self.params, other.params):
            if err := self_param.shape.not_conforms_to(other_param.shape):
                return err
        return None


@dataclass(eq=True, frozen=True)
class ErrorShape:
    error: error.Error

    @property
    def span(self) -> Span:
        return self.error.span

    def __str__(self) -> str:
        return str(self.error)

    def mangled_name(self) -> str:
        return "Error"

    def not_conforms_to(self, _other: Shape) -> error.Error | None:
        return self.error


Shape = PrimitiveShape | ProductShape | SumShape | FunShape | UnitShape | ErrorShape | ListShape


@dataclass
class TypeEnv:
    parent: TypeEnv | None
    node_shapes: dict[ast.NodeId, Shape]

    def set(self, node: ast.Node, shape: Shape) -> None:
        # If the type is unset in the root type_env, set it there, too.
        p = self
        while p.parent:
            if node.id in p.parent.node_shapes:
                break
            p = p.parent
        else:
            p.node_shapes[node.id] = shape
        self.node_shapes[node.id] = shape

    def get(self, node: ast.Node) -> Shape:
        shape = self.node_shapes.get(node.id)
        if shape is None and self.parent:
            shape = self.parent.get(node)
        if shape is None:
            raise KeyError(f"Shape for {node} not found")
        return shape


@dataclass
class Binding:
    value: Shape | Behaviour
    builtin: bool


@dataclass
class Scope:
    node: ast.Node | None
    parent: Scope | None
    bindings: dict[str, Binding]

    @staticmethod
    def empty() -> Scope:
        return Scope(None, None, {})

    @staticmethod
    def root() -> Scope:
        """The root scope with all the builtins."""
        scope = Scope(None, None, {})
        span = Span("<builtin>", "", 0, 0)
        bool_shape = PrimitiveShape("Bool", Behaviours(("@Bool",), scope), span)
        char_shape = PrimitiveShape("Char", Behaviours(("@Char",), scope), span)
        int_shape = PrimitiveShape("Int", Behaviours(("@Int",), scope), span)
        str_shape = PrimitiveShape("Str", Behaviours(("@Str",), scope), span)
        unit_shape = UnitShape(Behaviours((), scope), span)
        binding_defaults = {"builtin": True}
        fun_defaults = {"span": span, "builtin": True}
        to_str_shape = ProductShape.empty(span, scope)
        to_str_shape = replace(to_str_shape, behaviours=Behaviours(("@ToStr",), scope))
        scope.bindings["print"] = Binding(
            FunShape("print", (Param("obj", to_str_shape),), unit_shape, behaviour=None, **fun_defaults),
            **binding_defaults,
        )

        # Default behaviour interfaces.
        scope.bindings["@ToStr"] = Binding(
            Behaviour(
                "@ToStr",
                (
                    FunShape(
                        "to_str",
                        (Param("obj", ProductShape.empty(span, scope)),),
                        str_shape,
                        behaviour="@ToStr",
                        **fun_defaults,
                    ),
                ),
                interface=True,
            ),
            **binding_defaults,
        )

        # Default behaviours.
        scope.bindings["@Int"] = Binding(
            Behaviour(
                "@Int",
                (FunShape("to_str", (Param("i", int_shape),), str_shape, behaviour="@Int", **fun_defaults),),
                interface=False,
            ),
            **binding_defaults,
        )
        scope.bindings["@Bool"] = Binding(
            Behaviour(
                "@Bool",
                (FunShape("to_str", (Param("b", bool_shape),), str_shape, behaviour="@Bool", **fun_defaults),),
                interface=False,
            ),
            **binding_defaults,
        )
        scope.bindings["@Char"] = Binding(
            Behaviour(
                "@Char",
                (FunShape("to_str", (Param("c", char_shape),), str_shape, behaviour="@Char", **fun_defaults),),
                interface=False,
            ),
            **binding_defaults,
        )
        scope.bindings["@Str"] = Binding(
            Behaviour(
                "@Str",
                (FunShape("to_str", (Param("s", str_shape),), str_shape, behaviour="@Str", **fun_defaults),),
                interface=False,
            ),
            **binding_defaults,
        )

        scope.bindings["Int"] = Binding(int_shape, **binding_defaults)
        scope.bindings["Str"] = Binding(str_shape, **binding_defaults)
        scope.bindings["Bool"] = Binding(bool_shape, **binding_defaults)
        scope.bindings["Char"] = Binding(char_shape, **binding_defaults)
        scope.bindings["<unit>"] = Binding(unit_shape, **binding_defaults)
        return scope

    def builtin(self, shape: Literal["Bool", "Char", "Int", "Str", "<unit>"]) -> Shape:
        binding = self.lookup(shape)
        assert binding is not None, f"Builtin {shape} not found"
        assert isinstance(binding.value, Shape), f"Expected Shape, got {binding.value}"
        return binding.value

    def lookup(self, name: str) -> Binding | None:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def bind(self, name: str, shape: Shape | Behaviour) -> None:
        """Bind the name to the typ overwriting any existing binding."""
        self.bindings[name] = Binding(shape, builtin=False)

    def inside(self, node_typ: type[ast.Node]) -> ast.Node | None:
        s = self
        while s:
            if isinstance(s.node, node_typ):
                return s.node
            s = s.parent
        return None

    def is_child_of(self, other: Scope) -> bool:
        s = self
        while s:
            if s == other:
                return True
            s = s.parent
        return False


@dataclass
class FunSpec:
    type_env: TypeEnv
    fun_def: ast.FunDef
    # The base function type found at definition.
    base: FunShape
    # The specialized function type.
    specialized: FunShape

    def __str__(self) -> str:
        return str(self.specialized)


class TypeCheck:
    type_env: TypeEnv
    errors: list[error.Error]
    scope: Scope
    fun_specs: dict[FunShape, list[FunSpec]]
    fun_defs: dict[FunShape, ast.FunDef]
    nesting_level = 0

    Bool: Shape
    Char: Shape
    Int: Shape
    Str: Shape
    Unit: Shape

    def __init__(self) -> None:
        self.type_env = TypeEnv(None, {})
        self.errors = []
        self.scope = Scope.root()
        self.Bool = self.scope.builtin("Bool")
        self.Char = self.scope.builtin("Char")
        self.Int = self.scope.builtin("Int")
        self.Str = self.scope.builtin("Str")
        self.Unit = self.scope.builtin("<unit>")
        self.fun_specs = {}
        self.fun_defs = {}
        for name, fun in (x for x in inspect.getmembers(self, inspect.ismethod) if x[0].startswith("tc_")):

            def make_wrapper(fun: Callable) -> Any:
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    fun_desc = f"{fun.__name__}({', '.join(str(x) for x in args)})"
                    log("typechecker-trace", f">>> {fun_desc}", self.nesting_level)
                    self.nesting_level += 1
                    res = fun(*args, **kwargs)
                    self.nesting_level -= 1
                    log("typechecker-trace", f"<<< {fun_desc} = {res}", self.nesting_level)
                    return res

                return wrapper

            setattr(self, name, make_wrapper(fun))

    def fun_spec(self, fun: FunShape, call_args: list[ast.Expr]) -> FunSpec | None:
        """Try to find a FunSpec for the given function with the given parameter types."""
        specs = self.fun_specs.get(fun, [])
        param_types = [self.type_env.get(x) for x in call_args]
        for spec in specs:
            if spec.base == fun and [x.shape for x in spec.specialized.params] == param_types:
                return spec
        return None

    def build_specialized(self, base: FunShape, call_args: list[ast.Expr]) -> FunShape:
        params: list[Param] = []
        for param, arg in zip(base.params, call_args):
            shape = self.type_env.get(arg)
            params.append(Param(param.name, shape))
        return FunShape(base.name, tuple(params), base.result, base.behaviour, base.span, builtin=base.builtin)

    def specialize(self, base: FunShape, call_args: list[ast.Expr], span: Span) -> FunSpec | ErrorShape:
        with self.child_type_env():
            fun_def = self.fun_defs[base]
            specialized = self.build_specialized(base, call_args)

            specs = self.fun_specs.get(base, [])
            for spec in specs:
                if spec.specialized == specialized:
                    return spec

            log("typechecker-mono", f">>> Specializing {base} at call-site {span}", self.nesting_level)
            spec = FunSpec(self.type_env, fun_def, base, specialized)
            specs.append(spec)
            if fun_def.body is not None:
                self.fun_specs[base] = specs

            log(
                "typechecker-mono",
                f"Type checking {spec.base} at {spec.fun_def.span} with {spec.specialized} at call-site {span}",
                self.nesting_level,
            )
            error_mark = len(self.errors)
            shape = self.tc_fun_def_specialized(spec.fun_def, spec.specialized)
            if len(self.errors) > error_mark:
                # Rewind the error stack to the point where we started.
                errors = self.errors[error_mark:]
                self.errors = self.errors[:error_mark]
                self.error(
                    error.failed_to_specialize(str(spec.specialized), str(spec.base), span, spec.base.span, errors[0])
                )
            if isinstance(shape, ErrorShape):
                return ErrorShape(error.cascaded_error(shape.error, span))
            spec.specialized = replace(spec.specialized, result=shape.result)
            if err := spec.specialized.not_conforms_to(spec.base):
                return self.error(
                    error.does_not_conform_to(str(spec.specialized), str(spec.base), span, spec.base.span, err)
                )

            log(
                "typechecker-mono",
                f"<<< Specialized {spec.base} at call-site {span} as {spec.specialized}",
                self.nesting_level,
            )
            return spec

    @contextmanager
    def child_type_env(self) -> Generator[None]:
        prev = self.type_env
        self.type_env = TypeEnv(self.type_env, {})
        try:
            yield
        finally:
            self.type_env = prev

    @contextmanager
    def child_scope(self, node: ast.Node) -> Generator[None]:
        prev = self.scope
        self.scope = Scope(node, self.scope, {})
        try:
            yield
        finally:
            self.scope = prev

    def error(self, err: error.Error) -> ErrorShape:
        self.errors.append(err)
        return ErrorShape(err)

    def forward_declare(self, nodes: list[ast.Node]) -> error.Error | None:
        fun_defs = [x for x in nodes if isinstance(x, ast.FunDef)]
        shape_aliases = [x for x in nodes if isinstance(x, ast.ShapeAlias)]

        # Phase 1: Declare the names with basic shapes.
        for node in shape_aliases:
            shape = FunShape(node.name, (), self.Unit, None, node.span, builtin=False)
            log("typechecker-trace", f"Forward declaring phase 1: {shape}", self.nesting_level)
            self.scope.bind(node.name, shape)
        for node in fun_defs:
            shape = FunShape(node.name, (), self.Unit, None, node.span, builtin=False)
            log("typechecker-trace", f"Forward declaring phase 1: {shape}", self.nesting_level)
            self.scope.bind(node.name, shape)
            if node.behaviour:
                behaviour_binding = self.scope.lookup(node.behaviour)
                if behaviour_binding is None:
                    log(
                        "typechecker-trace",
                        f"Forward declaring phase 1: {node.behaviour}",
                        self.nesting_level,
                    )
                    self.scope.bind(node.behaviour, Behaviour(node.behaviour, (), interface=node.body is None))

        # Phase 2: Declare the correct shapes.
        for node in shape_aliases:
            shape = self.tc_shape_alias(node)
            self.scope.bind(node.name, shape)
            self.type_env.set(node, shape)
        for node in fun_defs:
            shape = self.tc_fun_decl(node)
            self.type_env.set(node, shape)
            if isinstance(shape, ErrorShape):
                return shape.error
            log("typechecker-trace", f"Forward declaring phase 2: {shape}", self.nesting_level)
            self.scope.bind(node.name, shape)
        return None

    def tc_assign(self, node: ast.Assign) -> Shape:
        self.visit(node.value, None)
        value = self.type_env.get(node.value)
        if isinstance(value, ErrorShape):
            return value
        log("typechecker-trace", f"Binding {node.target.name} to {value}", self.nesting_level)
        self.scope.bind(node.target.name, value)
        self.type_env.set(node.target, value)
        return self.Unit

    def tc_field(self, node: ast.Field) -> Shape:
        ast.walk(node, self.visit)
        shape = self.type_env.get(node.shape)
        if isinstance(shape, ErrorShape):
            return shape
        field = None
        if isinstance(node.shape, ast.ShapeRef):
            field = Field.with_shape_name(node.name, node.shape.name, self.scope)
        else:
            field = Field.with_shape(node.name, shape)
        return ProductShape(None, (field,), Behaviours((), self.scope), node.span)

    def tc_behaviour(self, _node: ast.Behaviour) -> Shape:
        return self.Unit

    def tc_binary_expr(self, node: ast.BinaryExpr) -> Shape:
        ast.walk(node, self.visit)
        lhs = self.type_env.get(node.lhs)
        rhs = self.type_env.get(node.rhs)
        if isinstance(lhs, ErrorShape):
            return lhs
        if isinstance(rhs, ErrorShape):
            return rhs
        if err := rhs.not_conforms_to(lhs):
            return self.error(error.does_not_conform_to(str(rhs), str(lhs), rhs.span, lhs.span, err))
        if node.op in (ast.BinaryOp.add, ast.BinaryOp.sub, ast.BinaryOp.mul, ast.BinaryOp.div):
            return lhs
        return self.Bool

    def tc_block(self, node: ast.Block) -> Shape:
        with self.child_scope(node):
            ast.walk(node, self.visit)
        if len(node.nodes) == 0:
            return self.Unit
        return self.type_env.get(node.nodes[-1])

    def tc_call(self, node: ast.Call) -> Shape:
        ast.walk(node, self.visit)
        callee = self.type_env.get(node.callee)
        if isinstance(callee, ErrorShape):
            return callee
        if not isinstance(callee, FunShape):
            return self.error(error.not_callable(node.callee.span, callee.span))

        args = node.args
        if callee.behaviour:
            assert isinstance(node.callee, ast.Member), f"Expected Member, got {node.callee}"
            args = [node.callee.target, *args]

        if callee.builtin:
            specialized = self.build_specialized(callee, args)
            if err := specialized.not_conforms_to(callee):
                return self.error(error.does_not_conform_to(str(specialized), str(callee), node.span, callee.span, err))
            return callee.result

        if callee.is_named:
            spec = self.specialize(callee, args, node.span)
            if isinstance(spec, ErrorShape):
                return ErrorShape(error.cascaded_error(spec.error, node.span))
            self.type_env.set(node.callee, spec.specialized)
            callee = spec.specialized

        return callee.result

    def tc_fun_decl(self, node: ast.FunDef) -> FunShape | ErrorShape:
        params: list[Param] = []
        for param in node.params:
            self.visit(param, node)
            param_shape = self.type_env.get(param)
            if isinstance(param_shape, ErrorShape):
                return param_shape
            params.append(Param(param.name, param_shape))
        self.visit(node.result, node)
        return_shape = self.type_env.get(node.result)
        if isinstance(return_shape, ErrorShape):
            return return_shape
        shape = FunShape(node.name, (*params,), return_shape, node.behaviour, node.span, builtin=False)
        log("typechecker-trace", f"Adding {shape} to fun_defs", self.nesting_level)
        self.fun_defs[shape] = node
        if node.behaviour:
            log("typechecker-trace", f"Adding {shape} to behaviours", self.nesting_level)
            behaviour_binding = self.scope.lookup(node.behaviour)
            behaviour_funs = []
            if behaviour_binding:
                behaviour = behaviour_binding.value
                assert isinstance(behaviour, Behaviour)
                if behaviour.interface != node.is_behaviour_interface_method():
                    if behaviour.interface:
                        return self.error(
                            error.cannot_add_method_to_interface_behaviour(str(behaviour), str(shape), node.span)
                        )
                    return self.error(
                        error.cannot_add_interface_method_to_non_interface_behaviour(
                            str(behaviour), str(shape), node.span
                        )
                    )
                behaviour_funs = list(behaviour.funs)
            behaviour_funs.append(shape)
            self.scope.bind(
                node.behaviour, Behaviour(node.behaviour, tuple(behaviour_funs), node.is_behaviour_interface_method())
            )
        return shape

    def tc_fun_def(self, node: ast.FunDef) -> Shape:
        shape = self.type_env.get(node)
        if isinstance(shape, ErrorShape):
            return shape
        assert isinstance(shape, FunShape)
        with self.child_scope(node):
            for param in shape.params:
                if err := self.scope.bind(param.name, param.shape):
                    return self.error(err)
            if node.body:
                self.visit(node.body, node)
        if node.body is None:
            # This is an interface method.
            return shape
        body_shape = self.type_env.get(node.body)
        if isinstance(body_shape, ErrorShape):
            return body_shape
        if err := body_shape.not_conforms_to(shape.result):
            return self.error(
                error.does_not_conform_to(str(body_shape), str(shape.result), node.span, shape.result.span, err)
            )
        if err := self.scope.bind(node.name, shape):
            return self.error(err)
        if node.name == "main":
            log("typechecker-trace", "Adding main to fun_defs", self.nesting_level)
            fun = shape
            if len(fun.params) != 0 or fun.result != self.Unit:
                if isinstance(fun.result, ErrorShape):
                    return ErrorShape(error.cascaded_error(fun.result.error, node.span))
                return self.error(error.invalid_main(node.span))
            self.fun_specs[fun] = [FunSpec(self.type_env, node, fun, fun)]
        return shape

    def tc_fun_def_specialized(self, node: ast.FunDef, fun: FunShape) -> FunShape | ErrorShape:
        with self.child_scope(node):
            for param in fun.params:
                if err := self.scope.bind(param.name, param.shape):
                    return self.error(err)
            ast.walk(node, self.visit)
        return_typ = self.type_env.get(node.body) if node.body else fun.result
        return replace(fun, result=return_typ)

    def tc_fun_param(self, node: ast.Param) -> Shape:
        ast.walk(node, self.visit)
        shape = self.type_env.get(node.shape)
        if isinstance(shape, ErrorShape):
            return shape
        return shape

    def tc_fun_shape(self, node: ast.FunShape) -> Shape:
        ast.walk(node, self.visit)
        params: list[Param] = []
        for param in node.params:
            shape = self.type_env.get(param.shape)
            if isinstance(shape, ErrorShape):
                return shape
            params.append(Param(param.name, shape))
        result = self.type_env.get(node.result)
        if isinstance(result, ErrorShape):
            return result
        return FunShape(None, (*params,), result, None, node.span, builtin=False)

    def tc_if(self, node: ast.If) -> Shape:
        ast.walk(node, self.visit)
        # todo: for now, all arms and the else block must have the same type.
        shape = self.type_env.get(node.arms[0])
        for arm in node.arms:
            arm_shape = self.tc_if_arm(arm)
            if shape != arm_shape:
                return self.error(error.is_not_same(str(shape), str(arm_shape), arm.span))
        if node.else_block:
            else_shape = self.tc_block(node.else_block)
            # todo: if/else with different types should create a union type.
            if shape != else_shape:
                return self.error(error.is_not_same(str(shape), str(else_shape), node.span))
        return shape

    def tc_if_arm(self, node: ast.IfArm) -> Shape:
        ast.walk(node, self.visit)
        return self.tc_block(node.block)

    def tc_list_index(self, node: ast.ListIndex) -> Shape:
        ast.walk(node, self.visit)
        shape = self.type_env.get(node.target)
        if isinstance(shape, ErrorShape):
            return shape
        if not isinstance(shape, ListShape):
            return self.error(error.unexpected_shape("a list", str(shape), node.target.span))
        index_shape = self.type_env.get(node.index)
        if isinstance(index_shape, ErrorShape):
            return index_shape
        if index_shape != self.Int:
            return self.error(error.unexpected_shape("an integer", str(index_shape), node.index.span))
        return shape.inner

    def tc_list_lit(self, node: ast.ListLit) -> Shape:
        ast.walk(node, self.visit)
        shape = None
        for value_node in node.values:
            value_shape = self.type_env.get(value_node)
            if isinstance(value_shape, ErrorShape):
                return value_shape
            if shape is None:
                shape = value_shape
            elif shape != value_shape:
                # todo: for now, all values in a list must have the same type.
                return self.error(error.is_not_same(str(shape), str(value_shape), value_node.span))
        if shape is None:
            shape = ProductShape.empty(node.span, self.scope)
        return ListShape(shape, Behaviours((), self.scope), node.span)

    def tc_list_shape(self, node: ast.ListShape) -> Shape:
        ast.walk(node, self.visit)
        shape = self.type_env.get(node.inner)
        if isinstance(shape, ErrorShape):
            return shape
        return ListShape(shape, Behaviours((), self.scope), node.span)

    def tc_member(self, node: ast.Member) -> Shape:
        ast.walk(node, self.visit)
        shape = self.type_env.get(node.target)
        if isinstance(shape, ErrorShape):
            return shape
        behaviour_fun = None
        if isinstance(shape, (ProductShape, PrimitiveShape)):
            behaviour_fun = shape.behaviours.fun(node.name)
        if isinstance(shape, PrimitiveShape):
            if behaviour_fun is None:
                return self.error(error.no_member(node.name, str(shape), node.target.span, node.span))
            return behaviour_fun
        if not isinstance(shape, ProductShape):
            return self.error(error.unexpected_shape(f"a shape with field `{node.name}`", str(shape), node.target.span))
        field = shape.field(node.name)
        if field:
            return field.shape
        if behaviour_fun:
            return behaviour_fun
        return self.error(error.no_member(node.name, str(shape), node.target.span, node.span))

    def tc_module(self, node: ast.Module) -> Shape:
        self.forward_declare(node.nodes)
        ast.walk(node, self.visit)
        return self.Unit

    def tc_name(self, node: ast.Name) -> Shape:
        name = self.scope.lookup(node.name)
        if name is None:
            return self.error(error.undefined_name(node.name, node.span))
        assert isinstance(name.value, Shape)
        return name.value

    def tc_product_shape(self, node: ast.ProductShape) -> Shape:
        ast.walk(node, self.visit)
        fields: list[Field] = []
        for field_node in node.fields:
            shape = self.type_env.get(field_node.shape)
            if isinstance(shape, ErrorShape):
                return shape
            field = None
            if isinstance(field_node.shape, ast.ShapeRef):
                field = Field.with_shape_name(field_node.name, field_node.shape.name, self.scope)
            else:
                field = Field.with_shape(field_node.name, shape)
            fields.append(field)
        behaviours = []
        for behaviour_node in node.behaviours:
            behaviour_binding = self.scope.lookup(behaviour_node.name)
            if not behaviour_binding:
                return self.error(error.undefined_name(behaviour_node.name, behaviour_node.span))
            assert isinstance(behaviour_binding.value, Behaviour)
            behaviours.append(behaviour_binding.value.name)
        return ProductShape(None, tuple(fields), Behaviours(tuple(behaviours), self.scope), node.span)

    def tc_product_shape_lit(self, node: ast.ProductShapeLit) -> Shape:
        ast.walk(node, self.visit)
        fields = []
        for field in node.fields:
            self.visit(field, node)
            shape = self.type_env.get(field.value)
            if isinstance(shape, ErrorShape):
                return shape
            fields.append(Field.with_shape(field.name, shape))
        behaviours = []
        for behaviour_node in node.behaviours:
            behaviour = self.scope.lookup(behaviour_node.name)
            if not behaviour:
                return self.error(error.undefined_name(behaviour_node.name, behaviour_node.span))
            assert isinstance(behaviour.value, Behaviour)
            behaviours.append(behaviour_node.name)
        for composite_node in node.composites:
            composite = self.type_env.get(composite_node)
            if isinstance(composite, ErrorShape):
                return composite
            assert isinstance(composite, ProductShape)
            assert not composite.behaviours
            # Merge fields from composite into the shape.
            for composite_fields in composite.fields:
                index = fields.index(composite_fields)
                if index < 0:
                    fields.append(composite_fields)
                else:
                    fields[index] = composite_fields
        shape = ProductShape(None, tuple(fields), Behaviours(tuple(behaviours), self.scope), node.span)
        if node.shape_ref:
            shape_ref = self.type_env.get(node.shape_ref)
            if isinstance(shape_ref, ErrorShape):
                return shape_ref
            if not isinstance(shape_ref, FunShape):
                shape = replace(shape, behaviours=shape.behaviours.merge(shape_ref.behaviours))
            if err := shape.not_conforms_to(shape_ref):
                return self.error(
                    error.does_not_conform_to(str(shape), node.shape_ref.name, node.span, node.shape_ref.span, err)
                )
            shape = replace(shape, name=node.shape_ref.name)
        return shape

    def tc_shape_lit_field(self, node: ast.ShapeLitField) -> Shape:
        ast.walk(node, self.visit)
        return self.Unit

    def tc_shape_alias(self, node: ast.ShapeAlias) -> Shape:
        ast.walk(node, self.visit)
        shape = self.type_env.get(node.shape)
        if isinstance(shape, ErrorShape):
            return shape
        return replace(shape, name=node.name)

    def tc_shape_ref(self, node: ast.ShapeRef) -> Shape:
        declared = self.scope.lookup(node.name)
        if declared is None:
            return self.error(error.undefined_name(node.name, node.span))
        assert isinstance(declared.value, Shape)
        return declared.value

    def tc_sum_shape(self, node: ast.SumShape) -> Shape:
        ast.walk(node, self.visit)
        variants = [self.type_env.get(variant) for variant in node.variants]
        behaviours = []
        for behaviour_node in node.behaviours:
            behaviour = self.scope.lookup(behaviour_node.name)
            if not behaviour:
                return self.error(error.undefined_name(behaviour_node.name, behaviour_node.span))
            assert isinstance(behaviour.value, Behaviour)
            behaviours.append(behaviour_node.name)
        return SumShape(None, tuple(variants), Behaviours(tuple(behaviours), self.scope), node.span)

    def visit(self, node: ast.Node, _parent: ast.Node | None) -> ast.Node:
        shape: Shape
        match node:
            case ast.Assign():
                shape = self.tc_assign(node)
            case ast.Field():
                shape = self.tc_field(node)
            case ast.Behaviour():
                shape = self.tc_behaviour(node)
            case ast.BinaryExpr():
                shape = self.tc_binary_expr(node)
            case ast.Block():
                shape = self.tc_block(node)
            case ast.BoolLit():
                shape = self.Bool
            case ast.Call():
                shape = self.tc_call(node)
            case ast.CharLit():
                shape = self.Char
            case ast.FunDef():
                shape = self.tc_fun_def(node)
            case ast.Param():
                shape = self.tc_fun_param(node)
            case ast.FunShape():
                shape = self.tc_fun_shape(node)
            case ast.If():
                shape = self.tc_if(node)
            case ast.IfArm():
                shape = self.tc_if_arm(node)
            case ast.IntLit():
                shape = self.Int
            case ast.ListShape():
                shape = self.tc_list_shape(node)
            case ast.ListIndex():
                shape = self.tc_list_index(node)
            case ast.ListLit():
                shape = self.tc_list_lit(node)
            case ast.Member():
                shape = self.tc_member(node)
            case ast.Module():
                shape = self.tc_module(node)
            case ast.Name():
                shape = self.tc_name(node)
            case ast.ProductShape():
                shape = self.tc_product_shape(node)
            case ast.ProductShapeLit():
                shape = self.tc_product_shape_lit(node)
            case ast.ShapeLitField():
                shape = self.tc_shape_lit_field(node)
            case ast.ShapeAlias():
                # Already handled in `forward_declare`.
                return node
            case ast.ShapeRef():
                shape = self.tc_shape_ref(node)
            case ast.StrLit():
                shape = self.Str
            case ast.SumShape():
                shape = self.tc_sum_shape(node)
            case ast.UnitShape():
                shape = self.Unit
            case _:
                raise AssertionError(f"Don't know how to type check: {node!r}")
        self.type_env.set(node, shape)
        return node


@dataclass
class TypeCheckResult:
    type_env: TypeEnv
    fun_specs: list[FunSpec]
    errors: list[error.Error]


def typecheck(node: ast.Node) -> TypeCheckResult:
    tc = TypeCheck()
    tc.visit(node, None)
    fun_specs = []
    for spec in tc.fun_specs.values():
        fun_specs.extend(list(spec))
    return TypeCheckResult(tc.type_env, fun_specs, tc.errors)
