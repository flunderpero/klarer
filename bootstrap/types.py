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

    def __str__(self) -> str:
        return f"@{self.name}"

    def is_same(self, other: Behaviour) -> bool:
        return self.name == other.name

    def mangled_name(self) -> str:
        return self.name

    def fun(self, name: str) -> FunShape | None:
        for fun in self.funs:
            if fun.name == name:
                return fun
        return None


@dataclass(eq=True, frozen=True)
class Behaviours:
    behaviours: tuple[Behaviour, ...]

    def is_same(self, other: Behaviours) -> bool:
        """Two behaviours are the same if the resulting functions are the same.
        If both, behaviour A and B expose a function x, then the order of A and B
        is important.
        """
        self_funs = self.functions()
        other_funs = other.functions()
        if len(self_funs) != len(other_funs):
            return False
        for self_fun, other_fun in zip(self_funs, other_funs):
            if not self_fun[0].is_same(other_fun[0]) or self_fun[1].is_same(other_fun[1]):
                return False
        return True

    def conforms_to(self, other: Behaviours) -> bool:
        """`self` conforms to `other` if it has at least all the behaviours of `other`
        in the same order.
        """
        self_funs = self.functions()
        other_funs = other.functions()
        if len(self_funs) < len(other_funs):
            return False
        for behaviour, fun in other_funs:
            if not any(x[0].is_same(behaviour) and x[1].is_same(fun) for x in self_funs):
                return False
        return True

    def functions(self) -> tuple[tuple[Behaviour, FunShape], ...]:
        funs = set()
        for behaviour in self.behaviours:
            for fun in behaviour.funs:
                funs.add((behaviour, fun))
        return tuple(funs)

    def fun(self, name: str) -> FunShape | None:
        for behaviour in reversed(self.behaviours):
            fun = behaviour.fun(name)
            if fun:
                return fun
        return None

    def merge(self, other: Behaviours) -> Behaviours:
        behaviours = list(self.behaviours)
        for behaviour in other.behaviours:
            if behaviour not in behaviours:
                behaviours.append(behaviour)
        return Behaviours(tuple(behaviours))

    def __str__(self) -> str:
        return " + ".join(str(x) for x in self.behaviours)


@dataclass(eq=True, frozen=True)
class PrimitiveShape:
    name: Literal["Bool", "Char", "Int", "Str"]
    behaviours: Behaviours
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return self.name

    def is_same(self, other: Shape) -> bool:
        # Behaviours of primitives are guaranteed to be the same.
        return isinstance(other, PrimitiveShape) and self.name == other.name

    def mangled_name(self) -> str:
        return self.name

    def conforms_to(self, other: Shape) -> bool:
        # For now, all primitives only conform to themselves, a sum shape, or the empty shape.
        # Later on, when we have different sized integers, I8 will conform to I16, etc.
        if self.is_same(other):
            return True
        if isinstance(other, SumShape):
            return any(self.conforms_to(x) for x in other.variants)
        return isinstance(other, ProductShape) and other.is_empty()


@dataclass(eq=True, frozen=True)
class UnitShape:
    behaviours = Behaviours(())
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return "_Unit"

    def is_same(self, other: Shape) -> bool:
        return isinstance(other, UnitShape)

    def mangled_name(self) -> str:
        return "_Unit"

    def conforms_to(self, other: Shape) -> bool:
        return isinstance(other, UnitShape)


@dataclass(eq=True, frozen=True)
class Attr:
    name: str
    shape: Shape

    def __str__(self) -> str:
        return self.name + " " + str(self.shape)

    def is_same(self, other: Attr) -> bool:
        return self.name == other.name and self.shape.is_same(other.shape)

    def mangled_name(self) -> str:
        return self.name + "_" + self.shape.mangled_name()

    def conforms_to(self, other: Attr) -> bool:
        return self.name == other.name and self.shape.conforms_to(other.shape)


def same_tuple(a: tuple[Any, ...], b: tuple[Any, ...]) -> bool:
    return sorted_tuple(a) == sorted_tuple(b)


def sorted_tuple(a: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(sorted(a, key=lambda x: x.mangled_name()))


@dataclass(eq=True, frozen=True)
class ProductShape:
    name: str | None
    attrs: tuple[Attr, ...]
    behaviours: Behaviours
    span: Span = field(compare=False, hash=False, repr=False)

    @staticmethod
    def empty(span: Span) -> ProductShape:
        return ProductShape(None, (), Behaviours(()), span)

    def is_named(self) -> bool:
        return self.name is not None

    def attr(self, name: str) -> Attr | None:
        return next((x for x in self.attrs if x.name == name), None)

    def __str__(self) -> str:
        name = f"{self.name}" if self.name else ""
        attrs = ", ".join(str(x) for x in self.attrs)
        return f"{name}{{{attrs}}}"

    def is_same(self, other: Shape) -> bool:
        return (
            isinstance(other, ProductShape)
            and same_tuple(self.attrs, other.attrs)
            and self.behaviours.is_same(other.behaviours)
        )

    @property
    def attrs_sorted(self) -> tuple[Attr, ...]:
        return tuple(sorted(self.attrs, key=lambda x: x.name))

    def mangled_name(self) -> str:
        name = [x.mangled_name() for x in sorted_tuple(self.attrs)]
        if self.name:
            return self.name + "_" + "_".join(name)
        return "_".join(name)

    def is_empty(self) -> bool:
        return not self.attrs

    def conforms_to(self, other: Shape) -> bool:
        """A product shape conforms the other shape if it has at least all the
        attributes of the other shape and the behaviours conform.

        The empty shape `{}` conforms any other shape.

        Examples:
        - {name Str, age Int} conforms to {name Str}
        - {} conforms any shape, function, or primitive

        If `other` is a sum shape, then the product shape conforms to the sum shape
        if any of the variants conform to the product shape.

        """
        if isinstance(other, SumShape):
            return any(x.conforms_to(other) for x in other.variants)

        if not isinstance(other, ProductShape):
            return False

        if not self.behaviours.conforms_to(other.behaviours):
            return False

        # All attributes of `other` must be present in `self`.
        return all(any(x.conforms_to(attr) for x in self.attrs) for attr in other.attrs)


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

    def is_same(self, other: Shape) -> bool:
        return (
            isinstance(other, SumShape)
            and same_tuple(self.variants, other.variants)
            and self.behaviours.is_same(other.behaviours)
        )

    def mangled_name(self) -> str:
        name = [x.mangled_name() for x in sorted_tuple(self.variants)]
        if self.name:
            return self.name + "_" + "_".join(name)
        return "_".join(name)

    def conforms_to(self, other: Shape) -> bool:
        """A sum shape conforms the other shape if it has at least all the
        variants of the other shape and the behaviours conform.

        The empty shape `{}` conforms any other shape.

        Examples:
        - {name Str, age Int} conforms to {name Str}
        - {} conforms any shape, function, or primitive

        """
        if not isinstance(other, SumShape):
            return False

        if not self.behaviours.conforms_to(other.behaviours):
            return False

        # All variants of `other` must be present in `self`.
        return all(any(x.conforms_to(variant) for x in self.variants) for variant in other.variants)


@dataclass(eq=True, frozen=True)
class FunShape:
    name: str | None
    params: tuple[Attr, ...]
    result: Shape
    namespace: str | None
    span: Span = field(compare=False, hash=False, repr=False)
    builtin: bool

    @property
    def is_named(self) -> bool:
        return self.name is not None

    def __str__(self) -> str:
        params = ", ".join(str(x) for x in self.params)
        name = f" {self.name}" if self.name else ""
        name = f" @{self.namespace}.{name[1:]}" if self.namespace else name
        return f"fun{name}({params}) -> {self.result}"

    def is_same(self, other: Shape) -> bool:
        return (
            isinstance(other, FunShape)
            and all(x.is_same(y) for x, y in zip(self.params, other.params))
            and self.result.is_same(other.result)
        )

    def mangled_name(self) -> str:
        if self.builtin:
            assert self.name is not None
            return self.name
        params = [x.mangled_name() for x in [*self.params, self.result]]
        name = ""
        if self.name:
            name = self.name + "__"
        if self.namespace:
            name = self.namespace + "__" + name
        return name + "_".join(params)

    def conforms_to(self, other: Shape) -> bool:
        """A function conforms the empty shape or another function if all
        its parameters and result conform the other function's parameters and result.

        Examples:
        - fun(a {name Str}) conforms to fun(a {})

        """
        if isinstance(other, ProductShape) and other.is_empty():
            return True
        if not isinstance(other, FunShape):
            return False
        if len(self.params) != len(other.params):
            return False
        if not self.result.conforms_to(other.result):
            return False
        for param, other_param in zip(self.params, other.params):  # noqa: SIM110
            if not param.conforms_to(other_param):
                return False
        return True


@dataclass(eq=True, frozen=True)
class ErrorShape:
    error: error.Error

    @property
    def span(self) -> Span:
        return self.error.span

    def __str__(self) -> str:
        return str(self.error)

    def is_same(self, _other: Shape) -> bool:
        return False

    def mangled_name(self) -> str:
        return "Error"

    def conforms_to(self, _other: Shape) -> bool:
        return False


Shape = PrimitiveShape | ProductShape | SumShape | FunShape | UnitShape | ErrorShape

builtin_span = Span("<builtin", "", 0, 0)
Bool = PrimitiveShape("Bool", Behaviours(()), builtin_span)
Char = PrimitiveShape("Char", Behaviours(()), builtin_span)
Int = PrimitiveShape("Int", Behaviours(()), builtin_span)
Str = PrimitiveShape("Str", Behaviours(()), builtin_span)
Unit = UnitShape(builtin_span)


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
        typ = self.node_shapes.get(node.id)
        if typ is None and self.parent:
            typ = self.parent.get(node)
        if typ is None:
            raise KeyError(f"Type for {node} not found")
        return typ


@dataclass
class Binding:
    shape: Shape
    builtin: bool
    is_fun_param: bool


@dataclass
class Scope:
    node: ast.Node | None
    parent: Scope | None
    bindings: dict[str, Binding]

    @staticmethod
    def root() -> Scope:
        """The root scope with all the builtins."""
        span = Span("<builtin>", "", 0, 0)
        binding_defaults = {"builtin": True, "is_fun_param": False}
        fun_defaults = {"namespace": None, "span": span, "builtin": True}
        scope = Scope(None, None, {})
        scope.bindings["print"] = Binding(
            FunShape("print", (Attr("s", Str),), Unit, **fun_defaults),
            **binding_defaults,
        )
        scope.bindings["int_to_str"] = Binding(
            FunShape("int_to_str", (Attr("i", Int),), Str, **fun_defaults),
            **binding_defaults,
        )
        scope.bindings["char_to_str"] = Binding(
            FunShape("char_to_str", (Attr("c", Char),), Str, **fun_defaults),
            **binding_defaults,
        )
        scope.bindings["bool_to_str"] = Binding(
            FunShape("bool_to_str", (Attr("b", Bool),), Str, **fun_defaults),
            **binding_defaults,
        )
        scope.bindings["Int"] = Binding(Int, **binding_defaults)
        scope.bindings["Str"] = Binding(Str, **binding_defaults)
        scope.bindings["Bool"] = Binding(Bool, **binding_defaults)
        scope.bindings["Char"] = Binding(Char, **binding_defaults)
        return scope

    def lookup(self, name: str) -> Binding | None:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def bind(
        self, name: str, shape: Shape, *, is_fun_param: bool = False, can_shadow_parent_scopes: bool = False
    ) -> error.Error | None:
        """Bind the name to the typ.
        An error is returned if the name is already bound to a different type.

        If `can_shadow_parent_scopes` is True, the name can be bound even if
        it is already bound in a parent scope but not in the current scope.
        """
        existing = self.lookup(name)
        if existing and (not can_shadow_parent_scopes or name in self.bindings):
            return error.duplicate_declaration(name, shape.span, existing.shape.span)
        self.bindings[name] = Binding(shape, builtin=False, is_fun_param=is_fun_param)
        return None

    def inside(self, node_typ: type[ast.Node]) -> ast.Node | None:
        s = self
        while s:
            if isinstance(s.node, node_typ):
                return s.node
            s = s.parent
        return None


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
    behaviours: dict[str, Behaviour]
    nesting_level = 0

    def __init__(self) -> None:
        self.type_env = TypeEnv(None, {})
        self.errors = []
        self.scope = Scope.root()
        self.fun_specs = {}
        self.fun_defs = {}
        self.behaviours = {}
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

    def specialize(self, fun: FunShape, call_args: list[ast.Expr], span: Span) -> FunSpec | ErrorShape:
        spec = self.fun_spec(fun, call_args)
        if spec:
            return spec
        with self.child_type_env():
            fun_def = self.fun_defs[fun]
            log("typechecker-mono", f">>> Specializing {fun} at call-site {span}", self.nesting_level)

            base = self.type_env.get(fun_def)
            assert isinstance(base, FunShape)
            params: list[Attr] = []
            for param, arg in zip(fun.params, call_args):
                shape = self.type_env.get(arg)
                params.append(Attr(param.name, shape))
            specialized = FunShape(fun.name, tuple(params), base.result, fun.namespace, fun.span, builtin=fun.builtin)

            spec = FunSpec(self.type_env, fun_def, base, specialized)

            if not spec.specialized.conforms_to(spec.base):
                return self.error(error.does_not_conform_to(str(spec.specialized), str(spec.base), span))

            log(
                "typechecker-mono",
                f"Type checking {spec.base} at {spec.fun_def.span} with {spec.specialized} at call-site {span}",
                self.nesting_level,
            )
            shape = self.tc_fun_def_specialized(spec.fun_def, spec.specialized)
            if isinstance(shape, ErrorShape):
                return ErrorShape(error.cascaded_error(shape.error, span))
            spec.specialized = replace(spec.specialized, result=shape.result)
            log(
                "typechecker-mono",
                f"<<< Specialized {spec.base} at call-site {span} as {spec.specialized}",
                self.nesting_level,
            )
            specs = self.fun_specs.get(fun, [])
            specs.append(spec)
            self.fun_specs[fun] = specs
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

    def tc_assign(self, node: ast.Assign) -> Shape:
        self.visit(node.value, None)
        value = self.type_env.get(node.value)
        if isinstance(value, ErrorShape):
            return value
        binding = self.scope.lookup(node.target.name)
        if binding is not None:
            if not binding.shape.is_same(value):
                return self.error(error.is_not_same(str(value), str(binding.shape), node.target.span))
        else:
            log("typechecker-trace", f"Binding {node.target.name} to {value}", self.nesting_level)
            self.scope.bind(node.target.name, value)
        self.type_env.set(node.target, value)
        return Unit

    def tc_attr(self, node: ast.Attr) -> Shape:
        ast.walk(node, self.visit)
        shape = self.type_env.get(node.shape)
        if isinstance(shape, ErrorShape):
            return shape
        return ProductShape(None, (Attr(node.name, shape),), Behaviours(()), node.span)

    def tc_behaviour(self, _node: ast.Behaviour) -> Shape:
        return Unit

    def tc_binary_expr(self, node: ast.BinaryExpr) -> Shape:
        ast.walk(node, self.visit)
        lhs = self.type_env.get(node.lhs)
        rhs = self.type_env.get(node.rhs)
        if isinstance(lhs, ErrorShape):
            return lhs
        if isinstance(rhs, ErrorShape):
            return rhs
        if not rhs.conforms_to(lhs):
            return self.error(error.does_not_conform_to(str(rhs), str(lhs), node.span))
        return Bool

    def tc_block(self, node: ast.Block) -> Shape:
        with self.child_scope(node):
            ast.walk(node, self.visit)
        if len(node.nodes) == 0:
            return Unit
        return self.type_env.get(node.nodes[-1])

    def tc_call(self, node: ast.Call) -> Shape:
        ast.walk(node, self.visit)
        callee = self.type_env.get(node.callee)
        if isinstance(callee, ErrorShape):
            return callee
        if not isinstance(callee, FunShape):
            return self.error(error.not_callable(node.callee.span, callee.span))

        # Build a specialized function if it is not a builtin.
        if callee.builtin:
            return callee.result

        args = node.args
        if callee.namespace:
            assert isinstance(node.callee, ast.Member), f"Expected Member, got {node.callee}"
            args = [node.callee.target, *args]

        spec = self.specialize(callee, args, node.span)
        if isinstance(spec, ErrorShape):
            return ErrorShape(error.cascaded_error(spec.error, node.span))
        self.type_env.set(node.callee, spec.specialized)

        # Now check the types.
        callee = spec.specialized
        return callee.result

    def tc_fun_def(self, node: ast.FunDef) -> Shape:
        params: list[Attr] = []
        with self.child_scope(node):
            for param in node.params:
                self.visit(param, node)
                param_shape = self.type_env.get(param)
                if err := self.scope.bind(param.name, param_shape, is_fun_param=True):
                    return self.error(err)
                params.append(Attr(param.name, param_shape))
            self.visit(node.body, node)
        return_shape = self.type_env.get(node.body)
        shape = FunShape(node.name, (*params,), return_shape, node.namespace, node.span, builtin=False)
        log("typechecker-trace", f"Adding {shape} to fun_defs", self.nesting_level)
        self.fun_defs[shape] = node
        if err := self.scope.bind(node.name, shape):
            return self.error(err)
        if node.name == "main":
            log("typechecker-trace", "Adding main to fun_defs", self.nesting_level)
            fun = shape
            if len(fun.params) != 0 or fun.result != Unit:
                if isinstance(fun.result, ErrorShape):
                    return ErrorShape(error.cascaded_error(fun.result.error, node.span))
                return self.error(error.invalid_main(node.span))
            self.fun_specs[fun] = [FunSpec(self.type_env, node, fun, fun)]
        if node.namespace:
            behaviour = self.behaviours.get(node.namespace)
            behaviour_funs = []
            if behaviour:
                behaviour_funs = list(behaviour.funs)
            behaviour_funs.append(shape)
            self.behaviours[node.namespace] = Behaviour(node.namespace, tuple(behaviour_funs))
        return shape

    def tc_fun_def_specialized(self, node: ast.FunDef, fun: FunShape) -> FunShape | ErrorShape:
        with self.child_scope(node):
            for param in fun.params:
                if err := self.scope.bind(param.name, param.shape, is_fun_param=True, can_shadow_parent_scopes=True):
                    return self.error(err)
            ast.walk(node, self.visit)
        return_typ = self.type_env.get(node.body)
        return replace(fun, result=return_typ)

    def tc_fun_param(self, node: ast.FunParam) -> Shape:
        ast.walk(node, self.visit)
        shape = self.type_env.get(node.shape)
        if isinstance(shape, ErrorShape):
            return shape
        return shape

    def tc_if(self, node: ast.If) -> Shape:
        ast.walk(node, self.visit)
        # todo: for now, all arms and the else block must have the same type.
        shape = self.type_env.get(node.arms[0])
        for arm in node.arms:
            arm_shape = self.tc_if_arm(arm)
            if not shape.is_same(arm_shape):
                return self.error(error.is_not_same(str(shape), str(arm_shape), arm.span))
        if node.else_block:
            else_shape = self.tc_block(node.else_block)
            # todo: if/else with different types should create a union type.
            if not shape.is_same(else_shape):
                return self.error(error.is_not_same(str(shape), str(else_shape), node.span))
        return shape

    def tc_if_arm(self, node: ast.IfArm) -> Shape:
        ast.walk(node, self.visit)
        return self.tc_block(node.block)

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
        attr = shape.attr(node.name)
        if attr:
            return attr.shape
        if behaviour_fun:
            return behaviour_fun
        return self.error(error.no_member(node.name, str(shape), node.target.span, node.span))

    def tc_module(self, node: ast.Module) -> Shape:
        ast.walk(node, self.visit)
        return Unit

    def tc_name(self, node: ast.Name) -> Shape:
        name = self.scope.lookup(node.name)
        if name is None:
            return self.error(error.undefined_name(node.name, node.span))
        return name.shape

    def tc_product_shape(self, node: ast.ProductShape) -> Shape:
        ast.walk(node, self.visit)
        attrs: list[Attr] = []
        for attr in node.attrs:
            shape = self.type_env.get(attr.shape)
            if isinstance(shape, ErrorShape):
                return shape
            attrs.append(Attr(attr.name, shape))
        behaviours = []
        for behaviour_node in node.behaviours:
            behaviour = self.behaviours.get(behaviour_node.name)
            if not behaviour:
                return self.error(error.undefined_name(behaviour_node.name, behaviour_node.span))
            behaviours.append(behaviour)
        return ProductShape(None, tuple(attrs), Behaviours(tuple(behaviours)), node.span)

    def tc_shape(self, node: ast.Shape) -> Shape:
        ast.walk(node, self.visit)
        raise NotImplementedError

    def tc_shape_lit(self, node: ast.ShapeLit) -> Shape:
        ast.walk(node, self.visit)
        attrs = []
        for attr in node.attrs:
            self.visit(attr, node)
            shape = self.type_env.get(attr.value)
            if isinstance(shape, ErrorShape):
                return shape
            attrs.append(Attr(attr.name, shape))
        behaviours = []
        for behaviour_node in node.behaviours:
            behaviour = self.behaviours.get(behaviour_node.name)
            if not behaviour:
                return self.error(error.undefined_name(behaviour_node.name, behaviour_node.span))
            behaviours.append(behaviour)
        for composite_node in node.composites:
            composite = self.type_env.get(composite_node)
            if isinstance(composite, ErrorShape):
                return composite
            assert isinstance(composite, ProductShape)
            assert not composite.behaviours
            # Merge attributes from composite into the shape.
            for composite_attr in composite.attrs:
                index = attrs.index(composite_attr)
                if index < 0:
                    attrs.append(composite_attr)
                else:
                    attrs[index] = composite_attr
        shape = ProductShape(None, tuple(attrs), Behaviours(tuple(behaviours)), node.span)
        if node.shape_ref:
            shape_ref = self.type_env.get(node.shape_ref)
            if isinstance(shape_ref, ErrorShape):
                return shape_ref
            if not isinstance(shape_ref, FunShape):
                shape = replace(shape, behaviours=shape.behaviours.merge(shape_ref.behaviours))
            if not shape.conforms_to(shape_ref):
                return self.error(error.does_not_conform_to(str(shape), node.shape_ref.name, node.span))
            shape = replace(shape, name=node.shape_ref.name)
        return shape

    def tc_shape_lit_attr(self, node: ast.ShapeLitAttr) -> Shape:
        ast.walk(node, self.visit)
        return Unit

    def tc_shape_alias(self, node: ast.ShapeAlias) -> Shape:
        ast.walk(node, self.visit)
        shape = self.type_env.get(node.shape)
        if isinstance(shape, ErrorShape):
            return shape
        shape = replace(shape, name=node.name)
        self.scope.bind(node.name, shape)
        return Unit

    def tc_shape_ref(self, node: ast.ShapeRef) -> Shape:
        declared = self.scope.lookup(node.name)
        if declared is None:
            return self.error(error.undefined_name(node.name, node.span))
        return declared.shape

    def tc_sum_shape(self, node: ast.SumShape) -> Shape:
        ast.walk(node, self.visit)
        variants = [self.type_env.get(variant) for variant in node.variants]
        behaviours = []
        for behaviour_node in node.behaviours:
            behaviour = self.behaviours.get(behaviour_node.name)
            if not behaviour:
                return self.error(error.undefined_name(behaviour_node.name, behaviour_node.span))
            behaviours.append(behaviour)
        return SumShape(None, tuple(variants), Behaviours(tuple(behaviours)), node.span)

    def visit(self, node: ast.Node, _parent: ast.Node | None) -> ast.Node:
        shape: Shape
        match node:
            case ast.Assign():
                shape = self.tc_assign(node)
            case ast.Attr():
                shape = self.tc_attr(node)
            case ast.Behaviour():
                shape = self.tc_behaviour(node)
            case ast.BinaryExpr():
                shape = self.tc_binary_expr(node)
            case ast.Block():
                shape = self.tc_block(node)
            case ast.BoolLit():
                shape = Bool
            case ast.Call():
                shape = self.tc_call(node)
            case ast.CharLit():
                shape = Char
            case ast.FunDef():
                shape = self.tc_fun_def(node)
            case ast.FunParam():
                shape = self.tc_fun_param(node)
            case ast.If():
                shape = self.tc_if(node)
            case ast.IfArm():
                shape = self.tc_if_arm(node)
            case ast.IntLit():
                shape = Int
            case ast.Member():
                shape = self.tc_member(node)
            case ast.Module():
                shape = self.tc_module(node)
            case ast.Name():
                shape = self.tc_name(node)
            case ast.ProductShape():
                shape = self.tc_product_shape(node)
            case ast.ShapeLit():
                shape = self.tc_shape_lit(node)
            case ast.ShapeLitAttr():
                shape = self.tc_shape_lit_attr(node)
            case ast.ShapeAlias():
                shape = self.tc_shape_alias(node)
            case ast.ShapeRef():
                shape = self.tc_shape_ref(node)
            case ast.StrLit():
                shape = Str
            case ast.SumShape():
                shape = self.tc_sum_shape(node)
            case ast.UnitShape():
                shape = Unit
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
