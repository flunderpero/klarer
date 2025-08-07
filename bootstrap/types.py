from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal

from . import ast, error
from .span import Span, log

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass
class Behaviour:
    name: str
    funs: list[Fun]

    def __str__(self) -> str:
        return f"@{self.name}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Behaviour) and self.name == other.name

    def __hash__(self) -> int:
        return hash(str(self))

    def mangled_name(self) -> str:
        return self.name


@dataclass
class Primitive:
    name: Literal["Bool", "Char", "Int", "Str", "Unit"]
    behaviours: list[Behaviour]
    span: Span

    def __str__(self) -> str:
        behaviours = " bind " + " + ".join(str(x) for x in self.behaviours) if self.behaviours else ""
        return f"{self.name}{behaviours}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Primitive) and self.name == other.name

    def __hash__(self) -> int:
        return hash(str(self))

    def mangled_name(self) -> str:
        return self.name


@dataclass
class Attr:
    name: str
    typ: Typ

    def __str__(self) -> str:
        return self.name + " " + str(self.typ)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Attr) and self.name == other.name and self.typ == other.typ

    def __hash__(self) -> int:
        return hash(str(self))

    def mangled_name(self) -> str:
        return self.name + "_" + self.typ.mangled_name()


@dataclass
class Shape:
    name: str | None
    attrs: list[Attr]
    variants: list[Typ]
    behaviours: list[Behaviour]
    span: Span

    def is_named(self) -> bool:
        return self.name is not None

    def attr(self, name: str) -> Attr | None:
        return next((x for x in self.attrs if x.name == name), None)

    def attr_index(self, name: str) -> int | None:
        return next((i for i, x in enumerate(self.attrs) if x.name == name), None)

    def __str__(self) -> str:
        variants = " | " + " | ".join(str(x) for x in self.variants) if self.variants else ""
        behaviours = " bind " + " + ".join(str(x) for x in self.behaviours) if self.behaviours else ""
        name = f" {self.name}" if self.name else ""
        return f"{name}{{{self.attrs}}}{variants}{behaviours}"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Shape)
            and self.attrs == other.attrs
            and self.variants == other.variants
            and self.behaviours == other.behaviours
        )

    def __hash__(self) -> int:
        return hash(str(self))

    def mangled_name(self) -> str:
        name = [x.mangled_name() for x in self.attrs + self.variants + self.behaviours]
        if self.name:
            return self.name + "_" + "_".join(name)
        return "_".join(name)


@dataclass
class Fun:
    name: str | None
    params: list[Attr]
    result: Typ
    span: Span
    builtin: bool

    @property
    def is_named(self) -> bool:
        return self.name is not None

    def __str__(self) -> str:
        params = ", ".join(str(x) for x in self.params)
        name = f" {self.name}" if self.name else ""
        return f"fun{name}({params}) -> {self.result}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Fun) and self.params == other.params and self.result == other.result

    def __hash__(self) -> int:
        return hash(str(self))

    def mangled_name(self) -> str:
        if self.builtin:
            assert self.name is not None
            return self.name
        params = [x.mangled_name() for x in [*self.params, self.result]]
        if self.name:
            return self.name + "_" + "_".join(params)
        return "_".join(params)


@dataclass
class Typ[T: Primitive | Shape | Fun | error.Error | None]:
    typ: T

    def __str__(self) -> str:
        return str(self.typ)

    @property
    def span(self) -> Span:
        if self.typ is None:
            return Span("<unknown>", "", 0, 0)
        return self.typ.span

    def is_error(self) -> bool:
        return self.typ is not None and isinstance(self.typ, error.Error)

    def is_none(self) -> bool:
        return self.typ is None

    def is_primitive(self) -> bool:
        return self.typ is not None and isinstance(self.typ, Primitive)

    def is_fun(self) -> bool:
        return self.typ is not None and isinstance(self.typ, Fun)

    def is_assignable_from(self, other: Typ) -> bool:
        if self.is_none():
            return other.is_none()
        if isinstance(self.typ, Primitive):
            return isinstance(other.typ, Primitive) and self.typ.name == other.typ.name
        # todo: check more types
        return False

    def merge(self, other: Typ) -> None:
        if self.is_none():
            self.typ = other.typ
        elif other.is_none():
            pass
        # todo: merge more types

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Typ) and self.typ == other.typ

    def __hash__(self) -> int:
        return hash(str(self))

    def mangled_name(self) -> str:
        if self.typ is None:
            return "none"
        assert not isinstance(self.typ, error.Error), f"Type {self.typ} must be resolved before mangling"
        return self.typ.mangled_name()


builtin_span = Span("<builtin", "", 0, 0)
BoolTyp = Typ(Primitive("Bool", [], builtin_span))
CharTyp = Typ(Primitive("Char", [], builtin_span))
IntTyp = Typ(Primitive("Int", [], builtin_span))
StrTyp = Typ(Primitive("Str", [], builtin_span))
UnitTyp = Typ(Primitive("Unit", [], builtin_span))


@dataclass
class TypeEnv:
    parent: TypeEnv | None
    node_types: dict[ast.NodeId, Typ]

    def set(self, node: ast.Node, typ: Typ) -> None:
        # If the type is unset in the root type_env, set it there, too.
        p = self
        while p.parent:
            if node.id in p.parent.node_types:
                break
            p = p.parent
        else:
            p.node_types[node.id] = typ
        self.node_types[node.id] = typ

    def get(self, node: ast.Node) -> Typ:
        typ = self.node_types.get(node.id)
        if typ is None and self.parent:
            typ = self.parent.get(node)
        if typ is None:
            raise KeyError(f"Type for {node} not found")
        return typ


@dataclass
class Binding:
    typ: Typ
    mut: bool
    builtin: bool


@dataclass
class Scope:
    node: ast.Node | None
    parent: Scope | None
    bindings: dict[str, Binding]

    @staticmethod
    def root() -> Scope:
        """The root scope with all the builtins."""
        span = Span("<builtin>", "", 0, 0)
        scope = Scope(None, None, {})
        scope.bindings["print"] = Binding(
            Typ(Fun("print", [Attr("s", StrTyp)], UnitTyp, span=span, builtin=True)),
            mut=False,
            builtin=True,
        )
        scope.bindings["Int"] = Binding(IntTyp, mut=False, builtin=True)
        scope.bindings["Str"] = Binding(StrTyp, mut=False, builtin=True)
        scope.bindings["Bool"] = Binding(BoolTyp, mut=False, builtin=True)
        scope.bindings["Char"] = Binding(CharTyp, mut=False, builtin=True)
        return scope

    def lookup(self, name: str) -> Binding | None:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def bind(self, name: str, typ: Typ, *, mut: bool) -> error.Error | None:
        existing = self.lookup(name)
        if existing:
            return error.duplicate_declaration(name, existing.typ.span, typ.span)
        self.bindings[name] = Binding(typ, mut, builtin=False)
        return None


@dataclass
class FunSpec:
    type_env: TypeEnv
    fun_def: ast.FunDef
    # The base function type found at definition.
    base: Fun
    # The specialized function type.
    specialized: Fun

    def __str__(self) -> str:
        return str(self.specialized)


class TypeCheck:
    type_env: TypeEnv
    errors: list[error.Error]
    scope: Scope
    fun_specs: dict[Fun, list[FunSpec]]
    fun_defs: dict[Fun, ast.FunDef]
    nesting_level = 0

    def __init__(self) -> None:
        self.type_env = TypeEnv(None, {})
        self.errors = []
        self.scope = Scope.root()
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

    def fun_spec(self, fun: Fun, call_args: list[ast.Expr]) -> FunSpec | None:
        """Try to find a FunSpec for the given function with the given parameter types."""
        specs = self.fun_specs.get(fun, [])
        param_types = [self.type_env.get(x) for x in call_args]
        for spec in specs:
            if spec.base == fun and spec.specialized.params == param_types:
                return spec
        return None

    def specialize(self, fun: Fun, call_args: list[ast.Expr], span: Span) -> FunSpec:
        spec = self.fun_spec(fun, call_args)
        if spec:
            return spec
        with self.child_type_env():
            fun_def = self.fun_defs[fun]
            log("typechecker-mono", f">>> Specializing {fun} at call-site {span}", self.nesting_level)

            params = [Attr(attr.name, self.type_env.get(arg)) for attr, arg in zip(fun.params, call_args)]
            specialized = Fun(fun.name, params, Typ(None), fun.span, builtin=fun.builtin)

            base = self.type_env.get(fun_def)
            spec = FunSpec(self.type_env, fun_def, base.typ, specialized)
            self.fun_specs.get(fun, []).append(spec)

            log(
                "typechecker-mono",
                f"Type checking {spec.base} at {spec.fun_def.span} with {spec.specialized} at call-site {span}",
                self.nesting_level,
            )
            typ = self.tc_fun_def_specialized(spec.fun_def, spec.specialized)
            spec.specialized.result = typ.typ.result
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

    def error(self, err: error.Error) -> Typ:
        self.errors.append(err)
        return Typ(err)

    def tc_assign(self, node: ast.Assign) -> Typ:
        self.visit(node.value, None)
        value = self.type_env.get(node.value)
        if value.is_error():
            return value
        match node.target:
            case ast.Name():
                binding = self.scope.lookup(node.target.name)
                if binding is not None:
                    if not binding.mut:
                        return self.error(error.not_mutable(node.target.name, node.target.span))
                    if not binding.typ.is_assignable_from(value):
                        return self.error(error.not_assignable_from(node.target.span, str(binding.typ), str(value)))
                else:
                    log("typechecker-trace", f"Binding {node.target.name} to {value}", self.nesting_level)
                    self.scope.bind(node.target.name, value, mut=node.mut)
                self.type_env.set(node.target, value)
            case ast.Member():
                ast.walk(node.target, self.visit)
                shape = self.type_env.get(node.target.target)
                if shape.is_error():
                    return shape
                attr = shape.typ.attr(node.target.name)
                if attr is None:
                    return self.error(
                        error.no_member(node.target.name, str(shape), node.target.span, node.target.target.span)
                    )
                if not attr.typ.is_assignable_from(value):
                    return self.error(error.not_assignable_from(node.target.span, str(attr.typ), str(value)))
            case _:
                raise AssertionError(f"Unsupported target type: {node.target}")
        return UnitTyp

    def tc_attr(self, node: ast.Attr) -> Typ:
        ast.walk(node, self.visit)
        typ = self.type_env.get(node.shape)
        if typ.is_error():
            return typ
        return Typ(Shape(None, [Attr(node.name, typ)], [], [], node.span))

    def tc_binary_expr(self, node: ast.BinaryExpr) -> Typ:
        ast.walk(node, self.visit)
        lhs = self.type_env.get(node.lhs)
        rhs = self.type_env.get(node.rhs)
        if lhs.is_error():
            return lhs
        if rhs.is_error():
            return rhs
        # We can directly merge the types of the two operands.
        lhs.merge(rhs)
        rhs.merge(lhs)
        if not lhs.is_assignable_from(rhs):
            return self.error(error.not_assignable_from(node.span, str(lhs), str(rhs)))
        return BoolTyp

    def tc_block(self, node: ast.Block) -> Typ:
        with self.child_scope(node):
            ast.walk(node, self.visit)
        if len(node.nodes) == 0:
            return UnitTyp
        return self.type_env.get(node.nodes[-1])

    def tc_call(self, node: ast.Call) -> Typ:
        ast.walk(node, self.visit)
        callee = self.type_env.get(node.callee)
        if callee.is_error():
            return callee
        if not isinstance(callee.typ, Fun):
            return self.error(error.not_callable(node.callee.span, callee.span))

        # Build a specialized function if it is not a builtin.
        if callee.typ.builtin:
            return callee.typ.result
        spec = self.specialize(callee.typ, node.args, node.span)
        self.type_env.set(node.callee, Typ(spec.specialized))
        return spec.specialized.result

    def tc_fun_def(self, node: ast.FunDef) -> Typ:
        params: list[Attr] = []
        with self.child_scope(node):
            for param in node.params:
                param_typ = Typ(None)
                if err := self.scope.bind(param, param_typ, mut=False):
                    return self.error(err)
                params.append(Attr(param, param_typ))
            ast.walk(node, self.visit)
        return_typ = self.type_env.get(node.body)
        typ = Typ(Fun(node.name, params, return_typ, node.span, builtin=False))
        log("typechecker-trace", f"Adding {typ.typ} to fun_defs", self.nesting_level)
        self.fun_defs[typ.typ] = node
        if err := self.scope.bind(node.name, typ, mut=False):
            return self.error(err)
        if node.name == "main":
            log("typechecker-trace", "Adding main to fun_defs", self.nesting_level)
            fun = typ.typ
            if len(fun.params) != 0 or fun.result != UnitTyp:
                return self.error(error.invalid_main(node.span))
            self.fun_specs[fun] = [FunSpec(self.type_env, node, fun, fun)]
        return typ

    def tc_member(self, node: ast.Member) -> Typ:
        ast.walk(node, self.visit)
        typ = self.type_env.get(node.target)
        if typ.is_error():
            return typ
        shape = typ.typ
        if not isinstance(shape, Shape):
            return self.error(error.unexpected_type(f"a shape with field `{node.name}`", str(shape), node.target.span))
        attr = shape.attr(node.name)
        if not attr:
            return self.error(error.no_member(node.name, str(shape), node.target.span, node.span))
        return attr.typ

    def tc_fun_def_specialized(self, node: ast.FunDef, fun: Fun) -> Typ:
        with self.child_scope(node):
            for param in fun.params:
                if err := self.scope.bind(param.name, param.typ, mut=False):
                    return self.error(err)
            ast.walk(node, self.visit)
        return_typ = self.type_env.get(node.body)
        fun.result = return_typ
        return Typ(fun)

    def tc_module(self, node: ast.Module) -> Typ:
        ast.walk(node, self.visit)
        return UnitTyp

    def tc_name(self, node: ast.Name) -> Typ:
        name = self.scope.lookup(node.name)
        if name is None:
            return self.error(error.undefined_name(node.name, node.span))
        return name.typ

    def tc_product_shape(self, node: ast.ProductShape) -> Typ:
        ast.walk(node, self.visit)
        attrs: list[Attr] = []
        for attr in node.attrs:
            typ = self.type_env.get(attr.shape)
            if typ.is_error():
                return typ
            attrs.append(Attr(attr.name, typ))
        return Typ(Shape(None, attrs, [], [], node.span))

    def tc_shape_literal(self, node: ast.ShapeLit) -> Typ:
        ast.walk(node, self.visit)
        attrs: list[Attr] = []
        for attr in node.attrs:
            typ = self.type_env.get(attr.value)
            if typ.is_error():
                return typ
            attrs.append(Attr(attr.name, typ))
        # todo: type check if node.shape_ref is set
        return Typ(Shape(None, attrs, [], [], node.span))

    def tc_shape_literal_attr(self, node: ast.ShapeLitAttr) -> Typ:
        ast.walk(node, self.visit)
        typ = self.type_env.get(node.value)
        if typ.is_error():
            return typ
        return Typ(Shape(None, [Attr(node.name, typ)], [], [], node.span))

    def tc_shape_decl(self, node: ast.ShapeDecl) -> Typ:
        ast.walk(node, self.visit)
        typ = self.type_env.get(node.shape)
        if typ.is_error():
            return typ
        if err := self.scope.bind(node.name, typ, mut=False):
            return self.error(err)
        return UnitTyp

    def tc_shape_ref(self, node: ast.ShapeRef) -> Typ:
        declared = self.scope.lookup(node.name)
        if declared is None:
            return self.error(error.undefined_name(node.name, node.span))
        return declared.typ

    def visit(self, node: ast.Node, _parent: ast.Node | None) -> ast.Node:
        typ: Typ
        match node:
            case ast.Assign():
                typ = self.tc_assign(node)
            case ast.Attr():
                typ = self.tc_attr(node)
            case ast.BinaryExpr():
                typ = self.tc_binary_expr(node)
            case ast.Block():
                typ = self.tc_block(node)
            case ast.BoolLit():
                typ = BoolTyp
            case ast.Call():
                typ = self.tc_call(node)
            case ast.CharLit():
                typ = CharTyp
            case ast.FunDef():
                typ = self.tc_fun_def(node)
            case ast.IntLit():
                typ = IntTyp
            case ast.Member():
                typ = self.tc_member(node)
            case ast.Module():
                typ = self.tc_module(node)
            case ast.Name():
                typ = self.tc_name(node)
            case ast.ProductShape():
                typ = self.tc_product_shape(node)
            case ast.ShapeLit():
                typ = self.tc_shape_literal(node)
            case ast.ShapeLitAttr():
                typ = self.tc_shape_literal_attr(node)
            case ast.ShapeDecl():
                typ = self.tc_shape_decl(node)
            case ast.ShapeRef():
                typ = self.tc_shape_ref(node)
            case ast.StrLit():
                typ = StrTyp
            case _:
                raise AssertionError(f"Don't know how to type check: {node!r}")
        self.type_env.set(node, typ)
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
