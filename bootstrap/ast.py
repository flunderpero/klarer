from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Literal, cast

if TYPE_CHECKING:
    from .span import FQN, Span

NodeId = int

nid_enabled = True


def to_str_withoud_nid(node: Node) -> str:
    global nid_enabled  # noqa: PLW0603
    ne = nid_enabled
    try:
        nid_enabled = False
        return str(node)
    finally:
        nid_enabled = ne


def nid(id: NodeId) -> str:
    if not nid_enabled:
        return ""
    return f"{id}:"


@dataclass
class StrLit:
    id: NodeId
    value: str
    span: Span

    def __str__(self) -> str:
        return nid(self.id) + f'"{self.value}"'


@dataclass
class CharLit:
    id: NodeId
    value: str
    span: Span

    def __str__(self) -> str:
        return nid(self.id) + f"'{self.value}'"


@dataclass
class IntLit:
    id: NodeId
    bits: int
    signed: bool
    value: int
    span: Span

    def __str__(self) -> str:
        if self.signed and self.bits == 64:
            return nid(self.id) + "Int"
        return nid(self.id) + f"I{self.bits}" if self.signed else f"U{self.bits}"


@dataclass
class BoolLit:
    id: NodeId
    value: bool
    span: Span

    def __str__(self) -> str:
        return nid(self.id) + str(self.value).lower()


@dataclass
class Name:
    id: NodeId
    name: str
    kind: Literal["ident", "type", "behaviour"]
    span: Span

    def __str__(self) -> str:
        if self.kind == "behaviour":
            return nid(self.id) + f"@{self.name}"
        return nid(self.id) + self.name


@dataclass
class Member:
    id: NodeId
    target: Expr
    name: str
    span: Span

    def __str__(self) -> str:
        return nid(self.id) + f"{self.target}.{self.name}"


@dataclass
class ShapeLitAttr:
    id: NodeId
    name: str
    value: Expr
    span: Span

    def __str__(self) -> str:
        return nid(self.id) + f"{self.name} = {self.value}"


@dataclass
class ShapeLit:
    id: NodeId
    shape_ref: ShapeRef | None
    attrs: list[ShapeLitAttr]
    span: Span

    def __str__(self) -> str:
        shape_ref = f"{self.shape_ref}." if self.shape_ref else ""
        return nid(self.id) + f"{shape_ref}{{{', '.join(str(x) for x in self.attrs)}}}"


@dataclass
class ShapeDecl:
    id: NodeId
    name: str
    shape: Shape
    behaviours: list[str]
    span: Span

    def __str__(self) -> str:
        behaviours = " with " + " + ".join(f"@{x}" for x in self.behaviours) if self.behaviours else ""
        return nid(self.id) + f"{self.name} = {self.shape}{behaviours}"


@dataclass
class ShapeRef:
    id: NodeId
    name: str
    span: Span

    def __str__(self) -> str:
        return nid(self.id) + self.name


@dataclass
class FunShape:
    id: NodeId
    params: list[Attr]
    result: Shape
    span: Span

    def __str__(self) -> str:
        params = ", ".join(str(x) for x in self.params)
        return nid(self.id) + f"fun({params}) -> {self.result}"


@dataclass
class ProductShape:
    id: NodeId
    attrs: list[Attr]
    span: Span

    def __str__(self) -> str:
        attrs = ", ".join(str(x) for x in self.attrs)
        return nid(self.id) + f"{{{attrs}}}"


@dataclass
class ProductShapeComp:
    id: NodeId
    lhs: Shape
    rhs: Shape
    span: Span

    def __str__(self) -> str:
        return nid(self.id) + f"{self.lhs} + {self.rhs}"


@dataclass
class SumShape:
    id: NodeId
    variants: list[Shape]
    span: Span

    def __str__(self) -> str:
        return nid(self.id) + " | ".join(str(x) for x in self.variants)


@dataclass
class Assign:
    id: NodeId
    target: Expr
    value: Expr
    span: Span
    mut: bool

    def __str__(self) -> str:
        mut = "mut " if self.mut else ""
        return nid(self.id) + f"{mut}{self.target} = {self.value}"


@dataclass
class Call:
    id: NodeId
    callee: Expr
    args: list[Expr]
    span: Span

    def __str__(self) -> str:
        return nid(self.id) + f"{self.callee}({', '.join(str(x) for x in self.args)})"


@dataclass
class If:
    id: NodeId
    arms: list[IfArm]
    else_block: Block | None
    span: Span

    def __str__(self) -> str:
        arms = "\n".join(str(x) for x in self.arms)
        else_ = " else do " + str(self.else_block) if self.else_block else ""
        return nid(self.id) + f"if {arms}{else_} end"


@dataclass
class IfArm:
    id: NodeId
    cond: Expr
    block: Block
    span: Span

    def __str__(self) -> str:
        return nid(self.id) + f"case {self.cond} do {self.block}"


@dataclass
class Loop:
    id: NodeId
    block: Block
    span: Span

    def __str__(self) -> str:
        return nid(self.id) + f"loop {self.block}"


class BinaryOp(Enum):
    add = "+"
    eq = "=="
    ne = "!="
    sub = "-"


@dataclass
class BinaryExpr:
    id: NodeId
    op: BinaryOp
    lhs: Expr
    rhs: Expr
    span: Span

    def __str__(self) -> str:
        return nid(self.id) + f"{self.lhs} {self.op} {self.rhs}"


@dataclass
class Block:
    id: NodeId
    nodes: list[Node]
    span: Span

    def __str__(self) -> str:
        values = "\n".join("    " + str(x) for x in self.nodes)
        return nid(self.id) + f"do\n{values}\nend"


@dataclass
class Attr:
    id: NodeId
    name: str
    shape: Shape
    span: Span

    def __str__(self) -> str:
        return nid(self.id) + f"{self.name} {self.shape}"


@dataclass
class FunDef:
    id: NodeId
    name: str
    namespace: str | None
    params: list[str]
    body: Block
    span: Span

    def __str__(self) -> str:
        ns = (f"@{self.namespace}.") if self.namespace is not None else ""
        return nid(self.id) + f"{ns}{self.name}({', '.join(str(x) for x in self.params)}){self.body}"


@dataclass
class Module:
    id: NodeId
    fqn: FQN
    nodes: list[Node]
    span: Span

    def __str__(self) -> str:
        return nid(self.id) + f"mod {self.fqn}\n" + "\n".join(str(x) for x in self.nodes)


Shape = ShapeRef | FunShape | ProductShape | ProductShapeComp | SumShape
Expr = BinaryExpr | Block | BoolLit | Call | CharLit | If | IntLit | Member | Name | StrLit | ShapeLit
Node = Assign | Expr | FunDef | Module | Loop | Shape | Attr | ShapeDecl | ShapeLitAttr | IfArm

ASTVisitor = Callable[[Node, Node | None], Node]


def walk(node: Node, visit_: ASTVisitor) -> bool:
    """Visit all children of the given node.

    @return True if the node contained children
    """

    def visit(node: Node, parent: Node | None) -> Node:
        res = visit_(node, parent)
        assert res is not None, f"`visit` must return a node, got {res}"
        return res

    match node:
        case Module():
            for i, n in enumerate(node.nodes):
                node.nodes[i] = visit(n, node)
        case Block():
            for i, n in enumerate(node.nodes):
                node.nodes[i] = visit(n, node)
        case FunDef():
            node.body = cast(Block, visit(node.body, node))
        case Call():
            node.callee = cast(Expr, visit(node.callee, node))
            for i, arg in enumerate(node.args):
                node.args[i] = cast(Expr, visit(arg, node))
        case BinaryExpr():
            node.lhs = cast(Expr, visit(node.lhs, node))
            node.rhs = cast(Expr, visit(node.rhs, node))
        case If():
            for i, arm in enumerate(node.arms):
                node.arms[i] = cast(IfArm, visit(arm, node))
            if node.else_block:
                node.else_block = cast(Block, visit(node.else_block, node))
        case IfArm():
            node.cond = cast(Expr, visit(node.cond, node))
            node.block = cast(Block, visit(node.block, node))
        case Loop():
            node.block = cast(Block, visit(node.block, node))
        case Assign():
            node.target = cast(Expr, visit(node.target, node))
            node.value = cast(Expr, visit(node.value, node))
        case Member():
            node.target = cast(Expr, visit(node.target, node))
        case ShapeDecl():
            node.shape = cast(Shape, visit(node.shape, node))
        case ShapeLit():
            if node.shape_ref:
                node.shape_ref = cast(ShapeRef, visit(node.shape_ref, node))
            for i, attr in enumerate(node.attrs):
                node.attrs[i] = cast(ShapeLitAttr, visit(attr, node))
        case ShapeLitAttr():
            node.value = cast(Expr, visit(node.value, node))
        case Attr():
            node.shape = cast(Shape, visit(node.shape, node))
        case ProductShape():
            for i, attr in enumerate(node.attrs):
                node.attrs[i] = cast(Attr, visit(attr, node))
        case ProductShapeComp():
            node.lhs = cast(Shape, visit(node.lhs, node))
            node.rhs = cast(Shape, visit(node.rhs, node))
        case SumShape():
            for i, variant in enumerate(node.variants):
                node.variants[i] = cast(Shape, visit(variant, node))
        case FunShape():
            for i, param in enumerate(node.params):
                node.params[i] = cast(Attr, visit(param, node))
            node.result = cast(Shape, visit(node.result, node))
        case Name() | IntLit() | CharLit() | StrLit() | BoolLit() | ShapeRef():
            return False
        case _:
            raise AssertionError(f"Don't know how to walk: {node}")
    return True
