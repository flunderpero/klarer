from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

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
    id: NodeId = field(compare=False, hash=False, repr=False)
    value: str
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return nid(self.id) + f'"{self.value}"'


@dataclass
class CharLit:
    id: NodeId = field(compare=False, hash=False, repr=False)
    value: str
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return nid(self.id) + f"'{self.value}'"


@dataclass
class IntLit:
    id: NodeId = field(compare=False, hash=False, repr=False)
    bits: int
    signed: bool
    value: int
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        if self.signed and self.bits == 64:
            return nid(self.id) + "Int"
        return nid(self.id) + f"I{self.bits}" if self.signed else f"U{self.bits}"


@dataclass
class BoolLit:
    id: NodeId = field(compare=False, hash=False, repr=False)
    value: bool
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return nid(self.id) + str(self.value).lower()


@dataclass
class Name:
    id: NodeId = field(compare=False, hash=False, repr=False)
    name: str
    kind: Literal["ident", "type", "behaviour"]
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        if self.kind == "behaviour":
            return nid(self.id) + f"@{self.name}"
        return nid(self.id) + self.name


@dataclass
class Member:
    id: NodeId = field(compare=False, hash=False, repr=False)
    target: Expr
    name: str
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return nid(self.id) + f"{self.target}.{self.name}"


@dataclass
class ShapeLitField:
    id: NodeId = field(compare=False, hash=False, repr=False)
    name: str
    value: Expr
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return nid(self.id) + f"{self.name} = {self.value}"


@dataclass
class ListLit:
    id: NodeId = field(compare=False, hash=False, repr=False)
    values: list[Expr]
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return nid(self.id) + f"[{self.values}]"


@dataclass
class ListIndex:
    id: NodeId = field(compare=False, hash=False, repr=False)
    target: Expr
    index: Expr
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return nid(self.id) + f"{self.target}[{self.index}]"


@dataclass
class CompoundShapeLit:
    id: NodeId = field(compare=False, hash=False, repr=False)
    shapes: list[ProductShapeLit]
    behaviours: list[Behaviour]
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        behaviours = " + " + " + ".join(str(x) for x in self.behaviours) if self.behaviours else ""
        return nid(self.id) + f"{{{', '.join(str(x) for x in self.shapes)}}}{behaviours}"


@dataclass
class ProductShapeLit:
    id: NodeId = field(compare=False, hash=False, repr=False)
    fields: list[ShapeLitField]
    shape_name: str | None
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        shape_name = f"{self.shape_name}" if self.shape_name else ""
        return nid(self.id) + f"{shape_name}{{{', '.join(str(x) for x in self.fields)}}}"


@dataclass
class Behaviour:
    id: NodeId = field(compare=False, hash=False, repr=False)
    name: str
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return nid(self.id) + self.name


@dataclass
class ShapeAlias:
    id: NodeId = field(compare=False, hash=False, repr=False)
    name: str
    shape: Shape
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return nid(self.id) + f"{self.name} = {self.shape}"


@dataclass
class ShapeRef:
    id: NodeId = field(compare=False, hash=False, repr=False)
    name: str
    behaviours: list[Behaviour]
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        behaviours = " + " + " + ".join(str(x) for x in self.behaviours) if self.behaviours else ""
        return nid(self.id) + self.name + behaviours


@dataclass
class UnitShape:
    id: NodeId = field(compare=False, hash=False, repr=False)
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return nid(self.id) + "Unit"


@dataclass
class FunShape:
    id: NodeId = field(compare=False, hash=False, repr=False)
    params: list[Param]
    result: Shape
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        params = ", ".join(str(x) for x in self.params)
        return nid(self.id) + f"fun({params}) -> {self.result}"


@dataclass
class ListShape:
    id: NodeId = field(compare=False, hash=False, repr=False)
    inner: Shape
    behaviours: list[Behaviour]
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        behaviours = " + " + " + ".join(str(x) for x in self.behaviours) if self.behaviours else ""
        return nid(self.id) + f"[{self.inner}]{behaviours}"


@dataclass
class CompoundShape:
    id: NodeId = field(compare=False, hash=False, repr=False)
    shapes: list[ProductShape | ShapeRef]
    behaviours: list[Behaviour]
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        behaviours = " + " + " + ".join(str(x) for x in self.behaviours) if self.behaviours else ""
        shapes = " + " + " + ".join(str(x) for x in self.shapes)
        return nid(self.id) + f"{shapes}{behaviours}"


@dataclass
class ProductShape:
    id: NodeId = field(compare=False, hash=False, repr=False)
    fields: list[Field]
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        fields = ", ".join(str(x) for x in self.fields)
        return nid(self.id) + f"{{{fields}}}"


@dataclass
class SumShape:
    id: NodeId = field(compare=False, hash=False, repr=False)
    variants: list[Shape]
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return nid(self.id) + " | ".join(str(x) for x in self.variants)


@dataclass
class Assign:
    id: NodeId = field(compare=False, hash=False, repr=False)
    target: Name
    value: Expr
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return nid(self.id) + f"{self.target} = {self.value}"


@dataclass
class Call:
    id: NodeId = field(compare=False, hash=False, repr=False)
    callee: Expr
    args: list[Expr]
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return nid(self.id) + f"{self.callee}({', '.join(str(x) for x in self.args)})"


@dataclass
class If:
    id: NodeId = field(compare=False, hash=False, repr=False)
    arms: list[IfArm]
    else_block: Block | None
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        arms = "\n".join(str(x) for x in self.arms)
        else_ = " else do " + str(self.else_block) if self.else_block else ""
        return nid(self.id) + f"if {arms}{else_} end"


@dataclass
class IfArm:
    id: NodeId = field(compare=False, hash=False, repr=False)
    cond: Expr
    block: Block
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return nid(self.id) + f"case {self.cond} do {self.block}"


@dataclass
class Loop:
    id: NodeId = field(compare=False, hash=False, repr=False)
    block: Block
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return nid(self.id) + f"loop {self.block}"


class BinaryOp(Enum):
    add = "+"
    div = "/"
    eq = "=="
    mul = "*"
    ne = "!="
    sub = "-"


@dataclass
class BinaryExpr:
    id: NodeId = field(compare=False, hash=False, repr=False)
    op: BinaryOp
    lhs: Expr
    rhs: Expr
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return nid(self.id) + f"{self.lhs} {self.op} {self.rhs}"


@dataclass
class Block:
    id: NodeId = field(compare=False, hash=False, repr=False)
    nodes: list[Node]
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        values = "\n".join("    " + str(x) for x in self.nodes)
        return nid(self.id) + f"do\n{values}\nend"


@dataclass
class Field:
    id: NodeId = field(compare=False, hash=False, repr=False)
    name: str
    shape: Shape
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return nid(self.id) + f"{self.name} {self.shape}"


@dataclass
class Param:
    id: NodeId = field(compare=False, hash=False, repr=False)
    name: str
    shape: Shape
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return nid(self.id) + f"{self.name} {self.shape}"


@dataclass
class FunDef:
    id: NodeId = field(compare=False, hash=False, repr=False)
    name: str
    behaviour: str | None
    params: list[Param]
    result: Shape
    body: Block | None
    span: Span = field(compare=False, hash=False, repr=False)

    def is_behaviour_interface_method(self) -> bool:
        return self.behaviour is not None and self.body is None

    def __str__(self) -> str:
        behaviour = (f"{self.behaviour}.") if self.behaviour is not None else ""
        body = "" if self.body is None else f" {self.body}"
        return nid(self.id) + f"{behaviour}{self.name}({', '.join(str(x) for x in self.params)}) {self.result}{body}"


@dataclass
class Module:
    id: NodeId = field(compare=False, hash=False, repr=False)
    fqn: FQN
    nodes: list[Node]
    span: Span = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return nid(self.id) + f"mod {self.fqn}\n" + "\n".join(str(x) for x in self.nodes)


Shape = ShapeRef | FunShape | ProductShape | SumShape | UnitShape | ListShape | CompoundShape
Expr = (
    BinaryExpr
    | Block
    | BoolLit
    | Call
    | CharLit
    | CompoundShapeLit
    | If
    | IntLit
    | Member
    | Name
    | StrLit
    | ProductShapeLit
    | ListLit
    | ListIndex
)
Node = Expr | Module | Shape | Field | ShapeAlias | ShapeLitField | IfArm | Param | Assign | FunDef | Behaviour | Shape

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
            for i, param in enumerate(node.params):
                node.params[i] = cast(Param, visit(param, node))
            node.result = cast(Shape, visit(node.result, node))
            if node.body:
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
            node.target = cast(Name, visit(node.target, node))
            node.value = cast(Expr, visit(node.value, node))
        case Member():
            node.target = cast(Expr, visit(node.target, node))
        case ShapeAlias():
            node.shape = cast(Shape, visit(node.shape, node))
        case ShapeRef():
            for i, behaviour in enumerate(node.behaviours):
                node.behaviours[i] = cast(Behaviour, visit(behaviour, node))
        case ProductShapeLit():
            for i, field in enumerate(node.fields):
                node.fields[i] = cast(ShapeLitField, visit(field, node))
        case CompoundShapeLit():
            for i, shape in enumerate(node.shapes):
                node.shapes[i] = cast(ProductShapeLit, visit(shape, node))
            for i, behaviour in enumerate(node.behaviours):
                node.behaviours[i] = cast(Behaviour, visit(behaviour, node))
        case ShapeLitField():
            node.value = cast(Expr, visit(node.value, node))
        case Field():
            node.shape = cast(Shape, visit(node.shape, node))
        case ListLit():
            for i, value in enumerate(node.values):
                node.values[i] = cast(Expr, visit(value, node))
        case ListIndex():
            node.target = cast(Expr, visit(node.target, node))
            node.index = cast(Expr, visit(node.index, node))
        case ListShape():
            node.inner = cast(Shape, visit(node.inner, node))
        case ProductShape():
            for i, field in enumerate(node.fields):
                node.fields[i] = cast(Field, visit(field, node))
        case CompoundShape():
            for i, shape in enumerate(node.shapes):
                node.shapes[i] = cast(Any, visit(shape, node))
            for i, behaviour in enumerate(node.behaviours):
                node.behaviours[i] = cast(Behaviour, visit(behaviour, node))
        case SumShape():
            for i, variant in enumerate(node.variants):
                node.variants[i] = cast(Shape, visit(variant, node))
        case FunShape():
            for i, param in enumerate(node.params):
                node.params[i] = cast(Param, visit(param, node))
            node.result = cast(Shape, visit(node.result, node))
        case Param():
            node.shape = cast(Shape, visit(node.shape, node))
        case Name() | IntLit() | CharLit() | StrLit() | BoolLit() | ShapeRef() | Behaviour() | UnitShape():
            return False
        case _:
            raise AssertionError(f"Don't know how to walk: {node}")
    return True
