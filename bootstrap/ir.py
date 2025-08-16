from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from . import ast, types
from .span import log

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass
class Int:
    bits: int
    signed: bool

    def __str__(self) -> str:
        return f"I{self.bits}" if self.signed else f"U{self.bits}"


@dataclass
class Str:
    def __str__(self) -> str:
        return "Str"


@dataclass
class Struct:
    fqn: str
    fields: list[Typ]

    def __str__(self) -> str:
        return f"{self.fqn}{{{', '.join(str(f) for f in self.fields)}}}"


@dataclass
class Fun:
    fqn: str
    params: list[Typ]
    result: Typ
    is_named: bool

    def __str__(self) -> str:
        name = f" {self.fqn}" if self.is_named else ""
        return f"fn{name}({', '.join(str(p) for p in self.params)}) -> {self.result}"


@dataclass
class Ptr:
    typ: Typ

    def __str__(self) -> str:
        return f"*{self.typ}"


@dataclass
class NoneTyp:
    def __str__(self) -> str:
        return "none"


Typ = Int | Struct | Fun | Ptr | Str | NoneTyp


I1 = Int(bits=1, signed=True)
I8 = Int(bits=8, signed=True)
U8 = Int(bits=8, signed=False)
I16 = Int(bits=16, signed=True)
U16 = Int(bits=16, signed=False)
I32 = Int(bits=32, signed=True)
U32 = Int(bits=32, signed=False)
I64 = Int(bits=64, signed=True)
U64 = Int(bits=64, signed=False)
Char = I64  # todo: Change it to U32 once we support int types other than I64

RegId = str


@dataclass
class Reg:
    id: RegId
    typ: Typ

    def __str__(self) -> str:
        return self.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Reg) and self.id == other.id


NoneReg = Reg("_none", NoneTyp())


@dataclass
class IntConst:
    reg: Reg
    value: int

    def __str__(self) -> str:
        return f"{self.reg.id} = {self.value}"

    def regs(self) -> list[Reg]:
        return [self.reg]


@dataclass
class GetPtr:
    reg: Reg
    src: Reg
    field: int = 0

    def __str__(self) -> str:
        return f"{self.reg} = getptr {self.src.typ} {self.src}, {self.reg.typ}, {self.field}"

    def regs(self) -> list[Reg]:
        return [self.reg, self.src]


@dataclass
class GetFnPtr:
    reg: Reg
    src: Fun

    def __str__(self) -> str:
        return f"{self.reg} = getfnptr {self.src.fqn}, {self.reg.typ}"

    def regs(self) -> list[Reg]:
        return [self.reg]


@dataclass
class Load:
    reg: Reg
    src: Reg

    def __str__(self) -> str:
        return f"{self.reg} = load {self.src.typ} {self.src}"

    def regs(self) -> list[Reg]:
        return [self.reg, self.src]


@dataclass
class Store:
    target: Reg
    src: Reg

    def __str__(self) -> str:
        return f"store {self.src.typ} {self.src}, {self.target}"

    @property
    def reg(self) -> Reg:
        """A store instruction does not create a new register."""
        return NoneReg

    def regs(self) -> list[Reg]:
        return [self.target, self.src]


@dataclass
class Call:
    reg: Reg
    callee: str | Reg
    args: list[Reg]

    def __str__(self) -> str:
        prefix = f"{self.reg} = call {self.reg.typ}" if self.reg != NoneReg else "call none"
        s = f"{prefix} {self.callee}"
        if self.args:
            s += f", {', '.join(f'{x.typ} {x.id}' for x in self.args)}"
        return s

    def regs(self) -> list[Reg]:
        regs = list(self.args)
        regs.append(self.reg)
        if isinstance(self.callee, Reg):
            regs.append(self.callee)
        return regs


@dataclass
class Alloc:
    reg: Reg
    args: list[Reg]

    def __str__(self) -> str:
        return f"{self.reg.id} = alloc {self.reg.typ}, {', '.join(f'{x.typ} {x.id}' for x in self.args)}"

    def regs(self) -> list[Reg]:
        return [self.reg, *self.args]


@dataclass
class IAddO:
    """Signed addition with overflow."""

    reg: Reg
    lhs: Reg
    rhs: Reg

    def __str__(self) -> str:
        return f"{self.reg} = iaddo {self.lhs.typ} {self.lhs}, {self.rhs.typ} {self.rhs}"

    def regs(self) -> list[Reg]:
        return [self.reg, self.lhs, self.rhs]


@dataclass
class ISubO:
    """Signed subtraction with overflow."""

    reg: Reg
    lhs: Reg
    rhs: Reg

    def __str__(self) -> str:
        return f"{self.reg} = isubo {self.lhs.typ} {self.lhs}, {self.rhs.typ} {self.rhs}"

    def regs(self) -> list[Reg]:
        return [self.reg, self.lhs, self.rhs]


class ICmpOp(Enum):
    eq = "eq"
    ne = "ne"


@dataclass
class ICmp:
    reg: Reg
    op: ICmpOp
    lhs: Reg
    rhs: Reg

    def __str__(self) -> str:
        return f"{self.reg} = icmp {self.op.value} {self.lhs.typ} {self.lhs}, {self.rhs.typ} {self.rhs}"

    def regs(self) -> list[Reg]:
        return [self.reg, self.lhs, self.rhs]


@dataclass
class PhiIn:
    reg: Reg
    block: Block

    def __str__(self) -> str:
        return f"[{self.reg}, {self.block.id}]"


@dataclass
class Phi:
    reg: Reg
    incoming: list[PhiIn]

    def __str__(self) -> str:
        return f"{self.reg} = phi {', '.join(str(reg) for reg in self.incoming)}"

    def regs(self) -> list[Reg]:
        return [x.reg for x in self.incoming] + [self.reg]


Inst = IntConst | GetPtr | GetFnPtr | Load | Store | Call | Alloc | IAddO | ISubO | ICmp | Phi

BlockId = int


@dataclass
class Block:
    id: BlockId
    insts: list[Inst]
    terminator: Terminator | None

    def __str__(self) -> str:
        insts = "\n".join(f"    {inst}" for inst in self.insts)
        if insts:
            insts += "\n"
        term = str(self.terminator) if self.terminator else "<TERMINATOR MISSING>"
        return f"{self.id}:\n{insts}    {term}"


@dataclass
class Branch:
    reg: Reg
    then_block: Block
    else_block: Block

    def __str__(self) -> str:
        return f"br {self.reg.typ} {self.reg.id}, {self.then_block.id}, {self.else_block.id}"

    def successors(self) -> list[Block]:
        return [self.then_block, self.else_block]

    def regs(self) -> list[Reg]:
        return [self.reg]


@dataclass
class Jump:
    target: Block

    def __str__(self) -> str:
        return f"b {self.target.id}"

    def successors(self) -> list[Block]:
        return [self.target]

    def regs(self) -> list[Reg]:
        return []


@dataclass
class Return:
    reg: Reg

    def __str__(self) -> str:
        return f"ret {self.reg.typ} {self.reg.id}"

    def successors(self) -> list[Block]:
        return []

    def regs(self) -> list[Reg]:
        return [self.reg]


Terminator = Branch | Jump | Return


@dataclass
class StrConst:
    reg: Reg
    value: str

    def __str__(self) -> str:
        return f'{self.reg.id} = "{self.value}"'


@dataclass
class IR:
    fn_irs: list[FunIR]
    constant_pool: dict[str, StrConst]
    structs: dict[str, Struct]

    def __str__(self) -> str:
        lines = []
        if self.constant_pool:
            lines.append("\n".join(str(const) for const in self.constant_pool.values()))
        if self.structs:
            lines.append("\n\n".join(str(struct) for struct in self.structs.values()))
        if self.fn_irs:
            lines.append("\n\n".join(str(fn_ir) for fn_ir in self.fn_irs))
        return "\n\n".join(lines)


@dataclass
class Param:
    reg: Reg
    typ: Typ

    def __str__(self) -> str:
        return f"{self.typ} {self.reg.id}"


@dataclass
class Scope:
    parent: Scope | None
    names: dict[str, Reg]

    def declare(self, name: str, reg: Reg) -> None:
        assert name not in self.names, f"Variable {name} already declared in scope"
        self.names[name] = reg

    def find(self, name: str) -> Reg | None:
        res = self.names.get(name)
        if res:
            return res
        if self.parent:
            return self.parent.find(name)
        return None


@dataclass
class LoopScope:
    continue_block: Block
    break_block: Block


@dataclass
class FunIR:
    fn_def: ast.FunDef
    fn_name: str
    params: list[Param]
    result: Typ
    blocks: list[Block]

    def __str__(self) -> str:
        params = ", ".join(str(param) for param in self.params)
        blocks = "\n".join(str(block) for block in self.blocks)
        return f"declare {self.fn_name}({params}) {self.result}:\n" + blocks


class FunGen:
    type_env: types.TypeEnv
    block: Block
    node_regs: dict[ast.NodeId, Reg]
    scope: Scope
    loop_scopes: list[LoopScope]
    ir: IR
    fun_ir: FunIR
    next_reg = 0
    next_block = 0

    def __init__(self, spec: types.FunSpec, ir: IR) -> None:
        self.type_env = spec.type_env
        self.ir = ir
        self.node_regs = {}
        self.loop_scopes = []
        self.scope = Scope(None, {})
        fun_typ = spec.specialized
        fun_def = spec.fun_def
        params: list[Param] = []
        if spec.specialized.params:
            for param in spec.specialized.params:
                typ = self.typ(param.shape)
                reg = self.reg(typ)
                self.scope.declare(param.name, reg)
                params.append(Param(reg, typ))
        else:
            for p in fun_typ.params:
                typ = self.typ(p.shape)
                reg = self.reg(typ)
                self.scope.declare(p.name, reg)
                params.append(Param(reg, typ))
        result = self.typ(fun_typ.result)
        name = self.fun_name(fun_typ) if fun_def.name != "main" else "main"
        self.fun_ir = FunIR(fun_def, name, params, result, [])
        self.block = self.new_block()

    def fun_name(self, fun: types.FunShape) -> str:
        if fun.builtin:
            assert fun.name is not None
            return fun.name
        return fun.mangled_name()

    def new_block(self) -> Block:
        self.next_block += 1
        res = Block(id=self.next_block, insts=[], terminator=None)
        self.fun_ir.blocks.append(res)
        return res

    @contextmanager
    def child_scope(self) -> Generator[None]:
        scope = self.scope
        self.scope = Scope(scope, {})
        yield
        self.scope = scope

    def typ(self, shape: types.Shape) -> Typ:
        match shape:
            case types.PrimitiveShape():
                match shape.name:
                    case "Bool":
                        return I1
                    case "Char":
                        return Char
                    case "Int":
                        return I64
                    case "Str":
                        return Str()
                    case _:
                        raise AssertionError(f"Unsupported primitive type: {shape.name}")
            case types.UnitShape():
                return NoneTyp()
            case types.ProductShape():
                name = shape.mangled_name()
                if existing := self.ir.structs.get(name):
                    return existing
                struct = Struct(name, [self.typ(x.shape) for x in shape.attrs_sorted])
                self.ir.structs[name] = struct
                return struct
            case types.FunShape():
                return Fun(
                    shape.mangled_name(),
                    [self.typ(x.shape) for x in shape.params],
                    self.typ(shape.result),
                    is_named=True,
                )
            case _:
                raise AssertionError(f"Unsupported type: {shape} ({shape.__class__})")

    def reg(self, typ: Typ, prefix: str = "_") -> Reg:
        if isinstance(typ, NoneTyp):
            return NoneReg
        self.next_reg += 1
        return Reg(id=f"{prefix}{self.next_reg}", typ=typ)

    def emit(self, inst: Inst, node: ast.Node | None) -> None:
        self.block.insts.append(inst)
        if node:
            assert node.id not in self.node_regs, f"Node {node.id} already has a register"
            self.node_regs[node.id] = inst.reg

    def generate(self, node: ast.Node, parent: ast.Node | None) -> ast.Node:
        match node:
            case ast.FunDef():
                ast.walk(node, self.generate)
                if self.block.terminator is None:
                    reg = NoneReg if isinstance(self.fun_ir.result, NoneTyp) else self.node_regs[node.body.id]
                    self.block.terminator = Return(reg)
            case ast.Block():
                with self.child_scope():
                    ast.walk(node, self.generate)
                    reg = NoneReg
                    if node.nodes:
                        reg = self.node_regs[node.nodes[-1].id]
                    self.node_regs[node.id] = reg
            case ast.If():
                log("ir-trace", f">>> if {node.span} ({len(node.arms)} arms, else: {node.else_block is not None})")
                # Add the else block to the list of arms to make the loop easier.
                arms: list[tuple[ast.Expr | None, ast.Block, ast.Node]] = [(x.cond, x.block, x) for x in node.arms]
                if node.else_block is not None:
                    arms.append((None, node.else_block, node.else_block))
                # Remember all the "then" blocks so we can set their terminators later.
                then_blocks = []
                next_block = None
                prev_block = self.block
                for cond, block, arm_node in arms:
                    if cond is not None:
                        self.generate(cond, arm_node)
                        then_block = self.new_block()
                    else:
                        then_block = prev_block
                    self.block = then_block
                    then_blocks.append(then_block)

                    self.generate(block, arm_node)

                    # Create a new block that will take the condition of the next arm.
                    next_block = self.new_block()
                    self.block = next_block
                    if prev_block is not None:
                        if cond is None:
                            prev_block.terminator = Jump(next_block)
                        else:
                            prev_block.terminator = Branch(self.node_regs[cond.id], then_block, next_block)
                    prev_block = next_block

                assert next_block is not None

                # Set the terminator of all the "then" blocks.
                for then_block in then_blocks:
                    then_block.terminator = Jump(next_block)

                # Insert another phi node for the result of the whole if expression.
                types_typ = self.type_env.get(node)
                if not isinstance(types_typ, types.UnitShape):
                    reg = self.reg(self.typ(types_typ))
                    phis = []
                    for arm, block in zip(arms, then_blocks):
                        block_reg = self.node_regs[arm[1].id]
                        phis.append(PhiIn(block_reg, block))
                    self.emit(Phi(reg, phis), node)
                else:
                    self.node_regs[node.id] = NoneReg
                log("ir-trace", f"<<< if {node.span}")
            case ast.StrLit():
                const = self.ir.constant_pool.get(node.value)
                if not const:
                    reg = Reg(f"s{len(self.ir.constant_pool)}", Str())
                    const = StrConst(reg, node.value)
                    self.ir.constant_pool[node.value] = const
                self.emit(GetPtr(reg=self.reg(Str()), src=const.reg), node)
            case ast.CharLit():
                reg = self.reg(Char)
                self.emit(IntConst(reg, value=ord(node.value)), node)
            case ast.IntLit():
                reg = self.reg(I64)
                self.emit(IntConst(reg, value=node.value), node)
            case ast.BoolLit():
                reg = self.reg(I1)
                self.emit(IntConst(reg, value=int(node.value)), node)
            case ast.ShapeLit():
                ast.walk(node, self.generate)
                # We need to sort the attributes by name because we did so in `typ()` when
                # construction the struct type.
                regs = [self.node_regs[x.id] for x in sorted(node.attrs, key=lambda x: x.name)]
                typ = self.typ(self.type_env.get(node))
                reg = self.reg(typ)
                self.emit(Alloc(reg, regs), node)
            case ast.ShapeLitAttr():
                ast.walk(node, self.generate)
                reg = self.node_regs[node.value.id]
                self.node_regs[node.id] = reg
            case ast.Name():
                reg = self.scope.find(node.name)
                if reg:
                    self.node_regs[node.id] = reg
                    return node
                # If this node is the callee of a call node then we don't want
                # to emit a GetFnPtr for named functions.
                if isinstance(parent, ast.Call) and parent.callee == node:
                    return node
                ir_typ = self.type_env.get(node)
                if not isinstance(ir_typ, types.FunShape) or not ir_typ.is_named:
                    return node
                # Emit a GetFnPtr if the identifier refers to a named function.
                getptr_reg = self.reg(Ptr(self.typ(ir_typ)))
                fn_typ = self.typ(ir_typ)
                assert isinstance(fn_typ, Fun), f"Expected Fn, got {fn_typ}"
                self.emit(GetFnPtr(getptr_reg, fn_typ), None)
                self.node_regs[node.id] = getptr_reg
            case ast.Member():
                ast.walk(node, self.generate)
                src = self.node_regs[node.target.id]
                types_src = self.type_env.get(node.target)
                assert isinstance(src.typ, Struct), f"Expected Struct, got {src.typ}"
                assert isinstance(types_src, types.ProductShape), f"Expected Shape, got {types_src}"
                attr = types_src.attr(node.name)
                if attr is not None:
                    attr_index = types_src.attrs_sorted.index(attr)
                    assert attr_index is not None, f"No member {node.name} in type {types_src}"
                    getptr_reg = self.reg(Ptr(src.typ.fields[attr_index]))
                    if isinstance(parent, ast.Assign):
                        self.emit(GetPtr(getptr_reg, src, attr_index), node)
                    else:
                        self.emit(GetPtr(getptr_reg, src, attr_index), None)
                        reg = self.reg(src.typ.fields[attr_index])
                        self.emit(Load(reg, getptr_reg), node)
                # If it's not an attribute, it has to be a behaviour function.
                # todo: emit a GetFunPtr if `parent` isn't ast.Call.
            case ast.Call():
                callee = self.type_env.get(node.callee)
                assert isinstance(callee, types.FunShape), f"Expected Fun, got {callee}"
                fun = callee
                ast.walk(node, self.generate)
                args = [self.node_regs[x.id] for x in node.args]
                if fun.namespace:
                    # This is a behaviour function call, prepend the receiver to the args.
                    assert isinstance(node.callee, ast.Member), f"Expected Member, got {node.callee}"
                    receiver = self.node_regs[node.callee.target.id]
                    args = [receiver, *args]
                if fun.is_named:
                    # Direct call by name.
                    reg = self.reg(self.typ(fun.result))
                    self.emit(Call(reg, self.fun_name(fun), args), node)
                else:
                    # Indirect call by register (either a `Fn` or a `Ptr<Fn>`).
                    src = self.node_regs[node.callee.id]
                    # Determine the result type of the call.
                    if isinstance(src.typ, Ptr):
                        fn = src.typ.typ
                        assert isinstance(fn, Fun), f"Expected Fn, got {fn}"
                    else:
                        assert isinstance(src.typ, Fun), f"Expected Fn, got {src.typ}"
                        fn = src.typ
                    reg = self.reg(fn.result)
                    self.emit(Call(reg, src, args), node)
            case ast.Assign():
                ast.walk(node, self.generate)
                reg = self.node_regs[node.value.id]
                src = node.target
                if not self.scope.find(src.name):
                    self.scope.declare(src.name, reg)
                self.node_regs[node.id] = NoneReg
            case ast.BinaryExpr():
                ast.walk(node, self.generate)
                lhs_reg = self.node_regs[node.lhs.id]
                rhs_reg = self.node_regs[node.rhs.id]
                match node.op:
                    case ast.BinaryOp.add:
                        reg = self.reg(I64)
                        self.emit(IAddO(reg, lhs_reg, rhs_reg), node)
                    case ast.BinaryOp.sub:
                        reg = self.reg(I64)
                        self.emit(ISubO(reg, lhs_reg, rhs_reg), node)
                    case ast.BinaryOp.eq | ast.BinaryOp.ne:
                        match lhs_reg.typ:
                            case Int():
                                op = ICmpOp.eq if node.op == ast.BinaryOp.eq else ICmpOp.ne
                                reg = self.reg(I1)
                                self.emit(ICmp(reg, op, lhs_reg, rhs_reg), node)
                            case _:
                                raise AssertionError(f"Unsupported type for equality comparison: {lhs_reg.typ}")
                    case _:
                        raise AssertionError(f"Unsupported binary op: {node.op}")
            case ast.ShapeRef() | ast.FunParam() | ast.UnitShape() | ast.Behaviour():
                pass
            case _:
                raise AssertionError(f"Unsupported node: {node.__class__}")
        return node


def generate_ir(specs: list[types.FunSpec]) -> IR:
    ir = IR(fn_irs=[], constant_pool={}, structs={})
    for spec in specs:
        gen = FunGen(spec, ir)
        gen.generate(spec.fun_def, None)
        ir.fn_irs.append(gen.fun_ir)

    return ir
