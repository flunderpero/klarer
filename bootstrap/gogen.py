from __future__ import annotations

import json
from dataclasses import dataclass

from . import ir

map_builtins = {
    "print": "Print",
}


@dataclass
class Code:
    indent_: int
    lines: list[str]

    def __str__(self) -> str:
        return "\n".join(x.rstrip() for x in self.lines)

    def indent(self) -> None:
        self.indent_ += 1

    def dedent(self) -> None:
        self.indent_ -= 1

    def newline(self) -> None:
        self.lines.append("")

    def write(self, s: str) -> None:
        if not self.lines:
            self.lines.append("")
        if self.lines[-1] == "":
            self.lines[-1] = self.indent_ * 4 * " "
        self.lines[-1] += s

    def writeln(self, s: str) -> None:
        self.write(s)
        self.newline()


def typ(ir_typ: ir.Typ) -> str:
    match ir_typ:
        case ir.Int():
            return "int"
        case ir.Char():
            return "rune"
        case ir.Bool():
            return "bool"
        case ir.Ptr():
            # todo: unify the handling of struct pointers
            ref = "*" if isinstance(ir_typ.typ, ir.Struct) else ""
            return f"{ref}{typ(ir_typ.typ)}"
        case ir.Str():
            return "string"
        case ir.Struct():
            return ir_typ.fqn
        case ir.NoneTyp():
            return "_"
        case ir.Fun():
            code = Code(0, [])
            emit_fun_signature("", [ir.Reg(f"p{i}", x) for i, x in enumerate(ir_typ.params)], ir_typ.result, code)
            return str(code)
        case ir.List():
            # todo: unify the handling of struct pointers
            ref = "*" if isinstance(ir_typ.typ, ir.Struct) else ""
            return f"[]{ref}{typ(ir_typ.typ)}"
        case _:
            raise NotImplementedError(f"Unsupported type: {ir_typ}")


def emit_fun_signature(name: str, params: list[ir.Reg], result: ir.Typ, code: Code) -> None:
    code.write(f"func {name}(")
    for param in params:
        ref = "*" if isinstance(param.typ, ir.Struct) else ""
        code.write(f"{param} {ref}{typ(param.typ)}")
        code.write(", ")
    code.write(") ")
    if not isinstance(result, ir.NoneTyp):
        ref = "*" if isinstance(result, ir.Struct) else ""
        code.write(f"{ref}{typ(result)} ")


class FuncGen:
    fun_ir: ir.FunIR
    getptrs: dict[ir.Reg, ir.GetPtr]
    # This represents PHI constraints. When we encounter a PHI node, we have
    # to make sure to use the same Go variable name for all incoming registers.
    reg_map: dict[ir.Reg, ir.Reg]

    def __init__(self, fun_ir: ir.FunIR) -> None:
        self.fun_ir = fun_ir
        self.getptrs = {}
        self.reg_map = {}

    def reg(self, reg: ir.Reg) -> ir.Reg:
        if reg in self.reg_map:
            return self.reg_map[reg]
        return reg

    def inst(self, inst: ir.Inst, code: Code) -> None:
        inst_reg = self.reg(inst.reg)
        match inst:
            case ir.Alloc():
                assert isinstance(inst_reg.typ, ir.Struct)
                struct_name = inst_reg.typ.fqn
                code.write(f"{inst_reg} = &{struct_name}{{")
                for i, arg_reg in enumerate(inst.args):
                    if i > 0:
                        code.write(", ")
                    code.write(f"_{i}: {self.reg(arg_reg)}")
                code.write("}")
                code.newline()
            case ir.Call():
                callee = inst.callee
                if isinstance(callee, str) and callee in map_builtins:
                    callee = map_builtins[callee]
                if inst_reg != ir.NoneReg:
                    code.write(f"{inst_reg} = ")
                code.write(f"{callee}(")
                code.writeln(", ".join(f"{self.reg(arg)}" for arg in inst.args) + ")")
            case ir.GetPtr():
                self.getptrs[inst_reg] = inst
                src_reg = self.reg(inst.src)
                if isinstance(inst.src.typ, ir.Struct):
                    code.write(f"{inst_reg} = {src_reg}._{inst.field}")
                else:
                    code.write(f"{inst_reg} = {src_reg}")
                code.newline()
            case ir.GetFunPtr():
                code.writeln(f"{inst_reg} = {inst.src.fqn}")
            case ir.IntConst():
                match inst.reg.typ:
                    case ir.Int():
                        code.writeln(f"{inst_reg} = {inst.value}")
                    case ir.Char():
                        code.writeln(f"{inst_reg} = {inst.value}")
                    case ir.Bool():
                        value = "true" if inst.value else "false"
                        code.writeln(f"{inst_reg} = {value}")
                    case _:
                        raise AssertionError(f"Unexpected type: {inst.reg.typ}")
            case ir.ListConst():
                assert isinstance(inst.reg.typ, ir.List)
                # todo: unify the handling of struct pointers
                ref = "*" if isinstance(inst.reg.typ.typ, ir.Struct) else ""
                code.write(f"{inst_reg} = []{ref}{typ(inst.reg.typ.typ)}{{")
                for i, value_reg in enumerate(inst.values):
                    if i > 0:
                        code.write(", ")
                    code.write(f"{self.reg(value_reg)}")
                code.writeln("}")
            case ir.ListConcat():
                lhs_reg = self.reg(inst.lhs)
                rhs_reg = self.reg(inst.rhs)
                code.writeln(f"{inst_reg} = append({lhs_reg}, {rhs_reg}...)")
            case ir.GetListPtr():
                src_reg = self.reg(inst.src)
                index_reg = self.reg(inst.index)
                code.writeln(f"{inst_reg} = {src_reg}[{index_reg}]")
            case ir.Load():
                src_reg = self.reg(inst.src)
                code.writeln(f"{inst_reg} = {src_reg}")
            case ir.Store():
                # inst.target has to be a GetPtr we have already seen.
                target_reg = self.reg(inst.target)
                getptr = self.getptrs[target_reg]
                getptr_src_reg = self.reg(getptr.src)
                code.writeln(f"{getptr_src_reg}._{getptr.field} = {inst.src}")
            case ir.IAddO():
                lhs_reg = self.reg(inst.lhs)
                rhs_reg = self.reg(inst.rhs)
                code.writeln(f"{inst_reg} = {lhs_reg} + {rhs_reg}")
            case ir.ISubO():
                lhs_reg = self.reg(inst.lhs)
                rhs_reg = self.reg(inst.rhs)
                code.writeln(f"{inst_reg} = {lhs_reg} - {rhs_reg}")
            case ir.IMulO():
                lhs_reg = self.reg(inst.lhs)
                rhs_reg = self.reg(inst.rhs)
                code.writeln(f"{inst_reg} = {lhs_reg} * {rhs_reg}")
            case ir.IDivO():
                lhs_reg = self.reg(inst.lhs)
                rhs_reg = self.reg(inst.rhs)
                code.writeln(f"{inst_reg} = {lhs_reg} / {rhs_reg}")
            case ir.ICmp():
                lhs_reg = self.reg(inst.lhs)
                rhs_reg = self.reg(inst.rhs)
                op = None
                match inst.op:
                    case ir.ICmpOp.eq:
                        op = "=="
                    case ir.ICmpOp.ne:
                        op = "!="
                    case _:
                        raise NotImplementedError(f"TODO: ICmp {inst.op.value} {lhs_reg}, {rhs_reg}")
                code.writeln(f"{inst_reg} = {lhs_reg} {op} {rhs_reg}")
            case ir.Phi():
                pass
            case _:
                raise NotImplementedError(f"TODO: {type(inst).__name__} {inst}")

    def block(self, block: ir.Block, code: Code) -> None:
        if len(self.fun_ir.blocks) > 1:
            code.writeln(f"case {block.id}:")
            code.indent()
        for inst in block.insts:
            self.inst(inst, code)
        match block.terminator:
            case ir.Return():
                if block.terminator.reg == ir.NoneReg:
                    code.writeln("return")
                else:
                    code.writeln(f"return {block.terminator.reg}")
            case ir.Branch():
                code.writeln(f"if {block.terminator.reg} {{")
                # todo: optimize if we detect a simple if-else chain and are
                #       sure that this is not a loop. In that case, we can
                #       just simply create an `if` statement and generate the blocks.
                code.indent()
                code.writeln(f"block = {block.terminator.then_block.id}")
                code.dedent()
                code.writeln("} else {")
                code.indent()
                code.writeln(f"block = {block.terminator.else_block.id}")
                code.dedent()
                code.writeln("}")
            case ir.Jump():
                code.writeln(f"block = {block.terminator.target.id}")
            case _:
                raise NotImplementedError(f"TODO: {type(block.terminator).__name__} {block.terminator}")
        if len(self.fun_ir.blocks) > 1:
            code.dedent()

    def connect_phi_registers(self) -> None:
        for block in self.fun_ir.blocks:
            for inst in block.insts:
                if isinstance(inst, ir.Phi):
                    for phi_in in inst.incoming:
                        self.reg_map[phi_in.reg] = inst.reg

    def declare_regs(self, code: Code) -> None:
        for block in self.fun_ir.blocks:
            for inst in block.insts:
                if isinstance(inst.reg.typ, ir.NoneTyp):
                    continue
                if inst.reg in self.reg_map:
                    continue
                ref = "*" if isinstance(inst.reg.typ, ir.Struct) else ""
                code.writeln(f"var {inst.reg} {ref}{typ(inst.reg.typ)}")

    def generate(self) -> str:
        code = Code(0, [])
        emit_fun_signature(self.fun_ir.fn_name, self.fun_ir.params, self.fun_ir.result, code)
        code.writeln("{")
        code.indent()
        self.connect_phi_registers()
        self.declare_regs(code)
        if len(self.fun_ir.blocks) > 1:
            code.writeln(f"block := {self.fun_ir.blocks[0].id}")
            code.writeln("for {")
            code.indent()
            code.writeln("switch block {")
        for block in self.fun_ir.blocks:
            self.block(block, code)
        if len(self.fun_ir.blocks) > 1:
            code.dedent()
            code.writeln("}}")
        code.dedent()
        code.writeln("}")
        return str(code)


def gogen(ir_: ir.IR) -> str:
    code = Code(0, [])
    code.writeln("package main")
    code.newline()
    for const in ir_.constant_pool.values():
        code.writeln(f"var {const.reg} = {json.dumps(const.value)}")
    if ir_.constant_pool:
        code.newline()
    for struct in ir_.structs.values():
        code.write(f"type {struct.fqn} struct{{")
        for i, field in enumerate(struct.fields):
            if i > 0:
                code.write("; ")
            code.write(f"_{i} ")
            if isinstance(field, ir.Struct):
                code.write(f"*{typ(field)}")
            else:
                code.write(typ(field))
        code.writeln("}")
    if ir_.structs:
        code.newline()
    for fun_ir in ir_.fn_irs:
        code.write(FuncGen(fun_ir).generate())
    return str(code)
