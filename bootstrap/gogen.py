from __future__ import annotations

import json
from dataclasses import dataclass

from . import ir

map_builtins = {
    "print": "Print",
    "int_to_str": "IntToStr",
    "bool_to_str": "BoolToStr",
    "char_to_str": "CharToStr",
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


def typ(typ: ir.Typ) -> str:
    match typ:
        case ir.Int():
            return "int"
        case ir.Str():
            return "string"
        case ir.Struct():
            return typ.fqn
        case ir.NoneTyp():
            return "_"
        case _:
            raise NotImplementedError(f"Unsupported type: {typ}")


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
        is_phi_reg = inst.reg.id != inst_reg.id
        assign = ":="
        if is_phi_reg:
            assign = "="
        match inst:
            case ir.Alloc():
                assert isinstance(inst_reg.typ, ir.Struct)
                struct_name = inst_reg.typ.fqn
                code.write(f"{inst_reg} {assign} &{struct_name}{{")
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
                    code.write(f"{inst_reg} {assign} ")
                code.write(f"{callee}(")
                code.writeln(", ".join(f"{self.reg(arg)}" for arg in inst.args) + ")")
            case ir.GetPtr():
                self.getptrs[inst_reg] = inst
                src_reg = self.reg(inst.src)
                if isinstance(inst.src.typ, ir.Struct):
                    code.write(f"{inst_reg} {assign} {src_reg}._{inst.field}")
                else:
                    code.write(f"{inst_reg} {assign} {src_reg}")
                if not is_phi_reg:
                    # todo: This is a hack because we shouldn't emit the code above
                    #       if the result is used in a `Store` only.
                    #       `Store` accesses `self.getptrs` because we cannot have
                    #       pointers to struct fields in Go.
                    code.write(f"; _ = {inst_reg}")
                code.newline()
            case ir.IntConst():
                code.writeln(f"{inst_reg} {assign} {inst.value}")
            case ir.Load():
                src_reg = self.reg(inst.src)
                code.writeln(f"{inst_reg} {assign} {src_reg}")
            case ir.Store():
                # inst.target has to be a GetPtr we have already seen.
                target_reg = self.reg(inst.target)
                getptr = self.getptrs[target_reg]
                getptr_src_reg = self.reg(getptr.src)
                code.writeln(f"{getptr_src_reg}._{getptr.field} = {inst.src}")
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
                code.writeln(f"if {block.terminator.reg} == 1 {{")
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

    def handle_phi_nodes(self, code: Code) -> None:
        for block in self.fun_ir.blocks:
            for inst in block.insts:
                if isinstance(inst, ir.Phi):
                    code.writeln(f"var {inst.reg} {typ(inst.reg.typ)}")
                    for phi_in in inst.incoming:
                        self.reg_map[phi_in.reg] = inst.reg

    def generate(self) -> str:
        code = Code(0, [])
        code.write(f"func {self.fun_ir.fn_name}(")
        for param in self.fun_ir.params:
            ref = "*" if isinstance(param.typ, ir.Struct) else ""
            code.write(f"{param.reg} {ref}{typ(param.typ)}")
            code.write(", ")
        code.write(") ")
        if not isinstance(self.fun_ir.result, ir.NoneTyp):
            code.write(f"{typ(self.fun_ir.result)} ")
        code.writeln("{")
        code.indent()
        self.handle_phi_nodes(code)
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
