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

    def __init__(self, fun_ir: ir.FunIR) -> None:
        self.fun_ir = fun_ir
        self.getptrs = {}

    def inst(self, inst: ir.Inst, code: Code) -> None:
        match inst:
            case ir.Alloc():
                assert isinstance(inst.reg.typ, ir.Struct)
                struct_name = inst.reg.typ.fqn
                code.write(f"{inst.reg.id} := &{struct_name}{{")
                for i, reg in enumerate(inst.args):
                    if i > 0:
                        code.write(", ")
                    code.write(f"_{i}: {reg}")
                code.write("}")
                code.newline()
            case ir.Call():
                callee = inst.callee
                if isinstance(callee, str) and callee in map_builtins:
                    callee = map_builtins[callee]
                if inst.reg != ir.NoneReg:
                    code.write(f"{inst.reg} := ")
                code.write(f"{callee}(")
                code.write(", ".join(f"{arg}" for arg in inst.args) + ")")
                code.newline()
            case ir.GetPtr():
                self.getptrs[inst.reg] = inst
            case ir.IntConst():
                code.write(f"{inst.reg} := {inst.value}")
                code.newline()
            case ir.Load():
                # inst.src has to be a GetPtr we have already seen.
                getptr = self.getptrs[inst.src]
                code.write(f"{inst.reg} := {getptr.src.id}._{getptr.field}")
                code.newline()
            case ir.Store():
                # inst.target has to be a GetPtr we have already seen.
                getptr = self.getptrs[inst.target]
                code.write(f"{getptr.src.id}._{getptr.field} = {inst.src}")
                code.newline()
            case _:
                raise NotImplementedError(f"TODO: {type(inst).__name__} {inst}")

    def generate(self) -> str:
        code = Code(0, [])
        code.write(f"func {self.fun_ir.fn_name}(")
        code.write(", ".join(f"{param.reg} {typ(param.typ)}" for param in self.fun_ir.params) + ") ")
        if not isinstance(self.fun_ir.result, ir.NoneTyp):
            code.write(f"{typ(self.fun_ir.result)} ")
        code.write("{")
        code.newline()
        code.indent()
        for block in self.fun_ir.blocks:
            for inst in block.insts:
                self.inst(inst, code)
            if isinstance(block.terminator, ir.Return) and block.terminator.reg != ir.NoneReg:
                code.write(f"return {block.terminator.reg}")
                code.newline()
        code.dedent()
        code.write("}")
        code.newline()
        return str(code)


def gogen(ir_: ir.IR) -> str:
    code = Code(0, [])
    code.write("package main")
    code.newline()
    code.newline()
    for const in ir_.constant_pool.values():
        code.write(f"var {const.reg} = {json.dumps(const.value)}")
        code.newline()
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
        code.write("}")
        code.newline()
    if ir_.structs:
        code.newline()
    for fun_ir in ir_.fn_irs:
        code.write(FuncGen(fun_ir).generate())
    return str(code)
