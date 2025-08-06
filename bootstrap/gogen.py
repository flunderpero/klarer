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


def typ(typ: ir.Typ) -> str:
    match typ:
        case ir.Int():
            return "int"
        case ir.Str():
            return "string"
        case ir.Struct():
            return "struct"
        case ir.NoneTyp():
            return "_"
        case _:
            raise NotImplementedError(f"Unsupported type: {typ}")


def gen_inst(inst: ir.Inst, code: Code) -> None:
    match inst:
        case ir.Call():
            callee = inst.callee
            if isinstance(callee, str) and callee in map_builtins:
                callee = map_builtins[callee]
            if inst.reg != ir.NoneReg:
                code.write(f"{inst.reg} := ")
            code.write(f"{callee}(")
            code.write(", ".join(f"{arg}" for arg in inst.args) + ")")
            code.newline()
        case _:
            raise NotImplementedError(f"TODO: {inst}")


def gen_fun(fun_ir: ir.FunIR) -> str:
    code = Code(0, [])
    code.write(f"func {fun_ir.fn_name}(")
    code.write(", ".join(f"{param.reg} {typ(param.typ)}" for param in fun_ir.params) + ") ")
    if not isinstance(fun_ir.result, ir.NoneTyp):
        code.write(f"{typ(fun_ir.result)} ")
    code.write("{")
    code.newline()
    code.indent()
    for block in fun_ir.blocks:
        for inst in block.insts:
            gen_inst(inst, code)
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
    for fun_ir in ir_.fn_irs:
        code.write(gen_fun(fun_ir))
    return str(code)
