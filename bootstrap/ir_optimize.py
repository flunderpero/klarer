from . import ir
from .span import log


def remove_unused(input: ir.IR) -> ir.IR:
    """Remove unused instructions, structs, and Str constants."""
    for fun in input.fn_irs:
        remove_unused_insts(fun)
    remove_unused_structs(input)
    remove_unused_str_constants(input)
    return input


def remove_unused_insts(input: ir.FunIR) -> None:
    def remove_pass() -> int:
        used: set[ir.RegId] = set()
        removed = 0

        def keep(inst: ir.Inst) -> bool:
            if isinstance(inst, ir.Store):
                return True
            if inst.reg == ir.NoneReg:
                return True
            return inst.reg.id in used

        # First, find all the registers that are used.
        for block in input.blocks:
            for inst in block.insts:
                for reg in inst.args():
                    used.add(reg.id)
            assert block.terminator is not None
            for reg in block.terminator.args():
                used.add(reg.id)

        # Then, remove all the unused instructions.
        for block in input.blocks:
            len_before = len(block.insts)
            block.insts = [x for x in block.insts if keep(x)]
            removed += len_before - len(block.insts)

        return removed

    while True:
        removed = remove_pass()
        log("ir-optimize-remove-unused", f"Removed {removed} unused instructions")
        if removed == 0:
            break


def remove_unused_structs(input: ir.IR) -> None:
    """Remove unused structs."""
    used_structs: set[str] = set()
    for fun in input.fn_irs:
        for block in fun.blocks:
            for inst in block.insts:
                for reg in [inst.reg, *inst.args()]:
                    if isinstance(reg.typ, ir.Struct):
                        used_structs.add(reg.typ.fqn)

    input.structs = {k: v for k, v in input.structs.items() if k in used_structs}


def remove_unused_str_constants(input: ir.IR) -> None:
    """Remove unused string constants."""
    used_constants: set[str] = set()
    for fun in input.fn_irs:
        for block in fun.blocks:
            for inst in block.insts:
                for reg in [inst.reg, *inst.args()]:
                    if ir.StrConst.is_str_const(reg):
                        used_constants.add(reg.id)
    input.constant_pool = {k: v for k, v in input.constant_pool.items() if v.reg.id in used_constants}
