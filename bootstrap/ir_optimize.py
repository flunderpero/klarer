from . import ir
from .span import log


def remove_unused(input: ir.IR) -> ir.IR:
    for fun in input.fn_irs:
        remove_unused_fun(fun)
    return input


def remove_unused_fun(input: ir.FunIR) -> None:
    """Remove unused instructions from the IR."""

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
