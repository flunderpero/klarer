from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from subprocess import run
from time import time
from typing import TYPE_CHECKING

from . import (
    ast,
    error,
    gogen,
    ir,
    parser,
    token,
    types,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass
class TokenStep:
    tokens: list[token.Token]
    errors: list[error.Error]
    duration: float

    def __str__(self) -> str:
        return "\n".join(str(x) for x in self.tokens)


@dataclass
class ParseStep:
    module: ast.Module
    errors: list[error.Error]
    duration: float

    def __str__(self) -> str:
        return str(self.module)


@dataclass
class TypecheckStep:
    module: ast.Module
    result: types.TypeCheckResult
    duration: float

    @property
    def errors(self) -> list[error.Error]:
        return self.result.errors

    def __str__(self) -> str:
        return self.debug()

    def __repr__(self) -> str:
        return self.debug()

    def debug(self) -> str:
        lines = []

        def visit(node: ast.Node, _parent: ast.Node | None) -> ast.Node:
            typ = self.result.type_env.node_shapes.get(node.id)
            typ_str = str(typ) if typ else "NOT_FOUND"
            node_str = ast.to_str_withoud_nid(node)
            code_str = node.span.lines(0)[1][0].strip()
            typ_str = typ_str.replace("\n", "\n    ")
            node_str = node_str.replace("\n", "\n    ")
            lines.append(str(node.span) + "    " + code_str)
            lines.append(f"    {node_str}")
            lines.append(f" => {typ_str}\n")
            ast.walk(node, visit)
            return node

        ast.walk(self.module, visit)
        return "\n".join(lines)


@dataclass
class AbortStep:
    errors: list[error.Error]
    duration: float

    def __str__(self) -> str:
        return "\n".join(str(x) for x in self.errors)


@dataclass
class IRStep:
    ir: ir.IR
    duration: float

    def __str__(self) -> str:
        return str(self.ir)


@dataclass
class CodeGenStep:
    code: str
    duration: float

    def __str__(self) -> str:
        return self.code


@dataclass
class CompileStep:
    returncode: int
    stdout: str
    stderr: str
    duration: float

    def __str__(self) -> str:
        return f"statuscode: {self.returncode}\nstdout: {self.stdout}\nstderr: {self.stderr}"


@dataclass
class RunStep:
    returncode: int
    stdout: str
    stderr: str
    duration: float

    def __str__(self) -> str:
        return f"statuscode: {self.returncode}\nstdout: {self.stdout}\nstderr: {self.stderr}"


CompilationStep = TokenStep | ParseStep | TypecheckStep | AbortStep | IRStep | CodeGenStep | CompileStep | RunStep


def compile(input: token.Input, outfile: str) -> Generator[CompilationStep]:  # noqa: A001
    cur_id = 0

    def next_id() -> int:
        nonlocal cur_id
        cur_id += 1
        return cur_id

    start = time()
    tokens, tokenize_errors = token.tokenize(input)
    yield TokenStep(tokens, tokenize_errors, time() - start)
    start = time()
    module, parse_errors = parser.parse(parser.Input(tokens, next_id))
    yield ParseStep(module, parse_errors, time() - start)
    start = time()
    tc_result = types.typecheck(module)
    yield TypecheckStep(module, tc_result, time() - start)
    start = time()
    if tokenize_errors or parse_errors or tc_result.errors:
        yield AbortStep(tokenize_errors + parse_errors + tc_result.errors, time() - start)
        return
    start = time()
    ir_ = ir.generate_ir(tc_result.fun_specs)
    yield IRStep(ir_, time() - start)
    start = time()
    code = gogen.gogen(ir_)
    yield CodeGenStep(code, time() - start)
    start = time()
    gofile = outfile + ".go"
    runtimefile = outfile + "_runtime.go"
    try:
        with open(gofile, "w") as f:
            f.write(code)
        with open(runtimefile, "w") as f:
            f.write(open(os.path.dirname(os.path.realpath(__file__)) + "/runtime.go").read())
        p = run(
            ["go", "build", "-o", outfile, gofile, runtimefile],
            input=code,
            text=True,
            check=False,
            capture_output=True,
        )
        yield CompileStep(p.returncode, p.stdout, p.stderr, time() - start)
        if p.returncode != 0:
            return
        start = time()
        p = run([outfile], check=False, capture_output=True, text=True)
        yield RunStep(p.returncode, p.stdout, p.stderr, time() - start)
    finally:
        for f in (gofile, runtimefile):
            with contextlib.suppress(BaseException):
                os.remove(f)
