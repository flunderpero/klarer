from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from . import ast, error, token
from .span import FQN, Span


@dataclass
class Input:
    tokens: list[token.Token]
    next_id: Callable[[], int]
    index: int = 0

    def next(self) -> token.Token:
        """Return the next token while skipping comments."""
        t = self.peek()
        if t.kind != token.Kind.eof:
            self.index += 1
        return t

    def peek(self) -> token.Token:
        """Return the next token without consuming it. Skips comments."""
        while self.tokens[self.index].kind == token.Kind.comment:
            self.index += 1
        return self.tokens[self.index]

    def span(self) -> Span:
        return self.peek().span

    def span_merge(self, span: Span) -> Span:
        if self.index > 0:
            return span.merge(self.tokens[self.index - 1].span)
        return span.merge(self.span())


class Parser:
    input: Input
    errors: list[error.Error]

    def __init__(self, input: Input) -> None:
        self.input = input
        self.errors = []

    def error(self, err: error.Error) -> None:
        """Record the error and find the next best place to resume parsing."""
        self.errors.append(err)
        while (t := self.input.next()).kind != token.Kind.eof:
            match t.kind:
                case token.Kind.paren_right | token.Kind.curly_right:
                    break
                case token.Kind.fun:
                    break

    def expect_ident(self) -> str | None:
        t = self.expect(token.Kind.ident)
        if t is None:
            return t
        return t.value_str()

    def expect_type_ident(self) -> str | None:
        t = self.expect(token.Kind.type_ident)
        if t is None:
            return t
        return t.value_str()

    def expect(self, *kind: token.Kind) -> token.Token | None:
        t = self.input.next()
        if t.kind in kind:
            return t
        self.error(error.unexpected_token(t.span, t.kind.value, *(x.value for x in kind)))
        return None

    def id(self) -> ast.NodeId:
        return self.input.next_id()

    def parse_name(self, expected_kind: token.Kind) -> ast.Name | None:
        span = self.input.span()
        ident = self.expect(expected_kind)
        if not ident:
            return None
        kind: Literal["ident", "type", "behaviour"]
        match expected_kind:
            case token.Kind.ident:
                kind = "ident"
            case token.Kind.type_ident:
                kind = "type"
            case token.Kind.behaviour_ns:
                kind = "behaviour"
            case _:
                raise AssertionError(f"Unexpected token kind: {expected_kind}")
        return ast.Name(self.id(), ident.value_str(), kind, self.input.span_merge(span))

    def parse_shape_decl(self) -> ast.ShapeDecl | None:
        span = self.input.span()
        name = self.expect_type_ident()
        if not name:
            return None
        if not self.expect(token.Kind.eq):
            return None
        shape = self.parse_shape()
        if not shape:
            return None
        return ast.ShapeDecl(self.id(), name, shape, self.input.span_merge(span))

    def parse_shape(self) -> ast.Shape | None:
        span = self.input.span()
        result: ast.Shape | None = None
        while True:
            t = self.input.peek()
            shape: ast.Shape | None
            match t.kind:
                case token.Kind.curly_left:
                    shape = self.parse_product_shape()
                case token.Kind.fun:
                    shape = self.parse_fun_shape()
                case token.Kind.type_ident:
                    match t.value_str():
                        case "Int" | "Bool" | "Str" | "Char":
                            self.input.next()
                            shape = ast.NominalType(self.id(), t.value_str(), self.input.span_merge(span))
                        case _:
                            return None
                case _:
                    self.error(error.unexpected_token(t.span, t.kind.value))
                    return None
            if not shape:
                return None
            if result is None:
                result = shape
            else:
                result.variants.append(shape)
            if self.input.peek().kind == token.Kind.pipe:
                self.input.next()
                if not isinstance(result, ast.SumShape):
                    result = ast.SumShape(self.id(), [result], self.input.span_merge(span))
            else:
                break

        return result

    def parse_fun_shape(self) -> ast.FunShape | None:
        span = self.input.span()
        if not self.expect(token.Kind.fun):
            return None
        params: list[ast.Attr] = []
        if not self.expect(token.Kind.paren_left):
            return None
        while self.input.peek().kind != token.Kind.paren_right:
            param_span = self.input.span()
            param_name = self.expect_ident()
            if not param_name:
                return None
            param_shape = self.parse_shape()
            if not param_shape:
                return None
            params.append(ast.Attr(self.id(), param_name, param_shape, self.input.span_merge(param_span)))
            match self.input.peek().kind:
                case token.Kind.comma:
                    self.input.next()
                case token.Kind.paren_right:
                    break
                case _:
                    self.error(error.unexpected_token(self.input.span(), self.input.peek().kind.value))
                    return None
        if not self.expect(token.Kind.paren_right):
            return None
        result = self.parse_shape()
        if not result:
            return None
        return ast.FunShape(self.id(), params, result, self.input.span_merge(span))

    def parse_product_shape(self) -> ast.ProductShape | None:
        span = self.input.span()
        if not self.expect(token.Kind.curly_left):
            return None
        attrs: list[ast.Attr] = []
        while self.input.peek().kind != token.Kind.curly_right:
            param_span = self.input.span()
            param_name = self.expect_ident()
            if not param_name:
                return None
            param_type = self.parse_shape()
            if not param_type:
                return None
            attrs.append(ast.Attr(self.id(), param_name, param_type, self.input.span_merge(param_span)))
            if self.input.peek().kind == token.Kind.comma:
                self.input.next()
        if not self.expect(token.Kind.curly_right):
            return None
        return ast.ProductShape(self.id(), attrs, self.input.span_merge(span))

    def parse_fun_def(self, name: str) -> ast.FunDef | None:
        span = self.input.span()
        if not self.expect(token.Kind.fun):
            return None
        params: list[str] = []
        if not self.expect(token.Kind.paren_left):
            return None
        while self.input.peek().kind != token.Kind.paren_right:
            param_name = self.expect_ident()
            if not param_name:
                return None
            params.append(param_name)
            match self.input.peek().kind:
                case token.Kind.comma:
                    self.input.next()
                case token.Kind.paren_right:
                    break
                case _:
                    self.error(error.unexpected_token(self.input.span(), self.input.peek().kind.value))
                    return None
        if not self.expect(token.Kind.paren_right):
            return None
        body = self.parse_block()
        if not body:
            return None
        return ast.FunDef(self.id(), name, None, params, body, self.input.span_merge(span))

    def parse_if(self) -> ast.If | None:
        span = self.input.span()
        if not self.expect(token.Kind.if_):
            return None
        cond = self.parse_expr()
        if not cond:
            return None
        then_block = self.parse_block()
        if not then_block:
            return None
        else_block: ast.Block | None = None
        if self.input.peek().kind == token.Kind.else_:
            self.input.next()
            else_block = self.parse_block()
            if not else_block:
                return None
        return ast.If(self.id(), cond, then_block, else_block, self.input.span_merge(span))

    def parse_loop(self) -> ast.Loop | None:
        span = self.input.span()
        if not self.expect(token.Kind.loop):
            return None
        block = self.parse_block()
        if not block:
            return None
        return ast.Loop(self.id(), block, self.input.span_merge(span))

    def parse_expr(self, min_precedence: int = 0) -> ast.Expr | None:
        lhs = self.parse_primary_expr()
        if not lhs:
            return None
        while True:
            t = self.input.peek()
            op_by_token = {
                token.Kind.plus: ast.BinaryOp.add,
                token.Kind.minus: ast.BinaryOp.sub,
                token.Kind.eqeq: ast.BinaryOp.eq,
                token.Kind.neq: ast.BinaryOp.ne,
            }
            precendence_by_op = {
                ast.BinaryOp.eq: 1,
                ast.BinaryOp.ne: 1,
                ast.BinaryOp.add: 2,
                ast.BinaryOp.sub: 2,
            }
            op = op_by_token.get(t.kind)
            if not op:
                return lhs
            precedence = precendence_by_op[op]
            if precedence < min_precedence:
                return lhs
            self.input.next()
            # `+ 1` because all of our binary operators are left associative, i.e. `1 + 3 - 4`
            # becomes `(1 + 3) - 4`.
            rhs = self.parse_expr(precedence + 1)
            if not rhs:
                return None
            lhs = ast.BinaryExpr(self.id(), op, lhs, rhs, self.input.span_merge(lhs.span))

    def parse_primary_expr(self) -> ast.Expr | None:
        t = self.input.peek()
        expr: ast.Expr | None
        match t.kind:
            case token.Kind.do:
                return self.parse_block()
            case token.Kind.ident | token.Kind.type_ident | token.Kind.behaviour_ns:
                expr = self.parse_name(t.kind)
            case token.Kind.str_lit:
                self.input.next()
                return ast.StrLit(self.id(), t.value_str(), t.span)
            case token.Kind.char_lit:
                self.input.next()
                return ast.CharLit(self.id(), t.value_str(), t.span)
            case token.Kind.int_lit:
                self.input.next()
                return ast.IntLit(
                    self.id(),
                    bits=64,
                    signed=True,
                    value=int(t.value_str()),
                    span=t.span,
                )
            case token.Kind.true | token.Kind.false:
                self.input.next()
                return ast.BoolLit(self.id(), value=t.kind == token.Kind.true, span=t.span)
            case token.Kind.if_:
                return self.parse_if()
            case _:
                self.error(error.unexpected_token(t.span, t.kind.value))
                return None
        if not expr:
            return None

        # while expr is not None:
        #     match self.input.peek().kind:
        #         case token.Kind.paren_left:
        #             expr = self.parse_call(expr)
        #         case token.Kind.dot:
        #             expr = self.parse_member(expr)
        #         case _:
        #             break
        return expr

    def parse_block(self) -> ast.Block | None:
        span = self.input.span()
        self.expect(token.Kind.do)
        nodes: list[ast.Node] = []
        while (t := self.input.peek()).kind != token.Kind.eof:
            if t.kind == token.Kind.end:
                self.input.next()
                break
            node = self.parse_block_node()
            if node:
                nodes.append(node)
        return ast.Block(self.id(), nodes, self.input.span_merge(span))

    def parse_block_node(self) -> ast.Node | None:
        t = self.input.peek()
        match t.kind:
            case token.Kind.loop:
                return self.parse_loop()
            case token.Kind.type_ident:
                return self.parse_shape_decl()
            case token.Kind.mut:
                self.input.next()
                assign = self.parse_expr()
                if not assign:
                    return None
                if not isinstance(assign, ast.Assign):
                    self.error(error.expected_assignment(t.span))
                    return None
                assign.mut = True
                return assign
            case _:
                expr = self.parse_expr()
                if not expr:
                    return None
                if self.input.peek().kind == token.Kind.eq:
                    # This is an assignment or a function definition.
                    self.input.next()
                    if self.input.peek().kind == token.Kind.fun:
                        if not isinstance(expr, ast.Name) or expr.kind != "ident":
                            self.error(error.expected_ident(str(expr), t.span))
                            return None
                        return self.parse_fun_def(expr.name)
                    rhs = self.parse_expr()
                    if not rhs:
                        return None
                    return ast.Assign(self.id(), expr, rhs, self.input.span_merge(expr.span), mut=False)
                return expr

    def parse_module(self) -> ast.Module:
        span = self.input.span()
        nodes: list[ast.Node] = []
        while t := self.input.peek():
            if t.kind == token.Kind.eof:
                break
            node = self.parse_block_node()
            if node:
                nodes.append(node)
        module_name = span.file.split("/")[-1].split(".")[0]
        return ast.Module(self.id(), FQN([module_name]), nodes, self.input.span_merge(span))


def parse(tokens: Input) -> tuple[ast.Module, list[error.Error]]:
    parser = Parser(tokens)
    return (parser.parse_module(), parser.errors)
