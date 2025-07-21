open Token

type expr_id = int

let cur_id = ref 0

let new_id () =
  let id = !cur_id in
  incr cur_id;
  id

type expr =
  | AssignExpr of expr_id * expr * expr
  | AtomExpr of expr_id * string
  | BlockExpr of expr_id * expr list
  | CallExpr of expr_id * expr * expr list
  | FuncExpr of expr_id * expr list * expr
  | IdentExpr of expr_id * string
  | IntExpr of expr_id * int
  | ParenExpr of expr_id * expr
  | StrExpr of expr_id * string

let rec expr_to_string e =
  match e with
  | AssignExpr (_, a, b) -> Printf.sprintf "%s = %s" (expr_to_string a) (expr_to_string b)
  | AtomExpr (_, s) -> Printf.sprintf ":%s" s
  | BlockExpr (_, l) -> Printf.sprintf "=> %s;" (String.concat "\n" (List.map expr_to_string l))
  | CallExpr (_, f, args) ->
      Printf.sprintf "%s(%s)" (expr_to_string f) (String.concat ", " (List.map expr_to_string args))
  | FuncExpr (_, args, body) ->
      Printf.sprintf "(%s) %s"
        (String.concat ", " (List.map expr_to_string args))
        (expr_to_string body)
  | IdentExpr (_, i) -> i
  | IntExpr (_, i) -> Printf.sprintf "%d" i
  | ParenExpr (_, e) -> Printf.sprintf "(%s)" (expr_to_string e)
  | StrExpr (_, s) -> Printf.sprintf "\"%s\"" s

let expr_id expr =
  match expr with
  | AssignExpr (id, _, _)
  | AtomExpr (id, _)
  | BlockExpr (id, _)
  | CallExpr (id, _, _)
  | FuncExpr (id, _, _)
  | IdentExpr (id, _)
  | IntExpr (id, _)
  | ParenExpr (id, _)
  | StrExpr (id, _) -> id

let rec traverse_expr expr f =
  f expr;
  match expr with
  | AssignExpr (id, lhs, rhs) ->
      f lhs;
      f rhs
  | BlockExpr (id, exprs) -> List.iter (fun e -> traverse_expr e f) exprs
  | CallExpr (id, target, args) ->
      f target;
      List.iter (fun e -> f e) args
  | _ -> ()

let rec parse_primary_expr tokens =
  match tokens with
  | [] -> failwith "unexpected EOF"
  | Atom s :: rest -> (AtomExpr (new_id (), s), rest)
  | FatArrow :: rest ->
      let expr, rest = parse_block_expr rest in
      (expr, rest)
  | Ident i :: rest -> (IdentExpr (new_id (), i), rest)
  | Int i :: rest -> (IntExpr (new_id (), i), rest)
  | LParen _ :: rest -> parse_func_or_paren_expr rest
  | Str s :: rest -> (StrExpr (new_id (), s), rest)
  | c :: rest -> failwith Printf.(sprintf "unexpected token: %s" (token_to_string c))

and parse_comma_separated_expr exprs tokens =
  match tokens with
  | [] -> failwith "unexpected EOF"
  | RParen :: rest -> (List.rev exprs, rest)
  | Comma :: rest ->
      let expr, rest = parse_binary_expr rest in
      parse_comma_separated_expr (expr :: exprs) rest
  | c :: rest ->
      if List.length exprs = 0 then
        let expr, rest = parse_binary_expr tokens in
        parse_comma_separated_expr [ expr ] rest
      else
        failwith Printf.(sprintf "unexpected token: %s" (token_to_string c))

and parse_func_or_paren_expr tokens =
  (* Look ahead to see if this is a function expression or just a parenthesis expression. 
     First, try to accumulate a comma separated list of parameters. *)
  let exprs, rest =
    match tokens with
    | [] -> failwith "unexpected EOF"
    | RParen :: rest -> ([], rest)
    | _ -> parse_comma_separated_expr [] tokens
  in
  match rest with
  | FatArrow :: rest ->
      let body, rest = parse_block_expr rest in
      (FuncExpr (new_id (), exprs, body), rest)
  | _ -> (
      match List.length exprs with
      | 0 -> failwith "empty parenthesis"
      | 1 -> (ParenExpr (new_id (), List.hd exprs), rest)
      | _ -> failwith "unexpected EOF")

and parse_postfix_expr tokens =
  let expr, tokens = parse_primary_expr tokens in
  let rec handle_postfix expr tokens =
    match tokens with
    | LParen true :: rest ->
        let args, rest = parse_comma_separated_expr [] rest in
        let call = CallExpr (new_id (), expr, args) in
        handle_postfix call rest (* Chain calls: f()() *)
    | _ -> (expr, tokens)
  in
  handle_postfix expr tokens

and parse_binary_expr tokens =
  let lhs, tokens = parse_postfix_expr tokens in
  match tokens with
  | [] -> (lhs, tokens)
  | Eq :: rest ->
      let rhs, rest = parse_binary_expr rest in
      (AssignExpr (new_id (), lhs, rhs), rest)
  | _ -> (lhs, tokens)

and parse_block_expr tokens =
  let rec loop acc tokens =
    match tokens with
    | [] -> (BlockExpr (new_id (), List.rev acc), [])
    | Semi :: rest -> (BlockExpr (new_id (), List.rev acc), rest)
    | _ ->
        let expr, rest = parse_binary_expr tokens in
        loop (expr :: acc) rest
  in
  loop [] tokens

let parse tokens =
  let expr, rest = parse_block_expr tokens in
  match rest with
  | [] -> expr
  | [ EOF ] -> expr
  | c :: rest -> failwith Printf.(sprintf "unexpected token: %s" (token_to_string c))
