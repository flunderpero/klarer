(* We are feeling fancy - let's compile straight to Go. *)

open Ast
open Typ

module Code = struct
  type t = {
    buf : Buffer.t;
    mutable indent : int;
    mutable indent_str : string;
  }

  let create () = { buf = Buffer.create 100; indent = 0; indent_str = "" }

  let indent t =
    t.indent <- t.indent + 1;
    t.indent_str <- String.make (t.indent * 4) ' '

  let dedent t =
    t.indent <- t.indent - 1;
    t.indent_str <- String.make (t.indent * 4) ' '

  let write t s = Buffer.add_string t.buf s
  let new_line t = Buffer.add_string t.buf ("\n" ^ t.indent_str)
  let contents t = Buffer.contents t.buf
end

let rec gencode_expr typ_map code parent_expr expr =
  match expr with
  | AssignExpr (_, lhs, rhs) ->
      gencode_expr typ_map code expr lhs;
      Code.write code ":= ";
      gencode_expr typ_map code expr rhs
  | AtomExpr (_, s) -> failwith "AtomExpr not implemented"
  | BlockExpr (_, exprs) ->
      Code.write code "{";
      Code.indent code;
      Code.new_line code;
      (match (parent_expr, List.rev exprs) with
      | FuncExpr _, [] -> Code.write code "return nil"
      | FuncExpr _, last :: rest ->
          List.iter
            (fun e ->
              gencode_expr typ_map code expr e;
              Code.new_line code)
            (List.rev rest);
          (* todo: don't return if the type of the last expression is Unit. *)
          Code.write code "return ";
          gencode_expr typ_map code expr last
      | _ ->
          List.iter
            (fun e ->
              gencode_expr typ_map code expr e;
              Code.new_line code)
            exprs);
      Code.dedent code;
      Code.new_line code;
      Code.write code "}";
      Code.new_line code
  | CallExpr (_, target, args) ->
      gencode_expr typ_map code expr target;
      Code.write code "(";
      List.iter
        (fun e ->
          gencode_expr typ_map code expr e;
          Code.write code ", ")
        args;
      Code.write code ") "
  | FuncExpr (_, args, body) ->
      Code.write code "func(";
      Code.write code
        (String.concat ", " (List.map (fun e -> Printf.sprintf "%s any" (expr_to_string e)) args));
      Code.write code ") any ";
      gencode_expr typ_map code expr body
  | IdentExpr (_, i) -> Code.write code (Printf.sprintf "%s " i)
  | IntExpr (_, i) -> Code.write code (Printf.sprintf "%d " i)
  | ParenExpr (_, e) ->
      Code.write code "(";
      gencode_expr typ_map code expr e;
      Code.write code ") "
  | StrExpr (_, s) -> Code.write code (Printf.sprintf "\"%s\" " s)

let gencode typ_map expr =
  let code = Code.create () in
  Code.write code "package main\n\n";
  Code.write code "func main() {\n";
  gencode_expr typ_map code (IdentExpr (0, "main")) expr;
  Code.write code "\n}\n";
  Code.contents code
