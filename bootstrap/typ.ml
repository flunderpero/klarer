open Ast

type expr_typ =
  | AtomTyp of string
  (* todo: model return type *)
  | CallTyp of expr_typ * expr_typ list
  | IntTyp
  | IdentTyp
  | AssignTyp of expr_typ * expr_typ
  | StrTyp
  (* Function parameter types are always IdentType. *)
  | FuncTyp of expr_typ list * expr_typ
  | UnitTyp

module TypMap = struct
  type t = (expr_id, expr_typ) Hashtbl.t

  let create () = Hashtbl.create 10
  let get t id = Hashtbl.find t id
  let set t id typ = Hashtbl.add t id typ
end

let rec typ_to_string t =
  match t with
  | AssignTyp (a, b) -> Printf.sprintf "Assign(%s, %s)" (typ_to_string a) (typ_to_string b)
  | AtomTyp s -> Printf.sprintf "Atom(%s)" s
  | CallTyp (f, args) ->
      Printf.sprintf "Call(%s, %s)" (typ_to_string f)
        (if args = [] then
           "Unit"
         else
           String.concat ", " (List.map typ_to_string args))
  | FuncTyp (args, body) ->
      Printf.sprintf "Func(%s, %s)"
        (if args = [] then
           "Unit"
         else
           String.concat ", " (List.map typ_to_string args))
        (typ_to_string body)
  | IdentTyp -> "Ident"
  | IntTyp -> "Int"
  | StrTyp -> "Str"
  | UnitTyp -> "Unit"

let typ_check expr =
  let typ_map = TypMap.create () in
  let rec typ_check_expr expr =
    let typ, id =
      match expr with
      | AssignExpr (id, lhs, rhs) ->
          let lhs_typ, _ = typ_check_expr lhs in
          let rhs_typ, _ = typ_check_expr rhs in
          (AssignTyp (lhs_typ, rhs_typ), id)
      | AtomExpr (id, s) -> (AtomTyp s, id)
      | BlockExpr (id, exprs) ->
          if List.length exprs = 0 then
            (UnitTyp, id)
          else
            let typs, ids = List.split (List.map typ_check_expr exprs) in
            (List.hd (List.rev typs), id)
      | CallExpr (id, target, args) ->
          let target_typ, _ = typ_check_expr target in
          let arg_typs, _ = List.split (List.map typ_check_expr args) in
          (CallTyp (target_typ, arg_typs), id)
      | FuncExpr (id, args, body) ->
          let arg_typs, _ = List.split (List.map typ_check_expr args) in
          let body_typ, _ = typ_check_expr body in
          (FuncTyp (List.rev arg_typs, body_typ), id)
      | IdentExpr (id, i) -> (IdentTyp, id)
      | IntExpr (id, i) -> (IntTyp, id)
      | ParenExpr (id, e) -> typ_check_expr e
      | StrExpr (id, s) -> (StrTyp, id)
    in
    TypMap.set typ_map id typ;
    (typ, id)
  in
  let typ, _ = typ_check_expr expr in
  (typ, typ_map)
