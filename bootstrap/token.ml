type token =
  | Atom of string
  | Comma
  | EOF
  | Eq
  | Error of string
  | FatArrow
  | Ident of string
  | Int of int
  | LParen of bool
  | Minus
  | Plus
  | RParen
  | Semi
  | Str of string

let token_to_string t =
  match t with
  | Atom s -> Printf.sprintf ":%s" s
  | Comma -> ","
  | EOF -> "EOF"
  | Eq -> "="
  | Error s -> Printf.sprintf "Error(%s)" s
  | FatArrow -> "=>"
  | Ident i -> i
  | Int i -> Printf.sprintf "%d" i
  | LParen imm ->
      if imm then
        "(i"
      else
        "("
  | Minus -> "-"
  | Plus -> "+"
  | RParen -> ")"
  | Semi -> ";"
  | Str s -> Printf.sprintf "\"%s\"" s

let tokens_to_string list = String.concat " " (List.map token_to_string list)

let tokenize s =
  let len = String.length s in
  let rec step i prev_ws =
    if i >= len then
      (EOF, i)
    else
      match String.get s i with
      | '+' -> (Plus, i + 1)
      | '-' -> (Minus, i + 1)
      | ';' -> (Semi, i + 1)
      | ',' -> (Comma, i + 1)
      | ':' ->
          let rec loop j =
            if j >= len then
              if j == i + 1 then
                (Error "empty atom", j)
              else
                (Atom (String.sub s (i + 1) (j - i - 1)), j)
            else
              match String.get s j with
              | 'a' .. 'z' | '_' | '0' .. '9' -> loop (j + 1)
              | _ -> (Atom (String.sub s (i + 1) (j - i - 1)), j)
          in
          loop (i + 1)
      | '=' ->
          if i + 1 < len && String.get s (i + 1) = '>' then
            (FatArrow, i + 2)
          else
            (Eq, i + 1)
      | '(' -> (LParen (prev_ws = false), i + 1)
      | ')' -> (RParen, i + 1)
      | '0' .. '9' ->
          let rec loop j =
            if j >= len then
              (Int (int_of_string (String.sub s i (j - i))), j)
            else
              match String.get s j with
              | '0' .. '9' -> loop (j + 1)
              | _ -> (Int (int_of_string (String.sub s i (j - i))), j)
          in
          loop (i + 1)
      | '"' ->
          let rec loop j =
            if j >= len then
              (Error "unterminated string literal", j)
            else
              match String.get s j with
              | '"' -> (Str (String.sub s (i + 1) (j - i - 1)), j + 1)
              | '\n' -> (Error "unterminated string literal", j)
              | _ -> loop (j + 1)
          in
          loop (i + 1)
      | '\n' | '\r' | ' ' -> step (i + 1) true
      | 'a' .. 'z' ->
          let rec loop j =
            if j >= len then
              (Ident (String.sub s i (j - i)), j + 1)
            else
              match String.get s j with
              | 'a' .. 'z' | '_' | '0' .. '9' | 'A' .. 'Z' -> loop (j + 1)
              | _ -> (Ident (String.sub s i (j - i)), j)
          in
          loop (i + 1)
      | c -> (Error (Printf.sprintf "unknown token: %c" c), i + 1)
  in
  let rec collect i acc =
    let t, i = step i false in
    match t with
    | EOF -> List.rev acc
    | _ -> collect i (t :: acc)
  in
  collect 0 []
