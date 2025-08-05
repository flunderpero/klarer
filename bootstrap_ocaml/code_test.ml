open Token
open Ast
open Typ
open Code

let failed = ref 0

let assert_equal ~loc a b =
  if a <> b then
    failwith (Printf.sprintf "expected `%s` but got `%s` at %s" a b loc)

let assert_true ~loc a =
  if not a then
    failwith (Printf.sprintf "expected `true` but got `false` at %s" loc)

let test name f =
  print_string name;
  try
    f ();
    print_endline " \027[32mPASS\027[0m"
  with Failure msg ->
    print_endline " \027[31mFAIL\027[0m";
    print_endline ("  " ^ msg);
    incr failed

let gen_expr src =
  let tokens = tokenize src in
  let expr, _ = parse_binary_expr tokens in
  let _, typ_map = typ_check expr in
  let code = Code.create () in
  gencode_expr typ_map code (IdentExpr (0, "main")) expr;
  Code.contents code

let gen_program src =
  let tokens = tokenize src in
  let expr, _ = parse_binary_expr tokens in
  let _, typ_map = typ_check expr in
  gencode typ_map expr

let test_simple_expr =
  test "int lit" (fun () ->
      let code = gen_expr "123" in
      assert_equal ~loc:__LOC__ "123 " code);

  test "str lit" (fun () ->
      let code = gen_expr "\"foo\"" in
      assert_equal ~loc:__LOC__ "\"foo\" " code);

  test "ident" (fun () ->
      let code = gen_expr "foo" in
      assert_equal ~loc:__LOC__ "foo " code);

  test "assign" (fun () ->
      let code = gen_expr "foo = 123" in
      assert_equal ~loc:__LOC__ "foo := 123 " code);

  test "block" (fun () ->
      let code = gen_expr "=> 123 foo;" in
      assert_equal ~loc:__LOC__ "{\n    123 \n    foo \n    \n}\n" code);

  test "paren" (fun () ->
      let code = gen_expr "(123)" in
      assert_equal ~loc:__LOC__ "(123 ) " code)

let test_call =
  test "call with no params" (fun () ->
      let code = gen_expr "id()" in
      assert_equal ~loc:__LOC__ "id () " code);
  test "call with single param" (fun () ->
      let code = gen_expr "id(a)" in
      assert_equal ~loc:__LOC__ "id (a , ) " code);
  test "call with multiple params" (fun () ->
      let code = gen_expr "foo(a, b)" in
      assert_equal ~loc:__LOC__ "foo (a , b , ) " code);
  test "call on any expression" (fun () ->
      let code = gen_expr "(42 = foo)(a)" in
      assert_equal ~loc:__LOC__ "(42 := foo ) (a , ) " code)

let test_program =
  test "simple program" (fun () ->
      let code = gen_program {|

      id = (a) => 
          b = 42
          a;

    |} in
      assert_equal ~loc:__LOC__
        {|package main

import (
    "fmt"
)

func main() {
id := func(a any) any {
    b := 42 
    return a 
}

}
|}
        code)

let () =
  if !failed = 0 then
    print_endline "All tests passed!"
  else (
    print_endline (Printf.sprintf "%d tests failed!" !failed);
    exit 1)
