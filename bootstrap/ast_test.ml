open Token
open Ast

let failed = ref 0

let assert_equal ~loc a b =
  if a <> b then
    failwith (Printf.sprintf "expected `%s` but got `%s` at %s" a b loc)

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
  expr_to_string expr

let test_simple_expr =
  test "int lit" (fun () ->
      let expr = gen_expr "123" in
      assert_equal ~loc:__LOC__ "123" expr);
  test "str lit" (fun () ->
      let expr = gen_expr "\"foo\"" in
      assert_equal ~loc:__LOC__ "\"foo\"" expr);
  test "ident" (fun () ->
      let expr = gen_expr "foo" in
      assert_equal ~loc:__LOC__ "foo" expr);
  test "atom" (fun () ->
      let expr = gen_expr ":foo" in
      assert_equal ~loc:__LOC__ ":foo" expr);
  test "assign" (fun () ->
      let expr = gen_expr "foo = 123" in
      assert_equal ~loc:__LOC__ "foo = 123" expr);
  test "block" (fun () ->
      let expr = gen_expr "=> 123 foo;" in
      assert_equal ~loc:__LOC__ "=> 123\nfoo;" expr);
  test "paren expr" (fun () ->
      let expr = gen_expr "(123)" in
      assert_equal ~loc:__LOC__ "(123)" expr);
  test "multilple simple exprs" (fun () ->
      let tokens = tokenize "123 \"foo\" foo" in
      let expr = parse tokens in
      assert_equal ~loc:__LOC__ "=> 123\n\"foo\"\nfoo;" (expr_to_string expr))

let test_function =
  test "function with no params" (fun () ->
      let expr = gen_expr "id = () => 42;" in
      assert_equal ~loc:__LOC__ "id = () => 42;" expr);
  test "function with singe param" (fun () ->
      let expr = gen_expr "id = (a) => a;" in
      assert_equal ~loc:__LOC__ "id = (a) => a;" expr);
  test "function with multiple params" (fun () ->
      let expr = gen_expr "foo = (a, b) => 42 a;" in
      assert_equal ~loc:__LOC__ "foo = (a, b) => 42\na;" expr);
  test "function with non-ident param" (fun () ->
      try
        gen_expr "foo = (a 42) => a;" |> ignore;
        failwith "expected failure"
      with Failure msg -> assert_equal ~loc:__LOC__ "unexpected token: 42" msg)

let test_call =
  test "call with no params" (fun () ->
      let expr = gen_expr "id()" in
      assert_equal ~loc:__LOC__ "id()" expr);
  test "call with single param" (fun () ->
      let expr = gen_expr "id(a)" in
      assert_equal ~loc:__LOC__ "id(a)" expr);
  test "call with multiple params" (fun () ->
      let expr = gen_expr "foo(a, b)" in
      assert_equal ~loc:__LOC__ "foo(a, b)" expr);
  test "call on any expression" (fun () ->
      let expr = gen_expr "(42 = foo)(a)" in
      assert_equal ~loc:__LOC__ "(42 = foo)(a)" expr)

let test_program =
  test "simple program" (fun () ->
      let tokens =
        tokenize {|

      id = (a) => 
          b = 42
          a;

      id(42)

    |}
      in
      let expr = parse tokens in
      assert_equal ~loc:__LOC__ "=> id = (a) => b = 42\na;\nid(42);" (expr_to_string expr))

let () =
  if !failed = 0 then
    print_endline "All tests passed!"
  else (
    print_endline (Printf.sprintf "%d tests failed!" !failed);
    exit 1)
