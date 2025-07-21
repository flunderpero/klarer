open Token
open Ast
open Typ

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
  let typ, _ = typ_check expr in
  typ_to_string typ

let test_simple_expr =
  test "int lit" (fun () ->
      let typ = gen_expr "123" in
      assert_equal ~loc:__LOC__ "Int" typ);
  test "str lit" (fun () ->
      let typ = gen_expr "\"foo\"" in
      assert_equal ~loc:__LOC__ "Str" typ);
  test "ident" (fun () ->
      let typ = gen_expr "foo" in
      assert_equal ~loc:__LOC__ "Ident" typ);
  test "atom" (fun () ->
      let typ = gen_expr ":foo" in
      assert_equal ~loc:__LOC__ "Atom(foo)" typ);
  test "assign" (fun () ->
      let typ = gen_expr "foo = 123" in
      assert_equal ~loc:__LOC__ "Assign(Ident, Int)" typ);
  test "paren" (fun () ->
      let typ = gen_expr "(123)" in
      assert_equal ~loc:__LOC__ "Int" typ)

let test_typ_map =
  test "typ map" (fun () ->
      let tokens = tokenize "foo = 123" in
      let expr, _ = parse_binary_expr tokens in
      let typ, typ_map = typ_check expr in
      let typs = ref [] in
      traverse_expr expr (fun e ->
          let t = TypMap.get typ_map (expr_id e) in
          let s = Printf.sprintf "%s: %s" (expr_to_string e) (typ_to_string t) in
          typs := s :: !typs);
      let result = String.concat "\n" (List.rev !typs) in
      let expected = "foo = 123: Assign(Ident, Int)\nfoo: Ident\n123: Int" in
      assert_equal ~loc:__LOC__ expected result)

let test_block =
  test "block" (fun () ->
      let typ = gen_expr "=> 123 foo;" in
      assert_equal ~loc:__LOC__ "Ident" typ);

  test "empty block" (fun () ->
      let typ = gen_expr "=>" in
      assert_equal ~loc:__LOC__ "Unit" typ)

let test_function =
  test "function" (fun () ->
      let typ = gen_expr "id = (a, b) => foo 42;" in
      assert_equal ~loc:__LOC__ "Assign(Ident, Func(Ident, Ident, Int))" typ);

  test "no param function" (fun () ->
      let typ = gen_expr "id = () => foo 42;" in
      assert_equal ~loc:__LOC__ "Assign(Ident, Func(Unit, Int))" typ)

let test_call =
  test "call with no params" (fun () ->
      let typ = gen_expr "id()" in
      assert_equal ~loc:__LOC__ "Call(Ident, Unit)" typ);
  test "call with single param" (fun () ->
      let typ = gen_expr "id(a)" in
      assert_equal ~loc:__LOC__ "Call(Ident, Ident)" typ);
  test "call with multiple params" (fun () ->
      let typ = gen_expr "foo(a, 42)" in
      assert_equal ~loc:__LOC__ "Call(Ident, Ident, Int)" typ);
  test "call on any expression" (fun () ->
      let typ = gen_expr "(foo = 42)(a, 42)" in
      assert_equal ~loc:__LOC__ "Call(Assign(Ident, Int), Ident, Int)" typ)

let () =
  if !failed = 0 then
    print_endline "All tests passed!"
  else (
    print_endline (Printf.sprintf "%d tests failed!" !failed);
    exit 1)
