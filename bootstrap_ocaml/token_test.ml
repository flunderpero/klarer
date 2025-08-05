open Token

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

let test_simple_tokens =
  test "all simple tokens" (fun () ->
      let tokens = tokenize " + - = ( ) =>" in
      assert_equal ~loc:__LOC__ "+ - = ( ) =>" (tokens_to_string tokens))

let test_int_lit =
  test "all is an int lit" (fun () ->
      let tokens = tokenize "123" in
      assert_equal ~loc:__LOC__ "123" (tokens_to_string tokens));
  test "int lit followed by other token" (fun () ->
      let tokens = tokenize "123+" in
      assert_equal ~loc:__LOC__ "123 +" (tokens_to_string tokens));
  test "int lit between other tokens" (fun () ->
      let tokens = tokenize "-123+" in
      assert_equal ~loc:__LOC__ "- 123 +" (tokens_to_string tokens))

let test_str_lit =
  test "all is a str lit" (fun () ->
      let tokens = tokenize "\"foo\"" in
      assert_equal ~loc:__LOC__ "\"foo\"" (tokens_to_string tokens));
  test "str lit followed by other token" (fun () ->
      let tokens = tokenize "\"foo\"+" in
      assert_equal ~loc:__LOC__ "\"foo\" +" (tokens_to_string tokens));
  test "str lit between other tokens" (fun () ->
      let tokens = tokenize "-\"foo\"+" in
      assert_equal ~loc:__LOC__ "- \"foo\" +" (tokens_to_string tokens));
  test "non-terminated str lit at eof" (fun () ->
      let tokens = tokenize "\"foo" in
      assert_equal ~loc:__LOC__ "Error(unterminated string literal)" (tokens_to_string tokens));
  test "non-terminated str lit at eol" (fun () ->
      let tokens = tokenize "\"foo\n" in
      assert_equal ~loc:__LOC__ "Error(unterminated string literal)" (tokens_to_string tokens))

let test_ident =
  test "all is an ident" (fun () ->
      let tokens = tokenize "foo" in
      assert_equal ~loc:__LOC__ "foo" (tokens_to_string tokens));
  test "ident followed by other token" (fun () ->
      let tokens = tokenize "foo+" in
      assert_equal ~loc:__LOC__ "foo +" (tokens_to_string tokens));
  test "ident between other tokens" (fun () ->
      let tokens = tokenize "-foo+" in
      assert_equal ~loc:__LOC__ "- foo +" (tokens_to_string tokens))

let test_atom =
  test "all is an atom" (fun () ->
      let tokens = tokenize ":foo" in
      assert_equal ~loc:__LOC__ ":foo" (tokens_to_string tokens));
  test "atom followed by other token" (fun () ->
      let tokens = tokenize ":foo+" in
      assert_equal ~loc:__LOC__ ":foo +" (tokens_to_string tokens));
  test "atom between other tokens" (fun () ->
      let tokens = tokenize "-:foo+" in
      assert_equal ~loc:__LOC__ "- :foo +" (tokens_to_string tokens))

let test_programs =
  test "simple program" (fun () ->
      let tokens = tokenize {|

      incr = (a) => a + 1;

      print (incr a)

    |} in
      assert_equal ~loc:__LOC__ "incr = ( a ) => a + 1 ; print ( incr a )" (tokens_to_string tokens))

let () =
  if !failed = 0 then
    print_endline "All tests passed!"
  else (
    print_endline (Printf.sprintf "%d tests failed!" !failed);
    exit 1)
