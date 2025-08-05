open Compiler

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

let build_and_run src =
  compile "test.go" src |> ignore;
  run "./test"

let test_program =
  test "simple program" (fun () ->
      let output = build_and_run {|

      id = (a) => print("PASS") a;

      id(42)

    |} in
      assert_equal ~loc:__LOC__ "PASS" output)

let () =
  if !failed = 0 then
    print_endline "All tests passed!"
  else (
    print_endline (Printf.sprintf "%d tests failed!" !failed);
    exit 1)
