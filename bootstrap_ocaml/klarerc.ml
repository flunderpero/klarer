open Compiler

(* Main entry point. *)
let () =
  let filename = Sys.argv.(1) in
  let f = open_in filename in
  let src = really_input_string f (in_channel_length f) in
  let _ = compile filename src in
  ()
