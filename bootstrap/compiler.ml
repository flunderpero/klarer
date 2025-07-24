open Unix
open Token
open Ast
open Typ
open Code

let run cmd =
  let inp = Unix.open_process_in (cmd ^ " 2>&1") in
  let r = In_channel.input_all inp in
  match Unix.close_process_in inp with
  | Unix.WEXITED 0 -> r
  | Unix.WEXITED n -> failwith (Printf.sprintf "Command failed with exit code %d: %s" n r)
  | Unix.WSIGNALED n -> failwith (Printf.sprintf "Command killed by signal %d" n)
  | Unix.WSTOPPED n -> failwith (Printf.sprintf "Command stopped by signal %d" n)

let compile filename src =
  let tokens = tokenize src in
  let expr = parse tokens in
  let _, typ_map = typ_check expr in
  let code = gencode typ_map expr in

  let base_filename = Filename.chop_extension filename in
  let target_filename = base_filename ^ ".go" in
  let file = open_out target_filename in
  output_string file code;
  close_out file;

  run
    (Printf.sprintf "go build -o %s %s" (Filename.quote base_filename)
       (Filename.quote target_filename))
