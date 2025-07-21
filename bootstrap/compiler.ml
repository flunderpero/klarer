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

  let file = open_out filename in
  output_string file code;
  close_out file;

  run ("go build " ^ Filename.quote filename)
