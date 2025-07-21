.PHONY: clean test precommit

ARGS = -I +unix unix.cma
OCAMLC = ocamlc $(ARGS)
OCAML = ocaml $(ARGS)

precommit: clean fmt test

fmt:
	ocamlformat -i --enable-outside-detected-project *.ml
 
test: token_test ast_test typ_test code_test compiler_test

token.cmo: token.ml
	$(OCAMLC) -c token.ml

token_test: token.cmo
	@echo "\n>>> Tokenizer tests"
	$(OCAML) token.cmo token_test.ml

ast.cmo: ast.ml token.cmo
	$(OCAMLC) -c ast.ml

ast_test: ast.cmo
	@echo "\n>>> AST tests"
	$(OCAML) token.cmo ast.cmo ast_test.ml

typ.cmo: token.cmo ast.cmo typ.ml
	$(OCAMLC) -c typ.ml

typ_test: typ.cmo
	@echo "\n>>> Type checker tests"
	$(OCAML) token.cmo ast.cmo typ.cmo typ_test.ml

code.cmo: token.cmo ast.cmo typ.cmo code.ml
	$(OCAMLC) -c code.ml

code_test: code.cmo
	@echo "\n>>> Code generator tests"
	$(OCAML) token.cmo ast.cmo typ.cmo code.cmo code_test.ml

compiler.cmo: token.cmo ast.cmo typ.cmo code.cmo compiler.ml
	$(OCAMLC) -c compiler.ml

compiler_test: compiler.cmo
	@echo "\n>>> Compiler tests"
	$(OCAML) token.cmo ast.cmo typ.cmo code.cmo compiler.cmo compiler_test.ml

clean:
	rm -f *.cmo *.cmi
