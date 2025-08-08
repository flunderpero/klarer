"""Run tests found in a markdown file."""

import contextlib
import re
import shutil
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from time import time

from . import compiler, error, token


@dataclass
class Case:
    headings: list[str]
    test_num: int
    line: int
    code: str
    expected_stdout: str
    only: bool = False

    def name(self) -> str:
        return " > ".join(self.headings) + f" ({self.test_num})"


def run_test(case: Case, print_code: str) -> list:
    def handle_errors(errors: list[error.Error]) -> list[error.Error]:
        if errors:
            for err in list(errors):
                line_number = err.span.start_line_col()[0]
                line = case.code.split("\n")[line_number - 1]
                try:
                    index = line.index("-- ERROR: ")
                except ValueError:
                    index = -1
                if index >= 0:
                    expected_error_message = line[index + len("-- ERROR: ") :].strip()
                    if expected_error_message in str(err).split("\n")[0]:
                        # This error is expected.
                        errors = [x for x in errors if x != err]
                        continue
        return errors

    tmp_dir = Path(tempfile.gettempdir(), re.sub(r"[^a-zA-Z0-9]", "_", case.name()))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / "test"
    try:
        for step in compiler.compile(token.Input("test.kl", case.code), str(tmp_file)):
            match step:
                case compiler.TokenStep():
                    if print_code == "tokens":
                        print()
                        print(step)
                        return handle_errors(step.errors)
                    if step.errors:
                        return handle_errors(step.errors)
                    if step.duration > 0.1:
                        print(f" [token:{step.duration * 1000:.0f}]", end="", flush=True)
                case compiler.ParseStep():
                    if print_code == "ast":
                        print()
                        print(step)
                        return handle_errors(step.errors)
                    if step.errors:
                        return handle_errors(step.errors)
                    if step.duration > 0.1:
                        print(f" [parse:{step.duration * 1000:.0f}]", end="", flush=True)
                case compiler.TypecheckStep():
                    if print_code == "types":
                        print()
                        print(step)
                        return handle_errors(step.errors)
                    if step.errors:
                        return handle_errors(step.errors)
                    if step.duration > 0.1:
                        print(f" [typecheck:{step.duration * 1000:.0f}]", end="", flush=True)
                case compiler.AbortStep():
                    pass
                case compiler.IRStep():
                    if print_code == "ir":
                        print()
                        print(step)
                        return []
                    if step.duration > 0.1:
                        print(f" [ir:{step.duration * 1000:.0f}]", end="", flush=True)
                case compiler.CodeGenStep():
                    if print_code == "code":
                        print()
                        print(step)
                        return []
                    if step.duration > 0.1:
                        print(f" [code:{step.duration * 1000:.0f}]", end="", flush=True)
                case compiler.CompileStep():
                    if step.returncode != 0:
                        return [f"Compilation (Go) failed with code {step.returncode}\n{step.stdout}\n{step.stderr}"]
                    if step.duration > 0.1:
                        print(f" [go:{step.duration * 1000:.0f}]", end="", flush=True)
                case compiler.RunStep():
                    if step.returncode != 0:
                        return [f"Test exited with code {step.returncode}\n{step.stdout}\n{step.stderr}"]
                    stdout = step.stdout.strip().replace("\0", "")
                    if stdout != case.expected_stdout:
                        return [f"Expected:\n\n`{case.expected_stdout}`\n\ngot:\n\n`{stdout}`"]
                    if step.duration > 0.1:
                        print(f" [run:{step.duration * 1000:.0f}]", end="", flush=True)
                case _:
                    raise ValueError(f"Unknown step: {step}")
    except (AssertionError, AttributeError):
        return ["Test failed with exception:", traceback.format_exc()]
    finally:
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(tmp_dir)

    return []


def find_tests(src: str, chapter: str) -> list[Case]:
    tests: list[Case] = []
    lines = src.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        headings = []
        if line.startswith("#") and chapter.lower() in line.lower():
            headings.append(line)
            heading_level = len(line.split(" ")[0])
            test_num = 1
            while i < len(lines):
                line = lines[i]
                if line.startswith("#"):
                    this_level = len(line.split(" ")[0])
                    if this_level <= heading_level:
                        break
                    headings = [x for x in headings if len(x.split(" ")[0]) < this_level]
                    headings.append(line)
                    test_num = 1
                i += 1
                if line.startswith("```klarer") and "!skip" not in line:
                    only = "!only" in line
                    test_line = i - 1
                    expected_stdout = []
                    code = []
                    while i < len(lines):
                        line = lines[i]
                        i += 1
                        if line.startswith("```"):
                            break
                        code.append(line)
                    code_str = "\n".join(code)
                    if lines[i].strip() == "" and len(lines) > i + 1 and lines[i + 1].strip() == "```":
                        i += 2
                        while i < len(lines):
                            line = lines[i]
                            if line.strip() == "```":
                                break
                            expected_stdout.append(line)
                            i += 1

                    test = Case(headings, test_num, test_line, code_str, "\n".join(expected_stdout), only)
                    test_num += 1
                    tests.append(test)
    return tests


def filter_tests(tests: list[Case]) -> list[Case]:
    # Filter sections with "!only" in the name.
    filtered_tests = []
    while True:
        only_sections = [i for i, x in enumerate(tests) if "!only" in x.name()]
        if not only_sections:
            break
        only_section = only_sections[0]
        section_heading = " ".join(tests[only_section].headings)
        section_tests = tests[only_section:]
        end_section = next(
            (i for i, x in enumerate(section_tests) if not " ".join(x.headings).startswith(section_heading)), None
        )
        if end_section:
            section_tests = section_tests[:end_section]
        for test in section_tests:
            for i, heading in enumerate(test.headings):
                test.headings[i] = heading.replace("!only", "")
        filtered_tests.extend(section_tests)
    if not filtered_tests:
        filtered_tests = tests

    # Filter tests marked with "!only".
    only_tests = [x for x in filtered_tests if x.only]
    if only_tests:
        filtered_tests = only_tests
    return filtered_tests


def main(args: list[str]) -> int:
    print_code = ""
    stages = ("tokens", "ast", "types", "ir", "code")
    for stage in stages:
        if f"--{stage}" in args:
            print_code = stage
            break
    print_error_stack = "--err-stack" in args
    bail = "--bail" in args
    args = [x for x in args if not x.startswith("--")]
    if len(args) == 1:
        print("Usage: md_tests.py <file> [chapter] [#test] [options]")
        print("  Options:")
        print("    --err-stack     Print error stack")
        print("    --signatures    Print signatures instead of debug info")
        for stage in stages:
            print(f"    --{stage}        Print {stage} output")
        return 1
    file = args[1]
    src = open(file).read()
    tests = find_tests(src, "" if len(args) == 2 else args[2])
    tests = filter_tests(tests)
    # print("aaa", "\n".join([x.name() for x in tests]))
    # return 0
    if len(args) > 3:
        test_num = int(args[3]) - 1
        tests = tests[test_num : test_num + 1]
    failed = 0
    start = time()
    for test in tests:
        print(test.name(), f"at {file}:{test.line}", end="")
        if errors := run_test(test, print_code):
            failed += 1
            print(" \033[0;31mFAIL\033[0m")
            for err in errors:
                print()
                print(err)
                if print_error_stack and not isinstance(err, str):
                    print("at", err.stacktrace)
            if bail:
                break
        elif not print_code:
            print(" \033[0;32mPASS\033[0m")
    duration = time() - start
    if failed:
        print(f"\n{failed}/{len(tests)} tests in {duration * 1000:.0f}ms \033[0;31mFAILED\033[0m")
        return 2
    if not print_code:
        print(f"\nAll {len(tests)} tests in {duration:.2f}s \033[0;32mPASSED\033[0m")
    return 0


def test_default() -> None:
    assert main(["md_tests.py", "TEST.md"]) == 0


if __name__ == "__main__":
    ret = main(sys.argv)
    if ret:
        sys.exit(ret)
