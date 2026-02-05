import ast
import asyncio
import os
import re
import secrets
import subprocess
import sys
import tempfile
import textwrap
import time

import numpy as np

from verl_patch.utils.tools.sandbox_fusion import execute_single_task

# 从环境变量获取超时配置
SANDBOX_RUN_TIMEOUT = int(os.getenv("SANDBOX_RUN_TIMEOUT", 15))
SANDBOX_CLIENT_TIMEOUT = int(os.getenv("SANDBOX_CLIENT_TIMEOUT", 15))
LOCAL_FALLBACK_RUN_TIMEOUT = int(os.getenv("LOCAL_FALLBACK_RUN_TIMEOUT", 75))


def _execute_code_locally(code: str, run_timeout: int):
    """当 sandbox 执行失败时在本地回退执行"""

    tmp_file = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    try:
        tmp_file.write(code)
        tmp_file.flush()
        tmp_path = tmp_file.name
    finally:
        tmp_file.close()

    start = time.time()
    try:
        completed = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=run_timeout,
        )
        duration = time.time() - start
        run_result = {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "return_code": completed.returncode,
            "execution_time": duration,
            "local_queue_time": 0.0,
            "sandbox_queue_time": 0.0,
            "executor": "local_fallback",
        }
    except subprocess.TimeoutExpired as exc:
        # killed because of timeout
        duration = time.time() - start
        stderr_content = exc.stderr or ""
        if stderr_content:
            stderr_content = f"{stderr_content.rstrip()}\n"
        stderr_content += f"Local execution timed out after {run_timeout} seconds."
        run_result = {
            "stdout": exc.stdout or "",
            "stderr": stderr_content,
            "return_code": -9,
            "execution_time": duration,
            "local_queue_time": 0.0,
            "sandbox_queue_time": 0.0,
            "executor": "local_fallback",
        }
    except Exception as exc:  # noqa: BLE001
        # probably killed because of OOM
        duration = time.time() - start
        run_result = {
            "stdout": "",
            "stderr": f"Local execution error: {exc}",
            "return_code": -1,
            "execution_time": duration,
            "local_queue_time": 0.0,
            "sandbox_queue_time": 0.0,
            "executor": "local_fallback",
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return run_result


def _run_with_sandbox_fallback(code_to_execute: str):
    run_result, run_status = asyncio.run(
        execute_single_task(
            code_to_execute,
            run_timeout=SANDBOX_RUN_TIMEOUT,
            client_timeout=SANDBOX_CLIENT_TIMEOUT,
            max_attempts=3,
        )
    )

    if run_result["return_code"] not in [0, 1]:
        print(
            f"[Code reward] sandbox returns {run_result['return_code']} after {run_result['execution_time']} seconds, run_result: {run_result}"
        )
        run_result = _execute_code_locally(code_to_execute, run_timeout=LOCAL_FALLBACK_RUN_TIMEOUT)
        print(
            f"[Code reward] local execution returns {run_result['return_code']} after {run_result['execution_time']} seconds, run_result: {run_result}"
        )

    return run_result


def _collect_used_names(*sources: str) -> set[str]:
    """Collect identifier names referenced across provided Python sources."""

    used: set[str] = set()
    for src in sources:
        if not src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
    return used


def prune_import_code(import_code: str, *code_fragments: str) -> str:
    """Remove unused import aliases based on names referenced in provided code."""

    if not import_code.strip():
        return import_code

    try:
        import_tree = ast.parse(import_code)
    except SyntaxError:
        return import_code

    used_names = _collect_used_names(import_code, *code_fragments)

    new_body = []
    changed = False

    for node in import_tree.body:
        if isinstance(node, ast.Import):
            kept_aliases = []
            for alias in node.names:
                if alias.name == "*":
                    kept_aliases.append(ast.alias(name=alias.name, asname=alias.asname))
                    continue
                introduced = alias.asname or alias.name.split(".")[0]
                if introduced in used_names:
                    kept_aliases.append(ast.alias(name=alias.name, asname=alias.asname))
            if kept_aliases:
                if len(kept_aliases) != len(node.names):
                    changed = True
                new_body.append(ast.copy_location(ast.Import(names=kept_aliases), node))
            else:
                changed = True
        elif isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                new_body.append(node)
                continue
            kept_aliases = []
            for alias in node.names:
                introduced = alias.asname or alias.name
                if introduced in used_names:
                    kept_aliases.append(ast.alias(name=alias.name, asname=alias.asname))
            if kept_aliases:
                if len(kept_aliases) != len(node.names):
                    changed = True
                new_body.append(
                    ast.copy_location(
                        ast.ImportFrom(module=node.module, names=kept_aliases, level=node.level),
                        node,
                    )
                )
            else:
                changed = True
        else:
            new_body.append(node)

    if not changed:
        return import_code

    new_module = ast.Module(body=new_body, type_ignores=[])
    ast.fix_missing_locations(new_module)

    try:
        return ast.unparse(new_module)
    except Exception:
        import astor

        return astor.to_source(new_module)


def strip_main_guard(src: str) -> str:
    """
    Remove top-level `if __name__ == "__main__": ...` blocks from `src` via AST,
    while preserving the `else:` branch (since under not-main it would run).
    We match both `__name__ == "__main__"` and `"__main__" == __name__`.
    Returns the transformed source code as a string.
    """

    def _is_main_guard(test: ast.AST) -> bool:
        # Match: (__name__ == "__main__") OR ("__main__" == __name__)
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
            return False
        if not isinstance(test.ops[0], ast.Eq):
            return False
        left, right = test.left, test.comparators[0]

        def _is_name(n):
            return isinstance(n, ast.Name) and n.id == "__name__"

        def _is_main(n):
            return isinstance(n, ast.Constant) and n.value == "__main__"

        return (_is_name(left) and _is_main(right)) or (_is_main(left) and _is_name(right))

    class Strip(ast.NodeTransformer):
        """Delete main-guard `if` blocks, keep their `else` suite."""

        def _flatten(self, stmts):
            out = []
            for s in stmts:
                res = self.visit(s)
                if res is None:
                    continue
                if isinstance(res, list):
                    out.extend(res)
                else:
                    out.append(res)
            return out

        def visit_If(self, node: ast.If):
            # First, transform children
            node = self.generic_visit(node)
            # If this `if` is the main guard, drop its body and keep `else`
            if _is_main_guard(node.test):
                return self._flatten(node.orelse)
            return node

    tree = ast.parse(src)
    tree = Strip().visit(tree)
    ast.fix_missing_locations(tree)

    try:
        return ast.unparse(tree)
    except Exception:
        import astor

        return astor.to_source(tree)


def instrument_asserts_and_count_with_success_tracking(src: str):
    """
    Instrument every `assert` with `__ASSERTS_RAN__ += 1` and *statically* compute
    how many times all asserts will execute, accounting for `for`-loops (including
    nesting). We only count loops whose iteration count can be determined at parse time.
    Tracks successful asserts.

    Supported iterables (statically countable):
      - range(k) / range(a, b[, step])   with integer literal args (including unary +/-)
      - literal containers: list/tuple/set/dict (length = number of elements/keys)
      - enumerate(<any of the above>)    (length = length of the underlying iterable)

    Unsupported (raise ValueError):
      - loops whose iteration count cannot be statically determined (e.g., names,
        attribute calls, comprehensions, function calls with unknown return lengths)
      - while-loops (by design)

    Returns:
      (instrumented_source: str, total_expected_asserts: int)
    """
    tree = ast.parse(src)

    # ---------- Helpers to evaluate constant integers and range lengths ----------

    def _const_int(node):
        """Return an int if `node` is an integer literal (supports unary +/-), else None."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)
        ):
            return +node.operand.value if isinstance(node.op, ast.UAdd) else -node.operand.value
        return None

    def _range_len(args):
        """
        Compute the length of range(...) given AST args already verified to be constant ints.
        Returns an int, or None if invalid (e.g., step == 0 or malformed).
        """
        if len(args) == 1:
            start, stop, step = 0, _const_int(args[0]), 1
        elif len(args) == 2:
            start, stop, step = _const_int(args[0]), _const_int(args[1]), 1
        elif len(args) == 3:
            start, stop, step = _const_int(args[0]), _const_int(args[1]), _const_int(args[2])
        else:
            return None

        if None in (start, stop, step) or step == 0:
            return None

        if step > 0:
            n = (stop - start + step - 1) // step
        else:
            # For negative steps, mirror the positive-step ceiling division logic.
            n = (start - stop - step - 1) // (-step)

        return max(0, n)

    def _static_iter_len(node):
        """
        Return the static iteration count for a `for` iterable expression, or None if unknown.

        Supports:
          - range(...)
          - enumerate(<supported iterable>)
          - literal list/tuple/set/dict
        """
        # range(...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
            # All args must be constant ints (allow unary +/-)
            ints_ok = True
            for a in node.args:
                if _const_int(a) is None:
                    ints_ok = False
                    break
            return _range_len(node.args) if ints_ok else None

        # enumerate(x)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "enumerate":
            if len(node.args) >= 1:
                return _static_iter_len(node.args[0])
            return None

        # Literal containers
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return len(node.elts)
        if isinstance(node, ast.Dict):
            return len(node.keys)

        # Unknown / dynamic
        return None

    # ---------- Node transformer with a loop-multiplier stack ----------

    class LoopAwareRewriterWithSuccess(ast.NodeTransformer):
        """
        Inserts `__INC_ASSERTS__()` before each `assert` and wraps assert in try-catch
        to track successful asserts with `__PASS_ASSERT__()`.
        """

        def __init__(self):
            super().__init__()
            self.count = 0
            self.multiplier_stack = [1]  # product of statically-known loop lengths

        def _flatten_block(self, stmts):
            """Visit a list of statements and flatten results (since visits may return lists)."""
            out = []
            for s in stmts:
                res = self.visit(s)
                if res is None:
                    continue
                if isinstance(res, list):
                    out.extend(res)
                else:
                    out.append(res)
            return out

        def visit_Assert(self, node: ast.Assert):
            self.count += self.multiplier_stack[-1]

            # Create the increment call
            inc = ast.parse("__INC_ASSERTS__()").body[0]

            # Create try-except block to track successful asserts
            try_block = ast.Try(
                body=[node],
                handlers=[
                    ast.ExceptHandler(
                        type=ast.Name(id='AssertionError', ctx=ast.Load()),
                        name=None,
                        body=[ast.parse("pass").body[0]],  # Just pass on assertion error
                    )
                ],
                orelse=[ast.parse("__PASS_ASSERT__()").body[0]],  # Call success tracker if no exception
                finalbody=[],
            )

            return [inc, try_block]

        def visit_For(self, node: ast.For):
            # Visit target and iter normally (they're expressions; will not produce statement lists).
            node.target = self.visit(node.target) or node.target
            node.iter = self.visit(node.iter) or node.iter

            # Determine static iteration count.
            iter_len = _static_iter_len(node.iter)
            if iter_len is None:
                raise ValueError("Cannot statically determine iteration count for this 'for' loop.")

            # Enter loop body with multiplied count.
            self.multiplier_stack.append(self.multiplier_stack[-1] * iter_len)
            node.body = self._flatten_block(node.body)
            self.multiplier_stack.pop()

            # `for ... else:` executes the else-block exactly once if the loop wasn't broken.
            # We conservatively count it once (no multiplier), as static break analysis is out of scope.
            node.orelse = self._flatten_block(node.orelse)
            return node

        # If you want to be explicit about unsupported async-for:
        def visit_AsyncFor(self, node: ast.AsyncFor):
            raise ValueError("Async for-loops are not supported for static assert counting.")

        # Fallback to default behavior for all other nodes.
        def generic_visit(self, node):
            return super().generic_visit(node)

        def visit_While(self, node: ast.While):
            raise ValueError("While-loops are not supported for static assert counting.")

    # Transform and unparse
    rw = LoopAwareRewriterWithSuccess()
    new_tree = rw.visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        new_src = ast.unparse(new_tree)  # Python 3.9+
    except Exception:
        import astor

        new_src = astor.to_source(new_tree)

    return new_src, rw.count


def build_hardened_code(
    solution_str: str,
    import_code: str,
    test_code: str,
):
    """
    Build a single Python program that:
      1) blocks premature exits (sys.exit, os._exit, quit/exit, self-kill)
      2) runs instrumented tests and reports success/failure counts
      3) prints detailed results including passed and failed assert counts
    """
    nonce = secrets.token_hex(16)

    # Preprocess the user solution OUTSIDE the generated program
    solution_no_main = strip_main_guard(solution_str)
    filtered_import_code = prune_import_code(import_code, solution_no_main, test_code).strip()

    # Instrument the test case with success tracking
    tests_instr, expected = instrument_asserts_and_count_with_success_tracking(test_code)
    tests_instr_indented = textwrap.indent(tests_instr, "    ")

    RUN_USER = f"{filtered_import_code}\n\n{solution_no_main}" if filtered_import_code else solution_no_main

    RUN_TESTS = f"""
# ==== RUN INSTRUMENTED TESTS ====
__ASSERTS_RAN__ = 0
__ASSERTS_PASSED__ = 0

def __INC_ASSERTS__():
    # Called before each assert in the instrumented tests
    global __ASSERTS_RAN__
    __ASSERTS_RAN__ += 1

def __PASS_ASSERT__():
    global __ASSERTS_PASSED__
    __ASSERTS_PASSED__ += 1

try:
    # ---- tests begin ----
{tests_instr_indented}
    # ---- tests end ----

    # Check if all asserts ran and report results
    if __ASSERTS_RAN__ != {expected}:
        print(f"__TEST_RESULT__:ERROR:only_ran_{{__ASSERTS_RAN__}}/{expected}_asserts")
    else:
        failed_asserts = __ASSERTS_RAN__ - __ASSERTS_PASSED__
        print(f"__TEST_RESULT__:SUCCESS:{{__ASSERTS_PASSED__}}_passed_{{failed_asserts}}_failed/{expected}_total")

except Exception as e:
    print(f"__TEST_RESULT__:ERROR:{{str(e)}}")

# Print final summary
print("__EVAL_OK__:{nonce}:1_test_{expected}_total_asserts")
"""

    return RUN_USER + RUN_TESTS


def build_io_validation_code(solution_str: str, inputs, outputs):
    solution_no_main = strip_main_guard(solution_str)

    inputs_literal = ",\n    ".join(repr(str(item)) for item in inputs)
    outputs_literal = ",\n    ".join(repr(str(item)) for item in outputs)

    return f"""
import io
import sys
import traceback

SOLUTION_CODE = {repr(solution_no_main)}

TEST_INPUTS = [
    {inputs_literal}
]

EXPECTED_OUTPUTS = [
    {outputs_literal}
]


def __run_solution__(stdin_text):
    stdin_backup = sys.stdin
    stdout_backup = sys.stdout
    sys.stdin = io.StringIO(stdin_text)
    buffer = io.StringIO()
    sys.stdout = buffer
    namespace = {{"__name__": "__main__", "__file__": "<solution>"}}

    try:
        exec(SOLUTION_CODE, namespace)
    except SystemExit:
        pass
    except Exception:
        sys.stdout = stdout_backup
        sys.stdin = stdin_backup
        print("__TEST_RESULT__:ERROR:solution_exception")
        traceback.print_exc()
        return None, False
    finally:
        sys.stdout = stdout_backup
        sys.stdin = stdin_backup

    return buffer.getvalue(), True


if len(TEST_INPUTS) != len(EXPECTED_OUTPUTS):
    print("__TEST_RESULT__:ERROR:mismatched_test_lengths")
else:
    passed = 0
    total = len(TEST_INPUTS)

    for stdin_text, expected_output in zip(TEST_INPUTS, EXPECTED_OUTPUTS):
        output_text, ok = __run_solution__(stdin_text)
        if not ok:
            break

        normalized_output = "" if output_text is None else str(output_text)
        expected_normalized = "" if expected_output is None else str(expected_output)

        if normalized_output.strip() == expected_normalized.strip():
            passed += 1
    else:
        failed = total - passed
        print(f"__TEST_RESULT__:SUCCESS:{{passed}}_passed_{{failed}}_failed/{{total}}_total")
"""


def extract_solution(solution_str, pattern):
    model_output = re.sub(
        r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL, count=1
    )
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "[END]"]
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()

    # # only extract the last 5000 chars from the end to avoid reward hacking
    # model_output = model_output[-5000:]

    # Parse code actions from responses
    matches = list(re.finditer(pattern, model_output))
    if matches:
        return matches[-1].group(1).strip()
    else:
        return None


def compute_score(solution_str, ground_truth, extra_info, pattern):
    extract_answer = extract_solution(solution_str=solution_str, pattern=pattern)

    if isinstance(extra_info, np.ndarray):
        extra_info = extra_info.item()

    has_code_piece = extract_answer is not None
    if not has_code_piece:
        return {
            "score": 0.0,
            "extra_info": {
                "score": 0.0,
                "has_code": 0,
                "valid_code": 0,
                "sandbox_failed": 0,
            },
        }

    if (
        isinstance(ground_truth, dict)
        and ground_truth.get("import_code", None) is not None
        and ground_truth.get("test_code", None) is not None
    ):
        try:
            # Build single code block that executes one test case
            code_to_execute = build_hardened_code(
                solution_str=extract_answer,
                import_code=ground_truth["import_code"],
                test_code=ground_truth["test_code"],
            )
        except Exception:
            # If building code fails (e.g., syntax error), return 0 score
            return {
                "score": 0.0,
                "extra_info": {
                    "score": 0.0,
                    "has_code": 1,
                    "valid_code": 0,
                    "sandbox_failed": 0,
                },
            }

        # Execute single code block
        run_result = _run_with_sandbox_fallback(code_to_execute)

        stdout_str = "" if run_result["stdout"] is None else str(run_result["stdout"])
        stderr_str = "" if run_result["stderr"] is None else str(run_result["stderr"])

        if run_result["return_code"] in [0, 1]:
            if len(stderr_str) != 0:
                return {
                    "score": 0.0,
                    "extra_info": {
                        "score": 0.0,
                        "has_code": 1,
                        "valid_code": 0,
                        "sandbox_failed": 0,
                    },
                }
            # Parse single test result: __TEST_RESULT__:SUCCESS:X_passed_Y_failed/Z_total
            test_pattern = r"__TEST_RESULT__:SUCCESS:(\d+)_passed_(\d+)_failed/(\d+)_total"
            m = re.search(test_pattern, stdout_str)
            if m:
                passed_asserts = int(m.group(1))
                failed_asserts = int(m.group(2))
                total_asserts = int(m.group(3))

                # Calculate score based on passed asserts
                score = passed_asserts / total_asserts if total_asserts > 0 else 0.0

                return {
                    "score": score,
                    "extra_info": {
                        "score": 0.0 if score < 1.0 else 1.0,
                        "has_code": 1,
                        "valid_code": 1,
                        "sandbox_failed": 0,
                    },
                }
            else:
                # No valid test result found
                return {
                    "score": 0.0,
                    "extra_info": {
                        "score": 0.0,
                        "has_code": 1,
                        "valid_code": 0,
                        "sandbox_failed": 0,
                    },
                }
        else:
            # Execution failed
            return {
                "score": 0.0,
                "extra_info": {
                    "score": 0.0,
                    "has_code": 1,
                    "valid_code": 0,
                    "is_filter": 1,
                    "sandbox_failed": 1,
                },
            }

    elif isinstance(ground_truth, dict) and "inputs" in ground_truth and "outputs" in ground_truth:
        stdin_list = []
        gt_stdout_list = []
        for test_stdin, test_stdout in zip(ground_truth["inputs"], ground_truth["outputs"]):
            if isinstance(test_stdin, np.ndarray):
                test_stdin = test_stdin.item()
            stdin_list.append(test_stdin)
            if isinstance(test_stdout, np.ndarray):
                test_stdout = test_stdout.item()
            gt_stdout_list.append(test_stdout)

        try:
            code_to_execute = build_io_validation_code(
                solution_str=extract_answer,
                inputs=stdin_list,
                outputs=gt_stdout_list,
            )
        except Exception:
            return {
                "score": 0.0,
                "extra_info": {
                    "score": 0.0,
                    "has_code": 1,
                    "valid_code": 0,
                    "sandbox_failed": 0,
                },
            }

        run_result = _run_with_sandbox_fallback(code_to_execute)

        stdout_str = "" if run_result["stdout"] is None else str(run_result["stdout"])
        stderr_str = "" if run_result["stderr"] is None else str(run_result["stderr"])

        if run_result["return_code"] in [0, 1]:
            if len(stderr_str) != 0:
                return {
                    "score": 0.0,
                    "extra_info": {
                        "score": 0.0,
                        "has_code": 1,
                        "valid_code": 0,
                        "sandbox_failed": 0,
                    },
                }
            test_pattern = r"__TEST_RESULT__:SUCCESS:(\d+)_passed_(\d+)_failed/(\d+)_total"
            m = re.search(test_pattern, stdout_str)
            if m:
                passed_tests = int(m.group(1))
                total_tests = int(m.group(3))
                score = passed_tests / total_tests if total_tests > 0 else 0.0
                return {
                    "score": score,
                    "extra_info": {
                        "score": 0.0 if score < 1.0 else 1.0,
                        "has_code": 1,
                        "valid_code": 1,
                        "sandbox_failed": 0,
                    },
                }
            else:
                # No valid test result found
                return {
                    "score": 0.0,
                    "extra_info": {
                        "score": 0.0,
                        "has_code": 1,
                        "valid_code": 0,
                        "sandbox_failed": 0,
                    },
                }
        else:
            # Execution failed
            return {
                "score": 0.0,
                "extra_info": {
                    "score": 0.0,
                    "has_code": 1,
                    "valid_code": 0,
                    "sandbox_failed": 1,
                },
            }

    else:
        raise ValueError(f"Ground truth should be a dict, but got {type(ground_truth)}")
