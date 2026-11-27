"""Grade generated code (runs ON a throwaway box: generated code is executed). For every outputs/<arm>/<id>.r0.txt:
extract the code (the fenced block that defines the entry point, else the first fenced block, else the raw text; an
unclosed fence = truncated reply, taken to the end), cut a top-level `if __name__ == "__main__":` tail, prepend the
HumanEval prompt when the entry point is not defined, append the tests, run `python3 -I` in a fresh temp dir with a 10 s
timeout and a 2 GB address-space cap. Pass = exit 0. Same rules for every arm.
Two grades per reply: `pass` = EvalPlus-style sanitized (drop trailing lines until the code parses, i.e. a reply cut at
the token cap keeps its complete part; keep only imports, defs, classes and assignments, so the model's own example
asserts/prints do not count); `pass_raw` = the code exactly as written (its own asserts included).
    python3 evaluate.py problems.jsonl <results dir> > <results dir>/graded.jsonl
"""
import ast
import io
import json
import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
import tokenize

# A closing fence occupies its own line and is at least as long as the opener.
# Inline backticks and shorter fences are part of the generated Python source.
FENCE = re.compile(r"^ {0,3}(`{3,})[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n"
                   r"(.*?)(?:^ {0,3}\1`*[ \t]*(?:\r?\n|\Z)|\Z)", re.M | re.S)


def extract(text: str, entry: str) -> str:
    blocks = [b for _, lang, b in FENCE.findall(text) if lang.lower() in ("", "python", "py", "python3")]
    if not blocks:
        return text
    for b in blocks:
        if entry and re.search(rf"^\s*def {re.escape(entry)}\s*\(", b, re.M):
            return b
    return blocks[0]


def sanitize(code: str) -> str:
    lines = code.splitlines()
    while lines:
        try:
            tree = ast.parse("\n".join(lines))
            break
        except SyntaxError:
            lines.pop()
    else:
        return ""
    keep = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)
    # Function/class source segments start at the definition, after any decorators.
    # Unparse the complete node so sanitized grading keeps decorator semantics.
    return "\n".join(ast.unparse(n) for n in tree.body if isinstance(n, keep))


def before_main(code: str) -> str:
    # A guard-looking line can be inside a string or docstring. Only a real
    # column-zero `if` token starts the top-level demo we want to discard.
    lines = code.splitlines(keepends=True)
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type == tokenize.NAME and token.string == "if" and token.start[1] == 0:
                if re.match(r"if __name__\s*==\s*['\"]__main__['\"]\s*:", token.line):
                    return "".join(lines[:token.start[0] - 1])
    except (tokenize.TokenError, IndentationError):
        pass  # sanitize() handles incomplete replies; raw grading keeps their errors.
    return code


def program(p: dict, text: str, clean: bool) -> str:
    code = extract(text, p["entry_point"])
    code = before_main(code)
    if clean:
        code = sanitize(code)
    if p["entry_point"] and not re.search(rf"^\s*def {re.escape(p['entry_point'])}\s*\(", code, re.M):
        code = p["src_prompt"] + code
    return code + "\n\n" + p["test"]


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    os.setsid()


def run(src: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "prog.py")
        open(f, "w").write(src)
        with subprocess.Popen([sys.executable, "-I", f], cwd=d, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              preexec_fn=limits, stdin=subprocess.DEVNULL, text=True) as proc:
            try:
                try:
                    _, stderr = proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.communicate()
                    return False, "timeout"
                if proc.returncode == 0:
                    return True, ""
                err = (stderr.strip().splitlines() or ["rc %d" % proc.returncode])[-1]
                return False, err[:200]
            finally:
                # limits() creates a session; children belong to this evaluation too.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()


def main():
    probs = {p["id"]: p for p in map(json.loads, open(sys.argv[1]))}
    root = os.path.join(sys.argv[2], "outputs")
    for arm in sorted(os.listdir(root)):
        for fn in sorted(os.listdir(os.path.join(root, arm))):
            pid = fn.rsplit(".r", 1)[0]
            text = open(os.path.join(root, arm, fn)).read()
            ok, err = run(program(probs[pid], text, True))
            ok_raw, err_raw = run(program(probs[pid], text, False))
            print(json.dumps({"arm": arm, "prompt_id": pid, "pass": ok, "err": err, "pass_raw": ok_raw,
                              "err_raw": err_raw}), flush=True)


if __name__ == "__main__":
    main()
