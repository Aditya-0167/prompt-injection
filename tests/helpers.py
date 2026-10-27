"""Helpers for the test suite. The experiment script is a single Colab cell with side effects at import,
so the tests extract only the pure functions and constants they need from its source."""
import ast, json, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "code" / "run_experiment_colab.py"


def load_script_symbols(names):
    wanted = set(names)
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    ns = {"re": re, "json": json}
    for node in tree.body:
        hit = isinstance(node, ast.FunctionDef) and node.name in wanted
        if isinstance(node, ast.Assign):
            hit = any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets)
        if hit:
            exec(compile(ast.Module([node], []), str(SCRIPT), "exec"), ns)
    missing = wanted - set(ns)
    assert not missing, f"not found in script: {missing}"
    return ns
