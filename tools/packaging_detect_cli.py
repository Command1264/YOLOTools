"""
CLI tool to detect packaging hints for PyInstaller and Nuitka.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class DetectResult:
    """Detection result container.

    Args:
        hidden_imports (List[str]): Suggested hidden imports.
        collect_submodules (List[str]): Suggested collect-submodules targets.
        nuitka_plugins (List[str]): Suggested Nuitka plugins.
        nuitka_includes (List[str]): Suggested Nuitka include-modules.
        static_imports (List[str]): Imports found by static scan.
        runtime_imports (List[str]): Imports found during runtime trace.
        errors (List[str]): Error messages during detection.
    """

    hidden_imports: List[str]
    collect_submodules: List[str]
    nuitka_plugins: List[str]
    nuitka_includes: List[str]
    static_imports: List[str]
    runtime_imports: List[str]
    errors: List[str]


PLUGIN_MAP = {
    "PySide6": "pyside6",
    "PyQt5": "pyqt5",
    "PyQt6": "pyqt6",
    "tkinter": "tk-inter",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect packaging hints for PyInstaller/Nuitka.",
    )
    parser.add_argument(
        "entry",
        help="Entry python file path.",
    )
    parser.add_argument(
        "--packager",
        choices=["pyinstaller", "nuitka", "both"],
        default="both",
        help="Target packager.",
    )
    parser.add_argument(
        "--project-root",
        default="",
        help="Project root for static scan. Defaults to entry's parent.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run entry once and capture runtime imports.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON for GUI integration.",
    )
    return parser.parse_args()


def _scan_imports_in_file(path: Path) -> Tuple[Set[str], Set[str], Set[str]]:
    static_imports: Set[str] = set()
    dynamic_imports: Set[str] = set()
    module_targets: Set[str] = set()
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return static_imports, dynamic_imports, module_targets
    try:
        tree = ast.parse(text, filename=str(path))
    except Exception:
        return static_imports, dynamic_imports, module_targets
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split(".")[0]
                static_imports.add(name)
                module_targets.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                name = node.module.split(".")[0]
                static_imports.add(name)
            if node.level and node.level > 0:
                if node.module:
                    module_targets.add("." * node.level + node.module)
                else:
                    for alias in node.names:
                        module_targets.add("." * node.level + alias.name)
            elif node.module:
                module_targets.add(node.module)
        elif isinstance(node, ast.Call):
            func_name = _resolve_call_name(node.func)
            if func_name in {"importlib.import_module", "__import__"}:
                mod = _first_str_arg(node)
                if mod:
                    dynamic_imports.add(mod.split(".")[0])
    return static_imports, dynamic_imports, module_targets


def _resolve_call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}"
    return ""


def _first_str_arg(node: ast.Call) -> str:
    if not node.args:
        return ""
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return ""


def _collect_static_imports(entry: Path, root: Path) -> Tuple[Set[str], Set[str]]:
    static_imports: Set[str] = set()
    dynamic_imports: Set[str] = set()
    visited: Set[Path] = set()
    queue: List[Path] = []
    entry_path = _resolve_entry_path(entry, root)
    if entry_path:
        queue.append(entry_path)
    while queue:
        current = queue.pop()
        if current in visited:
            continue
        visited.add(current)
        st, dyn, targets = _scan_imports_in_file(current)
        static_imports.update(st)
        dynamic_imports.update(dyn)
        for target in targets:
            next_path = _resolve_module_target(target, root, current)
            if next_path and next_path not in visited:
                queue.append(next_path)
    return static_imports, dynamic_imports


def _resolve_entry_path(entry: Path, root: Path) -> Optional[Path]:
    if entry.exists():
        return entry
    if entry.suffix != ".py":
        entry = entry.with_suffix(".py")
    if entry.exists():
        return entry
    return _resolve_module_target(entry.stem, root, None)


def _resolve_module_target(target: str, root: Path, current: Optional[Path]) -> Optional[Path]:
    if target.startswith("."):
        if current is None:
            return None
        return _resolve_relative_module_target(current, target, root)
    parts = target.split(".")
    candidate = root.joinpath(*parts)
    if candidate.is_dir():
        init_file = candidate / "__init__.py"
        if init_file.exists():
            return init_file
    py_file = candidate.with_suffix(".py")
    if py_file.exists():
        return py_file
    return None


def _resolve_relative_module_target(current: Path, target: str, root: Path) -> Optional[Path]:
    level = len(target) - len(target.lstrip("."))
    rel_module = target[level:]
    base = current.parent
    for _ in range(level - 1):
        base = base.parent
    if base is None:
        return None
    if rel_module:
        parts = rel_module.split(".")
        candidate = base.joinpath(*parts)
    else:
        candidate = base
    if candidate.is_dir():
        init_file = candidate / "__init__.py"
        if init_file.exists():
            return init_file
    py_file = candidate.with_suffix(".py")
    if py_file.exists():
        return py_file
    return None


def _run_trace_imports(entry: Path) -> Tuple[Set[str], List[str]]:
    errors: List[str] = []
    if not entry.exists():
        return set(), [f"Entry not found: {entry}"]
    helper = _build_runtime_helper()
    with tempfile.TemporaryDirectory() as tmp_dir:
        helper_path = Path(tmp_dir) / "import_trace_runner.py"
        output_path = Path(tmp_dir) / "imports.json"
        helper_path.write_text(helper, encoding="utf-8")
        cmd = [
            sys.executable,
            str(helper_path),
            str(entry),
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=False)
        except Exception as exc:
            errors.append(f"Failed to run trace: {exc}")
            return set(), errors
        if not output_path.exists():
            errors.append("Trace output not found.")
            return set(), errors
        try:
            data = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"Failed to read trace output: {exc}")
            return set(), errors
    imports = {str(item).split(".")[0] for item in data.get("imports", []) if item}
    return imports, errors


def _build_runtime_helper() -> str:
    return (
        "import builtins\n"
        "import importlib\n"
        "import json\n"
        "import runpy\n"
        "import sys\n"
        "\n"
        "entry = sys.argv[1]\n"
        "out_path = sys.argv[2]\n"
        "captured = set()\n"
        "\n"
        "orig_import = builtins.__import__\n"
        "def tracing_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
        "    if name:\n"
        "        captured.add(name)\n"
        "    return orig_import(name, globals, locals, fromlist, level)\n"
        "builtins.__import__ = tracing_import\n"
        "\n"
        "orig_import_module = importlib.import_module\n"
        "def tracing_import_module(name, package=None):\n"
        "    if name:\n"
        "        captured.add(name)\n"
        "    return orig_import_module(name, package)\n"
        "importlib.import_module = tracing_import_module\n"
        "\n"
        "try:\n"
        "    runpy.run_path(entry, run_name='__main__')\n"
        "except Exception:\n"
        "    pass\n"
        "\n"
        "with open(out_path, 'w', encoding='utf-8') as f:\n"
        "    json.dump({'imports': sorted(captured)}, f, ensure_ascii=False)\n"
    )


def _suggest_plugins(imports: Set[str]) -> List[str]:
    plugins = []
    for key, plugin in PLUGIN_MAP.items():
        if key in imports and plugin not in plugins:
            plugins.append(plugin)
    return plugins


def _build_result(
    static_imports: Set[str],
    dynamic_imports: Set[str],
    runtime_imports: Set[str],
    errors: List[str],
) -> DetectResult:
    hidden_imports = sorted(dynamic_imports | (runtime_imports - static_imports))
    collect_submodules = sorted(dynamic_imports)
    combined = static_imports | runtime_imports
    return DetectResult(
        hidden_imports=hidden_imports,
        collect_submodules=collect_submodules,
        nuitka_plugins=_suggest_plugins(combined),
        nuitka_includes=sorted(hidden_imports),
        static_imports=sorted(static_imports),
        runtime_imports=sorted(runtime_imports),
        errors=errors,
    )


def _as_json(result: DetectResult) -> Dict[str, object]:
    return {
        "version": 1,
        "hidden_imports": result.hidden_imports,
        "collect_submodules": result.collect_submodules,
        "nuitka_plugins": result.nuitka_plugins,
        "nuitka_includes": result.nuitka_includes,
        "static_imports": result.static_imports,
        "runtime_imports": result.runtime_imports,
        "errors": result.errors,
    }


def _print_text(result: DetectResult, packager: str) -> None:
    if result.errors:
        print("Errors:")
        for err in result.errors:
            print(f"- {err}")
        print("")
    if packager in {"pyinstaller", "both"}:
        print("[PyInstaller]")
        print("hidden-imports:")
        for item in result.hidden_imports:
            print(f"- {item}")
        print("collect-submodules:")
        for item in result.collect_submodules:
            print(f"- {item}")
        print("")
    if packager in {"nuitka", "both"}:
        print("[Nuitka]")
        print("plugins:")
        for item in result.nuitka_plugins:
            print(f"- {item}")
        print("include-modules:")
        for item in result.nuitka_includes:
            print(f"- {item}")
        print("")


def main() -> int:
    """Run CLI detection workflow.

    Returns:
        int: Exit code.
    """

    args = _parse_args()
    entry = Path(args.entry).resolve()
    if args.project_root:
        root = Path(args.project_root).resolve()
    else:
        root = entry.parent
    static_imports, dynamic_imports = _collect_static_imports(entry, root)
    runtime_imports: Set[str] = set()
    errors: List[str] = []
    if args.run:
        runtime_imports, run_errors = _run_trace_imports(entry)
        errors.extend(run_errors)
    result = _build_result(static_imports, dynamic_imports, runtime_imports, errors)
    if args.json:
        print(json.dumps(_as_json(result), ensure_ascii=False, indent=2))
        return 0
    _print_text(result, args.packager)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
