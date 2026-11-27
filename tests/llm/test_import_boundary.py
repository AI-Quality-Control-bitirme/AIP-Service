"""The Gemini SDK may only be imported by the Gemini adapter, so another provider can be added
without touching agent code (SoW §14)."""

import ast
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[2] / "app"
ALLOWED_DIR = APP_DIR / "llm" / "providers" / "gemini"


def imports_genai(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "google.genai" or alias.name.startswith("google.genai.") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "google.genai" or node.module.startswith("google.genai."):
                return True
            if node.module == "google" and any(alias.name == "genai" for alias in node.names):
                return True
    return False


def files_importing_genai() -> list[Path]:
    return [
        path
        for path in APP_DIR.rglob("*.py")
        if imports_genai(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    ]


def test_genai_imported_only_in_gemini_adapter():
    offenders = [path for path in files_importing_genai() if not path.is_relative_to(ALLOWED_DIR)]

    assert offenders == []


def test_boundary_check_is_not_vacuous():
    assert ALLOWED_DIR / "adapter.py" in files_importing_genai()
