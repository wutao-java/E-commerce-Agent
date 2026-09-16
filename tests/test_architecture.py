"""验证核心模块不反向依赖 HTTP 传输层。"""

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_PACKAGES = ("agent", "domain", "llm")


def imported_root_names(path: Path) -> set[str]:
    """返回 Python 文件直接导入的顶层包名。"""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_core_packages_do_not_depend_on_web() -> None:
    """Agent、领域契约和 LLM 适配层不能导入 Web 层。"""

    violations = [
        str(path.relative_to(PROJECT_ROOT))
        for package in CORE_PACKAGES
        for path in (PROJECT_ROOT / package).rglob("*.py")
        if "web" in imported_root_names(path)
    ]

    assert violations == []
