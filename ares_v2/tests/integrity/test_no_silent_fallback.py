"""Integrity Test: Enforce No Silent Fallbacks and No Bare Excepts.

Scans AST of all Python code in src/ to detect:
1. Bare 'except:'
2. 'except Exception: pass' or 'except: pass' without logging or reraise.
"""

import ast
from pathlib import Path
from ares.common.paths import ARES_ROOT


class FallbackVisitor(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.violations: list[str] = []

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        # 1. Bare except:
        if node.type is None:
            self.violations.append(
                f"{self.filename}:{node.lineno}: Bare 'except:' detected. Must catch specific exceptions."
            )
            
        # 2. except ...: pass
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            self.violations.append(
                f"{self.filename}:{node.lineno}: Silent exception pass detected. Must log or re-raise."
            )
            
        self.generic_visit(node)


def test_no_silent_fallback_ast():
    src_dir = ARES_ROOT / "ares_v2" / "src"
    violations = []
    
    for py_file in src_dir.rglob("*.py"):
        if "legacy" in py_file.parts:
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(py_file))
            visitor = FallbackVisitor(str(py_file.relative_to(ARES_ROOT)))
            visitor.visit(tree)
            violations.extend(visitor.violations)
        except SyntaxError as e:
            violations.append(f"{py_file}: SyntaxError: {e}")
            
    assert len(violations) == 0, (
        f"Found {len(violations)} silent fallback violations:\n" + "\n".join(violations)
    )
