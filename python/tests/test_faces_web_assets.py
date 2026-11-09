"""The faces web UI is a tree of vanilla ES modules: every file index.html / the modules reference must
ship as package data (`[tool.setuptools.package-data]` in pyproject.toml) and exist in the package."""

from __future__ import annotations

import posixpath
import re
import unittest
from importlib import resources
from pathlib import Path
from typing import Any

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
WEB_PREFIX = "faces/web"

_SCRIPT_SRC = re.compile(r"""<(?:script|link)\b[^>]*?\b(?:src|href)\s*=\s*["']/assets/([^"']+)["']""")
_IMPORT_FROM = re.compile(r"""(?:^|[;\s])(?:import|export)\b[^'"`;]*?\bfrom\s*["']([^"']+)["']""", re.M)
_IMPORT_BARE = re.compile(r"""(?:^|[;\s])import\s*["']([^"']+)["']""", re.M)
_IMPORT_DYNAMIC = re.compile(r"""\bimport\(\s*["']([^"']+)["']\s*\)""")


def _web() -> Any:
    return resources.files("kinderpedia_backup").joinpath(WEB_PREFIX)


def _referenced_assets() -> dict[str, set[str]]:
    """Map every asset path (relative to the web dir) -> the files that reference it."""
    web = _web()
    refs: dict[str, set[str]] = {}
    html = web.joinpath("index.html").read_text(encoding="utf-8")
    queue = []
    for rel in _SCRIPT_SRC.findall(html):
        refs.setdefault(rel, set()).add("index.html")
        queue.append(rel)
    seen: set[str] = set()
    while queue:
        rel = queue.pop()
        if rel in seen or not rel.endswith(".js"):
            continue
        seen.add(rel)
        node = web
        for part in rel.split("/"):
            node = node.joinpath(part)
        if not node.is_file():
            continue  # reported by the existence assertion
        src = node.read_text(encoding="utf-8")
        for spec in _IMPORT_FROM.findall(src) + _IMPORT_BARE.findall(src) + _IMPORT_DYNAMIC.findall(src):
            if not spec.startswith((".", "/")):
                raise AssertionError(f"{rel}: bare/remote import {spec!r} (the UI has zero dependencies)")
            target = posixpath.normpath(posixpath.join(posixpath.dirname(rel), spec))
            if target.startswith(".."):
                raise AssertionError(f"{rel}: import {spec!r} escapes the web dir")
            refs.setdefault(target, set()).add(rel)
            queue.append(target)
    return refs


def _package_data_globs() -> list[str]:
    # (no tomllib on py3.10) the table holds one `kinderpedia_backup = [...]` line
    text = PYPROJECT.read_text(encoding="utf-8")
    table = text.split("[tool.setuptools.package-data]", 1)[1].split("\n[", 1)[0]
    m = re.search(r"^kinderpedia_backup\s*=\s*\[([^\]]*)\]", table, re.M)
    assert m, "package-data table not found"
    return re.findall(r'"([^"]+)"', m.group(1))


def _glob_matches(pattern: str, path: str) -> bool:
    """setuptools package-data semantics: `*` stays within a segment, `**/` spans zero or more dirs."""
    rx = ""
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            rx += "(?:[^/]+/)*"
            i += 3
        elif pattern[i] == "*":
            rx += "[^/]*"
            i += 1
        else:
            rx += re.escape(pattern[i])
            i += 1
    return re.fullmatch(rx, path) is not None


class FacesWebAssetsTests(unittest.TestCase):
    def test_every_referenced_module_exists_and_ships_as_package_data(self) -> None:
        refs = _referenced_assets()
        self.assertIn("app.js", refs)
        self.assertIn("styles.css", refs)
        self.assertIn("views/batch.js", refs)  # nested modules really are reached from the entry point
        globs = _package_data_globs()
        web = _web()
        for rel, users in sorted(refs.items()):
            node = web
            for part in rel.split("/"):
                node = node.joinpath(part)
            self.assertTrue(node.is_file(), f"{rel} (referenced by {sorted(users)}) is missing")
            pkg_path = f"{WEB_PREFIX}/{rel}"
            self.assertTrue(
                any(_glob_matches(g, pkg_path) for g in globs),
                f"{pkg_path} is not covered by package-data globs {globs}",
            )
        self.assertTrue(any(_glob_matches(g, f"{WEB_PREFIX}/index.html") for g in globs))

    def test_every_module_is_reachable_and_js_tests_are_not_shipped(self) -> None:
        web_dir = Path(str(_web()))
        on_disk = {
            p.relative_to(web_dir).as_posix()
            for p in web_dir.rglob("*.js")
            if p.relative_to(web_dir).parts[0] != "tests"
        }
        self.assertEqual(on_disk - set(_referenced_assets()), set(), "unreferenced (dead) modules")
        globs = _package_data_globs()
        for test_file in (web_dir / "tests").glob("*"):
            pkg_path = f"{WEB_PREFIX}/tests/{test_file.name}"
            self.assertFalse(any(_glob_matches(g, pkg_path) for g in globs), f"{pkg_path} would ship")

    def test_glob_semantics(self) -> None:
        self.assertTrue(_glob_matches("faces/web/**/*.js", "faces/web/app.js"))
        self.assertTrue(_glob_matches("faces/web/**/*.js", "faces/web/views/batch.js"))
        self.assertFalse(_glob_matches("faces/web/**/*.js", "faces/web/tests/api.test.mjs"))
        self.assertFalse(_glob_matches("faces/web/*.css", "faces/web/sub/x.css"))


if __name__ == "__main__":
    unittest.main()
