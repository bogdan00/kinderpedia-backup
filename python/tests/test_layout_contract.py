"""Golden layout-contract conformance (shared with `bookmarklet/tests/layout_contract.test.mjs`).

Both the CLI and the Bookmarklet must produce byte-identical folder/file names so archives stay
cross-resumable. Any intentional change must update `fixtures/layout_contract.json` and both
implementations together.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any

from kinderpedia_backup.models import ChildContext
from kinderpedia_backup.sync import (
    album_folder_from_leaf,
    assign_album_folder_names,
    parse_date_prefix,
    plan_video_targets,
    resolve_child_folder_names,
    sanitize_name,
    stem_similarity,
)
from kinderpedia_backup.sync.naming import _normalize_match_text

FIXTURE = Path(__file__).parent / "fixtures" / "layout_contract.json"
CONTRACT: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))


def _int_keys(d: dict[str, Any]) -> dict[int, Any]:
    return {int(k): v for k, v in d.items()}


class LayoutContractTests(unittest.TestCase):
    def test_sanitize_name(self) -> None:
        for case in CONTRACT["sanitize_name"]:
            with self.subTest(case=case["input"]):
                self.assertEqual(sanitize_name(case["input"], case.get("max_len", 60)), case["expected"])

    def test_parse_date_prefix(self) -> None:
        for case in CONTRACT["parse_date_prefix"]:
            with self.subTest(case=case["input"]):
                prefix, ts = parse_date_prefix(case["input"])
                self.assertEqual(prefix, case["prefix"])
                expected = case["timestamp"]
                if expected == "local":
                    expected = datetime(*case["local"]).timestamp()
                if expected is None:
                    self.assertIsNone(ts)
                else:
                    assert ts is not None
                    self.assertAlmostEqual(ts, float(expected), places=3)

    def test_normalize_match_text_and_stem_similarity(self) -> None:
        for case in CONTRACT["normalize_match_text"]:
            with self.subTest(case=case["input"]):
                self.assertEqual(_normalize_match_text(case["input"]), case["expected"])
        for case in CONTRACT["stem_similarity"]:
            with self.subTest(case=case):
                self.assertEqual(stem_similarity(case["video"], case["album"]), case["expected"])

    def test_resolve_child_folder_names(self) -> None:
        for case in CONTRACT["resolve_child_folder_names"]:
            contexts = [ChildContext(**c) for c in case["contexts"]]
            got = {f"{cid}_{kid}": name for (cid, kid), name in resolve_child_folder_names(contexts).items()}
            self.assertEqual(got, case["expected"])

    def test_assign_album_folder_names(self) -> None:
        for case in CONTRACT["assign_album_folder_names"]:
            with self.subTest(case=case["name"]):
                got = assign_album_folder_names(case["albums"], _int_keys(case["existing"]))
                self.assertEqual(got, _int_keys(case["expected"]))

    def test_plan_video_targets(self) -> None:
        for case in CONTRACT["plan_video_targets"]:
            with self.subTest(case=case["name"]):
                folders = assign_album_folder_names(case["albums"], _int_keys(case["existing_albums"]))
                existing_videos = {k: (v[0], v[1]) for k, v in _int_keys(case["existing_videos"]).items()}
                got = plan_video_targets(case["videos"], case["albums"], folders, existing_videos)
                expected = {k: (v[0], v[1]) for k, v in _int_keys(case["expected"]).items()}
                self.assertEqual(got, expected)

    def test_album_folder_from_leaf(self) -> None:
        for case in CONTRACT["album_folder_from_leaf"]:
            self.assertEqual(album_folder_from_leaf(case["input"]), case["expected"])


if __name__ == "__main__":
    unittest.main()
