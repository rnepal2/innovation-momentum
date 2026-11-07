"""Local article and research links remain valid after directory moves."""

import importlib.util
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class LocalLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {"src", "href"} and value:
                self.targets.append(value)


class RepositoryLayoutTests(unittest.TestCase):
    def assert_links_exist(self, source, targets):
        for target in targets:
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            path = (ROOT if url.path.startswith("/") else source.parent) / unquote(url.path).lstrip(
                "/"
            )
            with self.subTest(source=source.relative_to(ROOT), target=target):
                self.assertTrue(path.exists(), f"Missing local link: {path}")

    @unittest.skipUnless(importlib.util.find_spec("markdown_it"), "Requires research extra")
    def test_markdown_links(self):
        from markdown_it import MarkdownIt

        parser = MarkdownIt("commonmark").enable("table")
        paths = [ROOT / "README.md", ROOT / "data/README.md", ROOT / "reports/evidence/README.md"]
        paths += list((ROOT / "research").rglob("*.md"))
        paths += [
            p
            for p in (ROOT / "traffic").rglob("*.md")
            if not {"data", "artifacts"}.intersection(p.relative_to(ROOT / "traffic").parts)
        ]
        for source in paths:
            targets = []
            for token in parser.parse(source.read_text()):
                for child in token.children or []:
                    if child.type in {"link_open", "image"}:
                        targets.append(child.attrGet("href") or child.attrGet("src"))
            self.assert_links_exist(source, targets)

    def test_article_links(self):
        for source in [
            ROOT / "index.html",
            ROOT / "reports/index.html",
            ROOT / "traffic/manuscript.html",
        ]:
            parser = LocalLinks()
            parser.feed(source.read_text())
            self.assert_links_exist(source, parser.targets)

    def test_study_directories_are_separate(self):
        self.assertFalse((ROOT / "research/2026-10").exists())
        self.assertFalse((ROOT / "research/traffic").exists())
        self.assertTrue((ROOT / "research/campaign/common.py").is_file())
        self.assertTrue((ROOT / "research/adoption/reproduce.py").is_file())
        self.assertTrue((ROOT / "traffic/tests/test_traffic_amplification.py").is_file())


if __name__ == "__main__":
    unittest.main()
