import re
from html.parser import HTMLParser

import pandas as pd

from paths import project_root

ROOT = project_root()


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def article_text() -> str:
    parser = TextExtractor()
    parser.feed((ROOT / "reports/index.html").read_text(encoding="utf-8"))
    return " ".join(" ".join(parser.parts).split())


def test_article_contains_required_design_caveats():
    text = article_text()
    required = [
        "current-taxonomy retrospective holdout",
        "centered log-ratio transform",
        "Partial 2026 data used as sensitivity only",
        "query-hit indices rather than paper counts",
        "Most graph-label randomizations produce an equal or higher composite correlation",
    ]
    for phrase in required:
        assert phrase in text


def test_article_avoids_prohibited_writing_patterns():
    text = article_text()
    patterns = [
        r"\bnot\s+[^.]{0,80}\s+but\s",
        r"\bless\s+[^.]{0,80}\s+more\b",
        r"here(?:'|’)s the (?:thing|kicker)",
        r"let(?:'|’)s (?:dive|unpack|break)",
        r"\bin (?:conclusion|summary)\b",
        r"\bultimately\b",
        r"at the end of the day",
        r"\bthis version\b",
        r"\brevised analysis\b",
        r"\bearlier project\b",
        r"\bthe analysis now\b",
        r"—",
    ]
    for pattern in patterns:
        assert re.search(pattern, text, flags=re.IGNORECASE) is None


def test_article_numbers_match_generated_evidence():
    text = article_text()
    inference = pd.read_csv(ROOT / "reports/evidence/holdout_inference.csv").set_index("feature")
    spectral = inference.loc["spectral_emergence_score", "spearman"]
    baseline = inference.loc["baseline_growth_3y", "spearman"]
    assert f"spectral composite reaches a Spearman correlation of {spectral:.2f}" in text
    assert f"CLR growth baseline reaches {baseline:.2f}" in text


def test_nested_selection_is_strictly_temporal():
    selected = pd.read_csv(ROOT / "reports/evidence/nested_model_selection.csv")
    assert (selected["eval_end_year"] == selected["cutoff_year"] + 3).all()
    assert selected["cutoff_year"].min() == 2010
    assert len(selected) == 13
