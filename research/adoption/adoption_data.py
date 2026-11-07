"""Typed concepts and chronological corpus preparation for the adoption pilot."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
DATA = ROOT / "data/adoption_2026"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
ONTOLOGY = json.loads((HERE / "ontology.json").read_text())
METHODS = list(ONTOLOGY["methods"])


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str) + "\n")


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def patterns(aliases):
    return [
        re.compile(r"(?<!\w)" + re.escape(normalize_text(alias)) + r"(?!\w)") for alias in aliases
    ]


def extract(text, domain):
    text = normalize_text(text)
    method = [
        i for i, key in enumerate(METHODS) if any(p.search(text) for p in METHOD_PATTERNS[key])
    ]
    apps = [
        i
        for i, key in enumerate(ONTOLOGY[domain])
        if any(p.search(text) for p in APP_PATTERNS[domain][key])
    ]
    return method, apps


METHOD_PATTERNS = {k: patterns(v["aliases"]) for k, v in ONTOLOGY["methods"].items()}
APP_PATTERNS = {
    domain: {k: patterns(v) for k, v in ONTOLOGY[domain].items()}
    for domain in ["robotics", "medical"]
}

ASSERTION = re.compile(
    r"\b(?:we|our|this (?:paper|work|study))\b.{0,180}\b(?:us\w*|propos\w*|present\w*|develop\w*|introduc\w*|appl\w*|design\w*|evaluat\w*|exploit\w*|validat\w*|show\w*)\b|\b(?:experiment\w*|empirical|case stud\w*|evaluat\w*|validat\w*|demonstrat(?:e|es|ed|ing))\b|\b(?:is|are|was|were) (?:used|applied|proposed|developed|employed)\b"
)
COMPARISON = re.compile(
    r"\b(?:compared (?:to|with)|outperform\w*|higher than|over|mainstream|alternative to|chosen over|previously proposed|replacement of|without|non)\b"
)


def asserted_concepts(title, abstract, domain):
    title_norm = normalize_text(title)
    if re.search(r"\b(?:survey|review)\b", title_norm) or re.search(
        r"\bthis (?:review|chapter)\b", normalize_text(abstract)[:200]
    ):
        return [], [], {}
    sentences = [normalize_text(s) for s in re.split(r"[.!?]\s+|[\r\n]+", abstract)]
    evidence = {}

    def accepted(key, pats):
        if any(
            match
            for p in pats
            for match in p.finditer(title_norm)
            if not COMPARISON.search(title_norm[max(0, match.start() - 40) : match.start()])
        ):
            evidence[key] = title
            return True
        for sentence in sentences:
            if not ASSERTION.search(sentence):
                continue
            for p in pats:
                for match in p.finditer(sentence):
                    if not COMPARISON.search(sentence[max(0, match.start() - 100) : match.start()]):
                        evidence[key] = sentence
                        return True
        return False

    methods = [i for i, k in enumerate(METHODS) if accepted("method:" + k, METHOD_PATTERNS[k])]
    apps = [
        i for i, k in enumerate(ONTOLOGY[domain]) if accepted("app:" + k, APP_PATTERNS[domain][k])
    ]
    return methods, apps, evidence


def key_title(p):
    return re.sub(r"[^a-z0-9]", "", p["title"].lower())


def key_doi(p):
    return (p.get("doi") or "").lower().removeprefix("https://doi.org/")


def read(path):
    with gzip.open(path, "rt") as f:
        return sorted((json.loads(line) for line in f), key=lambda p: (p["date"], p["id"]))


def load(domain):
    original = ROOT / "data/research_2026/robotics_works.jsonl.gz"
    path = original if domain == "robotics" else DATA / "medical.jsonl.gz"
    DATA.mkdir(exist_ok=True)
    digest = dict(
        raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ontology_sha256=hashlib.sha256((HERE / "ontology.json").read_bytes()).hexdigest(),
        parser_version=4,
    )
    cache, meta = DATA / f"{domain}_prepared.jsonl.gz", DATA / f"{domain}_prepared_meta.json"
    if cache.exists() and meta.exists() and json.loads(meta.read_text()) == digest:
        records = read(cache)
        for p in records:
            p["day"] = pd.Timestamp(p["date"])
        return records
    raw = read(path)
    exclude_ids, exclude_dois, exclude_titles = set(), set(), set()
    if domain == "medical":
        development = read(original)
        exclude_ids = {p["id"] for p in development}
        exclude_dois = {key_doi(p) for p in development if key_doi(p)}
        exclude_titles = {key_title(p) for p in development if len(key_title(p)) >= 40}
    kept, seen_id, seen_doi, seen_title, excluded = [], set(), set(), set(), {}
    for p in raw:
        title, doi = key_title(p), key_doi(p)
        author = next(iter(p.get("authors", [])), "")
        tkey = (title, author)
        reason = None
        if (
            p["id"] in exclude_ids
            or (doi and doi in exclude_dois)
            or (len(title) >= 40 and title in exclude_titles)
        ):
            reason = "development_overlap"
        elif (
            not "2010-01-01"
            <= p["date"]
            <= ("2025-12-31" if domain == "robotics" else "2026-10-14")
        ):
            reason = "date"
        elif len(p.get("abstract", "").split()) < 20:
            reason = "short_abstract"
        elif (
            p["id"] in seen_id
            or (doi and doi in seen_doi)
            or (len(title) >= 40 and author and tkey in seen_title)
        ):
            reason = "duplicate"
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        seen_id.add(p["id"])
        if doi:
            seen_doi.add(doi)
        seen_title.add(tkey)
        p["methods"], p["applications"], p["role_evidence"] = asserted_concepts(
            p["title"], p["abstract"], domain
        )
        p["day"] = pd.Timestamp(p["date"])
        kept.append(p)
    audit = dict(
        domain=domain,
        records=len(kept),
        raw_records=len(raw),
        exclusions=excluded,
        missing_authors=sum(not p["authors"] for p in kept),
        missing_references=sum(not p["references"] for p in kept),
        method_papers=sum(bool(p["methods"]) for p in kept),
        app_papers=sum(bool(p["applications"]) for p in kept),
        joint_papers=sum(bool(p["methods"]) and bool(p["applications"]) for p in kept),
        arxiv_links=sum(bool(p.get("arxiv_ids")) for p in kept),
        raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ontology_sha256=hashlib.sha256((HERE / "ontology.json").read_bytes()).hexdigest(),
    )
    save(OUT / f"{domain}_audit.json", audit)
    with gzip.open(cache, "wt") as f:
        for p in kept:
            f.write(json.dumps({k: v for k, v in p.items() if k != "day"}) + "\n")
    save(meta, digest)
    return kept


def incidence(records, key, size):
    rows, cols = [], []
    for i, p in enumerate(records):
        rows.extend([i] * len(p[key]))
        cols.extend(p[key])
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(len(records), size)
    )


def review_sample():
    records = [
        p
        for p in load("robotics")
        if p["date"] <= "2017-12-31" and p["methods"] and p["applications"]
    ]
    initial = {
        p["id"]
        for name in ["development_review_initial.json", "development_review_role1.json"]
        if (OUT / name).exists()
        for p in json.loads((OUT / name).read_text())
    }
    sample = sorted(
        [p for p in records if p["id"] not in initial],
        key=lambda p: hashlib.sha256(p["id"].encode()).hexdigest(),
    )[:20]
    save(
        OUT / "development_review.json",
        [
            dict(
                id=p["id"],
                title=p["title"],
                abstract=p["abstract"],
                methods=[METHODS[i] for i in p["methods"]],
                applications=[list(ONTOLOGY["robotics"])[i] for i in p["applications"]],
            )
            for p in sample
        ],
    )
    pd.DataFrame(
        [
            dict(id=p["id"], reviewer="", method_used="", application_clear="", notes="")
            for p in sample
        ]
    ).to_csv(OUT / "human_review.csv", index=False)


if __name__ == "__main__":
    review_sample()
