"""Shared corpus preparation. Vocabulary and text context stop at the end of 2015."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/research_2026"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def month_index(date):
    return (int(date[:4]) - 2010) * 12 + int(date[5:7]) - 1


def load_corpus(source):
    filename = "robotics_works.jsonl.gz" if source == "openalex" else "arxiv_robotics.jsonl.gz"
    path = DATA / filename
    records, excluded, seen_id, seen_doi, seen_title = [], {}, set(), set(), set()
    with gzip.open(path, "rt") as f:
        candidates = sorted((json.loads(line) for line in f), key=lambda p: (p["date"], p["id"]))
    # Retain the earliest dated manifestation, independent of API result ordering.
    for p in candidates:
        reason = None
        doi = (p.get("doi") or "").lower().removeprefix("https://doi.org/")
        title = re.sub(r"[^a-z0-9]", "", (p.get("title") or "").lower())
        author = next(iter(p.get("authors", [])), "")
        title_key = (title, author)
        if p["id"] in seen_id:
            reason = "id"
        elif doi and doi in seen_doi:
            reason = "doi"
        elif len(title) >= 40 and author and title_key in seen_title:
            reason = "title_first_author"
        elif not (0 <= month_index(p["date"]) < 192):
            reason = "date"
        elif len(p.get("abstract", "").split()) < 20:
            reason = "short_abstract"
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        seen_id.add(p["id"])
        if doi:
            seen_doi.add(doi)
        seen_title.add(title_key)
        p["month"] = month_index(p["date"])
        records.append(p)
    records.sort(key=lambda p: (p["date"], p["id"]))
    audit = {
        "source": source,
        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "retained": len(records),
        "excluded": excluded,
        "by_year": pd.Series([p["date"][:4] for p in records])
        .value_counts()
        .sort_index()
        .to_dict(),
        "missing_authors": sum(not p["authors"] for p in records),
        "missing_references": sum(not p["references"] for p in records),
        "january_1_fraction": np.mean([p["date"][5:] == "01-01" for p in records]),
        "date_resolution": "publication date" if source == "openalex" else "identifier month",
        "vintage_warning": "Current metadata reconstruct historical text and identities; this is not a vintage archive.",
    }
    save_json(OUT / f"{source}_audit.json", audit)
    return records


def prepare(source, size=512, vocabulary_end=71, records=None):
    records = load_corpus(source) if records is None else records
    texts = [p["abstract"].lower() for p in records]
    months = np.array([p["month"] for p in records])
    train = months <= vocabulary_end
    vec = CountVectorizer(
        ngram_range=(2, 2),
        binary=True,
        min_df=20,
        max_df=0.35,
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
        dtype=np.int32,
    )
    fit = vec.fit_transform(np.array(texts, dtype=object)[train])
    names = vec.get_feature_names_out()
    generic = set(
        "paper propose proposed present presents presented results experimental experiments demonstrate demonstrate demonstrates shown show using used use based approach approaches method methods study studies compared consider considered problem problems provide provides different novel new existing recent several various also obtained respect respectively however therefore".split()
    )
    banned = set(ENGLISH_STOP_WORDS) | generic
    valid = np.array([all(w not in banned for w in name.split()) for name in names])
    count = np.asarray(fit.sum(axis=0)).ravel()
    order = sorted(np.flatnonzero(valid), key=lambda i: (-count[i], names[i]))[:size]
    vocab = names[order].tolist()
    fixed = CountVectorizer(
        vocabulary=vocab,
        ngram_range=(2, 2),
        binary=True,
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
        dtype=np.int32,
    )
    incidence = fixed.transform(texts).tocsr()
    monthly = np.zeros((192, len(vocab)), dtype=float)
    for t in range(192):
        monthly[t] = np.asarray(incidence[months == t].sum(axis=0)).ravel()
    # Context vectors use ordinary words observed in the vocabulary-training period only.
    words = CountVectorizer(
        max_features=4000, min_df=10, max_df=0.8, stop_words="english", dtype=np.float64
    )
    context = words.fit_transform(np.array(texts, dtype=object)[train])
    df = np.asarray((context > 0).sum(axis=0)).ravel()
    context = context.multiply(np.log((1 + train.sum()) / (1 + df)) + 1)
    profiles = normalize(incidence[train].T @ context)
    similarity = (profiles @ profiles.T).toarray()
    suffix = "" if vocabulary_end == 71 else f"_vocab{2010 + vocabulary_end // 12}"
    pd.DataFrame({"concept": vocab, "training_papers": count[order]}).to_csv(
        OUT / f"{source}{suffix}_vocabulary.csv", index=False
    )
    save_json(
        OUT / f"{source}{suffix}_coverage.json",
        {
            "vocabulary_size": len(vocab),
            "training_papers": int(train.sum()),
            "paper_coverage": float(np.mean(np.diff(incidence.indptr) > 0)),
            "median_phrases_per_paper": float(np.median(np.diff(incidence.indptr))),
            "monthly_zero_fraction_128": float(np.mean(monthly[:, :128] == 0)),
            "vocabulary_training_end": f"{2010 + vocabulary_end // 12}-12-31",
            "generic_filter": sorted(generic),
        },
    )
    return records, months, incidence, monthly, vocab, similarity


def graph(incidence, months, cutoff, size, records=None, kind="cooccurrence", k=7):
    past = incidence[months <= cutoff, :size].astype(np.float64)
    if kind == "references":
        subset = [p for p in records if p["month"] <= cutoff]
        reference_index, rows, cols = {}, [], []
        for row, p in enumerate(subset):
            for ref in set(p["references"]):
                col = reference_index.setdefault(ref, len(reference_index))
                rows.append(row)
                cols.append(col)
        references = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(len(subset), len(reference_index))
        )
        profiles = normalize(past.T @ references)
        weights = (profiles @ profiles.T).toarray()
    else:
        co = (past.T @ past).toarray()
        d = np.maximum(np.diag(co), 1)
        weights = co / np.sqrt(d[:, None] * d[None, :])
    np.fill_diagonal(weights, 0)
    selected = np.zeros_like(weights)
    for i in range(size):
        j = np.argsort(weights[i])[-k:]
        selected[i, j] = weights[i, j]
    weights = np.maximum(selected, selected.T)
    d = np.maximum(weights.sum(axis=1), 1e-12)
    return weights / np.sqrt(d[:, None] * d[None, :])


def permutation_within_activity(counts, seed):
    rng = np.random.default_rng(seed)
    groups = np.array_split(np.argsort(counts), 4)
    perm = np.arange(len(counts))
    for group in groups:
        perm[group] = rng.permutation(group)
    return perm
