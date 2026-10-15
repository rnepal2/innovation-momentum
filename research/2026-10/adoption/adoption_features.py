"""Past-only exposure networks and genuinely new-author outcomes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from adoption_data import DATA, METHODS, ONTOLOGY, OUT, incidence, save
from scipy import sparse
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def disjoint_witness(papers):
    groups = [(p, set(p["authors"])) for p in papers if p["authors"]]
    for i in range(len(groups) - 2):
        choices = [j for j in range(i + 1, len(groups)) if not groups[i][1] & groups[j][1]]
        for j, k in combinations(choices, 2):
            if not groups[j][1] & groups[k][1]:
                return [groups[i][0]["id"], groups[j][0]["id"], groups[k][0]["id"]]
    return []


def outcome(records, method, app, cutoff, forbidden, months=24):
    end = cutoff + pd.DateOffset(months=months)
    future = [
        p
        for p in records
        if cutoff < p["day"] <= end
        and method in p["methods"]
        and app in p["applications"]
        and p["authors"]
        and len(p["authors"]) <= 50
        and not set(p["authors"]) & forbidden
    ]
    future.sort(key=lambda p: (p["day"], p["id"]))
    bins = {
        next(i for i in range(4) if p["day"] <= cutoff + pd.DateOffset(months=6 * (i + 1)))
        for p in future
    }
    witness = disjoint_witness(future) if len(bins) >= 2 else []
    event_day = None
    if witness:
        lo, hi = 3, len(future)
        while lo < hi:
            mid = (lo + hi) // 2
            prefix = future[:mid]
            occupied = {
                next(i for i in range(4) if p["day"] <= cutoff + pd.DateOffset(months=6 * (i + 1)))
                for p in prefix
            }
            if len(occupied) >= 2 and disjoint_witness(prefix):
                hi = mid
            else:
                lo = mid + 1
        event_day = future[lo - 1]["date"]
    return dict(
        event=int(bool(witness)),
        one_group=int(bool(future)),
        papers=len(future),
        bins=len(bins),
        witness=witness,
        event_day=event_day,
    )


def author_network(past):
    g = nx.Graph()
    oversized = 0
    for p in past:
        authors = sorted(set(p["authors"]))
        g.add_nodes_from(authors)
        if len(authors) > 50:
            oversized += 1
            continue
        weight = 2 / max(len(authors) * (len(authors) - 1), 1)
        for a, b in combinations(authors, 2):
            if g.has_edge(a, b):
                g[a][b]["weight"] += weight
            else:
                g.add_edge(a, b, weight=weight)
    groups = (
        nx.community.louvain_communities(g, seed=42, resolution=1, weight="weight")
        if g.number_of_edges()
        else [{a} for a in g]
    )
    labels = {author: i for i, group in enumerate(groups) for author in sorted(group)}

    def majority(paper):
        if len(paper["authors"]) > 50:
            return -1
        counts = Counter(labels[a] for a in set(paper["authors"]))
        return min(counts, key=lambda c: (-counts[c], c)) if counts else -1

    paper_labels = np.array([majority(p) for p in past])
    return (
        g,
        labels,
        paper_labels,
        dict(
            authors=len(g),
            edges=g.number_of_edges(),
            communities=len(groups),
            excluded_large_authorship_papers=oversized,
            largest_community_fraction=max(map(len, groups), default=0) / max(len(g), 1),
        ),
    )


def row_transition(flow):
    return sparse.diags(1 / np.maximum(np.asarray(flow.sum(axis=1)).ravel(), 1)) @ flow


def propagation(q, transition, r):
    state = q.copy()
    values = []
    for _ in range(3):
        state = state @ transition
        values.append(float(state @ r))
    return values


def breadth(authors, labels):
    counts = np.array(list(Counter(labels[a] for a in authors if a in labels).values()), float)
    if not len(counts):
        return [0.0, 0.0, 0.0]
    p = counts / counts.sum()
    return [
        np.log1p(len(counts)),
        float(-np.sum(p * np.log(p)) / max(np.log(len(counts)), 1e-12)),
        float(p.max()),
    ]


def snapshot(records, domain, cutoff, nulls=10, include_outcomes=True):
    cutoff = pd.Timestamp(cutoff)
    cache = DATA / f"{domain}_{cutoff.date()}_snapshot.json"
    fingerprint = dict(
        version=5,
        nulls=nulls,
        include_outcomes=include_outcomes,
        corpus=json.loads((DATA / f"{domain}_prepared_meta.json").read_text()),
        anchors=hashlib.sha256((OUT / "anchors.json").read_bytes()).hexdigest(),
    )
    if cache.exists():
        result = json.loads(cache.read_text())
        if result.get("fingerprint") == fingerprint:
            return result
    past = [p for p in records if p["day"] <= cutoff]
    nm, na = len(METHODS), len(ONTOLOGY[domain])
    m, a = incidence(past, "methods", nm), incidence(past, "applications", na)
    mm, aa = m.toarray().astype(bool), a.toarray().astype(bool)
    total_m, total_a = mm.sum(axis=0), aa.sum(axis=0)
    graph, labels, paper_labels, network_audit = author_network(past)
    print(
        f"{domain} {cutoff.date()}: author communities {network_audit['communities']}, papers {len(past)}",
        flush=True,
    )
    nc = network_audit["communities"]
    ids = {p["id"]: i for i, p in enumerate(past)}
    anchors = json.loads((OUT / "anchors.json").read_text())["methods"]
    ref_methods, ref_dates = {}, {}
    for p in past:
        ref_methods[p["id"]] = sum(1 << i for i in p["methods"])
        ref_dates[p["id"]] = p["day"]
    for mi, method in enumerate(METHODS):
        for anchor in anchors[method]:
            identifier = anchor["id"].rsplit("/", 1)[-1]
            ref_methods[identifier] = ref_methods.get(identifier, 0) | (1 << mi)
            ref_dates[identifier] = pd.Timestamp(anchor["publication_date"])
    cited_methods = np.zeros((len(past), nm), bool)
    flow_rows, flow_cols, reference_rows, reference_cols = [], [], [], []
    ref_index, bad_dates, observed_edges = {}, 0, 0
    for i, p in enumerate(past):
        bits = 0
        for ref in set(p["references"]):
            if ref in ref_dates and ref_dates[ref] > p["day"]:
                bad_dates += 1
                continue
            bits |= ref_methods.get(ref, 0)
            reference_rows.append(i)
            reference_cols.append(ref_index.setdefault(ref, len(ref_index)))
            if ref in ids:
                observed_edges += 1
                c1, c2 = paper_labels[ids[ref]], paper_labels[i]
                if c1 >= 0 and c2 >= 0 and c1 != c2:
                    flow_rows.append(c1)
                    flow_cols.append(c2)
        cited_methods[i] = [(bits >> mi) & 1 for mi in range(nm)]
    flow = sparse.csr_matrix((np.ones(len(flow_rows)), (flow_rows, flow_cols)), shape=(nc, nc))
    directed, undirected, reverse = [row_transition(f) for f in [flow, flow + flow.T, flow.T]]
    b = sparse.csr_matrix(
        (np.ones(len(reference_rows)), (reference_rows, reference_cols)),
        shape=(len(past), len(ref_index)),
    )
    if b.shape[1]:
        mp, ap = normalize(m.T @ b), normalize(a.T @ b)
        reference_similarity = (mp @ ap.T).toarray()
    else:
        reference_similarity = np.zeros((nm, na))
    tfidf = TfidfVectorizer(
        max_features=2500, min_df=3, max_df=0.9, stop_words="english"
    ).fit_transform([p["abstract"] for p in past])
    semantic = (normalize(m.T @ tfidf) @ normalize(a.T @ tfidf).T).toarray()
    paper_days = np.array([p["day"].value for p in past])
    recent = paper_days > (cutoff - pd.DateOffset(months=12)).value
    older = (paper_days > (cutoff - pd.DateOffset(months=24)).value) & ~recent
    first_m = [
        min((p["day"] for p in past if i in p["methods"]), default=cutoff) for i in range(nm)
    ]
    first_a = [
        min((p["day"] for p in past if i in p["applications"]), default=cutoff) for i in range(na)
    ]
    method_authors = [
        {author for p in past if i in p["methods"] for author in p["authors"]} for i in range(nm)
    ]
    app_authors = [
        {author for p in past if i in p["applications"] for author in p["authors"]}
        for i in range(na)
    ]
    neighbors_m = [
        {b for author in authors for b in graph.neighbors(author)} for authors in method_authors
    ]
    permutations = []
    authors = list(labels)
    activity = Counter(author for p in past for author in set(p["authors"]))
    activity_groups = np.array_split(np.argsort([activity[x] for x in authors], kind="stable"), 4)
    for seed in range(nulls):
        rng = np.random.default_rng(5000 + seed)
        permuted = np.array([labels[x] for x in authors])
        for group in activity_groups:
            permuted[group] = rng.permutation(permuted[group])
        permutations.append(dict(zip(authors, permuted.tolist())))
    rows = []
    for mi in range(nm):
        for ai in range(na):
            if total_m[mi] < 20 or total_a[ai] < 20:
                continue
            joint = mm[:, mi] & aa[:, ai]
            if joint.sum() > 10:
                continue
            seeds = {author for p, keep in zip(past, joint) if keep for author in p["authors"]}
            collaborators = seeds | {other for author in seeds for other in graph.neighbors(author)}
            app_only, method_only = aa[:, ai] & ~mm[:, mi], mm[:, mi] & ~aa[:, ai]
            exposure_rows = np.flatnonzero(app_only & cited_methods[:, mi])
            exposure_rows = [
                i
                for i in exposure_rows
                if len(past[i]["authors"]) <= 50 and not set(past[i]["authors"]) & seeds
            ]
            exposed = {author for i in exposure_rows for author in past[i]["authors"]} - seeds
            c_exposed = (app_authors[ai] & neighbors_m[mi]) - seeds
            q = np.bincount(paper_labels[method_only & (paper_labels >= 0)], minlength=nc).astype(
                float
            )
            r = np.bincount(paper_labels[app_only & (paper_labels >= 0)], minlength=nc).astype(
                float
            )
            q /= max(q.sum(), 1)
            r /= max(r.sum(), 1)
            shared = method_authors[mi] & app_authors[ai]
            joint_first = min((p["day"] for p, keep in zip(past, joint) if keep), default=cutoff)
            base = [
                np.log1p(total_m[mi]),
                np.log1p(total_a[ai]),
                np.log1p(mm[recent, mi].sum()),
                np.log1p(aa[recent, ai].sum()),
                np.log1p(mm[recent, mi].sum()) - np.log1p(mm[older, mi].sum()),
                np.log1p(aa[recent, ai].sum()) - np.log1p(aa[older, ai].sum()),
                np.log1p(joint.sum()),
                np.log1p(joint[recent].sum()),
                (cutoff - first_m[mi]).days / 365.25,
                (cutoff - first_a[ai]).days / 365.25,
                (cutoff - joint_first).days / 365.25,
                semantic[mi, ai],
                np.log1p(len(seeds)),
                np.log1p(len(past)),
                float(np.mean([not past[i]["references"] for i in np.flatnonzero(aa[:, ai])])),
            ]
            static = [
                np.log1p(len(shared)),
                len(shared) / max(len(method_authors[mi] | app_authors[ai]), 1),
                float(q @ r) / (max(np.linalg.norm(q) * np.linalg.norm(r), 1e-12)),
                reference_similarity[mi, ai],
            ]
            exposure = [
                np.log1p(len(exposure_rows)),
                np.log1p(len(exposed)),
                np.log1p(len(c_exposed)),
                np.log1p(
                    len(
                        {
                            ref
                            for i in exposure_rows
                            for ref in past[i]["references"]
                            if ref_methods.get(ref, 0) & (1 << mi)
                            and ref_dates[ref] <= past[i]["day"]
                        }
                    )
                ),
            ]
            label = outcome(records, mi, ai, cutoff, seeds) if include_outcomes else None
            strict = outcome(records, mi, ai, cutoff, collaborators) if include_outcomes else None
            rows.append(
                dict(
                    method=METHODS[mi],
                    application=list(ONTOLOGY[domain])[ai],
                    method_index=mi,
                    app_index=ai,
                    prior=int(joint.sum()),
                    base=base,
                    static=static,
                    exposure=exposure,
                    breadth=breadth(exposed, labels),
                    null=[breadth(exposed, lab) for lab in permutations],
                    undirected=propagation(q, undirected, r),
                    directed=propagation(q, directed, r),
                    reverse=propagation(q, reverse, r),
                    label=label,
                    strict=strict,
                    seed_authors=sorted(seeds),
                    exposed_authors=len(exposed),
                    burst_score=float(
                        -np.log10(
                            max(
                                binomtest(
                                    int(joint[recent].sum()),
                                    int(joint[recent | older].sum()),
                                    0.5,
                                    alternative="greater",
                                ).pvalue,
                                1e-12,
                            )
                        )
                    )
                    if joint[recent | older].sum()
                    else 0.0,
                )
            )
    audit = dict(
        domain=domain,
        cutoff=str(cutoff.date()),
        past_papers=len(past),
        rows=len(rows),
        **network_audit,
        internal_citation_edges=observed_edges,
        intercommunity_citations=len(flow_rows),
        future_dated_references_removed=bad_dates,
        feature_groups=dict(base=15, static=4, exposure=4, breadth=3, undirected=3, directed=3),
    )
    save(OUT / f"{domain}_{cutoff.date()}_network.json", audit)
    result = dict(
        domain=domain, cutoff=str(cutoff.date()), rows=rows, audit=audit, fingerprint=fingerprint
    )
    save(cache, result)
    return result
