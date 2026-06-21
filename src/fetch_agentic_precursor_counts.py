#!/usr/bin/env python3
"""Fetch annual OpenAlex counts for agentic-AI precursor terms."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from openalex import (
    grouped_counts_by_year,
    load_json,
    parse_grouped_year_counts,
    write_json,
)
from paths import project_root

ROOT = project_root()


def main() -> None:
    config_path = ROOT / "configs" / "openalex_agentic_precursor_terms.json"
    config = load_json(config_path)
    source = config["source"]
    years = list(range(source["year_start"], source["year_end"] + 1))
    endpoint = source["works_endpoint"]
    field_filter = source["field_filter"]
    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    print("Fetching total AI-subfield annual counts from OpenAlex...")
    total_payload = grouped_counts_by_year(
        endpoint=endpoint,
        filters=[field_filter],
        year_start=source["year_start"],
        year_end=source["year_end"],
    )
    total_counts = parse_grouped_year_counts(total_payload, years)

    raw: dict[str, object] = {
        "fetched_at": fetched_at,
        "config": config,
        "total_ai_subfield": {
            "query": field_filter,
            "payload_meta": total_payload.get("meta", {}),
            "counts": total_counts,
        },
        "terms": {},
    }
    rows = []
    for term in config["terms"]:
        term_id = term["id"]
        query = term["query"]
        phrase_filter = f'{source["query_filter"]}:"{query}"'
        print(f"Fetching {term_id}: {query}")
        payload = grouped_counts_by_year(
            endpoint=endpoint,
            filters=[field_filter, phrase_filter],
            year_start=source["year_start"],
            year_end=source["year_end"],
        )
        counts = parse_grouped_year_counts(payload, years)
        raw["terms"][term_id] = {
            "term": term,
            "query_filter": phrase_filter,
            "payload_meta": payload.get("meta", {}),
            "counts": counts,
        }
        for year in years:
            total = total_counts[year]
            count = counts[year]
            rows.append(
                {
                    "year": year,
                    "term_id": term_id,
                    "label": term["label"],
                    "family": term["family"],
                    "query": query,
                    "count": count,
                    "total_ai_subfield_count": total,
                    "share_per_million_ai": (count / total * 1_000_000.0) if total else 0.0,
                    "source": source["name"],
                    "fetched_at": fetched_at,
                }
            )

    write_json(ROOT / "data" / "raw" / "openalex_agentic_precursor_counts_raw.json", raw)
    out = ROOT / "data" / "processed" / "openalex_agentic_precursor_counts.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
