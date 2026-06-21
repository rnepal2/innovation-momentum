#!/usr/bin/env python3
"""Fetch annual OpenAlex counts for the curated AI/ML phrase panel."""

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
    config_path = ROOT / "configs" / "openalex_ai_ml_terms.json"
    config = load_json(config_path)
    source = config["source"]
    years = list(range(source["year_start"], source["year_end"] + 1))
    endpoint = source["works_endpoint"]
    field_filter = source["field_filter"]

    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    raw: dict[str, object] = {
        "fetched_at": fetched_at,
        "config": config,
        "total_ai_subfield": None,
        "topics": {},
    }

    print("Fetching total AI-subfield annual counts from OpenAlex...")
    total_payload = grouped_counts_by_year(
        endpoint=endpoint,
        filters=[field_filter],
        year_start=source["year_start"],
        year_end=source["year_end"],
    )
    total_counts = parse_grouped_year_counts(total_payload, years)
    raw["total_ai_subfield"] = {
        "query": field_filter,
        "payload_meta": total_payload.get("meta", {}),
        "counts": total_counts,
    }

    rows = []
    for topic in config["topics"]:
        topic_id = topic["id"]
        query = topic["query"]
        phrase_filter = f'{source["query_filter"]}:"{query}"'
        filters = [field_filter, phrase_filter]
        print(f"Fetching {topic_id}: {query}")
        payload = grouped_counts_by_year(
            endpoint=endpoint,
            filters=filters,
            year_start=source["year_start"],
            year_end=source["year_end"],
        )
        counts = parse_grouped_year_counts(payload, years)
        raw["topics"][topic_id] = {
            "topic": topic,
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
                    "topic_id": topic_id,
                    "label": topic["label"],
                    "cluster": topic["cluster"],
                    "query": query,
                    "count": count,
                    "total_ai_subfield_count": total,
                    "share_per_million_ai": (count / total * 1_000_000.0) if total else 0.0,
                    "source": source["name"],
                    "fetched_at": fetched_at,
                }
            )

    raw_path = ROOT / "data" / "raw" / "openalex_ai_ml_counts_raw.json"
    processed_path = ROOT / "data" / "processed" / "openalex_ai_ml_topic_counts.csv"
    write_json(raw_path, raw)
    df = pd.DataFrame(rows)
    df.to_csv(processed_path, index=False)
    print(f"Wrote {raw_path}")
    print(f"Wrote {processed_path}")


if __name__ == "__main__":
    main()
