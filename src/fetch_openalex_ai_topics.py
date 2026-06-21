#!/usr/bin/env python3
"""Fetch OpenAlex AI-subfield topic metadata and annual topic counts.

The primary analysis uses `primary_topic.id` counts so numerator and denominator
refer to the same OpenAlex topic taxonomy. The broader `topics.id` within the AI
subfield is kept as a robustness check because it can count secondary-topic
assignments on papers whose primary topic is elsewhere in AI.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from openalex import (
    grouped_counts_by_year,
    openalex_get,
    parse_grouped_year_counts,
    write_json,
)
from paths import project_root

ROOT = project_root()

WORKS_ENDPOINT = "https://api.openalex.org/works"
TOPICS_ENDPOINT = "https://api.openalex.org/topics"
SUBFIELD_FILTER = "primary_topic.subfield.id:1702"
TOPIC_FILTER = "subfield.id:1702"
YEAR_START = 1990
YEAR_END = 2026


def short_openalex_id(openalex_url: str) -> str:
    return openalex_url.rstrip("/").split("/")[-1]


def main() -> None:
    years = list(range(YEAR_START, YEAR_END + 1))
    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    print("Fetching AI-subfield topic metadata from OpenAlex...")
    topics_payload = openalex_get(
        TOPICS_ENDPOINT,
        {"filter": TOPIC_FILTER, "per-page": "200"},
    )
    topics = topics_payload.get("results", [])
    if not topics:
        raise RuntimeError("OpenAlex returned no AI-subfield topics.")

    print("Fetching total AI-subfield annual counts...")
    total_payload = grouped_counts_by_year(
        endpoint=WORKS_ENDPOINT,
        filters=[SUBFIELD_FILTER],
        year_start=YEAR_START,
        year_end=YEAR_END,
    )
    total_counts = parse_grouped_year_counts(total_payload, years)

    topic_rows = []
    count_rows = []
    raw = {
        "fetched_at": fetched_at,
        "topic_filter": TOPIC_FILTER,
        "total_ai_subfield": {
            "query": SUBFIELD_FILTER,
            "payload_meta": total_payload.get("meta", {}),
            "counts": total_counts,
        },
        "topics_payload_meta": topics_payload.get("meta", {}),
        "topics": {},
    }
    for topic in topics:
        topic_id = short_openalex_id(topic["id"])
        keywords = topic.get("keywords") or []
        topic_rows.append(
            {
                "topic_id": topic_id,
                "openalex_id": topic["id"],
                "display_name": topic.get("display_name"),
                "description": topic.get("description"),
                "keywords": "; ".join(keywords),
                "domain_id": (topic.get("domain") or {}).get("id"),
                "domain_name": (topic.get("domain") or {}).get("display_name"),
                "field_id": (topic.get("field") or {}).get("id"),
                "field_name": (topic.get("field") or {}).get("display_name"),
                "subfield_id": (topic.get("subfield") or {}).get("id"),
                "subfield_name": (topic.get("subfield") or {}).get("display_name"),
                "works_count": topic.get("works_count"),
                "cited_by_count": topic.get("cited_by_count"),
                "works_api_url": topic.get("works_api_url"),
                "updated_date": topic.get("updated_date"),
                "created_date": topic.get("created_date"),
                "source": "OpenAlex Topics API",
                "fetched_at": fetched_at,
            }
        )
        print(f"Fetching annual counts for {topic_id}: {topic.get('display_name')}")
        primary_payload = grouped_counts_by_year(
            endpoint=WORKS_ENDPOINT,
            filters=[f"primary_topic.id:{topic_id}"],
            year_start=YEAR_START,
            year_end=YEAR_END,
            polite_sleep=0.06,
        )
        any_ai_payload = grouped_counts_by_year(
            endpoint=WORKS_ENDPOINT,
            filters=[SUBFIELD_FILTER, f"topics.id:{topic_id}"],
            year_start=YEAR_START,
            year_end=YEAR_END,
            polite_sleep=0.06,
        )
        primary_counts = parse_grouped_year_counts(primary_payload, years)
        any_ai_counts = parse_grouped_year_counts(any_ai_payload, years)
        raw["topics"][topic_id] = {
            "topic": topic,
            "primary_topic_query_filter": f"primary_topic.id:{topic_id}",
            "primary_topic_payload_meta": primary_payload.get("meta", {}),
            "primary_topic_counts": primary_counts,
            "ai_subfield_any_topic_query_filter": f"{SUBFIELD_FILTER},topics.id:{topic_id}",
            "ai_subfield_any_topic_payload_meta": any_ai_payload.get("meta", {}),
            "ai_subfield_any_topic_counts": any_ai_counts,
        }
        for year in years:
            total = total_counts[year]
            primary_count = primary_counts[year]
            any_ai_count = any_ai_counts[year]
            count_rows.append(
                {
                    "year": year,
                    "topic_id": topic_id,
                    "display_name": topic.get("display_name"),
                    "primary_topic_count": primary_count,
                    "ai_subfield_any_topic_count": any_ai_count,
                    "total_ai_subfield_count": total,
                    "primary_share_per_million_ai": (primary_count / total * 1_000_000.0) if total else 0.0,
                    "ai_any_share_per_million_ai": (any_ai_count / total * 1_000_000.0) if total else 0.0,
                    "source": "OpenAlex Works API primary_topic.id and AI-subfield topics.id grouped counts",
                    "fetched_at": fetched_at,
                }
            )

    raw_path = ROOT / "data" / "raw" / "openalex_ai_topics_raw.json"
    topics_path = ROOT / "data" / "processed" / "openalex_ai_topics.csv"
    counts_path = ROOT / "data" / "processed" / "openalex_ai_topic_year_counts.csv"
    write_json(raw_path, raw)
    pd.DataFrame(topic_rows).sort_values("works_count", ascending=False).to_csv(topics_path, index=False)
    pd.DataFrame(count_rows).to_csv(counts_path, index=False)
    print(f"Wrote {topics_path}")
    print(f"Wrote {counts_path}")


if __name__ == "__main__":
    main()
