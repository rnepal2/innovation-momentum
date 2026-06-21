"""Small OpenAlex client for grouped-count research queries."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def openalex_get(endpoint: str, params: dict[str, str], retries: int = 3) -> dict[str, Any]:
    encoded = urllib.parse.urlencode(params)
    url = f"{endpoint}?{encoded}"
    headers = {
        "User-Agent": "innovation-momentum/0.1 (public OpenAlex API client)"
    }
    request = urllib.request.Request(url, headers=headers)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - retain the final network/API error.
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"OpenAlex request failed after {retries} attempts: {url}") from last_error


def grouped_counts_by_year(
    *,
    endpoint: str,
    filters: list[str],
    year_start: int,
    year_end: int,
    polite_sleep: float = 0.12,
) -> dict[str, Any]:
    full_filters = [
        f"from_publication_date:{year_start}-01-01",
        f"to_publication_date:{year_end}-12-31",
        *filters,
    ]
    payload = openalex_get(
        endpoint,
        {
            "filter": ",".join(full_filters),
            "group_by": "publication_year",
            "per-page": "200",
        },
    )
    time.sleep(polite_sleep)
    return payload


def parse_grouped_year_counts(payload: dict[str, Any], years: list[int]) -> dict[int, int]:
    counts = {year: 0 for year in years}
    for row in payload.get("group_by", []):
        key = row.get("key")
        if key is None:
            continue
        year = int(key)
        if year in counts:
            counts[year] = int(row.get("count", 0))
    return counts
