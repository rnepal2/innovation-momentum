"""Bounded first-version checks; successful downloads never imply vintage coverage."""

import hashlib
import io
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from xml.etree import ElementTree as ET

from adoption_data import DATA, OUT, asserted_concepts, extract, load, save
from pypdf import PdfReader

LAST = 0.0


def get(url):
    global LAST
    time.sleep(max(0.0, 3.1 - (time.monotonic() - LAST)))
    LAST = time.monotonic()
    request = urllib.request.Request(
        url, headers={"User-Agent": "innovation-momentum-research/0.3"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(30_000_000)
        return data, None
    except urllib.error.HTTPError as error:
        return None, f"HTTP {error.code}"
    except (urllib.error.URLError, TimeoutError):
        return None, "connection failure"


def first_abstract(text):
    match = re.search(
        r"(?is)\babstract\s*[:.\-\u2014]?\s*(.*?)(?=\n\s*(?:1\s*\.?\s+|I\.\s+)?(?:introduction|background)|\n\s*(?:keywords|index terms)\b)",
        text,
    )
    return match.group(1).strip() if match and len(match.group(1).split()) >= 30 else None


def main():
    folder = DATA / "version_audit"
    folder.mkdir(exist_ok=True)
    records = [
        p
        for p in load("medical")
        if p.get("arxiv_ids") and p["methods"] and p["applications"] and p["date"] <= "2023-12-31"
    ]
    records.sort(key=lambda p: hashlib.sha256(p["id"].encode()).hexdigest())
    rows = []
    for p in records[:16]:
        identifier = p["arxiv_ids"][0]
        safe = identifier.replace("/", "_")
        row = dict(
            id=p["id"],
            arxiv_id=identifier,
            openalex_date=p["date"],
            version_url=f"https://arxiv.org/pdf/{identifier}v1",
        )
        cache = folder / f"{safe}.xml"
        if cache.exists():
            raw, error = cache.read_bytes(), None
        else:
            url = "https://oaipmh.arxiv.org/oai?" + urllib.parse.urlencode(
                dict(
                    verb="GetRecord",
                    identifier=f"oai:arXiv.org:{identifier}",
                    metadataPrefix="arXivRaw",
                )
            )
            raw, error = get(url)
            if raw:
                cache.write_bytes(raw)
        if raw:
            root = ET.fromstring(raw)
            versions = [v for v in root.iter() if v.tag.endswith("}version") or v.tag == "version"]
            row["versions"] = len(versions)
            first = next((v for v in versions if v.attrib.get("version") == "v1"), None)
            row["first_version_date"] = (
                next((e.text for e in first if e.tag.endswith("}date")), None)
                if first is not None
                else None
            )
        else:
            row["metadata_failure"] = error
        pdf = folder / f"{safe}v1.pdf"
        if pdf.exists():
            data, error = pdf.read_bytes(), None
        else:
            data, error = get(row["version_url"])
            if data and data.startswith(b"%PDF"):
                pdf.write_bytes(data)
        if not data or not data.startswith(b"%PDF"):
            row["pdf_failure"] = error or "non-PDF response"
        else:
            row["pdf_sha256"] = hashlib.sha256(data).hexdigest()
            try:
                reader = PdfReader(io.BytesIO(data))
                text = "\n".join(page.extract_text() or "" for page in reader.pages[:3])
                abstract = first_abstract(text)
                if abstract:
                    current = extract(p["abstract"], "medical")
                    earlier = extract(abstract, "medical")
                    strict_current = asserted_concepts("", p["abstract"], "medical")[:2]
                    strict_earlier = asserted_concepts("", abstract, "medical")[:2]
                    row.update(
                        abstract_extracted=True,
                        plain_alias_agreement=current == earlier,
                        asserted_alias_agreement=strict_current == strict_earlier,
                        current_methods=current[0],
                        v1_methods=earlier[0],
                        current_apps=current[1],
                        v1_apps=earlier[1],
                    )
                    (folder / f"{safe}v1_abstract.txt").write_text(abstract)
                else:
                    row["text_failure"] = "first-version abstract boundary unavailable"
            except Exception as error:
                row["text_failure"] = type(error).__name__
        rows.append(row)
        print(f"Version audit {len(rows)}/{min(16, len(records))}: {identifier}", flush=True)
    save(
        OUT / "version_audit.json",
        dict(
            eligible_papers=len(records),
            sample=len(rows),
            cases=rows,
            interpretation="Small deterministic metadata/text audit. Failures are missing evidence, and agreement does not establish historical author/reference metadata or full-corpus vintage safety.",
        ),
    )


if __name__ == "__main__":
    main()
