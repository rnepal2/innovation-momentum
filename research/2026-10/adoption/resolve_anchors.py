"""Resolve declared canonical references without searching for successful outcomes."""

import importlib.util
import json

from adoption_data import HERE, ONTOLOGY, OUT, save


def main():
    if (OUT / "anchors.json").exists():
        print("Frozen canonical references preserved.")
        return
    spec = importlib.util.spec_from_file_location(
        "_anchor_api", HERE.parent / "campaign/fetch_corpus.py"
    )
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    dois = sorted(
        {doi.lower() for method in ONTOLOGY["methods"].values() for doi in method["references"]}
    )
    client.FILTER = "doi:" + "|".join(dois)
    client.SELECT = "id,doi,title,publication_date"
    payload = client.get_page("*")
    result = payload["results"]
    lookup = {p["doi"].lower(): p for p in result if p.get("doi")}
    anchors = {
        key: [lookup[doi.lower()] for doi in value["references"] if doi.lower() in lookup]
        for key, value in ONTOLOGY["methods"].items()
    }
    save(
        OUT / "anchors.json",
        dict(
            methods=anchors,
            unresolved=[doi for doi in dois if doi not in lookup],
            reported_cost_usd=payload["meta"].get("cost_usd"),
            source="OpenAlex exact DOI lookup",
        ),
    )
    print(
        json.dumps(
            {
                "resolved": len(result),
                "unresolved": len(dois) - len(lookup),
                "titles": [p["title"] for p in result],
            }
        )
    )


if __name__ == "__main__":
    main()
