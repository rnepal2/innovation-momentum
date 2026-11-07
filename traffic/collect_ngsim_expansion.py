"""Fixed nonoverlapping official NGSIM blocks with unchanged spatial supports."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from collect_ngsim import COLUMNS, DATA, acquire, profile, save_json, url
from empirical_resolution import SUPPORTS, fixed_field, flagged_traces, spectrum
from provenance import archive_sources

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(out, raw):
    if (out.exists() and any(out.iterdir())) or (raw.exists() and any(raw.iterdir())):
        raise FileExistsError("Choose fresh snapshot directories")
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    old = HERE / "results/ngsim"
    metadata = json.loads((old / "manifest.json").read_text())["8ect-6jqj_metadata.json"]
    assert (
        hashlib.sha256((DATA / "8ect-6jqj_metadata.json").read_bytes()).hexdigest()
        == metadata["sha256"]
    )
    save_json(
        out / "freeze.json",
        {
            "sources": archive_sources(out, HERE),
            "protocol": "ngsim-expansion-followup.md",
            "official_metadata_receipt": metadata,
            "start_offsets_s": [240, 420],
            "duration_s": 180,
            "width_m": 25,
            "time_bin_s": 5,
            "supports_m": SUPPORTS,
        },
    )
    original = json.loads((old / "profiles.json").read_text())
    receipt, rows, profiles, failures = {}, [], {}, []
    for location, length in SUPPORTS.items():
        earliest = original[location]["start_epoch_ms"] - 60000
        for offset in [240, 420]:
            identity = f"{location}_offset{offset}"
            start = earliest + 1000 * offset
            where = f"location='{location}' AND global_time >= {start} AND global_time < {start + 180000}"
            try:
                count = acquire(
                    url({"$select": "count(*) as rows", "$where": where}),
                    identity + "_count.json",
                    receipt,
                    data_root=raw,
                )
                save_json(out / "manifest.json", receipt)
                expected = int(json.loads(count.read_text())[0]["rows"])
                if not 0 < expected < 500000:
                    raise ValueError("Empty block or row cap exceeded")
                path = acquire(
                    url(
                        {
                            "$select": COLUMNS,
                            "$where": where,
                            "$order": "global_time,vehicle_id,frame_id",
                            "$limit": 500000,
                        },
                        "csv",
                    ),
                    identity + ".csv",
                    receipt,
                    limit=64 * 1024**2,
                    data_root=raw,
                )
                save_json(out / "manifest.json", receipt)
                frame = pd.read_csv(path)
                if len(frame) != expected:
                    raise ValueError("Server count mismatch")
                prof = profile(frame)
                assert prof["duplicate_keys"] == 0
                flagged = flagged_traces(frame)
                prof.update(
                    {
                        "start_epoch_ms": start,
                        "offset_s": offset,
                        "location": location,
                        "flagged_vehicle_ids": flagged.tolist(),
                        "excluded_vehicles": len(flagged),
                        "excluded_rows": int(frame.vehicle_id.isin(flagged).sum()),
                    }
                )
                profiles[identity] = prof
                for quality, f in [
                    ("unfiltered", frame),
                    ("flagged_trace_exclusion", frame[~frame.vehicle_id.isin(flagged)]),
                ]:
                    field, eligibility = fixed_field(f, start, length, 25, 5)
                    for centered in [False, True]:
                        row = {
                            "id": identity,
                            "location": location,
                            "offset_s": offset,
                            "quality": quality,
                            "centered": centered,
                            **eligibility,
                        }
                        if eligibility["eligible"]:
                            power, error = spectrum(field, 5, centered)
                            row.update(
                                {
                                    "power_mps2": power[1:].sum(),
                                    "first_two_share": power[1:3].sum() / power[1:].sum(),
                                    "integration_error": error,
                                }
                            )
                            pd.DataFrame({"mode": range(len(power)), "power_mps2": power}).to_csv(
                                out / f"{identity}_{quality}_center{centered}_powers.csv",
                                index=False,
                            )
                        rows.append(row)
                    if eligibility["eligible"]:
                        field.to_csv(out / f"{identity}_{quality}_field.csv")
                save_json(out / "profiles.json", profiles)
                pd.DataFrame(rows).to_csv(out / "spectra.csv", index=False)
                print("Accepted", identity, len(frame), flush=True)
            except Exception as error:
                failures.append({"id": identity, "error": str(error), "type": type(error).__name__})
                save_json(out / "failures.json", failures)
                raise
    save_json(out / "failures.json", failures)
    save_json(
        out / "acquisition_summary.json",
        {
            "new_rows": sum(p["rows"] for p in profiles.values()),
            "bytes": sum(r["bytes"] for r in receipt.values()),
            "blocks": len(profiles),
            "independent_days": False,
            "raw_directory": str(raw.relative_to(ROOT)),
            "license_id_from_accepted_metadata": "CC_30_BY_SA",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "results/ngsim_expansion")
    parser.add_argument(
        "--raw", type=Path, default=ROOT / "traffic/data/ngsim_expansion"
    )
    args = parser.parse_args()
    run(args.out, args.raw.resolve())
