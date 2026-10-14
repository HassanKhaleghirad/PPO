#!/usr/bin/env python3

import csv
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/h/Documents/PPO")

ARCHIVES = {
    "dataReplay": ROOT / "InTAS_highway_2_dataReplay.zip",
    "suddenConstantSpeed": ROOT / "InTAS_highway_2_suddenConstantSpeed.zip",
}

SPLITS = ("Train", "Validation", "Test")
PAIRS = (
    ("Train", "Validation"),
    ("Train", "Test"),
    ("Validation", "Test"),
)

FIELDS = (
    "sender_id",
    "sender_alias",
    "messageID",
    "rcvTime",
    "sendTime",
    "sender",
    "receiver",
)


def signature(rec):
    # Exclude attacker so identical payloads with different labels
    # can be detected as potential label conflicts.
    return json.dumps(
        {field: rec.get(field) for field in FIELDS},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def read_scenario(path):
    data = {split: [] for split in SPLITS}

    with zipfile.ZipFile(path) as outer:
        for outer_name in outer.namelist():
            normalized = outer_name.replace("\\", "/")
            if not normalized.lower().endswith(".zip"):
                continue

            split = next(
                (p for p in normalized.split("/") if p in SPLITS),
                None,
            )
            if split is None:
                continue

            with zipfile.ZipFile(
                io.BytesIO(outer.read(outer_name))
            ) as inner:
                for filename in inner.namelist():
                    if not filename.lower().endswith(".json"):
                        continue

                    try:
                        obj = json.loads(inner.read(filename))
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        print(f"WARNING: {filename}: {exc}")
                        continue

                    if isinstance(obj, dict):
                        records = (
                            obj["messages"]
                            if isinstance(obj.get("messages"), list)
                            else [obj]
                        )
                    elif isinstance(obj, list):
                        records = obj
                    else:
                        continue

                    data[split].extend(
                        r for r in records if isinstance(r, dict)
                    )

    return data


def label_counts(records):
    result = defaultdict(Counter)
    for rec in records:
        result[signature(rec)][str(rec.get("attacker"))] += 1
    return result


def compare(scenario, split_a, split_b, data):
    a = label_counts(data[split_a])
    b = label_counts(data[split_b])

    keys_a = set(a)
    keys_b = set(b)
    common = keys_a & keys_b

    shared_occurrences = 0
    same_label_pairs = 0

    for key in common:
        total_a = sum(a[key].values())
        total_b = sum(b[key].values())
        shared_occurrences += min(total_a, total_b)

        for label in set(a[key]) | set(b[key]):
            same_label_pairs += min(a[key][label], b[key][label])

    conflicts = shared_occurrences - same_label_pairs

    row = {
        "scenario": scenario,
        "split_a": split_a,
        "split_b": split_b,
        "records_a": len(data[split_a]),
        "records_b": len(data[split_b]),
        "unique_payloads_a": len(keys_a),
        "unique_payloads_b": len(keys_b),
        "common_payload_signatures": len(common),
        "shared_record_occurrences": shared_occurrences,
        "same_label_matches": same_label_pairs,
        "different_label_matches": conflicts,
        "signatures_only_a": len(keys_a - keys_b),
        "signatures_only_b": len(keys_b - keys_a),
    }

    print(
        f"\n{scenario}: {split_a} vs {split_b}\n"
        + "\n".join(f"  {k}: {v}" for k, v in row.items()
                    if k not in ("scenario", "split_a", "split_b"))
    )
    return row


def main():
    rows = []

    for scenario, path in ARCHIVES.items():
        print(f"\nLoading {scenario}: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)

        data = read_scenario(path)

        for split in SPLITS:
            counts = Counter(signature(r) for r in data[split])
            print(
                f"{scenario} {split}: records={len(data[split])}, "
                f"unique_payloads={len(counts)}, "
                f"repeated_occurrences="
                f"{len(data[split]) - len(counts)}"
            )

        for split_a, split_b in PAIRS:
            rows.append(
                compare(scenario, split_a, split_b, data)
            )

    output = ROOT / "cross_split_duplicate_audit.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nReport written to: {output}")


if __name__ == "__main__":
    main()
