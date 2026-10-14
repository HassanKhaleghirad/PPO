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

# The label is deliberately excluded from the payload signature.
PAYLOAD_FIELDS = (
    "sender_id",
    "sender_alias",
    "messageID",
    "rcvTime",
    "sendTime",
    "sender",
    "receiver",
)


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def payload_signature(rec):
    return canonical({
        field: rec.get(field)
        for field in PAYLOAD_FIELDS
    })


def read_records(path):
    result = {split: [] for split in SPLITS}

    with zipfile.ZipFile(path) as outer:
        for outer_name in outer.namelist():
            normalized = outer_name.replace("\\", "/")

            if not normalized.lower().endswith(".zip"):
                continue

            split = next(
                (part for part in normalized.split("/") if part in SPLITS),
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

                    for rec in records:
                        if isinstance(rec, dict):
                            result[split].append(rec)

    return result


def summarize_split(split, left_records, right_records):
    left = defaultdict(Counter)
    right = defaultdict(Counter)

    for rec in left_records:
        left[payload_signature(rec)][str(rec.get("attacker"))] += 1

    for rec in right_records:
        right[payload_signature(rec)][str(rec.get("attacker"))] += 1

    left_signatures = set(left)
    right_signatures = set(right)
    common = left_signatures & right_signatures

    identical_payload_pairs = 0
    same_label_pairs = 0
    different_label_pairs = 0
    conflicting_payloads = 0

    # For each exact payload, match identical labels first.
    # Remaining occurrences, if any, are cross-label matches.
    for sig in common:
        a = left[sig].copy()
        b = right[sig].copy()

        identical = sum((a & b).values())
        identical_payload_pairs += min(sum(a.values()), sum(b.values()))
        same_label_pairs += identical

        for label in set(a) | set(b):
            matched = min(a[label], b[label])
            a[label] -= matched
            b[label] -= matched

        remaining_a = +a
        remaining_b = +b
        mismatches = min(sum(remaining_a.values()), sum(remaining_b.values()))

        if mismatches:
            conflicting_payloads += 1
            different_label_pairs += mismatches

    row = {
        "split": split,
        "dataReplay_records": len(left_records),
        "suddenConstantSpeed_records": len(right_records),
        "dataReplay_unique_payloads": len(left_signatures),
        "sudden_unique_payloads": len(right_signatures),
        "common_exact_payloads": len(common),
        "payloads_only_dataReplay": len(left_signatures - right_signatures),
        "payloads_only_sudden": len(right_signatures - left_signatures),
        "matched_payload_occurrences": identical_payload_pairs,
        "same_label_occurrences": same_label_pairs,
        "different_label_occurrences": different_label_pairs,
        "payloads_with_label_conflicts": conflicting_payloads,
    }

    print("\n" + "=" * 72)
    print("SPLIT:", split)
    for key, value in row.items():
        if key != "split":
            print(f"{key}: {value}")

    return row


def main():
    for path in ARCHIVES.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    print("Reading dataReplay...")
    replay = read_records(ARCHIVES["dataReplay"])

    print("Reading suddenConstantSpeed...")
    sudden = read_records(ARCHIVES["suddenConstantSpeed"])

    rows = [
        summarize_split(split, replay[split], sudden[split])
        for split in SPLITS
    ]

    output = ROOT / "verified_payload_label_comparison.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nCSV written to:", output)


if __name__ == "__main__":
    main()
