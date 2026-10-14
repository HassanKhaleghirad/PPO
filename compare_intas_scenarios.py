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


def records_from_json(data):
    obj = json.loads(data)

    if isinstance(obj, dict):
        if isinstance(obj.get("messages"), list):
            yield from (
                item for item in obj["messages"]
                if isinstance(item, dict)
            )
        else:
            yield obj

    elif isinstance(obj, list):
        yield from (
            item for item in obj
            if isinstance(item, dict)
        )


def stable_value(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def record_key(rec):
    # Key for aligning corresponding messages across scenarios.
    return (
        str(rec.get("sender_id", "")),
        str(rec.get("rcvTime", "")),
        str(rec.get("sendTime", "")),
        str(rec.get("messageID", "")),
        str(rec.get("sender_alias", "")),
    )


def exact_signature(rec):
    # Compare core message fields and endpoint data.
    fields = (
        "sender_id",
        "sender_alias",
        "messageID",
        "rcvTime",
        "sendTime",
        "sender",
        "receiver",
    )
    return stable_value({
        field: rec.get(field)
        for field in fields
    })


def load_scenario(path):
    result = {split: defaultdict(list) for split in SPLITS}

    if not path.is_file():
        raise FileNotFoundError(path)

    with zipfile.ZipFile(path) as outer:
        for outer_name in outer.namelist():
            normalized = outer_name.replace("\\", "/")

            if not normalized.lower().endswith(".zip"):
                continue

            parts = normalized.split("/")
            split = next(
                (part for part in parts if part in SPLITS),
                None,
            )

            if split is None:
                continue

            nested_data = outer.read(outer_name)

            with zipfile.ZipFile(io.BytesIO(nested_data)) as inner:
                for filename in inner.namelist():
                    if not filename.lower().endswith(".json"):
                        continue

                    try:
                        raw = inner.read(filename)

                        for rec in records_from_json(raw):
                            key = record_key(rec)
                            result[split][key].append(rec)

                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        print(
                            f"WARNING: Could not parse "
                            f"{outer_name}/{filename}: {exc}"
                        )

    return result


def compare_split(split, left, right):
    left_map = left[split]
    right_map = right[split]

    left_keys = set(left_map)
    right_keys = set(right_map)
    common_keys = left_keys & right_keys

    left_count = sum(len(items) for items in left_map.values())
    right_count = sum(len(items) for items in right_map.values())

    # For keys that occur more than once, compare all matching records
    # by exact signature and label counts.
    exact_record_matches = 0
    label_equal = 0
    label_different = 0
    unmatched_duplicate_excess = 0
    signatures_differ = 0
    matched_label_pairs = Counter()

    for key in common_keys:
        left_records = left_map[key]
        right_records = right_map[key]

        left_signatures = Counter(
            (exact_signature(rec), str(rec.get("attacker")))
            for rec in left_records
        )
        right_signatures = Counter(
            (exact_signature(rec), str(rec.get("attacker")))
            for rec in right_records
        )

        common_signatures = left_signatures & right_signatures
        exact_record_matches += sum(common_signatures.values())

        left_labels = Counter(str(rec.get("attacker")) for rec in left_records)
        right_labels = Counter(str(rec.get("attacker")) for rec in right_records)

        # Compare labels for records aligned by the same key.
        # If duplicates exist, pair records deterministically by signature.
        left_ordered = sorted(
            left_records,
            key=lambda rec: (
                exact_signature(rec),
                str(rec.get("attacker")),
            ),
        )
        right_ordered = sorted(
            right_records,
            key=lambda rec: (
                exact_signature(rec),
                str(rec.get("attacker")),
            ),
        )

        pair_count = min(len(left_ordered), len(right_ordered))

        for i in range(pair_count):
            a = str(left_ordered[i].get("attacker"))
            b = str(right_ordered[i].get("attacker"))
            matched_label_pairs[(a, b)] += 1

            if a == b:
                label_equal += 1
            else:
                label_different += 1

        unmatched_duplicate_excess += abs(
            len(left_records) - len(right_records)
        )

        if left_signatures != right_signatures:
            signatures_differ += 1

    row = {
        "split": split,
        "dataReplay_records": left_count,
        "suddenConstantSpeed_records": right_count,
        "dataReplay_unique_keys": len(left_keys),
        "suddenConstantSpeed_unique_keys": len(right_keys),
        "common_keys": len(common_keys),
        "keys_only_dataReplay": len(left_keys - right_keys),
        "keys_only_suddenConstantSpeed": len(right_keys - left_keys),
        "exact_core_record_matches": exact_record_matches,
        "matched_label_equal": label_equal,
        "matched_label_different": label_different,
        "keys_with_core_field_differences": signatures_differ,
        "unmatched_duplicate_excess": unmatched_duplicate_excess,
        "matched_label_pairs": json.dumps(
            {
                f"{a}->{b}": count
                for (a, b), count in sorted(matched_label_pairs.items())
            },
            ensure_ascii=False,
        ),
    }

    print("\n" + "=" * 78)
    print("SPLIT:", split)

    for key, value in row.items():
        if key != "split":
            print(f"{key}: {value}")

    return row


def main():
    print("Loading dataReplay...")
    replay = load_scenario(ARCHIVES["dataReplay"])

    print("Loading suddenConstantSpeed...")
    sudden = load_scenario(ARCHIVES["suddenConstantSpeed"])

    rows = [
        compare_split(split, replay, sudden)
        for split in SPLITS
    ]

    output_path = ROOT / "scenario_record_comparison.csv"

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nReport written to:", output_path)


if __name__ == "__main__":
    main()
