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


def signature(rec):
    fields = (
        "sender_id",
        "sender_alias",
        "messageID",
        "rcvTime",
        "sendTime",
        "sender",
        "receiver",
    )
    return json.dumps(
        {field: rec.get(field) for field in fields},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def load_records(path):
    result = {split: [] for split in SPLITS}

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

                    result[split].extend(
                        rec for rec in records if isinstance(rec, dict)
                    )

    return result


def summarize(scenario, split, group, records):
    labels = Counter(str(r.get("attacker")) for r in records)
    total = len(records)

    row = {
        "scenario": scenario,
        "split": split,
        "group": group,
        "records": total,
        "benign_0": labels.get("0", 0),
        "attack_1": labels.get("1", 0),
        "other_labels": sum(
            count for label, count in labels.items()
            if label not in ("0", "1")
        ),
        "attack_pct": (
            100.0 * labels.get("1", 0) / total if total else 0.0
        ),
    }

    print(
        f"{scenario:22} {split:10} {group:13} "
        f"n={total:6} benign={row['benign_0']:6} "
        f"attack={row['attack_1']:6} "
        f"attack_pct={row['attack_pct']:.2f}%"
    )
    return row


def main():
    for path in ARCHIVES.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    print("Loading dataReplay...")
    replay = load_records(ARCHIVES["dataReplay"])

    print("Loading suddenConstantSpeed...")
    sudden = load_records(ARCHIVES["suddenConstantSpeed"])

    rows = []

    print("\nScenario / split / group label distributions")
    print("-" * 100)

    for split in SPLITS:
        maps = {}

        for scenario, dataset in (
            ("dataReplay", replay),
            ("suddenConstantSpeed", sudden),
        ):
            # Payload signatures are unique in the preceding audit,
            # but lists preserve records safely if duplicates occur.
            grouped = defaultdict(list)
            for rec in dataset[split]:
                grouped[signature(rec)].append(rec)
            maps[scenario] = grouped

        common = set(maps["dataReplay"]) & set(maps["suddenConstantSpeed"])

        for scenario, dataset in (
            ("dataReplay", replay),
            ("suddenConstantSpeed", sudden),
        ):
            grouped = maps[scenario]

            matched = [
                rec
                for sig in common
                for rec in grouped[sig]
            ]
            scenario_only = [
                rec
                for sig, items in grouped.items()
                if sig not in common
                for rec in items
            ]

            rows.append(
                summarize(scenario, split, "matched", matched)
            )
            rows.append(
                summarize(scenario, split, "scenario_only", scenario_only)
            )

    output = ROOT / "matched_unmatched_label_distribution.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nReport written to:", output)


if __name__ == "__main__":
    main()
