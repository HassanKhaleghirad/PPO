
import csv
import io
import json
import zipfile
from pathlib import Path
from collections import Counter

ROOT = Path("/home/h/Documents/PPO")

ARCHIVES = {
    "dataReplay": ROOT / "InTAS_highway_2_dataReplay.zip",
    "suddenConstantSpeed": ROOT / "InTAS_highway_2_suddenConstantSpeed.zip",
}

SPLITS = ("Train", "Validation", "Test")


def number(value):
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.strip())
    except (ValueError, TypeError):
        pass
    return None


def position_values(value):
    """Parse a position stored as a list, tuple, or comma-separated string."""
    if isinstance(value, (list, tuple)):
        return [number(x) for x in value]

    if isinstance(value, str):
        return [number(x) for x in value.split(",")]

    return []


def records_from_json(data):
    obj = json.loads(data)

    if isinstance(obj, dict):
        if isinstance(obj.get("messages"), list):
            yield from (x for x in obj["messages"] if isinstance(x, dict))
        else:
            yield obj

    elif isinstance(obj, list):
        yield from (x for x in obj if isinstance(x, dict))


def iter_archive_records(archive_path):
    with zipfile.ZipFile(archive_path) as outer:
        for outer_name in outer.namelist():
            normalized = outer_name.replace("\\", "/")
            parts = normalized.split("/")

            if not normalized.lower().endswith(".zip"):
                continue

            split = next(
                (p for p in parts if p in SPLITS),
                None
            )
            if split is None:
                continue

            nested_bytes = outer.read(outer_name)

            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner:
                for filename in inner.namelist():
                    if not filename.lower().endswith(".json"):
                        continue

                    raw = inner.read(filename)

                    try:
                        for rec in records_from_json(raw):
                            yield split, filename, rec
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        print(
                            f"WARNING: JSON error: "
                            f"{outer_name}/{filename}: {exc}"
                        )


def summarize(archive_path, scenario):
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    states = {}

    for split in SPLITS:
        states[split] = {
            "records": 0,
            "labels": Counter(),
            "senders": set(),
            "times": {"rcvTime": [], "sendTime": []},
            "positions": {
                "sender.pos[0]": [],
                "sender.pos[1]": [],
                "receiver.pos[0]": [],
                "receiver.pos[1]": [],
            },
        }

    for split, filename, rec in iter_archive_records(archive_path):
        s = states[split]
        s["records"] += 1

        attack = rec.get("attacker")
        s["labels"][str(attack)] += 1

        sender_id = rec.get("sender_id")
        if sender_id is not None:
            s["senders"].add(str(sender_id))

        for key in ("rcvTime", "sendTime"):
            v = number(rec.get(key))
            if v is not None:
                s["times"][key].append(v)

        for endpoint in ("sender", "receiver"):
            nested = rec.get(endpoint)
            if not isinstance(nested, dict):
                continue

            pos = position_values(nested.get("pos"))

            for i in range(min(2, len(pos))):
                v = pos[i]
                key = f"{endpoint}.pos[{i}]"
                if v is not None:
                    s["positions"][key].append(v)

    output = []

    def bounds(values):
        if not values:
            return None, None
        return min(values), max(values)

    for split in SPLITS:
        s = states[split]

        print("\n" + "=" * 78)
        print(f"Scenario: {scenario} | Split: {split}")
        print("Records:", s["records"])
        print("Attacker labels:", dict(s["labels"]))
        print("Unique sender IDs:", len(s["senders"]))

        row = {
            "scenario": scenario,
            "split": split,
            "records": s["records"],
            "unique_sender_ids": len(s["senders"]),
            "labels_json": json.dumps(
                dict(s["labels"]), ensure_ascii=False
            ),
        }

        for key, values in s["times"].items():
            lo, hi = bounds(values)
            print(f"{key}: min={lo} max={hi} n={len(values)}")
            row[f"{key}_min"] = lo
            row[f"{key}_max"] = hi
            row[f"{key}_n"] = len(values)

        for key, values in s["positions"].items():
            lo, hi = bounds(values)
            print(f"{key}: min={lo} max={hi} n={len(values)}")
            row[f"{key}_min"] = lo
            row[f"{key}_max"] = hi
            row[f"{key}_n"] = len(values)

        output.append(row)

    return output


def main():
    all_rows = []

    for scenario, path in ARCHIVES.items():
        print("\n\nARCHIVE:", path)
        all_rows.extend(summarize(path, scenario))

    output_path = ROOT / "raw_split_ranges.csv"

    if all_rows:
        with output_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=list(all_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(all_rows)

    print("\nReport written to:", output_path)


if __name__ == "__main__":
    main()
