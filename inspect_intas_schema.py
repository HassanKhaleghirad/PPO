import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/h/Documents/PPO")
ARCHIVES = [
    ("dataReplay", ROOT / "InTAS_highway_2_dataReplay.zip"),
    ("suddenConstantSpeed", ROOT / "InTAS_highway_2_suddenConstantSpeed.zip"),
]

def records_from_json(obj):
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
    elif isinstance(obj, dict):
        if "attacker" in obj:
            yield obj
        else:
            for value in obj.values():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            yield item

def type_name(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__

def inspect_archive(scenario, archive_path, out):
    print(f"\n{'=' * 76}\nSCENARIO: {scenario}\nARCHIVE: {archive_path.name}", file=out)

    counts = Counter()
    labels = defaultdict(Counter)
    paths = defaultdict(Counter)
    examples = {}
    filenames = defaultdict(set)
    errors = []

    with zipfile.ZipFile(archive_path) as outer:
        split_archives = [
            name for name in outer.namelist()
            if name.lower().endswith(".zip")
            and any(f"/{s}/" in name.replace("\\", "/")
                    for s in ("Train", "Validation", "Test"))
        ]

        for nested_name in sorted(split_archives):
            normalized = nested_name.replace("\\", "/")
            split = next(
                (s for s in ("Train", "Validation", "Test")
                 if f"/{s}/" in normalized),
                "UNKNOWN"
            )

            try:
                nested_bytes = outer.read(nested_name)
                with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner:
                    json_names = [
                        n for n in inner.namelist()
                        if n.lower().endswith(".json")
                    ]

                    for json_name in json_names:
                        try:
                            obj = json.loads(inner.read(json_name))
                            for record in records_from_json(obj):
                                counts[split] += 1
                                label = str(record.get("attacker", "MISSING")).strip()
                                labels[split][label] += 1

                                filenames[split].add(json_name)

                                for key, value in record.items():
                                    path = key
                                    paths[(split, path)][type_name(value)] += 1
                                    examples.setdefault(
                                        (split, path), repr(value)[:180]
                                    )

                                    if isinstance(value, dict):
                                        for subkey, subvalue in value.items():
                                            subpath = f"{key}.{subkey}"
                                            paths[(split, subpath)][
                                                type_name(subvalue)
                                            ] += 1
                                            examples.setdefault(
                                                (split, subpath),
                                                repr(subvalue)[:180]
                                            )
                        except Exception as exc:
                            errors.append(
                                f"{split}/{json_name}: {type(exc).__name__}: {exc}"
                            )
            except Exception as exc:
                errors.append(
                    f"{nested_name}: {type(exc).__name__}: {exc}"
                )

    print("\nRECORD AND LABEL COUNTS", file=out)
    for split in ("Train", "Validation", "Test", "UNKNOWN"):
        if split not in counts:
            continue
        total = counts[split]
        print(
            f"{split}: records={total:,}, JSON files={len(filenames[split]):,}, "
            f"labels={dict(labels[split])}",
            file=out
        )
        if total:
            for label, count in sorted(labels[split].items()):
                print(f"  attacker={label}: {count / total:.2%}", file=out)

    print("\nFIELD PATHS AND TYPES", file=out)
    for (split, path), type_counts in sorted(paths.items()):
        print(
            f"{split:10s} {path:32s} types={dict(type_counts)} "
            f"example={examples[(split, path)]}",
            file=out
        )

    if errors:
        print(f"\nERRORS ({len(errors)})", file=out)
        for error in errors[:30]:
            print(error, file=out)

if __name__ == "__main__":
    report = ROOT / "schema_report.txt"
    with report.open("w", encoding="utf-8") as out:
        for scenario, path in ARCHIVES:
            if not path.exists():
                print(f"Missing archive: {path}", file=out)
                continue
            inspect_archive(scenario, path, out)

    print(f"Report saved: {report}")
    print("\n========== REPORT PREVIEW ==========")
    print(report.read_text(encoding="utf-8")[:24000])
