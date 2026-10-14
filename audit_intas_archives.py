#!/usr/bin/env python3

import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

BASE = Path("/home/h/Documents/PPO")

ARCHIVES = {
    "dataReplay": {
        "file": "InTAS_highway_2_dataReplay.zip",
        "md5": "9515b57c25dd34304642654dd22df66b",
    },
    "suddenConstantSpeed": {
        "file": "InTAS_highway_2_suddenConstantSpeed.zip",
        "md5": "32351002170f4690460a9f585d7d6048",
    },
}

MAX_NESTING = 5
MAX_SAMPLE_RECORDS = 3

SPLIT_NAMES = {
    "train": "Train",
    "training": "Train",
    "validation": "Validation",
    "valid": "Validation",
    "val": "Validation",
    "test": "Test",
    "testing": "Test",
}


def md5_file(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_from_path(path):
    for part in re.split(r"[/!\\]+", path.lower()):
        if part in SPLIT_NAMES:
            return SPLIT_NAMES[part]
    return "UNKNOWN"


def json_records(obj, path="$"):
    """Find dictionaries containing an attacker field at any nesting level."""
    if isinstance(obj, dict):
        if "attacker" in obj:
            yield path, obj

        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                yield from json_records(value, f"{path}.{key}")

    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            if isinstance(value, (dict, list)):
                yield from json_records(value, f"{path}[{index}]")


def json_documents_from_zip(zf, prefix="", depth=0):
    """Read JSON documents recursively without extracting archive contents."""
    if depth > MAX_NESTING:
        print(f"NESTING_LIMIT | {prefix}")
        return

    for info in zf.infolist():
        if info.is_dir():
            continue

        name = f"{prefix}{info.filename}"

        try:
            raw = zf.read(info)
        except Exception as exc:
            print(f"READ_ERROR | {name} | {exc}")
            continue

        if info.filename.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as nested:
                    yield from json_documents_from_zip(
                        nested, prefix=name + "!/", depth=depth + 1
                    )
            except Exception as exc:
                print(f"NESTED_ZIP_ERROR | {name} | {exc}")

        elif info.filename.lower().endswith(".json"):
            try:
                obj = json.loads(raw.decode("utf-8-sig"))
                yield name, obj
            except Exception as exc:
                print(f"JSON_ERROR | {name} | {exc}")


def compact(value, limit=160):
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def find_fields(record):
    """Report plausible identifiers and timestamps without assuming a schema."""
    interesting = re.compile(
        r"attacker|label|class|vehicle|sender|receiver|station|"
        r"pseudonym|id$|time|timestamp|position|message|event",
        re.IGNORECASE,
    )
    return {
        key: compact(value)
        for key, value in record.items()
        if interesting.search(str(key))
    }


def audit_archive(scenario, config, report, csv_writer):
    path = BASE / config["file"]

    print("\n" + "=" * 88, file=report)
    print(f"SCENARIO: {scenario}", file=report)
    print(f"ARCHIVE: {path.name}", file=report)

    if not path.is_file():
        print("STATUS: MISSING ARCHIVE", file=report)
        print(f"ERROR: Missing file: {path}")
        return

    actual_md5 = md5_file(path)
    expected_md5 = config["md5"]
    checksum_ok = actual_md5 == expected_md5

    print(f"SIZE_BYTES: {path.stat().st_size}", file=report)
    print(f"EXPECTED_MD5: {expected_md5}", file=report)
    print(f"ACTUAL_MD5: {actual_md5}", file=report)
    print(f"MD5_OK: {checksum_ok}", file=report)

    if not checksum_ok:
        print("STATUS: STOPPED - CHECKSUM MISMATCH", file=report)
        print(f"ERROR: Checksum mismatch: {path.name}")
        return

    try:
        zf = zipfile.ZipFile(path)
    except Exception as exc:
        print(f"ZIP_OPEN_ERROR: {exc}", file=report)
        return

    with zf:
        bad_member = zf.testzip()
        print(f"ZIP_INTEGRITY: {'OK' if bad_member is None else bad_member}",
              file=report)

        if bad_member is not None:
            print(f"ERROR: Corrupt ZIP member: {bad_member}")
            return

        all_paths = [
            info.filename for info in zf.infolist() if not info.is_dir()
        ]

        print(f"TOP_LEVEL_FILE_COUNT: {len(all_paths)}", file=report)
        print("ARCHIVE_PATHS (first 100):", file=report)

        for member in all_paths[:100]:
            print(f"  {member}", file=report)

        extension_counts = Counter(
            Path(p.split("!/")[-1]).suffix.lower() or "[no extension]"
            for p in all_paths
        )
        print(f"TOP_LEVEL_EXTENSIONS: {dict(extension_counts)}", file=report)

        split_paths = Counter(split_from_path(p) for p in all_paths)
        print(f"TOP_LEVEL_PATH_SPLITS: {dict(split_paths)}", file=report)

        file_count = Counter()
        document_count = Counter()
        record_count = Counter()
        attacker_values = Counter()
        missing_attacker = Counter()
        record_key_counts = Counter()
        sample_records = Counter()
        errors = Counter()
        split_record_keys = {}
        seen_fingerprints = set()
        duplicate_records = Counter()

        for member, obj in json_documents_from_zip(zf):
            split = split_from_path(member)
            file_count[split] += 1
            document_count[split] += 1

            records = list(json_records(obj))
            if not records:
                missing_attacker[split] += 1

            for record_path, record in records:
                record_count[split] += 1
                attacker = record.get("attacker")
                attacker_values[(split, compact(attacker))] += 1

                keys = tuple(sorted(map(str, record.keys())))
                record_key_counts[(split, keys)] += 1

                fingerprint = json.dumps(
                    record, sort_keys=True, ensure_ascii=False, default=str
                )
                if fingerprint in seen_fingerprints:
                    duplicate_records[split] += 1
                else:
                    seen_fingerprints.add(fingerprint)

                if sample_records[split] < MAX_SAMPLE_RECORDS:
                    sample_records[split] += 1
                    print(
                        f"SAMPLE_RECORD | split={split} | file={member} | "
                        f"path={record_path}",
                        file=report,
                    )
                    print(f"  KEYS: {list(record.keys())}", file=report)
                    print(f"  FIELDS: {find_fields(record)}", file=report)

                csv_writer.writerow({
                    "scenario": scenario,
                    "split": split,
                    "json_file": member,
                    "record_path": record_path,
                    "attacker": compact(attacker),
                    "keys": "|".join(map(str, record.keys())),
                    "fields": json.dumps(
                        find_fields(record), ensure_ascii=False
                    ),
                    "fingerprint": hashlib.sha256(
                        fingerprint.encode("utf-8")
                    ).hexdigest(),
                })

        print("\nJSON AUDIT:", file=report)
        print(f"JSON_DOCUMENTS_BY_SPLIT: {dict(document_count)}", file=report)
        print(f"RECORDS_WITH_ATTACKER_BY_SPLIT: {dict(record_count)}", file=report)
        print(f"JSON_FILES_WITHOUT_ATTACKER_RECORDS: {dict(missing_attacker)}",
              file=report)
        print(f"ATTACKER_DISTRIBUTION: {dict(attacker_values)}", file=report)
        print(f"EXACT_DUPLICATE_RECORDS_WITHIN_ARCHIVE: {dict(duplicate_records)}",
              file=report)

        print("RECORD_SCHEMA_COUNTS:", file=report)
        for (split, keys), count in record_key_counts.most_common(30):
            print(f"  split={split} count={count} keys={keys}", file=report)

        print("\nSPLIT SUMMARY:", file=report)
        for split in ("Train", "Validation", "Test", "UNKNOWN"):
            print(
                f"  {split}: JSON documents={document_count[split]}, "
                f"records with attacker={record_count[split]}",
                file=report,
            )

        print(
            "\nNOTE: UNKNOWN means the split was not identifiable from the "
            "JSON member path. It does not prove that the split is absent.",
            file=report,
        )


def main():
    report_path = BASE / "audit_report.txt"
    records_path = BASE / "audit_records.csv"

    BASE.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as report, \
         records_path.open("w", encoding="utf-8", newline="") as csvfile:

        fields = [
            "scenario", "split", "json_file", "record_path", "attacker",
            "keys", "fields", "fingerprint",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()

        print("InTAS archive audit", file=report)
        print(f"Working directory: {BASE}", file=report)

        for scenario, config in ARCHIVES.items():
            audit_archive(scenario, config, report, writer)

    print("\nAudit finished.")
    print(f"Report:  {report_path}")
    print(f"Records: {records_path}")


if __name__ == "__main__":
    main()
