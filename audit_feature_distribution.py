#!/usr/bin/env python3
import csv
import io
import json
import math
import statistics
import zipfile
import logging
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ARCHIVES = {
    "dataReplay": ROOT / "InTAS_highway_2_dataReplay.zip",
    "suddenConstantSpeed": ROOT / "InTAS_highway_2_suddenConstantSpeed.zip",
}

SPLITS = ("Train", "Validation", "Test")

# شناسه‌ها و برچسب‌ها نباید به‌عنوان ویژگی رفتاری تفسیر شوند.
EXCLUDED_FEATURES = {
    "attacker",
    "sender_id",
    "sender_alias",
    "messageID",
}

logging.basicConfig(
    filename=ROOT / "audit_feature_distribution.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def split_from_path(name):
    parts = name.replace("\\", "/").split("/")
    for part in parts:
        if part in SPLITS:
            return part
    return None


def flatten_record(obj, prefix=""):
    """Flatten nested dictionaries and comma-separated numeric vectors."""
    result = {}

    if not isinstance(obj, dict):
        return result

    for key, value in obj.items():
        name = f"{prefix}.{key}" if prefix else key

        if name in EXCLUDED_FEATURES or key in EXCLUDED_FEATURES:
            continue

        if isinstance(value, dict):
            result.update(flatten_record(value, name))
            continue

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)):
                result[name] = float(value)
            continue

        # بعضی نسخه‌ها بردارها را به صورت "x,y,z" ذخیره می‌کنند.
        if isinstance(value, str) and "," in value:
            parts = value.split(",")
            try:
                numbers = [float(p.strip()) for p in parts]
                if all(math.isfinite(x) for x in numbers):
                    for i, number in enumerate(numbers):
                        result[f"{name}[{i}]"] = number
            except ValueError:
                pass

    return result


def iter_records(outer_zip, scenario):
    """Read outer ZIP, then the ZIP belonging to each split."""
    with zipfile.ZipFile(outer_zip) as outer:
        for inner_name in outer.namelist():
            if not inner_name.lower().endswith(".zip"):
                continue

            split = split_from_path(inner_name)
            if split not in SPLITS:
                continue

            try:
                inner_bytes = outer.read(inner_name)
                with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                    for member in inner.namelist():
                        if not member.lower().endswith(".json"):
                            continue

                        try:
                            raw = inner.read(member)
                            data = json.loads(raw.decode("utf-8-sig"))
                        except Exception as exc:
                            log.warning(
                                "JSON read failure: %s!/%s: %s",
                                inner_name, member, exc
                            )
                            continue

                        if isinstance(data, dict):
                            data = [data]

                        if not isinstance(data, list):
                            continue

                        for record in data:
                            if isinstance(record, dict):
                                yield scenario, split, member, record

            except Exception:
                log.exception("Nested ZIP failure: %s", inner_name)


def safe_stats(values):
    if not values:
        return {
            "n": 0, "missing": 0,
            "mean": "", "std": "", "min": "",
            "q01": "", "q25": "", "median": "",
            "q75": "", "q99": "", "max": "",
        }

    values = sorted(values)
    n = len(values)

    def quantile(q):
        if n == 1:
            return values[0]
        pos = (n - 1) * q
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return values[lo] * (1 - frac) + values[hi] * frac

    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if n > 1 else 0.0

    return {
        "n": n,
        "missing": "",
        "mean": mean,
        "std": std,
        "min": values[0],
        "q01": quantile(0.01),
        "q25": quantile(0.25),
        "median": quantile(0.50),
        "q75": quantile(0.75),
        "q99": quantile(0.99),
        "max": values[-1],
    }


def main():
    label_counts = Counter()
    feature_values = defaultdict(list)
    feature_presence = Counter()
    vehicle_ids = defaultdict(set)
    record_counts = Counter()
    schema_counts = Counter()
    time_ranges = defaultdict(lambda: {"min": None, "max": None})

    for scenario, archive in ARCHIVES.items():
        if not archive.exists():
            raise FileNotFoundError(f"Archive not found: {archive}")

        with zipfile.ZipFile(archive) as z:
            bad = z.testzip()
            if bad:
                raise RuntimeError(f"ZIP integrity error: {archive}: {bad}")

        log.info("Processing %s", scenario)

        for _, split, member, record in iter_records(archive, scenario):
            key = (scenario, split)
            record_counts[key] += 1
            schema_counts[(scenario, split, tuple(sorted(record.keys())))] += 1

            label = str(record.get("attacker", "")).strip()
            if label in ("0", "1"):
                label_counts[(scenario, split, label)] += 1
            else:
                label_counts[(scenario, split, "UNKNOWN")] += 1

            sender_id = record.get("sender_id")
            if sender_id is not None:
                vehicle_ids[key].add(str(sender_id))

            # زمان خام برای تحلیل جداگانه؛ شناسه زمانی مدل نیست.
            for time_field in ("rcvTime", "sendTime"):
                try:
                    t = float(record[time_field])
                    if math.isfinite(t):
                        slot = time_ranges[(scenario, split, time_field)]
                        slot["min"] = t if slot["min"] is None else min(slot["min"], t)
                        slot["max"] = t if slot["max"] is None else max(slot["max"], t)
                except (KeyError, TypeError, ValueError):
                    pass

            features = flatten_record(record)

            # نرخ حضور هر ویژگی برای تشخیص مقادیر گمشده
            for name in features:
                feature_presence[(scenario, split, name)] += 1
                feature_values[(scenario, split, name)].append(features[name])

    # 1) Class distribution
    label_rows = []
    for scenario in ARCHIVES:
        for split in SPLITS:
            n = record_counts[(scenario, split)]
            positives = label_counts[(scenario, split, "1")]
            negatives = label_counts[(scenario, split, "0")]
            unknown = label_counts[(scenario, split, "UNKNOWN")]

            label_rows.append({
                "scenario": scenario,
                "split": split,
                "records": n,
                "attacker_0": negatives,
                "attacker_1": positives,
                "unknown_label": unknown,
                "positive_rate_pct": (
                    100 * positives / (positives + negatives)
                    if positives + negatives else ""
                ),
            })

    write_csv(
        ROOT / "label_distribution.csv",
        label_rows,
        ["scenario", "split", "records", "attacker_0",
         "attacker_1", "unknown_label", "positive_rate_pct"],
    )

    # 2) Feature distributions
    feature_rows = []
    for (scenario, split, feature), values in sorted(feature_values.items()):
        stats = safe_stats(values)
        total = record_counts[(scenario, split)]
        present = feature_presence[(scenario, split, feature)]

        feature_rows.append({
            "scenario": scenario,
            "split": split,
            "feature": feature,
            "records": total,
            "present": present,
            "missing": total - present,
            "missing_pct": 100 * (total - present) / total if total else "",
            **stats,
        })

    write_csv(
        ROOT / "feature_distribution.csv",
        feature_rows,
        ["scenario", "split", "feature", "records", "present",
         "missing", "missing_pct", "n", "mean", "std", "min",
         "q01", "q25", "median", "q75", "q99", "max"],
    )

    # 3) Compare Train vs Validation/Test using mean and std.
    # This is a descriptive effect-size audit, not a formal hypothesis test.
    shift_rows = []
    for scenario in ARCHIVES:
        features = sorted({
            feature
            for (s, split, feature) in feature_values
            if s == scenario
        })

        for feature in features:
            train = feature_values.get((scenario, "Train", feature), [])
            if not train:
                continue

            train_mean = statistics.fmean(train)
            train_std = statistics.pstdev(train) if len(train) > 1 else 0.0
            train_median = statistics.median(train)

            for split in ("Validation", "Test"):
                target = feature_values.get((scenario, split, feature), [])
                if not target:
                    shift_rows.append({
                        "scenario": scenario, "feature": feature,
                        "comparison": f"Train_vs_{split}",
                        "train_n": len(train), "target_n": 0,
                        "train_mean": train_mean, "target_mean": "",
                        "standardized_mean_difference": "",
                        "train_median": train_median,
                        "target_median": "",
                        "median_difference": "",
                    })
                    continue

                target_mean = statistics.fmean(target)
                target_std = (
                    statistics.pstdev(target) if len(target) > 1 else 0.0
                )
                target_median = statistics.median(target)

                pooled = math.sqrt(
                    (train_std ** 2 + target_std ** 2) / 2
                )
                smd = (target_mean - train_mean) / pooled if pooled else ""

                shift_rows.append({
                    "scenario": scenario,
                    "feature": feature,
                    "comparison": f"Train_vs_{split}",
                    "train_n": len(train),
                    "target_n": len(target),
                    "train_mean": train_mean,
                    "target_mean": target_mean,
                    "standardized_mean_difference": smd,
                    "train_median": train_median,
                    "target_median": target_median,
                    "median_difference": target_median - train_median,
                })

    write_csv(
        ROOT / "feature_shift.csv",
        shift_rows,
        ["scenario", "feature", "comparison", "train_n", "target_n",
         "train_mean", "target_mean", "standardized_mean_difference",
         "train_median", "target_median", "median_difference"],
    )

    # 4) Vehicle ID overlap. Interpret carefully: repeated IDs are not
    # automatically leakage if the official split is temporal.
    overlap_rows = []
    for scenario in ARCHIVES:
        for left, right in (
            ("Train", "Validation"),
            ("Train", "Test"),
            ("Validation", "Test"),
        ):
            a = vehicle_ids[(scenario, left)]
            b = vehicle_ids[(scenario, right)]
            overlap = a & b

            overlap_rows.append({
                "scenario": scenario,
                "split_a": left,
                "split_b": right,
                "unique_ids_a": len(a),
                "unique_ids_b": len(b),
                "overlap_ids": len(overlap),
                "overlap_pct_of_a": 100 * len(overlap) / len(a) if a else "",
                "overlap_pct_of_b": 100 * len(overlap) / len(b) if b else "",
            })

    write_csv(
        ROOT / "vehicle_overlap.csv",
        overlap_rows,
        ["scenario", "split_a", "split_b", "unique_ids_a",
         "unique_ids_b", "overlap_ids",
         "overlap_pct_of_a", "overlap_pct_of_b"],
    )

    # 5) Compact console summary
    print("\n=== LABEL DISTRIBUTION ===")
    for row in label_rows:
        print(
            f"{row['scenario']:22s} {row['split']:10s} "
            f"n={row['records']:6d} "
            f"attack={row['attacker_1']:6d} "
            f"rate={row['positive_rate_pct']}"
        )

    print("\n=== VEHICLE OVERLAP ===")
    for row in overlap_rows:
        print(row)

    print("\n=== OUTPUT FILES ===")
    for name in (
        "label_distribution.csv",
        "feature_distribution.csv",
        "feature_shift.csv",
        "vehicle_overlap.csv",
        "audit_feature_distribution.log",
    ):
        print(ROOT / name)

    print("\nAudit completed. No model was trained.")


if __name__ == "__main__":
    main()
