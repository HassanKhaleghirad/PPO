#!/usr/bin/env python3
"""Independent InTAS loader for TrustRL.

Reads official Train/Validation/Test splits from nested ZIP archives.
Features, labels and identity/time metadata remain separate.
Uses Python standard library only.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SPLITS = ("Train", "Validation", "Test")

# Counts verified during the previous archive audit.
EXPECTED = {
    "dataReplay": {
        "Train": (37701, 6541),
        "Validation": (17977, 3212),
        "Test": (44259, 8743),
    },
    "suddenConstantSpeed": {
        "Train": (37701, 1528),
        "Validation": (17977, 565),
        "Test": (44259, 1571),
    },
}

METADATA_FIELDS = (
    "sender_id",
    "sender_alias",
    "messageID",
    "rcvTime",
    "sendTime",
)


@dataclass(frozen=True)
class Sample:
    """One message, with prediction inputs isolated from its label."""

    features: dict[str, float | str | None]
    label: int
    metadata: dict[str, Any]


def walk_records(obj: Any):
    """Yield message records containing an attacker field."""
    if isinstance(obj, dict):
        if "attacker" in obj:
            yield obj
        else:
            for value in obj.values():
                yield from walk_records(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from walk_records(item)


def numeric(value: Any) -> float | None:
    """Return finite numeric values, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    number = float(value)
    return number if math.isfinite(number) else None


def parse_vector(value: Any) -> list[float] | None:
    """Parse a 3D vector represented as a list or comma-separated string."""
    if isinstance(value, (list, tuple)):
        parts = value
    elif isinstance(value, str):
        parts = value.split(",")
    else:
        return None

    try:
        numbers = [float(part.strip()) for part in parts]
    except (ValueError, TypeError, AttributeError):
        return None

    if len(numbers) < 3 or not all(
        math.isfinite(number) for number in numbers[:3]
    ):
        return None

    return numbers[:3]


def make_features(record: dict[str, Any]) -> dict[str, float | str | None]:
    """Build predictive features without IDs, timestamps or attacker labels."""
    features: dict[str, float | str | None] = {}

    received = numeric(record.get("rcvTime"))
    sent = numeric(record.get("sendTime"))

    # Assumes timestamps use microseconds, as in the existing baseline.
    if received is not None and sent is not None:
        features["message_delay_s"] = (received - sent) / 1_000_000.0

    positions: dict[str, list[float] | None] = {}

    for object_name in ("sender", "receiver"):
        obj = record.get(object_name, {})
        if not isinstance(obj, dict):
            obj = {}

        profile = obj.get("driversProfile")
        features[f"{object_name}_driversProfile"] = (
            str(profile) if profile is not None else None
        )

        for key, value in obj.items():
            if key in ("pos", "pos_noise", "driversProfile"):
                continue

            number = numeric(value)
            if number is not None:
                features[f"{object_name}_{key}"] = number

        positions[object_name] = parse_vector(obj.get("pos"))
        noise = parse_vector(obj.get("pos_noise"))

        if noise is not None:
            for axis, number in zip(("x", "y", "z"), noise):
                features[f"{object_name}_pos_noise_{axis}"] = number

    sender_pos = positions.get("sender")
    receiver_pos = positions.get("receiver")

    if sender_pos is not None and receiver_pos is not None:
        dx = sender_pos[0] - receiver_pos[0]
        dy = sender_pos[1] - receiver_pos[1]
        dz = sender_pos[2] - receiver_pos[2]

        features["relative_pos_dx"] = dx
        features["relative_pos_dy"] = dy
        features["relative_pos_dz"] = dz
        features["relative_distance"] = math.sqrt(
            dx * dx + dy * dy + dz * dz
        )

    for key in ("spd", "acl", "spd_noise", "acl_noise"):
        sender_value = features.get(f"sender_{key}")
        receiver_value = features.get(f"receiver_{key}")

        if isinstance(sender_value, float) and isinstance(receiver_value, float):
            features[f"delta_{key}"] = sender_value - receiver_value

    sender_heading = features.get("sender_hed")
    receiver_heading = features.get("receiver_hed")

    if isinstance(sender_heading, float) and isinstance(receiver_heading, float):
        features["heading_difference"] = abs(
            (sender_heading - receiver_heading + 180) % 360 - 180
        )

    # Defense in depth: labels and raw identity/time fields are never features.
    forbidden = {"attacker", *METADATA_FIELDS}
    leaked = forbidden.intersection(features)
    if leaked:
        raise AssertionError(f"Forbidden fields in features: {sorted(leaked)}")

    return features


def scenario_name(path: Path) -> str:
    name = path.name.lower()
    if "datareplay" in name:
        return "dataReplay"
    if "suddenconstantspeed" in name:
        return "suddenConstantSpeed"
    raise ValueError(f"Cannot infer scenario from archive name: {path.name}")


def load_scenario(
    archive_path: str | Path,
    *,
    validate_counts: bool = True,
) -> dict[str, list[Sample]]:
    """Read all official splits directly from the outer and nested ZIPs."""
    path = Path(archive_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    scenario = scenario_name(path)
    frames: dict[str, list[Sample]] = {split: [] for split in SPLITS}

    with zipfile.ZipFile(path) as outer:
        outer_names = outer.namelist()

        for split in SPLITS:
            nested_names = []
            for name in outer_names:
                if not name.lower().endswith(".zip"):
                    continue

                parts = name.replace("\\", "/").split("/")
                if any(part.casefold() == split.casefold() for part in parts):
                    nested_names.append(name)

            if not nested_names:
                raise RuntimeError(
                    f"No nested ZIP found for split {split!r} in {path}"
                )

            for nested_name in sorted(nested_names):
                nested_bytes = outer.read(nested_name)

                with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner:
                    json_names = sorted(
                        name for name in inner.namelist()
                        if name.lower().endswith(".json")
                    )

                    for json_name in json_names:
                        try:
                            obj = json.loads(inner.read(json_name))
                        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                            raise ValueError(
                                f"Invalid JSON: {nested_name}/{json_name}"
                            ) from exc

                        for record in walk_records(obj):
                            raw_label = numeric(record.get("attacker"))
                            if raw_label not in (0.0, 1.0):
                                raise ValueError(
                                    f"Invalid attacker label {raw_label!r} "
                                    f"in {nested_name}/{json_name}"
                                )

                            metadata = {
                                key: record.get(key)
                                for key in METADATA_FIELDS
                            }

                            # Preserve optional explicit receiver identifiers
                            # only as metadata, never as predictive features.
                            if "receiver_id" in record:
                                metadata["receiver_id"] = record["receiver_id"]

                            frames[split].append(
                                Sample(
                                    features=make_features(record),
                                    label=int(raw_label),
                                    metadata=metadata,
                                )
                            )

    for split in SPLITS:
        samples = frames[split]
        counts = Counter(sample.label for sample in samples)
        positives = counts.get(1, 0)
        negatives = counts.get(0, 0)

        print(
            f"{scenario:22s} {split:10s} "
            f"records={len(samples):,} "
            f"benign={negatives:,} attack={positives:,}"
        )

        if validate_counts:
            expected_total, expected_positive = EXPECTED[scenario][split]
            if len(samples) != expected_total or positives != expected_positive:
                raise RuntimeError(
                    f"Audit mismatch for {scenario}/{split}: "
                    f"got total={len(samples)}, attack={positives}; "
                    f"expected total={expected_total}, "
                    f"attack={expected_positive}"
                )

    return frames


def main() -> None:
    root = Path(__file__).resolve().parent
    archives = (
        root / "InTAS_highway_2_dataReplay.zip",
        root / "InTAS_highway_2_suddenConstantSpeed.zip",
    )

    for archive in archives:
        frames = load_scenario(archive)
        print(f"Validated: {archive.name}")

        # Sanity-check feature/label/metadata separation.
        sample = next(
            sample
            for split in SPLITS
            for sample in frames[split]
        )
        assert "attacker" not in sample.features
        assert "sender_id" not in sample.features
        assert "messageID" not in sample.features
        assert "rcvTime" not in sample.features
        assert "attacker" not in sample.metadata

        print(
            f"Separation check passed; example feature count="
            f"{len(sample.features)}, label={sample.label}, "
            f"metadata keys={sorted(sample.metadata)}"
        )
        del frames


if __name__ == "__main__":
    main()
