#!/usr/bin/env python3
"""Build temporal vehicle graphs from official nested InTAS ZIP archives.

Graph construction is label-blind:
- attacker labels are written only to a separate audit/label stream.
- sender IDs are node identifiers, never predictive features.
- absolute positions are used only for optional proximity-edge construction.
- proximity edges are inferred spatial relations, not observed network links.
- timestamps and distance thresholds use raw dataset units unless verified.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from trustrl_data import SPLITS, make_features, numeric, parse_vector, walk_records


ARCHIVES = {
    "dataReplay": "InTAS_highway_2_dataReplay.zip",
    "suddenConstantSpeed": "InTAS_highway_2_suddenConstantSpeed.zip",
}


def read_records(archive_path: Path, split: str):
    """Yield records from the requested official split without extraction."""
    with zipfile.ZipFile(archive_path) as outer:
        nested_names = []
        for name in outer.namelist():
            if not name.lower().endswith(".zip"):
                continue
            parts = name.replace("\\", "/").split("/")
            if any(part.casefold() == split.casefold() for part in parts):
                nested_names.append(name)

        if not nested_names:
            raise RuntimeError(f"No nested ZIP for split={split}: {archive_path}")

        for nested_name in sorted(nested_names):
            with zipfile.ZipFile(io.BytesIO(outer.read(nested_name))) as inner:
                for json_name in sorted(inner.namelist()):
                    if not json_name.lower().endswith(".json"):
                        continue
                    try:
                        obj = json.loads(inner.read(json_name))
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise ValueError(
                            f"Invalid JSON: {nested_name}/{json_name}"
                        ) from exc
                    yield from walk_records(obj)


def sender_state(record: dict[str, Any]) -> dict[str, Any]:
    """Extract sender observations; omit absolute position from node features."""
    sender = record.get("sender")
    if not isinstance(sender, dict):
        sender = {}

    features: dict[str, Any] = {}
    profile = sender.get("driversProfile")
    features["driversProfile"] = str(profile) if profile is not None else None

    for key in (
        "spd", "spd_noise", "acl", "acl_noise",
        "hed", "hed_noise", "distance_to_road_edge",
    ):
        value = numeric(sender.get(key))
        if value is not None:
            features[key] = value

    pos = parse_vector(sender.get("pos"))
    # Coordinates are retained only as construction metadata for edge creation.
    return {"features": features, "position": pos}


def aggregate_features(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate each vehicle's observations inside one temporal window."""
    numeric_values: dict[str, list[float]] = defaultdict(list)
    categories: dict[str, Counter] = defaultdict(Counter)

    for obs in observations:
        for key, value in obs.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                numeric_values[key].append(float(value))
            elif isinstance(value, str):
                categories[key][value] += 1

    result: dict[str, Any] = {}
    for key, values in sorted(numeric_values.items()):
        result[f"{key}_mean"] = statistics.fmean(values)
        result[f"{key}_last"] = values[-1]
    for key, counts in sorted(categories.items()):
        result[key] = sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[0][0]

    return result


def build_snapshot(
    window_index: int,
    window_start: float,
    window_end: float,
    records: list[dict[str, Any]],
    distance_threshold: float,
) -> dict[str, Any]:
    """Build one graph snapshot and separately return label audit rows."""
    by_sender: dict[str, list[dict[str, Any]]] = defaultdict(list)
    latest_state: dict[str, dict[str, Any]] = {}
    label_rows = []

    ordered = sorted(
        records,
        key=lambda rec: (
            numeric(rec.get("rcvTime"))
            if numeric(rec.get("rcvTime")) is not None else math.inf,
            str(rec.get("sender_id", "")),
            str(rec.get("messageID", "")),
        ),
    )

    for record in ordered:
        sender_id = record.get("sender_id")
        timestamp = numeric(record.get("rcvTime"))

        if sender_id is None or timestamp is None:
            raise ValueError("Record missing sender_id or valid rcvTime")

        sender_id = str(sender_id)
        state = sender_state(record)
        by_sender[sender_id].append(state["features"])
        latest_state[sender_id] = state

        label = numeric(record.get("attacker"))
        if label not in (0.0, 1.0):
            raise ValueError(f"Invalid attacker label: {label!r}")

        # Labels are written separately; never included in graph structure/features.
        label_rows.append({
            "window_index": window_index,
            "sender_id": sender_id,
            "messageID": record.get("messageID"),
            "rcvTime": record.get("rcvTime"),
            "sendTime": record.get("sendTime"),
            "label": int(label),
        })

    nodes = []
    for sender_id in sorted(by_sender):
        positions = [
            state["position"]
            for state in []
        ]
        del positions

        node = {
            "id": sender_id,
            "features": aggregate_features(by_sender[sender_id]),
            "message_count": len(by_sender[sender_id]),
        }
        nodes.append(node)

    # Keep the last valid position for each sender within this window.
    positions_by_sender = {}
    for sender_id in sorted(latest_state):
        position = latest_state[sender_id]["position"]
        if position is not None:
            positions_by_sender[sender_id] = position

    edges = []
    positioned = sorted(positions_by_sender)
    threshold_sq = distance_threshold * distance_threshold

    for i, source in enumerate(positioned):
        p1 = positions_by_sender[source]
        for target in positioned[i + 1:]:
            p2 = positions_by_sender[target]
            distance_sq = sum((a - b) ** 2 for a, b in zip(p1, p2))
            if distance_sq <= threshold_sq:
                edges.append({
                    "source": source,
                    "target": target,
                    "relation": "inferred_spatial_proximity",
                    "distance_coordinate_units": math.sqrt(distance_sq),
                })

    graph = {
        "window_index": window_index,
        "window_start_raw": window_start,
        "window_end_raw": window_end,
        "timestamp_unit": "unverified_raw_dataset_units",
        "edge_semantics": "inferred spatial proximity; not a recorded network link",
        "distance_threshold_coordinate_units": distance_threshold,
        "nodes": nodes,
        "edges": edges,
    }
    return graph, label_rows


def process_split(
    archive_path: Path,
    scenario: str,
    split: str,
    output_root: Path,
    window_size: float,
    distance_threshold: float,
) -> dict[str, int]:
    records = []
    labels_seen = Counter()
    timestamps = []

    for record in read_records(archive_path, split):
        timestamp = numeric(record.get("rcvTime"))
        if timestamp is None:
            raise ValueError(f"Invalid rcvTime in {scenario}/{split}")
        if record.get("sender_id") is None:
            raise ValueError(f"Missing sender_id in {scenario}/{split}")

        label = numeric(record.get("attacker"))
        if label not in (0.0, 1.0):
            raise ValueError(f"Invalid attacker label in {scenario}/{split}")

        labels_seen[int(label)] += 1
        timestamps.append(timestamp)
        records.append(record)

    if not records:
        raise RuntimeError(f"No records loaded for {scenario}/{split}")

    minimum = min(timestamps)
    maximum = max(timestamps)

    windows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        timestamp = numeric(record["rcvTime"])
        index = int((timestamp - minimum) // window_size)
        windows[index].append(record)

    output_dir = output_root / scenario / split
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = output_dir / "snapshots.jsonl"
    labels_path = output_dir / "message_labels.jsonl"

    total_nodes = 0
    total_edges = 0
    graph_count = 0
    all_label_rows = []

    with graph_path.open("w", encoding="utf-8") as graph_file:
        for index in sorted(windows):
            start = minimum + index * window_size
            end = start + window_size

            graph, label_rows = build_snapshot(
                index,
                start,
                end,
                windows[index],
                distance_threshold,
            )
            graph_file.write(json.dumps(graph, ensure_ascii=False) + "\n")
            all_label_rows.extend(label_rows)

            graph_count += 1
            total_nodes += len(graph["nodes"])
            total_edges += len(graph["edges"])

    with labels_path.open("w", encoding="utf-8") as labels_file:
        for row in all_label_rows:
            labels_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "records": len(records),
        "benign": labels_seen[0],
        "attack": labels_seen[1],
        "snapshots": graph_count,
        "node_instances": total_nodes,
        "inferred_edges": total_edges,
        "unique_senders": len({str(r["sender_id"]) for r in records}),
    }

    print(
        f"{scenario:22s} {split:10s} "
        f"records={summary['records']:,} "
        f"senders={summary['unique_senders']:,} "
        f"snapshots={summary['snapshots']:,} "
        f"node_instances={summary['node_instances']:,} "
        f"spatial_edges={summary['inferred_edges']:,} "
        f"benign={summary['benign']:,} attack={summary['attack']:,}"
    )
    print(f"  Graphs: {graph_path}")
    print(f"  Labels: {labels_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-size",
        type=float,
        default=1_000_000_000,
        help="Window width in raw rcvTime units (default: 1e9; unit unverified).",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=50.0,
        help="Spatial edge threshold in raw coordinate units (default: 50).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("graph_outputs"),
        help="Output root directory.",
    )
    args = parser.parse_args()

    if not math.isfinite(args.window_size) or args.window_size <= 0:
        parser.error("--window-size must be a finite positive number")
    if not math.isfinite(args.distance_threshold) or args.distance_threshold < 0:
        parser.error("--distance-threshold must be finite and non-negative")

    root = Path(__file__).resolve().parent

    for scenario, filename in ARCHIVES.items():
        archive_path = root / filename
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)

        print("\n" + "=" * 78)
        print(f"SCENARIO: {scenario}")
        print(f"window_size_raw={args.window_size}")
        print(f"distance_threshold_coordinate_units={args.distance_threshold}")

        for split in SPLITS:
            process_split(
                archive_path,
                scenario,
                split,
                args.output,
                args.window_size,
                args.distance_threshold,
            )


if __name__ == "__main__":
    main()
