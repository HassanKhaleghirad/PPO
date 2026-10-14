#!/usr/bin/env python3

import io
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ARCHIVES = [
    Path("InTAS_highway_2_dataReplay.zip"),
    Path("InTAS_highway_2_suddenConstantSpeed.zip"),
]
SPLITS = ("Train", "Validation", "Test")


def walk_records(obj):
    if isinstance(obj, dict):
        if "attacker" in obj:
            yield obj
        else:
            for value in obj.values():
                yield from walk_records(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from walk_records(item)


def number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def summarize(values, name):
    values = sorted(values)
    print(f"\n{name}: count={len(values):,}")
    if not values:
        return

    for percentile in (0, 1, 10, 25, 50, 75, 90, 99, 100):
        index = round((len(values) - 1) * percentile / 100)
        print(f"  p{percentile:03d} = {values[index]:.6f}")


def inspect_archive(path):
    print("\n" + "=" * 78)
    print("ARCHIVE:", path)

    if not path.is_file():
        print("ERROR: archive missing")
        return

    for split in SPLITS:
        timestamps = []
        send_times = []
        delays = []
        message_ids = []
        sender_counts = Counter()
        sender_aliases = defaultdict(set)
        sender_positions = defaultdict(list)
        label_counts = Counter()
        missing_sender = 0
        missing_rcv = 0
        missing_send = 0
        invalid_labels = 0
        total = 0

        with zipfile.ZipFile(path) as outer:
            nested_names = []
            for name in outer.namelist():
                if not name.lower().endswith(".zip"):
                    continue
                parts = name.replace("\\", "/").split("/")
                if any(part.casefold() == split.casefold() for part in parts):
                    nested_names.append(name)

            for nested_name in sorted(nested_names):
                with zipfile.ZipFile(io.BytesIO(outer.read(nested_name))) as inner:
                    for json_name in inner.namelist():
                        if not json_name.lower().endswith(".json"):
                            continue

                        obj = json.loads(inner.read(json_name))
                        for rec in walk_records(obj):
                            total += 1

                            sender = rec.get("sender_id")
                            if sender is None or str(sender).strip() == "":
                                missing_sender += 1
                            else:
                                sender = str(sender)
                                sender_counts[sender] += 1

                                alias = rec.get("sender_alias")
                                if alias is not None:
                                    sender_aliases[sender].add(str(alias))

                                sender_obj = rec.get("sender")
                                if isinstance(sender_obj, dict):
                                    pos = sender_obj.get("pos")
                                    if isinstance(pos, str):
                                        sender_positions[sender].append(pos)

                            rcv = number(rec.get("rcvTime"))
                            send = number(rec.get("sendTime"))

                            if rcv is None:
                                missing_rcv += 1
                            else:
                                timestamps.append(rcv)

                            if send is None:
                                missing_send += 1
                            else:
                                send_times.append(send)

                            if rcv is not None and send is not None:
                                delays.append(rcv - send)

                            mid = number(rec.get("messageID"))
                            if mid is not None:
                                message_ids.append(mid)

                            label = number(rec.get("attacker"))
                            if label in (0.0, 1.0):
                                label_counts[int(label)] += 1
                            else:
                                invalid_labels += 1

        print(f"\n--- {split} ---")
        print(f"records: {total:,}")
        print(f"unique sender_id: {len(sender_counts):,}")
        print(f"records missing sender_id: {missing_sender:,}")
        print(f"records missing rcvTime: {missing_rcv:,}")
        print(f"records missing sendTime: {missing_send:,}")
        print(f"invalid labels: {invalid_labels:,}")
        print(f"label counts: {dict(label_counts)}")

        if timestamps:
            print(
                "rcvTime range:",
                f"{min(timestamps):.0f} .. {max(timestamps):.0f}",
            )
            print("unique rcvTime:", len(set(timestamps)))
            summarize(
                [b - a for a, b in zip(sorted(timestamps), sorted(timestamps)[1:])],
                "Adjacent sorted rcvTime deltas (raw units)",
            )

        if send_times:
            print(
                "sendTime range:",
                f"{min(send_times):.0f} .. {max(send_times):.0f}",
            )
            print("unique sendTime:", len(set(send_times)))

        if delays:
            summarize(delays, "rcvTime - sendTime (raw units)")
            positive = sum(value >= 0 for value in delays)
            print(f"non-negative delays: {positive:,}/{len(delays):,}")

        if message_ids:
            print(
                "messageID range:",
                f"{min(message_ids):.0f} .. {max(message_ids):.0f}",
            )
            print("unique messageID:", len(set(message_ids)))

        print("\nMost frequent senders:")
        for sender, count in sender_counts.most_common(10):
            print(
                f"  {sender}: records={count}, "
                f"aliases={sorted(sender_aliases[sender])[:4]}"
            )

        multi_alias = {
            sender: sorted(aliases)
            for sender, aliases in sender_aliases.items()
            if len(aliases) > 1
        }
        print("senders with multiple aliases:", len(multi_alias))
        for sender, aliases in list(multi_alias.items())[:5]:
            print(f"  {sender}: {aliases[:6]}")

        position_counts = Counter()
        for values in sender_positions.values():
            position_counts[len(values)] += 1
        print("per-sender position sample-count distribution (count: vehicles):")
        print(dict(position_counts.most_common(10)))


def main():
    for archive in ARCHIVES:
        inspect_archive(archive)


if __name__ == "__main__":
    main()
