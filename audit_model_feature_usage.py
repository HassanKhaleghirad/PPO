#!/usr/bin/env python3

import re
from pathlib import Path

ROOT = Path("/home/h/Documents/PPO")
OUTPUT = ROOT / "model_feature_usage_audit.txt"

EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__",
    "site-packages", "node_modules",
}

PATTERN = re.compile(
    r"attacker|sender_id|sender_alias|messageID|"
    r"rcvTime|sendTime|sender|receiver|"
    r"feature|target|label|column|"
    r"\bX_train\b|\bX_test\b|\by_train\b|\by_test\b|"
    r"\bX_val\b|\by_val\b|\bfeatures\b|\bstate\b|"
    r"\bobs\b|\breward\b|\baction\b|"
    r"ColumnTransformer|OneHotEncoder|SimpleImputer|"
    r"StandardScaler|MinMaxScaler|train_test_split|"
    r"flatten|select_dtypes|drop\(|"
    r"observation_space|observation|predict_proba|"
    r"fit_transform|\.fit\(|\.transform\(|"
    r"RandomForest|XGBClassifier|PPO\(",
    re.IGNORECASE,
)

def excluded(path):
    return any(part in EXCLUDED_DIRS for part in path.parts)

def main():
    files = sorted(
        p for p in ROOT.rglob("*.py")
        if not excluded(p.relative_to(ROOT))
    )

    output = []
    output.append("MODEL FEATURE USAGE AUDIT")
    output.append(f"Project: {ROOT}")
    output.append(f"Python source files found: {len(files)}")
    output.append("=" * 100)

    total_matches = 0

    for path in files:
        try:
            lines = path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        except OSError as exc:
            output.append(f"\nREAD ERROR: {path}: {exc}")
            continue

        matches = [
            (number, line)
            for number, line in enumerate(lines, start=1)
            if PATTERN.search(line)
        ]

        if not matches:
            continue

        total_matches += len(matches)
        output.append(
            f"\n\n{'=' * 100}\n"
            f"FILE: {path.relative_to(ROOT)}\n"
            f"MATCHING LINES: {len(matches)}\n"
            f"{'=' * 100}"
        )

        for number, line in matches:
            output.append(f"{number:5}: {line.rstrip()}")

    output.append("\n\n" + "=" * 100)
    output.append(f"TOTAL MATCHING LINES: {total_matches}")
    output.append("END OF AUDIT")

    OUTPUT.write_text("\n".join(output) + "\n", encoding="utf-8")

    print(f"Python files scanned: {len(files)}")
    print(f"Matching source lines: {total_matches}")
    print(f"Full report: {OUTPUT}")
    print("\nPython files potentially implementing baseline/PPO:")

    candidates = [
        p for p in files
        if re.search(
            r"baseline|ppo|train|agent|environment|env|trust|"
            r"feature|model|dataset|veremi|intas",
            p.name,
            re.IGNORECASE,
        )
    ]

    for path in candidates:
        print(" ", path.relative_to(ROOT))

    print("\nFirst 100 matching lines:")
    shown = 0
    for line in output:
        if re.match(r"\s*\d+:", line):
            print(line)
            shown += 1
            if shown >= 100:
                break

    print(f"\nFor the complete report, run:")
    print(f"  less {OUTPUT}")

if __name__ == "__main__":
    main()
