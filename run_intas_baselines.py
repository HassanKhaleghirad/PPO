import io
import json
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path("/home/h/Documents/PPO")
OUT = ROOT / "baseline_results"
OUT.mkdir(exist_ok=True)

SCENARIOS = {
    "dataReplay": ROOT / "InTAS_highway_2_dataReplay.zip",
    "suddenConstantSpeed": ROOT / "InTAS_highway_2_suddenConstantSpeed.zip",
}
SPLITS = ("Train", "Validation", "Test")
CAT_COLS = ["sender_driversProfile", "receiver_driversProfile"]


def walk_records(obj):
    """Find records containing an attacker label in JSON lists/dicts."""
    if isinstance(obj, dict):
        if "attacker" in obj:
            yield obj
        else:
            for value in obj.values():
                yield from walk_records(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_records(value)


def parse_vector(value):
    """Parse a 3D vector represented as a comma-separated string."""
    if isinstance(value, (list, tuple)):
        parts = value
    elif isinstance(value, str):
        parts = value.split(",")
    else:
        return None

    try:
        nums = [float(x.strip()) for x in parts]
        if len(nums) >= 3 and all(np.isfinite(x) for x in nums[:3]):
            return nums[:3]
    except (ValueError, TypeError):
        pass
    return None


def numeric(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.number)):
        value = float(value)
        return value if np.isfinite(value) else None
    return None


def make_features(record):
    features = {}

    # Exclude identifiers and absolute timestamps.
    # Use elapsed message delay instead of raw rcvTime/sendTime.
    rcv = numeric(record.get("rcvTime"))
    send = numeric(record.get("sendTime"))
    if rcv is not None and send is not None:
        features["message_delay_s"] = (rcv - send) / 1_000_000.0

    vectors = {}

    for obj_name in ("sender", "receiver"):
        obj = record.get(obj_name, {})
        if not isinstance(obj, dict):
            obj = {}

        # Categorical fields are encoded inside the ML pipeline.
        profile = obj.get("driversProfile")
        features[f"{obj_name}_driversProfile"] = (
            str(profile) if profile is not None else None
        )

        for key, value in obj.items():
            if key in ("pos", "pos_noise", "driversProfile"):
                continue

            number = numeric(value)
            if number is not None:
                features[f"{obj_name}_{key}"] = number

        # Do not feed absolute coordinates to the model.
        # Keep position-noise components as observable features.
        pos = parse_vector(obj.get("pos"))
        noise = parse_vector(obj.get("pos_noise"))
        vectors[obj_name] = pos

        if noise is not None:
            for axis, number in zip(("x", "y", "z"), noise):
                features[f"{obj_name}_pos_noise_{axis}"] = number

    # Relative geometry is more meaningful than absolute map location.
    s_pos = vectors.get("sender")
    r_pos = vectors.get("receiver")
    if s_pos is not None and r_pos is not None:
        dx = s_pos[0] - r_pos[0]
        dy = s_pos[1] - r_pos[1]
        dz = s_pos[2] - r_pos[2]
        features["relative_pos_dx"] = dx
        features["relative_pos_dy"] = dy
        features["relative_pos_dz"] = dz
        features["relative_distance"] = float(
            np.sqrt(dx * dx + dy * dy + dz * dz)
        )

    # Explicit relative-motion features.
    for key in ("spd", "acl", "spd_noise", "acl_noise"):
        s = features.get(f"sender_{key}")
        r = features.get(f"receiver_{key}")
        if s is not None and r is not None:
            features[f"delta_{key}"] = s - r

    sh = features.get("sender_hed")
    rh = features.get("receiver_hed")
    if sh is not None and rh is not None:
        # Smallest absolute angular difference in degrees.
        features["heading_difference"] = abs((sh - rh + 180) % 360 - 180)

    return features


def load_scenario(archive_path):
    frames = {}

    with zipfile.ZipFile(archive_path) as outer:
        for split in SPLITS:
            split_marker = f"/{split}/"
            nested_names = [
                name for name in outer.namelist()
                if name.lower().endswith(".zip")
                and split_marker in name.replace("\\", "/")
            ]

            rows = []
            if not nested_names:
                raise RuntimeError(
                    f"No nested ZIP found for split={split} "
                    f"in {archive_path.name}"
                )

            for nested_name in nested_names:
                raw = outer.read(nested_name)

                with zipfile.ZipFile(io.BytesIO(raw)) as inner:
                    json_names = [
                        name for name in inner.namelist()
                        if name.lower().endswith(".json")
                    ]

                    for json_name in json_names:
                        obj = json.loads(inner.read(json_name))
                        for record in walk_records(obj):
                            label = numeric(record.get("attacker"))
                            if label not in (0.0, 1.0):
                                raise ValueError(
                                    f"Invalid attacker label: {label}"
                                )

                            row = make_features(record)
                            row["label"] = int(label)
                            rows.append(row)

            if not rows:
                raise RuntimeError(f"No records loaded for {split}")

            df = pd.DataFrame(rows)
            df["label"] = df["label"].astype("int8")
            frames[split] = df

            counts = df["label"].value_counts().to_dict()
            print(
                f"  {split:10s}: records={len(df):,}, "
                f"labels={counts}, features={df.shape[1] - 1}"
            )

    return frames


def make_pipeline(model, numeric_cols, categorical_cols):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])

    return Pipeline([
        ("preprocess", prep),
        ("model", model),
    ])


def select_threshold(y_true, probabilities):
    # Select the threshold using Validation F1 only.
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    if len(thresholds) == 0:
        return 0.5

    p = precision[:-1]
    r = recall[:-1]
    denom = p + r
    f1 = np.divide(
        2 * p * r,
        denom,
        out=np.zeros_like(denom),
        where=denom != 0,
    )

    # In ties, choose the threshold with the best F1; no Test labels used.
    return float(thresholds[int(np.argmax(f1))])


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "fpr": float(fpr),
        "pr_auc_ap": float(average_precision_score(
            y_true, probabilities
        )),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    all_results = []

    for scenario, archive in SCENARIOS.items():
        print(f"\n{'=' * 72}")
        print(f"SCENARIO: {scenario}")

        if not archive.exists():
            raise FileNotFoundError(archive)

        frames = load_scenario(archive)

        train = frames["Train"]
        val = frames["Validation"]
        test = frames["Test"]

        # Keep the same columns for all splits.
        feature_cols = sorted(
            (set(train.columns) | set(val.columns) | set(test.columns))
            - {"label"}
        )

        for df in frames.values():
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = np.nan

        # Drop features that contain no usable values in Train.
        usable_cols = [
            col for col in feature_cols
            if train[col].notna().any()
        ]
        categorical_cols = [
            col for col in CAT_COLS if col in usable_cols
        ]
        numeric_cols = [
            col for col in usable_cols if col not in categorical_cols
        ]

        X_train = train[usable_cols]
        y_train = train["label"].to_numpy()
        X_val = val[usable_cols]
        y_val = val["label"].to_numpy()
        X_test = test[usable_cols]
        y_test = test["label"].to_numpy()

        positive = int(y_train.sum())
        negative = len(y_train) - positive
        scale_pos_weight = negative / max(positive, 1)

        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="aucpr",
                tree_method="hist",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ),
        }

        for model_name, estimator in models.items():
            print(f"\nTraining {model_name} ...")

            pipe = make_pipeline(
                estimator, numeric_cols, categorical_cols
            )
            pipe.fit(X_train, y_train)

            val_prob = pipe.predict_proba(X_val)[:, 1]
            threshold = select_threshold(y_val, val_prob)

            val_metrics = evaluate(y_val, val_prob, threshold)

            # Test is evaluated only after threshold selection.
            test_prob = pipe.predict_proba(X_test)[:, 1]
            test_metrics = evaluate(y_test, test_prob, threshold)

            print(f"  Selected threshold: {threshold:.6f}")
            print("  Validation:", json.dumps(val_metrics))
            print("  Test:      ", json.dumps(test_metrics))

            result = {
                "scenario": scenario,
                "model": model_name,
                "train_records": len(train),
                "validation_records": len(val),
                "test_records": len(test),
                "train_positive_rate": float(y_train.mean()),
                "validation_positive_rate": float(y_val.mean()),
                "test_positive_rate": float(y_test.mean()),
                "features_used": len(usable_cols),
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            all_results.append(result)

            pred_df = pd.DataFrame({
                "y_true": y_test,
                "probability": test_prob,
                "prediction": (test_prob >= threshold).astype(int),
            })
            pred_path = OUT / f"{scenario}_{model_name}_test_predictions.csv"
            pred_df.to_csv(pred_path, index=False)

            del pipe

        del frames

    results_df = pd.DataFrame(all_results)
    results_path = OUT / "baseline_metrics.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\n{'=' * 72}")
    print("FINAL TEST COMPARISON")
    cols = [
        "scenario", "model", "features_used",
        "test_accuracy", "test_precision", "test_recall",
        "test_f1", "test_fpr", "test_pr_auc_ap",
        "test_tp", "test_fp", "test_fn", "test_tn",
    ]
    print(results_df[cols].to_string(index=False))
    print(f"\nMetrics saved to: {results_path}")
    print(f"Predictions saved under: {OUT}")


if __name__ == "__main__":
    main()
