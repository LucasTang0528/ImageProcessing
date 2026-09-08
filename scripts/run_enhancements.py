"""Phase 5 - build the three enhancements and score each against the best technique.

Report section 3.8. One shared extraction pass produces the T1, T2 and T3
matrices and derives E1 (feature-level fusion) and E2 (blemish-aware regional
weighting); E3 (weighted decision-level fusion) is fitted from the three
individual matrices. Every row is scored on the same 5-fold leakage-safe
cross-validation and the same held-out test partition the individual techniques
use, so an enhancement's gain over the strongest technique is a paired
comparison over identical folds.

Run from the project root::

    python scripts/run_enhancements.py --per-class 200 --no-augment --tag pilot
    python scripts/run_enhancements.py --tag phase5          # the reported run

Outputs land in ``results/<tag>/``:

* ``enhancement_matrix.csv``  - one row per technique and enhancement
* ``enhancement_vs_best.csv`` - each E* against the best individual, paired t-test
* ``ablation.csv``            - E1 and E3 with one technique removed at a time
* ``E{1,2,3}_confusion.png``  - normalised confusion matrix per enhancement
* ``run_metadata.json``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from compare import compare, background_only_accuracy, paired_fold_t_test  # noqa: E402
from config import Config, get_config  # noqa: E402
from data import ImageRecord, load_primary  # noqa: E402
from enhance import (  # noqa: E402
    build_fusion_pipeline,
    build_technique_matrices,
    cross_validate_decision_fusion,
    cross_validate_matrix,
    fit_decision_fusion,
    macro_f1_weights,
    stack_feature_matrices,
)
from evaluate import (  # noqa: E402
    classification_metrics,
    PER_CLASS_TARGET,
    save_confusion_matrix,
    save_dataframe,
    save_per_class_outputs,
)
from harness import build_pipeline, set_global_seed, stratified_split  # noqa: E402


def balanced_subset(records: Sequence[ImageRecord], per_class: int, config: Config) -> List[ImageRecord]:
    """Draw ``per_class`` records from each class, matching the other drivers."""
    rng = np.random.default_rng(config.seed)
    chosen: List[int] = []
    for label in range(config.primary.n_classes):
        pool = [i for i, record in enumerate(records) if record.label == label]
        take = min(per_class, len(pool))
        chosen.extend(int(pool[i]) for i in rng.choice(len(pool), size=take, replace=False))
    return [records[i] for i in sorted(chosen)]


def _progress(name: str):
    def report(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"    {name}: {done}/{total}", end="\r", flush=True)
    return report


def _row(name: str, kind: str, dim: int, cv, test_report) -> Dict[str, object]:
    return {
        "technique": name,
        "kind": kind,
        "dimensionality": dim,
        "cv_mean_accuracy": cv.mean_accuracy,
        "cv_std_accuracy": cv.std_accuracy,
        "cv_mean_macro_f1": cv.mean_macro_f1,
        "test_accuracy": test_report.accuracy,
        "test_macro_f1": test_report.macro_f1,
        "test_weighted_f1": test_report.weighted_f1,
        **{f"cv_fold{i + 1}_accuracy": float(s) for i, s in enumerate(cv.accuracy_folds)},
    }


def main(argv: Sequence[str] | None = None) -> int:
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class", type=int, default=0)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--tag", default="phase5")
    parser.set_defaults(augment=True)
    args = parser.parse_args(argv)

    set_global_seed(config)
    records = load_primary(config)
    if args.per_class:
        records = balanced_subset(records, args.per_class, config)
    partition = stratified_split(records, config)

    mode = "PILOT" if (args.per_class or not args.augment) else "FULL"
    print("=" * 74)
    print(f"Phase 5 enhancements - {mode}")
    print("=" * 74)
    print(f"  images       : {len(records)} ({args.per_class or 'all'} per class)")
    print(f"  train / test : {len(partition.train)} / {len(partition.test)}")
    print(f"  augmentation : {'on' if args.augment else 'OFF'}")
    if mode == "PILOT":
        print("  NOTE         : reduced data. Indicative only - not the reported figures.")

    print("\n  Building training matrices (one shared segmentation pass) ...")
    started = time.perf_counter()
    train = build_technique_matrices(
        partition.train, config, augment=args.augment, progress=_progress("train"),
    )
    print(f"\n  Building test matrices ...")
    test = build_technique_matrices(
        partition.test, config, augment=False, progress=_progress("test"),
    )
    print(f"\n  {train['T1'].X.shape[0]} train rows, {test['T1'].X.shape[0]} test rows "
          f"in {time.perf_counter() - started:.1f}s "
          f"({train.excluded} + {test.excluded} excluded by segmentation)")

    # E1 needs the concatenated matrix on both partitions.
    train.matrices["E1"] = stack_feature_matrices(
        [train["T1"], train["T2"], train["T3"]], technique="E1"
    )
    test.matrices["E1"] = stack_feature_matrices(
        [test["T1"], test["T2"], test["T3"]], technique="E1"
    )

    display = list(config.primary.display_names)
    rows: List[Dict[str, object]] = []
    reports = {}
    # Kept so that per-class metrics can be written for the cross-validated
    # partition too. The fold confusions live on the report; discarding it
    # here is what previously made per-class CV figures unrecoverable.
    cv_reports = {}

    # ---- individual techniques ---------------------------------------- #
    for name in ("T1", "T2", "T3"):
        cv = cross_validate_matrix(train[name], config, technique=name)
        fitted = build_pipeline(config).fit(train[name].X, train[name].y)
        report = classification_metrics(
            test[name].y, fitted.predict(test[name].X), display, technique=name
        )
        reports[name] = report
        cv_reports[name] = cv
        rows.append(_row(name, "individual", train[name].dim, cv, report))
        print(f"    {name}: CV {cv.mean_accuracy:.4f}  test {report.accuracy:.4f}")

    # ---- E1 feature-level fusion ------------------------------------- #
    e1_pipeline = build_fusion_pipeline(config)
    cv_e1 = cross_validate_matrix(train["E1"], config, technique="E1", pipeline=e1_pipeline)
    e1_fitted = build_fusion_pipeline(config).fit(train["E1"].X, train["E1"].y)
    e1_report = classification_metrics(
        test["E1"].y, e1_fitted.predict(test["E1"].X), display, technique="E1"
    )
    reports["E1"] = e1_report
    cv_reports["E1"] = cv_e1
    rows.append(_row("E1", "enhancement", train["E1"].dim, cv_e1, e1_report))
    print(f"    E1: CV {cv_e1.mean_accuracy:.4f}  test {e1_report.accuracy:.4f} "
          f"(PCA kept {e1_fitted.named_steps['pca'].n_components_} components)")

    # ---- E2 blemish-aware regional weighting ----------------------- #
    cv_e2 = cross_validate_matrix(train["E2"], config, technique="E2")
    e2_fitted = build_pipeline(config).fit(train["E2"].X, train["E2"].y)
    e2_report = classification_metrics(
        test["E2"].y, e2_fitted.predict(test["E2"].X), display, technique="E2"
    )
    reports["E2"] = e2_report
    cv_reports["E2"] = cv_e2
    rows.append(_row("E2", "enhancement", train["E2"].dim, cv_e2, e2_report))
    print(f"    E2: CV {cv_e2.mean_accuracy:.4f}  test {e2_report.accuracy:.4f}")

    # ---- E3 weighted decision-level fusion ------------------------- #
    # The cross-validation deliberately does NOT receive these weights. They
    # are the cross-validated macro F1 over the whole training partition, so
    # every fold's validation rows helped set them; scoring a fold with them
    # would let each fold be judged partly on its own answers. Passing None
    # makes each fold derive its weights from its own training rows alone.
    # The weights below are fitted on the full training partition and used for
    # the held-out test prediction, which is legitimate - the test rows played
    # no part in them.
    cv_e3 = cross_validate_decision_fusion(train.individual, config)
    weights = macro_f1_weights(train.individual, config)
    fusion = fit_decision_fusion(train.individual, config, weights=weights)
    e3_pred = fusion.predict({name: test[name] for name in ("T1", "T2", "T3")})
    e3_report = classification_metrics(test["T1"].y, e3_pred, display, technique="E3")
    reports["E3"] = e3_report
    cv_reports["E3"] = cv_e3
    rows.append(_row("E3", "enhancement", -1, cv_e3, e3_report))
    print(f"    E3: CV {cv_e3.mean_accuracy:.4f}  test {e3_report.accuracy:.4f}  "
          f"weights " + ", ".join(f"{k}={v:.3f}" for k, v in weights.items()))

    matrix = pd.DataFrame(rows).set_index("technique")

    # ---- comparison: every E* against the best individual ---------- #
    fold_scores = {
        str(name): matrix.loc[name, [c for c in matrix.columns if c.endswith("_accuracy")
                                     and c.startswith("cv_fold")]].to_numpy(dtype=float)
        for name in matrix.index
    }
    control = background_only_accuracy(config.paths.results_root, "primary")
    report = compare(
        fold_scores,
        test_accuracy={n: float(matrix.loc[n, "test_accuracy"]) for n in matrix.index},
        dimensionality={n: int(matrix.loc[n, "dimensionality"]) for n in matrix.index},
        control_accuracy=control,
        # The 80% bar is set against the best individual technique. This
        # ranking also carries E1/E2/E3, and an enhancement usually tops it,
        # so the scope has to be stated or the target gets read against the
        # wrong row.
        target_techniques=("T1", "T2", "T3"),
    )
    best_individual = max(("T1", "T2", "T3"), key=lambda n: fold_scores[n].mean())
    vs_best = pd.DataFrame(
        [
            {
                **paired_fold_t_test(
                    fold_scores[e], fold_scores[best_individual], e, best_individual
                ).__dict__,
                "meets_3pt_target": (fold_scores[e].mean() - fold_scores[best_individual].mean()) >= 0.03,
            }
            for e in ("E1", "E2", "E3")
        ]
    )

    # ---- ablation: drop one technique at a time -------------------- #
    ablation_rows: List[Dict[str, object]] = []
    for pair in (("T1", "T2"), ("T1", "T3"), ("T2", "T3")):
        stacked = stack_feature_matrices([train[p] for p in pair], technique="+".join(pair))
        cv_pair = cross_validate_matrix(stacked, config, technique="+".join(pair),
                                        pipeline=build_fusion_pipeline(config))
        ablation_rows.append({
            "strategy": "E1_feature_fusion", "techniques": "+".join(pair),
            "dropped": (set(("T1", "T2", "T3")) - set(pair)).pop(),
            "cv_mean_accuracy": cv_pair.mean_accuracy, "cv_std_accuracy": cv_pair.std_accuracy,
        })
        sub = {p: train[p] for p in pair}
        cv_dec = cross_validate_decision_fusion(sub, config, technique="+".join(pair))
        ablation_rows.append({
            "strategy": "E3_decision_fusion", "techniques": "+".join(pair),
            "dropped": (set(("T1", "T2", "T3")) - set(pair)).pop(),
            "cv_mean_accuracy": cv_dec.mean_accuracy, "cv_std_accuracy": cv_dec.std_accuracy,
        })

    # ---- write everything ---------------------------------------- #
    output = config.paths.results_subdir(args.tag)
    save_dataframe(matrix, output / "enhancement_matrix.csv")
    save_dataframe(vs_best, output / "enhancement_vs_best.csv", index=False)
    save_dataframe(pd.DataFrame(ablation_rows), output / "ablation.csv", index=False)
    for name in ("E1", "E2", "E3"):
        save_confusion_matrix(reports[name], output / f"{name}_confusion.png")
    # Per-class serialisation. Everything written here was already
    # computed above; nothing is refitted, so no published figure moves.
    passes = save_per_class_outputs(reports, cv_reports, display, output)
    summary = report.summary_text()
    summary += (
        "\n\nPER-CLASS TARGET (precision and recall >= "
        f"{PER_CLASS_TARGET:.2f} for every class)\n"
    )
    summary += passes.to_string(index=False) + "\n"
    (output / "enhancement_ranking.txt").write_text(summary, encoding="utf-8")

    (output / "run_metadata.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "per_class": args.per_class or "all",
                "augmentation": args.augment,
                "n_train_rows": int(train["T1"].X.shape[0]),
                "n_test_rows": int(test["T1"].X.shape[0]),
                "excluded_train": train.excluded,
                "excluded_test": test.excluded,
                "e3_weights": weights,
                "best_individual": best_individual,
                "seed": config.seed,
                "segmentation_method": config.segmentation.method,
                "background_only_control": control,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 74)
    print("ENHANCEMENT MATRIX")
    print("=" * 74)
    print(matrix[["kind", "dimensionality", "cv_mean_accuracy", "test_accuracy",
                  "test_macro_f1"]].to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n" + report.summary_text())
    print(f"\n  Written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
