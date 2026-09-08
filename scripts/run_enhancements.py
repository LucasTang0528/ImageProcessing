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
* ``ablation_enhancements.csv`` - E2 with one half of its split removed, and
  against a control that keeps the split and moves it somewhere arbitrary
* ``per_class.csv``           - precision, recall, F1 and support per class, for
  every configuration including the ablation arms
* ``confusion/<config>.csv``  - the held-out normalised confusion matrix as
  numbers, beside the existing PNG
* ``confusion/cv/<config>.csv`` - the same matrix from pooled out-of-fold
  validation predictions, which is the only form an ablation arm can have
* ``E{1,2,3}_confusion.png``  - normalised confusion matrix per enhancement
* ``run_metadata.json``

Two provenances, and the difference matters
-------------------------------------------

``T1``, ``T2``, ``T3``, ``E1``, ``E2`` and ``E3`` are fitted on the whole
training partition and predict the held-out test split, so they have a genuine
held-out confusion matrix. An **ablation arm never does either**: it is only
ever cross-validated, which is why ``ablation.csv`` carries a CV accuracy and
no test column. Giving those arms a per-class breakdown therefore cannot come
from the test split, and inventing a test prediction for them would quietly add
six new uses of data the study has deliberately left alone.

Instead every configuration is also scored on its **pooled out-of-fold
validation predictions** - the predictions the cross-validation already made
and threw away. Each is made by a model that never saw that image, and
:func:`~harness.leakage_safe_folds` puts only originals on the validation side,
so the pool holds one prediction per source image. ``per_class.csv`` carries a
``source`` column saying which of the two any row came from; nothing here mixes
them silently.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
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
    OutOfFoldPredictions,
    build_fusion_pipeline,
    build_technique_matrices,
    cross_validate_decision_fusion_detailed,
    cross_validate_matrix_detailed,
    e2_block_columns,
    fit_decision_fusion,
    macro_f1_weights,
    stack_feature_matrices,
)
from evaluate import (  # noqa: E402
    ClassificationReport,
    classification_metrics,
    save_confusion_matrix,
    save_dataframe,
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


class MetricStore:
    """Accumulates the per-class metrics and confusion matrices to be written.

    Nothing here computes a score. Every value it is handed was produced by
    :func:`~evaluate.classification_metrics` from predictions the driver had
    already made, so adding this store cannot move a number in
    ``enhancement_matrix.csv``, ``enhancement_vs_best.csv`` or ``ablation.csv``.
    """

    def __init__(self, class_names: Sequence[str]) -> None:
        self.class_names = list(class_names)
        self.rows: List[Dict[str, object]] = []
        self.confusions: Dict[str, Dict[str, ClassificationReport]] = {
            "test": {}, "cv_out_of_fold": {}
        }

    def add(self, config: str, kind: str, source: str, report: ClassificationReport) -> None:
        """Record one configuration's breakdown under one provenance."""
        for label in self.class_names:
            entry = report.per_class.loc[label]
            self.rows.append({
                "configuration": config,
                "kind": kind,
                "source": source,
                "class": label,
                "precision": float(entry["precision"]),
                "recall": float(entry["recall"]),
                "f1": float(entry["f1"]),
                "support": int(entry["support"]),
                "accuracy": report.accuracy,
                "macro_f1": report.macro_f1,
            })
        self.confusions[source][config] = report

    def add_out_of_fold(
        self, config: str, kind: str, pooled: OutOfFoldPredictions
    ) -> None:
        """Score one configuration on the predictions its folds already made."""
        self.add(
            config,
            kind,
            "cv_out_of_fold",
            classification_metrics(
                pooled.y_true, pooled.y_pred, self.class_names, technique=config
            ),
        )

    def write(self, output: Path) -> List[Path]:
        """Write ``per_class.csv`` and every confusion matrix. Returns the paths."""
        written = [save_dataframe(pd.DataFrame(self.rows), output / "per_class.csv",
                                  index=False)]
        for source, reports in self.confusions.items():
            directory = output / "confusion" if source == "test" else output / "confusion" / "cv"
            directory.mkdir(parents=True, exist_ok=True)
            for config, report in reports.items():
                frame = pd.DataFrame(
                    report.confusion_normalised,
                    index=pd.Index(self.class_names, name="true"),
                    columns=self.class_names,
                )
                written.append(save_dataframe(frame, directory / f"{_slug(config)}.csv"))
        return written


def _slug(config: str) -> str:
    """Make a configuration name safe to use as a filename."""
    return config.replace("/", "-").replace("\\", "-").replace(" ", "_")


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


def guard_existing_results(output: Path, tag: str, force: bool) -> None:
    """Refuse to overwrite a results directory that already holds figures.

    results/phase5 is the run the report is written against, and it has been
    checked figure by figure. The tag defaults to ``phase5``, so running this
    script with no arguments would silently replace it - and a rerun that
    differs even slightly leaves the report describing numbers that no longer
    exist anywhere, with nothing to compare against.

    Regenerating is still one flag away. The guard exists so that overwriting
    is a decision rather than a side effect of not passing an argument.

    Raises:
        SystemExit: If the directory already holds results and force is unset.
    """
    existing = sorted(
        path.name for path in output.glob("*")
        if path.suffix in {".csv", ".json", ".txt"}
    )
    if not existing or force:
        return
    raise SystemExit(
        f"results/{tag} already holds {len(existing)} result file(s), starting "
        f"with {existing[0]}. Refusing to overwrite it: this is the directory "
        f"the report is written against.\n"
        f"  To write somewhere else:  --tag {tag}_rerun\n"
        f"  To replace it anyway:     --force"
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class", type=int, default=0)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--tag", default="phase5")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing results/<tag> directory.")
    parser.set_defaults(augment=True)
    args = parser.parse_args(argv)

    # Checked before any work is done. Refusing after an hours-long run would
    # be a worse failure than not running at all.
    guard_existing_results(config.paths.results_subdir(args.tag), args.tag, args.force)

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
        # E2's ablation control needs a second regional descriptor per image,
        # built from a relocated blemish mask. Asking for it here keeps it on
        # the one segmentation pass; the test partition does not need it,
        # because the enhancement ablation is cross-validated only.
        include_random_control=True,
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
    store = MetricStore(display)

    # ---- individual techniques ---------------------------------------- #
    for name in ("T1", "T2", "T3"):
        cv, pooled = cross_validate_matrix_detailed(train[name], config, technique=name)
        fitted = build_pipeline(config).fit(train[name].X, train[name].y)
        report = classification_metrics(
            test[name].y, fitted.predict(test[name].X), display, technique=name
        )
        reports[name] = report
        rows.append(_row(name, "individual", train[name].dim, cv, report))
        store.add(name, "individual", "test", report)
        store.add_out_of_fold(name, "individual", pooled)
        print(f"    {name}: CV {cv.mean_accuracy:.4f}  test {report.accuracy:.4f}")

    # ---- E1 feature-level fusion ------------------------------------- #
    e1_pipeline = build_fusion_pipeline(config)
    cv_e1, pooled_e1 = cross_validate_matrix_detailed(
        train["E1"], config, technique="E1", pipeline=e1_pipeline
    )
    e1_fitted = build_fusion_pipeline(config).fit(train["E1"].X, train["E1"].y)
    e1_report = classification_metrics(
        test["E1"].y, e1_fitted.predict(test["E1"].X), display, technique="E1"
    )
    reports["E1"] = e1_report
    rows.append(_row("E1", "enhancement", train["E1"].dim, cv_e1, e1_report))
    store.add("E1", "enhancement", "test", e1_report)
    store.add_out_of_fold("E1", "enhancement", pooled_e1)
    print(f"    E1: CV {cv_e1.mean_accuracy:.4f}  test {e1_report.accuracy:.4f} "
          f"(PCA kept {e1_fitted.named_steps['pca'].n_components_} components)")

    # ---- E2 blemish-aware regional weighting ----------------------- #
    cv_e2, pooled_e2 = cross_validate_matrix_detailed(train["E2"], config, technique="E2")
    e2_fitted = build_pipeline(config).fit(train["E2"].X, train["E2"].y)
    e2_report = classification_metrics(
        test["E2"].y, e2_fitted.predict(test["E2"].X), display, technique="E2"
    )
    reports["E2"] = e2_report
    rows.append(_row("E2", "enhancement", train["E2"].dim, cv_e2, e2_report))
    store.add("E2", "enhancement", "test", e2_report)
    store.add_out_of_fold("E2", "enhancement", pooled_e2)
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
    cv_e3, pooled_e3 = cross_validate_decision_fusion_detailed(train.individual, config)
    weights = macro_f1_weights(train.individual, config)
    fusion = fit_decision_fusion(train.individual, config, weights=weights)
    e3_pred = fusion.predict({name: test[name] for name in ("T1", "T2", "T3")})
    e3_report = classification_metrics(test["T1"].y, e3_pred, display, technique="E3")
    reports["E3"] = e3_report
    rows.append(_row("E3", "enhancement", -1, cv_e3, e3_report))
    store.add("E3", "enhancement", "test", e3_report)
    store.add_out_of_fold("E3", "enhancement", pooled_e3)
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
        cv_pair, pooled_pair = cross_validate_matrix_detailed(
            stacked, config, technique="+".join(pair),
            pipeline=build_fusion_pipeline(config),
        )
        ablation_rows.append({
            "strategy": "E1_feature_fusion", "techniques": "+".join(pair),
            "dropped": (set(("T1", "T2", "T3")) - set(pair)).pop(),
            "cv_mean_accuracy": cv_pair.mean_accuracy, "cv_std_accuracy": cv_pair.std_accuracy,
        })
        store.add_out_of_fold(
            f"E1_feature_fusion__{'+'.join(pair)}", "ablation", pooled_pair
        )
        sub = {p: train[p] for p in pair}
        cv_dec, pooled_dec = cross_validate_decision_fusion_detailed(
            sub, config, technique="+".join(pair)
        )
        ablation_rows.append({
            "strategy": "E3_decision_fusion", "techniques": "+".join(pair),
            "dropped": (set(("T1", "T2", "T3")) - set(pair)).pop(),
            "cv_mean_accuracy": cv_dec.mean_accuracy, "cv_std_accuracy": cv_dec.std_accuracy,
        })
        store.add_out_of_fold(
            f"E3_decision_fusion__{'+'.join(pair)}", "ablation", pooled_dec
        )

    # ---- ablation: remove one half of E2's split, then move the split -- #
    #
    # The brief asks for each enhancement *strategy* to be ablated, not only
    # each technique. E2 is the study's best configuration and its novel
    # contribution, and until now it was the one thing never taken apart.
    #
    # Three arms. Two drop half of the split, which answers "does describing
    # both halves beat describing either". The third is the one that matters:
    # it keeps the split, keeps all 154 columns, keeps the areas, and moves the
    # blemish mask to an arbitrary part of the same fruit. E2 doubles the
    # dimensionality of T1 and T2 by splitting, so a gain over them could be
    # bought entirely by the extra columns; only a control that holds the
    # columns fixed and varies the *placement* can tell the two apart.
    print("\n  Ablating E2 ...")
    blocks = e2_block_columns(train["T1"].dim, train["T2"].dim)
    e2_dim = 2 * train["T1"].dim + 2 * train["T2"].dim
    if train["E2"].dim != e2_dim:
        raise AssertionError(
            f"E2 is {train['E2'].dim} columns but its four blocks total {e2_dim}; "
            f"the regional layout has changed and these arms would slice the wrong ones"
        )

    healthy = np.concatenate([blocks["t1_healthy"], blocks["t2_healthy"]])
    blemished = np.concatenate([blocks["t1_blemished"], blocks["t2_blemished"]])
    e2_arms = {
        "E2_healthy_only": (
            replace(train["E2"], X=train["E2"].X[:, healthy], technique="E2_healthy_only"),
            "healthy sub-region descriptors only",
        ),
        "E2_blemished_only": (
            replace(train["E2"], X=train["E2"].X[:, blemished], technique="E2_blemished_only"),
            "blemished sub-region descriptors only",
        ),
        "E2_random_mask": (
            train["E2_random"],
            "split kept, blemish mask relocated on the same fruit",
        ),
    }

    enhancement_ablation: List[Dict[str, object]] = [{
        "strategy": "E2_regional_weighting",
        "arm": "E2_full",
        "description": "both sub-regions, real blemish mask",
        "dimensionality": train["E2"].dim,
        "cv_mean_accuracy": cv_e2.mean_accuracy,
        "cv_std_accuracy": cv_e2.std_accuracy,
        "cv_mean_macro_f1": cv_e2.mean_macro_f1,
        "cv_std_macro_f1": cv_e2.std_macro_f1,
        "delta_vs_full_e2": 0.0,
        **{f"fold{i + 1}_accuracy": float(v) for i, v in enumerate(cv_e2.accuracy_folds)},
        "rows": int(train["E2"].X.shape[0]),
    }]

    for arm, (arm_matrix, description) in e2_arms.items():
        cv_arm, pooled_arm = cross_validate_matrix_detailed(arm_matrix, config, technique=arm)
        store.add_out_of_fold(arm, "ablation", pooled_arm)
        enhancement_ablation.append({
            "strategy": "E2_regional_weighting",
            "arm": arm,
            "description": description,
            "dimensionality": arm_matrix.dim,
            "cv_mean_accuracy": cv_arm.mean_accuracy,
            "cv_std_accuracy": cv_arm.std_accuracy,
            "cv_mean_macro_f1": cv_arm.mean_macro_f1,
            "cv_std_macro_f1": cv_arm.std_macro_f1,
            "delta_vs_full_e2": cv_e2.mean_accuracy - cv_arm.mean_accuracy,
            **{f"fold{i + 1}_accuracy": float(v) for i, v in enumerate(cv_arm.accuracy_folds)},
            "rows": int(arm_matrix.X.shape[0]),
        })
        print(f"    {arm:<20} dim {arm_matrix.dim:>4}  CV {cv_arm.mean_accuracy:.4f} "
              f"(E2 full is {cv_e2.mean_accuracy - cv_arm.mean_accuracy:+.4f} on it)")

    store.add_out_of_fold("E2_full", "ablation", pooled_e2)

    # ---- write everything ---------------------------------------- #
    output = config.paths.results_subdir(args.tag)
    save_dataframe(matrix, output / "enhancement_matrix.csv")
    save_dataframe(vs_best, output / "enhancement_vs_best.csv", index=False)
    save_dataframe(pd.DataFrame(ablation_rows), output / "ablation.csv", index=False)
    save_dataframe(pd.DataFrame(enhancement_ablation),
                   output / "ablation_enhancements.csv", index=False)
    written = store.write(output)
    print(f"    per-class metrics: {len(store.rows)} rows across "
          f"{len(written) - 1} confusion matrices")
    for name in ("E1", "E2", "E3"):
        save_confusion_matrix(reports[name], output / f"{name}_confusion.png")
    (output / "enhancement_ranking.txt").write_text(report.summary_text() + "\n", encoding="utf-8")

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
