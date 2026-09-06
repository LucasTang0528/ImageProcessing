"""Phase 4 - rank the benchmarked techniques and test every pairwise gap.

Reads the ``benchmark_matrix.csv`` that :mod:`scripts.run_benchmarks` wrote for
one run and turns it into a ranking backed by a paired t-test over the five
cross-validation folds (Demsar, 2006). Nothing here reads the test partition or
fits a model; it only consumes per-fold scores that already exist.

Run from the project root::

    python scripts/run_benchmarks.py --tag phase3      # produces the input
    python scripts/run_comparison.py --tag phase3      # consumes it

Outputs land next to the input, in ``results/<tag>/``:

* ``comparison_matrix.csv``  - one row per technique, CV and test metrics
* ``pairwise_ttests.csv``    - every pair: mean gap, t, p, Holm-adjusted p
* ``ranking.txt``            - the ranked list, the verdict and the controls
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from compare import (  # noqa: E402
    background_only_accuracy,
    compare,
    load_fold_scores,
    load_test_metrics,
)
from config import get_config  # noqa: E402
from evaluate import save_dataframe  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="phase3",
                        help="Sub-directory of results/ holding benchmark_matrix.csv.")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance threshold for the Holm-adjusted p-values.")
    parser.add_argument("--metric", default="accuracy", choices=["accuracy", "macro_f1"],
                        help="Per-fold metric the ranking is built on.")
    args = parser.parse_args(argv)

    directory = config.paths.results_root / args.tag
    matrix_csv = directory / "benchmark_matrix.csv"
    if not matrix_csv.exists():
        parser.error(
            f"no benchmark_matrix.csv in {directory}. "
            f"Run scripts/run_benchmarks.py --tag {args.tag} first."
        )

    fold_scores = load_fold_scores(matrix_csv, metric=args.metric)
    test_metrics = load_test_metrics(matrix_csv)
    test_accuracy = {
        str(name): float(test_metrics.loc[name, "accuracy"])
        for name in test_metrics.index
        if "accuracy" in test_metrics.columns
    }
    dimensionality = {
        str(name): int(test_metrics.loc[name, "dimensionality"])
        for name in test_metrics.index
        if "dimensionality" in test_metrics.columns
    }
    control = background_only_accuracy(config.paths.results_root, dataset="primary")

    report = compare(
        fold_scores,
        metric=args.metric,
        alpha=args.alpha,
        test_accuracy=test_accuracy,
        dimensionality=dimensionality,
        control_accuracy=control,
    )

    print("=" * 74)
    print(f"Phase 4 comparison - {args.tag}")
    print("=" * 74)
    print(f"  techniques : {', '.join(report.ranking)}")
    print(f"  folds      : {report.fold_scores[report.best].size}")
    print(f"  metric     : {args.metric}")
    print(f"  alpha      : {args.alpha}")
    if control is None:
        print("  control    : background-only audit not found; run scripts/audit_dataset.py")
    print()
    print(report.summary_text())

    save_dataframe(report.matrix_frame(), directory / "comparison_matrix.csv")
    save_dataframe(report.pairwise_frame(), directory / "pairwise_ttests.csv", index=False)
    (directory / "ranking.txt").write_text(report.summary_text() + "\n", encoding="utf-8")

    print(f"\n  Written to {directory}:")
    for filename in ("comparison_matrix.csv", "pairwise_ttests.csv", "ranking.txt"):
        print(f"    {filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
