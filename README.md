# FreshSight

**A Comparative and Enhancement Study of Classical Feature Extraction Techniques for Apple Ripeness and Blemish Assessment**

BMDS2133 Image Processing — Mode A: Comparative & Enhancement Study.

Three classical descriptor families are compared for three-stage apple grading
(Unripe, Ripe, Rotten) under a single shared experimental harness. No deep
learning is used anywhere in this repository, by design: the study is about
classical techniques, and a CNN would answer a different question.

| Technique | Descriptor family | Dimensionality | Status |
| :-- | :-- | :-- | :-- |
| T1 | Colour distribution descriptors | 105 | Phase 2 |
| T2 | GLCM texture descriptors | 40 | Phase 2 |
| T3 | LBP + morphological blemish descriptors | 34 | Phase 2 |

---

## The experimental claim, and how the code enforces it

The study compares three techniques, so its results are only meaningful if the
**feature extraction function is the sole difference between the three runs**.
Preprocessing, segmentation, the data partition, the cross-validation folds,
the augmentation plan, the classifier and its hyperparameters are identical
throughout.

This is enforced architecturally rather than by convention:

- Every technique implements one interface,
  `extract_features(bgr_image, fruit_mask) -> np.ndarray`, defined in
  [features/base.py](features/base.py).
- [harness.py](harness.py) calls techniques through the structural
  `FeatureExtractorLike` protocol. It never imports a concrete technique, so
  the dependency runs one way only.
- A technique receives **copies** of the image and mask. It has no reference to
  the configuration, the partition, the classifier, or any other technique, so
  it cannot reach into or modify the harness. This is covered by the test
  `test_a_technique_cannot_corrupt_the_shared_sample`.
- Every experimental parameter lives in [config.json](config.json). Nothing is
  hardcoded in source, and no absolute path appears anywhere — enforced by
  `test_no_absolute_paths_are_hardcoded_in_source`.
- The seed is 42, everywhere randomness occurs.

---

## Reproducing every reported number

### 1. Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Verified against Python 3.13.13 with NumPy 2.4.4, SciPy 1.17.1, pandas 3.0.2,
Matplotlib 3.10.8, OpenCV 5.0.0, scikit-image 0.26.0 and scikit-learn 1.8.0.

### 2. Datasets

The datasets are not committed to this repository — they are large and
separately licensed. `data/` ships with empty class folders only.

#### Kaggle credentials — each teammate needs their own

The API token is personal and **must never be committed or shared**. Every team
member creates their own:

1. Sign in at [kaggle.com](https://www.kaggle.com), open **Settings → API**, and
   click **Create New Token**. A `kaggle.json` downloads.
2. Move it to `~/.kaggle/kaggle.json` — on Windows that is
   `C:\Users\<you>\.kaggle\kaggle.json`.
3. On macOS or Linux, restrict it: `chmod 600 ~/.kaggle/kaggle.json`.
4. Open the [dataset page](https://www.kaggle.com/datasets/dudinurdiyansah/fruit-ripeness-dataset)
   once while signed in and accept its terms, or the download returns 403.

`.gitignore` already excludes `.kaggle/`, `kaggle.json`, and common
credential-file patterns. It deliberately does **not** blanket-ignore `*.json`,
because `config.json` is the experiment's single source of truth and must stay
tracked.

#### Staging the primary dataset

[scripts/fetch_dataset.py](scripts/fetch_dataset.py) runs in two stages,
because the published folder names are undocumented and it will not guess.

```bash
python scripts/fetch_dataset.py          # stage 1: download and survey, copies nothing
```

This prints the downloaded directory tree with per-folder image counts and the
class mapping it proposes, then stops. **Read the mapping before continuing.**
If a class was not matched, point at it explicitly:

```bash
python scripts/fetch_dataset.py --copy --map UnripeApple=<actual folder name>
```

Otherwise:

```bash
python scripts/fetch_dataset.py --copy   # stage 2: copy the apple classes into data/primary
```

What the copy stage guarantees:

- Only the **three apple classes** are copied; every other fruit is ignored.
- If the dataset ships a `train`/`test` split, **only `train` is used**. The
  harness performs its own seeded 80:20 stratified split, so folding in a
  publisher's test set would silently change the fixed partition.
- The kagglehub cache is **only read from** — never moved, modified or deleted.
- Re-running is **idempotent**: files already staged at the same size are
  skipped. Filename collisions between source folders get a deterministic
  `<folder>__<name>` suffix, never a counter, so repeat runs cannot multiply
  the dataset.
- Every staged file is checked to **decode** via OpenCV. Any that fail are
  listed and the script exits non-zero, rather than leaving the harness to
  choke on them mid-run.
- Per-class counts and any duplicate filenames are reported at the end.

The generalisation and robustness sets (Phase 6) are staged by hand for now.
Arrange all three roots as below.

```
data/
  primary/                       # Fruit Ripeness Dataset (Nurdiyansah, 2024), apple subset
    UnripeApple/
    RipeApple/
    RottenApple/
  generalisation/                # Fresh and Stale Classification (swoyam2609, 2023)
    FreshApple/
    RottenApple/
  robustness/                    # varied lighting and backgrounds
    Good/
    Bad/
    Mixed/
```

Only `data/primary` is needed for Phases 1–5; the other two roots are used in
Phase 6. Accepted file extensions are listed under `datasets.image_extensions`
in [config.json](config.json).

To confirm the loader can see the data:

```bash
python data.py
```

| Script | Role |
| :-- | :-- |
| `scripts/fetch_dataset.py` | Downloads and stages `data/primary` from Kaggle. |
| `scripts/sanity_check_segmentation.py` | Visual verification of the harness. |

### 3. Phase 1 — verify the harness

```bash
python config.py                                # print the active configuration
python -m pytest tests -v                       # 64 tests, no dataset required
python scripts/sanity_check_segmentation.py     # visual check, dataset required
python scripts/audit_dataset.py                 # dataset confound audit, dataset required
```

The sanity check writes to `results/phase1/`:

| File | Contents |
| :-- | :-- |
| `segmentation_primary_<class>.png` | 12 random images per class: original, preprocessed, mask, contour and bounding box. Failed rows are outlined in red. |
| `segmentation_coverage_primary.csv` | Per-class mask coverage statistics and polarity counts. |
| `segmentation_failures_primary.csv` | Every image flagged as a segmentation failure, with its coverage and reason. |

Useful options:

```bash
python scripts/sanity_check_segmentation.py --per-class 20
python scripts/sanity_check_segmentation.py --source generalisation
```

**Inspect the PNGs before proceeding to Phase 2.** If the masks are wrong, every
number in every later phase is wrong, and no amount of downstream analysis will
reveal it.

### 4. Audit the dataset before adopting it

A comparative study inherits every flaw in its data. The audit measures the
flaws that accuracy figures cannot reveal, and prints a pass/fail verdict:

```bash
python scripts/audit_dataset.py                                          # data/primary
python scripts/audit_dataset.py --source generalisation
python scripts/audit_dataset.py --root data/candidate --classes A,B,C    # vet a candidate
```

| Check | What it answers |
| :-- | :-- |
| Background-only classification | Train the shared SVM on the border ring alone, which holds no fruit pixels. Near chance is healthy; well above chance means imaging style is confounded with the label. |
| Segmentation behaviour per class | Coverage, chosen polarity and failure rate describe the segmenter, not the fruit, so they should barely move between classes. Drift means any exclusion policy removes images class-dependently. |
| Background uniformity per class | Separates plain studio backdrops from cluttered scenes, and explains a failure of the first check. |
| Duplicates and near-duplicates | A duplicate spanning the train/test split leaks the answer; one spanning two classes means the labels contradict each other. |

Duplicate detection runs a difference hash as a cheap filter and then confirms
each candidate against a colour thumbnail. The confirmation is not optional: a
difference hash keys on the silhouette, so every centred apple on a white
backdrop hashes alike, and matching on the hash alone reported 465 fictitious
cross-class duplicates on the primary set. With confirmation the true figure is
45 pairs, none crossing a class boundary. See `test_dataset_audit.py`.

Exit status is 0 when the dataset passes and 2 when it does not, so the audit
can gate a pipeline.

#### Result for the current primary dataset

The Fruit Ripeness Dataset (Nurdiyansah, 2024) **fails** this audit:

| Finding | Measurement |
| :-- | :-- |
| Imaging style predicts the label | Background pixels alone classify at **74.0%** against a 33.3% chance level; Unripe recall 92.5% |
| Segmentation behaves differently per class | Failure rate 0.0% Unripe, 11.0% Ripe, 12.5% Rotten; "dark" polarity chosen for 13.4%, 45.3%, 76.1% |
| Class-dependent exclusion | `on_failure: "exclude"` would drop 0 Unripe but 100 Rotten images, silently rebalancing the test set |
| Duplicates | 45 within-class pairs, none cross-class |

The cause is that the set was scraped: unripe apples are photographed on the
tree, rotten apples as studio product shots. Only 138 of its 2400 images have a
plain background and 128 of those are Rotten, so restricting to studio shots is
not available either — it would leave no Unripe images at all. A different
primary dataset is required; the harness itself is unaffected.

### Phases 2–6

Not yet implemented. Each phase is built and verified in turn.

| Phase | Deliverable | Status |
| :-- | :-- | :-- |
| 1 | Shared harness, evaluation module, visual sanity check | **Complete** |
| 2 | The three feature extractors, with unit tests | Pending |
| 3 | Individual benchmarks | Pending |
| 4 | Comparison, paired t-tests, ranking | Pending |
| 5 | Sub-comparisons (colour space, bins, GLCM parameters) | Pending |
| 6 | Held-out generalisation and robustness sets | Pending |

---

## The shared harness

Pipeline order for one sample:

```
read -> (augment, training only) -> preprocess -> segment -> extract features
```

**Preprocessing** ([`harness.preprocess`](harness.py))

1. Resize to 224 × 224 (`cv2.INTER_AREA`).
2. Gaussian blur, 5 × 5 kernel.
3. Convert to CIE L\*a\*b\*, CLAHE on the L\* channel only (clip limit 2.0,
   8 × 8 tiles), convert back to BGR. Equalising L\* alone normalises
   illumination without disturbing the a\* and b\* chromaticity the colour
   descriptors read.

**Segmentation** ([`harness.segment_fruit`](harness.py))

4. Otsu threshold on the HSV V channel.
5. Morphological closing, 5 × 5 elliptical element.
6. Keep the largest connected component only.
7. Fill interior holes, so the mask is the solid fruit silhouette and a dark
   blemish is not punched out of the region it is measured against (see
   deviation 2 below). Disable with `segmentation.fill_holes: false`.
8. Return the mask, its bounding box and its outer contour.
9. Flag any mask covering < 5% or > 95% of the frame as a **segmentation
   failure**, log it, and — under the default `on_failure: "exclude"` policy —
   drop it rather than pass a broken mask to a feature extractor. Because
   segmentation is shared, the same images are dropped for every technique, so
   the comparison stays like-for-like.

**Partition** ([`harness.stratified_split`](harness.py), [`harness.make_cv`](harness.py))

10. 80:20 stratified train/test split, `random_state=42`.
11. 5-fold stratified cross-validation on the **training partition only**,
    with folds drawn over **source images** rather than over feature rows
    (`harness.leakage_safe_folds`). An image and all of its augmented variants
    always land on the same side of a fold; training folds carry originals and
    variants, validation folds carry **originals only**.
12. Augmentation — horizontal flip, ±15° rotation, brightness jitter ±0.2 —
    applied to **training images only**, and only *after* the split.

**Classifier** ([`harness.build_pipeline`](harness.py))

13. `Pipeline([StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale',
    probability=True, random_state=42)])`. The scaler lives inside the
    pipeline, so under cross-validation it is fitted on each training fold
    alone and never sees a validation fold or the test partition.

---

## Three deviations from the brief, and why

All three are recorded here rather than buried in the code, because they change
what the harness does. All three are applied identically to every technique, so
none affects the validity of the comparison, and each can be switched off in
`config.json`.

**1. Automatic Otsu polarity (`segmentation.polarity`, default `"auto"`).**
A plain Otsu threshold labels the *brighter* side as foreground. Real apple
photographs appear on both light and dark backgrounds, so a fixed polarity
returns an inverted mask — the background, not the fruit — on a substantial
fraction of the dataset. With `"auto"`, both sides of the threshold are
evaluated and the candidate with plausible coverage that touches the frame
border least is kept. Set `"bright"` or `"dark"` in `config.json` to force a
fixed polarity. The choice is made inside the shared harness, so it is
identical for every technique, and the chosen polarity is reported per image in
the coverage CSV.

**2. Interior holes are filled (`segmentation.fill_holes`, default `true`).**
This was found by inspecting the Phase 1 sanity-check output, and it is the
deviation that matters most. A dark blemish on a fruit photographed against a
*dark* background falls on the background side of the Otsu threshold, and a
5 × 5 closing is far too small to bridge it — so the blemish is punched out of
the fruit mask as a hole. That is exactly backwards: T3 reports blemish ratio
as blemished pixels over **mask** pixels, and pixels outside the mask are never
examined, so the most severely rotten fruit would report the *least*
blemishing. It also made the mask a function of the backdrop — the same fruit
gave a holed mask on a dark ground and a solid one on a light ground, so every
mask-restricted descriptor would partly measure the background. Filling the
region enclosed by the outer contour removes both problems. Covered by
`test_dark_blemishes_stay_inside_the_fruit_mask` and
`test_the_mask_does_not_depend_on_the_background`.

**3. Augmentation is applied before preprocessing, not after.**
A brightness-jittered variant is put through the same CLAHE illumination
normalisation a real image would be, which is what makes the augmentation a
meaningful test of the pipeline rather than of the descriptor alone. Rotation
uses replicated borders instead of zero padding, so no artificial black corners
are introduced for Otsu to mistake for fruit.

---

## Errors this codebase actively prevents

| Failure mode | Guard | Test |
| :-- | :-- | :-- |
| Scaler or hyperparameter fitted on test data | Scaler is a pipeline step; CV runs on the training partition only | `test_pipeline_scales_inside_the_pipeline` |
| Augmenting before splitting, leaking a flipped test image into training | Augmentation plan is built from the training partition after the split; `augment=False` for test | `test_feature_matrix_augments_training_only`, `test_no_test_image_appears_in_the_training_matrix` |
| A flipped copy of a **validation** image sitting in its own training fold | CV folds are drawn over source images, not rows; validation folds hold originals only | `test_no_augmented_variant_of_a_validation_image_reaches_a_training_fold`, `test_validation_folds_contain_only_original_images`, `test_naive_row_wise_cross_validation_leaks` |
| An empty mask producing a vector of zeros or NaNs | Coverage bounds flag the image; `require_non_empty_mask` raises | `test_empty_mask_is_refused_rather_than_producing_zeros` |
| NaN or Inf reaching the classifier | `validate_vector` checks every returned vector | `test_validate_vector_rejects_nan_and_inf` |
| Timing the first call, charging import overhead to the algorithm | A discarded warm-up call precedes timing, then times are averaged | `test_extraction_timing_excludes_the_warm_up_call` |
| Accuracy reported alone on an imbalanced test set | `classification_metrics` always returns macro and weighted F1 | — |
| An implicit difference between techniques | Techniques receive copies and hold no harness reference | `test_a_technique_cannot_corrupt_the_shared_sample`, `test_two_techniques_receive_identical_inputs` |
| Scoring the photography rather than the fruit | `audit_dataset.py` trains the shared classifier on background pixels alone and fails the dataset if it beats chance | `test_verdict_fails_a_dataset_whose_background_predicts_the_class` |
| A segmenter that behaves differently on different classes | The audit compares coverage, polarity and failure rate across classes | `test_verdict_fails_class_dependent_segmentation` |
| Duplicate images leaking across the split, or carrying two labels | Difference hash filter confirmed by a colour thumbnail | `test_same_silhouette_different_colour_is_not_a_duplicate`, `test_verdict_fails_on_contradictory_labels` |

---

## Repository layout

```
config.json      every experimental parameter and path; the single source of truth
config.py        loads and validates config.json into an immutable Config
data.py          dataset discovery and image loading, deterministically ordered
harness.py       preprocessing, segmentation, augmentation, partition, pipeline,
                 and the polymorphic feature-matrix builder
evaluate.py      metrics, cross-validation, timing, confusion matrices, CSV/PNG output
features/
  base.py        the FeatureExtractor interface every technique must implement
  t1_colour.py         (Phase 2)
  t2_glcm.py           (Phase 2)
  t3_lbp_blemish.py    (Phase 2)
compare.py       (Phase 4) benchmark matrix, paired t-tests, ranking
scripts/
  fetch_dataset.py               downloads and stages the primary dataset
  sanity_check_segmentation.py   visual verification of the harness
  audit_dataset.py               confound audit; vets a dataset before adoption
tests/
  test_phase1_harness.py         48 tests, runnable without the dataset
  test_dataset_audit.py          16 tests for the audit and its duplicate detector
results/         all generated CSVs and PNGs
```

## Team

| Member | Student ID | Technique |
| :-- | :-- | :-- |
| Ong Song Wei | 2414328 | T1 — Colour distribution descriptors |
| Tang Khuan Zhi | 2414351 | T2 — GLCM texture descriptors |
| Tang Yue Hann | 2414352 | T3 — LBP and morphological blemish descriptors |

## References

- Haralick, R.M., Shanmugam, K. & Dinstein, I. (1973) 'Textural features for image classification', *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-3(6), pp. 610–621.
- Ojala, T., Pietikäinen, M. & Mäenpää, T. (2002) 'Multiresolution grey-scale and rotation invariant texture classification with local binary patterns', *IEEE TPAMI*, 24(7), pp. 971–987.
- Otsu, N. (1979) 'A threshold selection method from grey-level histograms', *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), pp. 62–66.
- Zuiderveld, K. (1994) 'Contrast limited adaptive histogram equalisation', in *Graphics Gems IV*. Academic Press, pp. 474–485.
- Cortes, C. & Vapnik, V. (1995) 'Support-vector networks', *Machine Learning*, 20(3), pp. 273–297.
