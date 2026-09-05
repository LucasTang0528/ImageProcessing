# FreshSight

**A Comparative and Enhancement Study of Classical Feature Extraction Techniques for Apple Ripeness and Blemish Assessment**

BMDS2133 Image Processing — Mode A: Comparative & Enhancement Study.

Three classical descriptor families are compared for three-stage apple grading
(Unripe, Ripe, Rotten) under a single shared experimental harness. No deep
learning is used anywhere in this repository, by design: the study is about
classical techniques, and a CNN would answer a different question.

| Technique | Descriptor family | Dimensionality | Status |
| :-- | :-- | :-- | :-- |
| T1 | MPEG-7 dominant colour descriptor | 37 | **Built** |
| T2 | GLCM texture descriptors | 40 | **Built** |
| T3 | Multiscale morphological descriptors | 36 | Phase 2 |

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
python -m pytest tests -v                      # 174 tests, no dataset required
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
| Background-only classification | Train the shared SVM on the border ring alone, which holds no fruit pixels. Near chance is healthy; well above chance means imaging style is confounded with the label. This is the *ceiling* of the confound. |
| Background reaching the descriptors | What share of **mask** pixels are indistinguishable from that image's own background, by hue-saturation backprojection. A descriptor only ever sees pixels inside the mask, so this decides how much of the ceiling is actually reachable. |
| Segmentation behaviour per class | Coverage, chosen polarity and failure rate describe the segmenter, not the fruit, so they should barely move between classes. Drift means any exclusion policy removes images class-dependently. |
| Background uniformity per class | Separates plain studio backdrops from cluttered scenes, and explains a failure of the first check. |
| Duplicates and near-duplicates | A duplicate spanning the train/test split leaks the answer; one spanning two classes means the labels contradict each other. |

The second check deliberately does **not** measure how much of the mask falls
in the frame border. That version is worthless for the GrabCut segmenter, which
seeds the border as *definite* background — a label GrabCut can never overturn
— so it returns zero whatever the mask contains. It is pinned as a regression by
`test_background_leakage_is_not_forced_to_zero_by_the_frame_border`. The
backprojection version reads high both when a mask really does contain
background and when the fruit genuinely shares the background's colour, so it
marks masks that cannot be trusted to exclude the confound rather than proving a
segmentation mistake.

Duplicate detection runs a difference hash as a cheap filter and then confirms
each candidate against a colour thumbnail. The confirmation is not optional: a
difference hash keys on the silhouette, so every centred apple on a white
backdrop hashes alike, and matching on the hash alone reported 465 fictitious
cross-class duplicates on the primary set. With confirmation the true figure is
45 pairs, none crossing a class boundary. See `test_dataset_audit.py`.

Exit status is 0 when the dataset passes and 2 when it does not, so the audit
can gate a pipeline.

#### Result for the current primary dataset

The Fruit Ripeness Dataset (Nurdiyansah, 2024) was scraped, and each class was
photographed differently: unripe apples on the tree, rotten apples as studio
product shots. Only 138 of its 2400 images have a plain background and 128 of
those are Rotten.

| Finding | Otsu-on-V | Seeded GrabCut |
| :-- | :-- | :-- |
| Background alone predicts the label | **74.0%** against a 33.3% chance level (Unripe recall 92.5%) | unchanged — a property of the dataset, not the segmenter |
| Segmentation failure rate per class | 0.0 / 11.0 / 12.5% — spread **12.5 pts** | 4.5 / 0.75 / 0.5% — spread **4.0 pts** |
| Polarity chosen per class | 13.4 / 45.3 / 76.1% "dark" — spread **62.8 pts** | not applicable |
| Mean mask coverage spread | 0.172 | **0.076** |
| Total segmentation failures | 188 | **46** |
| Mask pixels indistinguishable from background (Un/Ri/Ro) | 25.3 / 21.1 / 15.3% — spread 10.0 pts | 21.7 / **4.8** / **3.0%** — spread **18.7 pts** |
| Duplicates | 45 within-class pairs, none cross-class | — |

All figures are over the full 2400 images.

Replacing the segmenter removed the class-correlated behaviour that made the
comparison invalid, and cut background contamination on Ripe and Rotten by
roughly four-fifths — mean leakage across the dataset fell from 20.6% to 9.8%.

But **Unripe barely moved, from 25.3% to 21.7%, and the disparity between
classes therefore widened**, from 10.0 to 18.7 points. That is the honest
result and it is why the audit still returns `NOT SUITABLE`. A green apple
photographed among green leaves shares its colour with its own background, so
part of that 21.7% is not a segmentation error at all — it is the fruit
genuinely being the colour of the foliage — and no classical segmenter
separates the two cleanly.

Two consequences for the write-up, both of which belong in Results &
Discussion rather than being quietly absorbed:

* **The comparison between techniques remains valid.** All three see identical
  masks from one shared pipeline, so the ranking of T1, T2 and T3 — the
  assignment's actual objective — is unaffected.
* **The absolute accuracies carry a ceiling.** The 74% background-only figure
  is the control to report them against, and any claim that colour separates
  Unripe well has to be read alongside the fact that its background is the
  same colour as the fruit.

Three replacement datasets were staged and audited before this conclusion was
reached; all three scored worse (see below).

#### Vetting a replacement

Stage each candidate into its own directory, audit it, and adopt only one that
passes. `data/primary` is left untouched throughout, so nothing is lost if a
candidate turns out to be worse.

```bash
python scripts/fetch_dataset.py --slug hilton                      # survey only
python scripts/fetch_dataset.py --slug hilton --copy --dest data/candidate_hilton
python scripts/audit_dataset.py --root data/candidate_hilton \
    --classes UnripeApple,RipeApple,RottenApple
```

`--slug` takes a Kaggle `owner/dataset` identifier or one of the shorthands in
`CANDIDATE_SLUGS`:

| Shorthand | Dataset | Audit result |
| :-- | :-- | :-- |
| `current` | `dudinurdiyansah/fruit-ripeness-dataset` | 74.0% background-only; **adopted**, with the caveats above |
| `leftin` | `leftin/fruit-ripeness-unripe-ripe-and-rotten` | **84.3%** background-only. Has the right three class names and stable segmentation, but its Unripe class is 100% publisher-augmented (`aug_` prefix on all 1934 files) while Ripe and Rotten are ~23% augmented (`saltandpepper_`, `translation_`) — a different pipeline per class, which is what drives the score |
| `hilton` | `davidhilton/apple-ripeness-levels-image-dataset` | **91.0%** background-only against a 20% chance level. Five percentage levels rather than three stages and no rotten class; labels do not track ripeness (bright green apples are labelled "100% ripe"), several images are heavily colour-manipulated, and it holds exact duplicate files across only 500 images |
| `shawhy` | `shawhy/datasets-of-fruit-ripeness-identification` | Not usable: no ripeness class folders at all — one `mixed apple` directory plus a COCO `train2017`/`annotations` layout |

Public apple-ripeness datasets are confounded near-universally, because each
class tends to be collected in one session or from one source. The audit is
worth running on any new candidate before adopting it.

To adopt the winner, point `paths.primary_root` in `config.json` at its
directory, or move it to `data/primary`. Class folder names are read from
`datasets.primary.classes`, so a dataset using different names needs either a
`--map` at staging time or an edit to that list.

### Recorded for Phase 4 — hypotheses and reruns

Raised during Phase 2 review and deliberately **not acted on yet**. Each is
settled against Phase 3 output rather than by argument.

**H1 — Flip augmentation interacts differently with each technique.** T1 is
strictly flip-invariant by construction, so a horizontally flipped training
image is an exact duplicate row in T1's feature space. T2's GLCM at 45° and
135° is orientation-sensitive, so the same augmentation yields genuinely
distinct rows there. The harness is identical for all three and invariance to
a transform is a property under comparison rather than a confound — but
duplicate rows do affect SVM margins. *Action: rerun all three with flip
augmentation disabled and report whether the ranking changes.*

**H2 — T2's error structure may be complementary to T1's, not opposite.** The
brief predicts colour separates Unripe/Ripe while confusing Ripe/Rotten, with
texture showing the reverse. Early T2 figures point elsewhere: contrast at
distance 1, angle 0 runs Unripe 2.75, Rotten 2.63, Ripe 2.07 — a U-shape in
which **Ripe is the smoothest class and Unripe and Rotten are both rougher**.
If that holds, T2 confuses Unripe with Rotten, which is the pair colour
separates most cleanly, making the two techniques complementary rather than
opposite. *Action: test against the Phase 3 confusion matrices; do not
speculate beyond what they support.*

**H3 — Absolute accuracy is bounded by the dataset confound.** Background
pixels alone classify at 74.0% against a 33.3% chance level. Report every
headline accuracy against that control, and treat Unripe-specific colour
claims with particular care: 21.7% of Unripe mask pixels are
indistinguishable from their own image's background. *Action: report the
control alongside the benchmark matrix, and run the ranking on a
low-leakage subset as a sensitivity check.*

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

## Four deviations from the brief, and why

All four are recorded here rather than buried in the code, because they change
what the harness does. All four are applied identically to every technique, so
none affects the validity of the comparison, and each can be switched off in
`config.json`.

**0. Segmentation is seeded GrabCut, not Otsu on V (`segmentation.method`,
default `"grabcut"`).** This is the largest deviation and the brief names the
Otsu method explicitly, so the reasoning matters.

Otsu on the value channel does not separate fruit from background on this
dataset, and it fails *by class*. Over all 2400 images it flagged 0% of Unripe
against 12.5% of Rotten, and chose the "dark" side of the threshold for 13.4%
of Unripe against 76.1% of Rotten. On an orchard photograph the mask is sunlit
foliage; on a rotten apple the dark side of the threshold is the rot patch, so
the mask becomes the very blemish T3 is supposed to measure *inside* it. Under
`on_failure: "exclude"` it would also have dropped 100 Rotten images and no
Unripe ones, quietly rebalancing the test set.

GrabCut is seeded with an explicit label image rather than the usual bounding
rectangle: the frame border is marked definite background, a central core
definite foreground, and the ellipse between them probable foreground. A bare
rectangle leaves no background to model when the fruit fills the frame and
collapsed to an empty mask on three of twelve awkward images; the label seed
removes both failure modes. Across those same images coverage tightened from
4–77% to 15–36%.

Two consequences are handled explicitly. `cv2.grabCut` fits its colour models
with k-means drawn from OpenCV's global RNG, so it is **not deterministic** by
default — two techniques would be described from different masks, which is the
one difference between techniques this study exists to exclude; the RNG is
reseeded before every call. And because the seeded core is definite foreground,
GrabCut can never return an empty mask, so the minimum-coverage bound can no
longer detect an image holding no fruit; `grabcut.min_seed_growth` flags a mask
that grew less than 1.5× its seed (a blank frame grows about 1.27×, a real
fruit several times over).

Set `"method": "otsu_v"` to run the brief's method instead. It is still
implemented, still tested, and `scripts/audit_dataset.py` reports both.

**1. Automatic Otsu polarity (`segmentation.polarity`, default `"auto"`).**
Applies to `otsu_v` only.
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
| Scoring the photography rather than the fruit | `audit_dataset.py` trains the shared classifier on background pixels alone, then measures how much background survives inside the masks | `test_verdict_fails_a_dataset_whose_background_predicts_the_class`, `test_verdict_passes_a_confounded_dataset_whose_masks_do_not_leak` |
| A "leakage" metric the segmenter satisfies by construction | Leakage is measured by backprojecting the image's own background histogram, never by position within the frame | `test_background_leakage_is_not_forced_to_zero_by_the_frame_border` |
| Two techniques described from different masks | GrabCut's k-means initialisation draws on OpenCV's global RNG, which is reseeded before every call | `test_grabcut_is_reproducible_across_calls`, `test_two_techniques_receive_identical_inputs` |
| A seeded mask passing as a fruit on an image containing none | GrabCut cannot return an empty mask, so a mask that grew less than `min_seed_growth` times its seed is flagged | `test_grabcut_flags_a_frame_that_holds_no_fruit`, `test_grabcut_flags_a_fruit_smaller_than_its_seed` |
| Staging a repacked archive twice, leaking copies across the split | Matched source folders are digested by filename and size, keeping one per distinct listing | `test_a_nested_repack_of_the_same_folder_is_dropped` |
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
  t1_dominant_colour.py  MPEG-7 dominant colour descriptor, 37 dimensions
  t2_glcm.py             GLCM texture descriptors, 40 dimensions
  t3_morphology.py       (Phase 2)
compare.py       (Phase 4) benchmark matrix, paired t-tests, ranking
scripts/
  fetch_dataset.py               downloads and stages the primary dataset
  sanity_check_segmentation.py   visual verification of the harness
  audit_dataset.py               confound audit; vets a dataset before adoption
tests/
  test_phase1_harness.py         54 tests, runnable without the dataset
  test_dataset_audit.py          22 tests for the audit, its leakage metric and duplicates
  test_fetch_dataset.py          22 tests for dataset staging and class matching
  test_t1_dominant_colour.py     41 tests for T1, its ordering and its angular statistics
  test_t2_glcm.py                35 tests for T2, background exclusion and degenerate matrices
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
