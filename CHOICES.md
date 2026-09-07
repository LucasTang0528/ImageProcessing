# CHOICES.md

Every place the specification is silent and this codebase had to decide something.

The brief fixes a great deal — three techniques, one shared harness, an SVM with
`rbf`/`C=1.0`/`gamma='scale'`, an 80:20 split, five folds — and it leaves a
great deal open. A parameter that was never chosen is indistinguishable in the
results from one that was chosen carefully, so this file separates them: for
each decision it records **what was picked, why, where it lives, and whether
any experiment has actually tested it**.

Two things this file is not:

- It is **not** the deviations list. A deviation is somewhere the brief said one
  thing and the code does another; there are four of those and they are in
  [README.md](README.md#four-deviations-from-the-brief-and-why), with the
  evidence that motivated each. This file is about silence, not disagreement.
- It is **not** a record of tuning. Nothing here was fitted against the test
  split or against the report's acceptance targets. Where a sweep exists, it
  runs on the training partition only and its result is reported whether or not
  it flatters the configuration that shipped.

`config.json` is the single source of truth for everything marked **config**.
Editing it is the only supported way to change any of these.

---

## The status column

| Status | Meaning |
| :-- | :-- |
| **Measured** | A sub-experiment sweeps it and the result is in `results/`. The shipped value may or may not be the winner — see the note. |
| **Reasoned** | Chosen from a stated argument about the data or the method, not from a sweep. Defensible, not empirical. |
| **Inherited** | A convention taken from the literature or a library default. Nobody in this project has tested it. |

An **Inherited** value is not a bug. It is an honest label: it says the number
is load-bearing and unexamined, which is the thing a reader cannot otherwise
tell from a results table.

---

## 1. Segmentation

The brief names Otsu on the HSV value channel. The code uses seeded GrabCut,
which is deviation 0 in the README and argued there at length. The choices
below are the ones *inside* that decision, where the brief says nothing at all.

| Choice | Value | Status | Why |
| :-- | :-- | :-- | :-- |
| `segmentation.method` | `grabcut` | **Reasoned** | Otsu on V failed by class: 0% of Unripe flagged against 12.5% of Rotten, and it selected the dark side of the threshold for 76.1% of Rotten images, making the mask the rot patch itself. Full argument and figures in README deviation 0. |
| `grabcut.iterations` | 5 | **Inherited** | OpenCV's usual figure. The mask stops moving well before it on this data; not measured. |
| `grabcut.core_scale` | 0.28 | **Reasoned** | The seed's certain-foreground ellipse, as a fraction of the frame. Small enough to sit inside the fruit on a centred photograph. **This is the weakest choice in the pipeline**: the audit finds the seeded core lands off the fruit on roughly 8% of the dataset, and those images produce masks that are more than a quarter background. |
| `grabcut.border_scale` | 0.06 | **Reasoned** | Frame margin taken as certain background. |
| `grabcut.probable_scale` | 0.78 | **Reasoned** | Everything between core and border is left probable, so GrabCut decides it. |
| `min_mask_fraction` / `max_mask_fraction` | 0.05 / 0.95 | **Reasoned** | A mask outside these bounds is not a fruit. Rejecting rather than describing it stops a vector of background being learned from. |
| `on_failure` | `exclude` | **Reasoned** | A failed segmentation drops the image rather than substituting a rectangle. Exclusions are logged and counted in every `run_config.json`, because silently dropping images can rebalance a class. |
| `fill_holes` | `true` | **Reasoned** | A specular highlight can punch a hole through the middle of a fruit mask; a fruit is simply connected. |
| `cache_masks` | `true` | — | **Configured but not implemented.** Nothing in `harness.py` reads this key. Every driver re-segments from scratch, which is why three sub-experiment runners each pay ~13 minutes of duplicate GrabCut. Left in the config because caching is the right fix and is scoped as a future task, but it is dead today and this file says so rather than letting a reader assume otherwise. |

---

## 2. T1 — dominant colour

| Choice | Value | Status | Why |
| :-- | :-- | :-- | :-- |
| `n_colours` | 4 | **Measured** — E1.2 | Swept over N = 3, 4, 5, 6 in `results/e1/e1_2.csv`, reported against dimensionality (8N + 5) rather than padded to a common length. |
| `space` | `LAB` | **Measured** — E1.3 | LAB, HSV and RGB in `results/e1/e1_3.csv`. LAB was chosen a priori because it is perceptually uniform and because the ripeness indices are defined in CIE terms; the sweep is what tests that. |
| `exclude_specular` | `true` | **Measured** — E1.4 | On against off in `results/e1/e1_4.csv`, with the share of pixels each arm discards reported beside the accuracy, because an accuracy difference is uninterpretable without it. |
| `specular_lightness` / `specular_chroma` | 240 (8-bit L) / 12 | **Reasoned** | A highlight is bright *and* colourless; requiring both stops a genuinely bright yellow fruit being discarded. The thresholds themselves are **unmeasured** — E1.4 tests whether exclusion helps, not where the boundary should sit. |
| `sample_size` | 5000 | **Inherited** | Enough pixels to fit k-means inside the time budget. Assignment runs on the full mask, so the shares stay exact regardless. Not measured. |
| `coherency_min_fraction` | 0.01 | **Inherited** | A component under 1% of the fruit counts as speckle rather than a lesion. Not measured. |
| `green_a_max` | 0.0 | **Reasoned** | The neutral axis of a\*. Not a tuned value — it is the definition of "green side". |
| `decay_chroma_max` / `decay_lightness_max` | 25 / 45 (CIE L\*) | **Reasoned, and under suspicion** | Fixed a priori, before the descriptor met a real image, and never refitted — so free of any tuning-on-test concern, and equally not optimised. On held-out exemplars the resulting index reads 26.5% on an **Unripe** apple and 0.0% on a **Rotten** one, backwards from its design: a dark low-chroma *green* cluster (L\* 14.7, a\* −10.4, b\* 10.3) satisfies both tests, so on a green apple in shadow the index detects the shadow. E1.5's fourth arm measures whether the dimension contributes anything; it does not change the thresholds. Refitting them belongs in Phase 5 on training folds alone. |
| `blocks` | `ABC` | **Measured** — E1.5 | A alone, A+B, A+B+C, and A+B+C without the decay dimension, in `results/e1/e1_5.csv`. |
| Cluster ordering | descending share, ties by ascending lightness | **Reasoned** | k-means labels clusters arbitrarily, so without a canonical order the same fruit yields the same numbers in a different arrangement — noise in every dimension, invisible to every check except a permutation test. |
| k-means sample drawn from **colour-sorted** pixels | — | **Reasoned** | Drawing by array position makes the fitted centroids depend on where pixels sit in the frame. The harness augments by flipping and rotating, every one of which rearranges pixels without changing colour content, so a flipped apple would otherwise get genuinely different dominant colours rather than a permutation of the same ones. |
| `KMeans(n_init=10)` set explicitly | — | **Reasoned** | scikit-learn's `"auto"` has changed meaning across releases; leaving it unset would make the descriptor's definition depend on the installed version. |

### E1.1's baseline

| Choice | Value | Status | Why |
| :-- | :-- | :-- | :-- |
| Histogram bins | 32 per channel | **Inherited** | The conventional fixed-bin quantisation the descriptor is meant to improve on. |
| Colour moments | mean, SD, cube-root skewness | **Reasoned** | Stricker & Orengo (1995). The cube root keeps the statistic in the channel's own units, which is that paper's convention — deliberately different from the standardised skewness the T2 baseline uses, because the two follow different literatures. |
| Baseline applies the **same** specular exclusion | `true` | **Reasoned** | E1.1 is only about the descriptor family if both arms see the same pixels. The baseline calls T1's own `specular_selector` rather than reimplementing it, so the two cannot drift apart. |
| Hue treated linearly in HSV | — | **Reasoned** | Left uncorrected on purpose. A conventional histogram baseline does not handle the wrap point, and correcting it would import one of the descriptor's ideas into the arm meant to lack it. |

---

## 3. T2 — GLCM texture

| Choice | Value | Status | Why |
| :-- | :-- | :-- | :-- |
| `levels` | 32 | **Measured** — E2.3 | 16, 32 and 64 in `results/e2/e2_3.csv`. 32 was inherited from the Haralick literature and is a default, not a measured choice — the sweep exists to say which. |
| `distances` | [1, 2] | **Measured** — E2.2 | [1], [1,2], [1,2,3] in `results/e2/e2_2.csv`, reported against dimensionality. |
| `angles_deg` | 0, 45, 90, 135 | **Inherited** | The four Haralick directions. The *set* is not swept; whether to keep them separate is. |
| `angle_averaged` | `false` | **Measured** — E2.4 | Per-angle against the rotation-invariant average in `results/e2/e2_4.csv`. Averaging costs three quarters of the length; the harness already augments by rotating, which may make the invariance redundant. |
| `symmetric` | `true` | **Inherited** | The standard Haralick convention: each pair counted in both directions. |
| Quantisation spans 0–255, not the fruit's own range | — | **Reasoned** | Rescaling per image would normalise away absolute brightness, which is a real difference between a pale unripe apple and a dark rotten one, and would make the descriptor's meaning depend on the image it came from. |
| Background as ignore level 0, then row/column 0 **deleted** | — | **Reasoned** | Zeroing the background instead would make 0 an enormous perfectly uniform grey level whose co-occurrences with the fruit boundary swamp contrast and dissimilarity — the measured "texture" would be the *shape of the mask*. Deleting the level removes every background pair from numerator and denominator alike. |
| Empty co-occurrence matrix → correlation 0.0 | — | **Reasoned** | scikit-image returns 1.0, asserting perfect correlation on the strength of no measurement at all. A neutral value is better than a fabricated extreme one. A uniform *real* crop keeps scikit-image's 1.0, because that convention is meaningful. |

### E2.1's baseline

| Choice | Value | Status | Why |
| :-- | :-- | :-- | :-- |
| Four moments | mean, variance, standardised skewness, excess kurtosis | **Inherited** | The conventional first-order texture set (Gonzalez & Woods, 2018, §11.3). Note this is *not* the cube-root skewness the T1 baseline uses; each follows its own literature. |
| Taken over the same pixels the GLCM pairs | — | **Reasoned** | Same mask, background excluded rather than zero-filled, for the same reason zero-filling breaks the co-occurrence matrix. |

---

## 4. T3 — multiscale morphology

| Choice | Value | Status | Why |
| :-- | :-- | :-- | :-- |
| `r_max` | 7 | **Measured** — E3.3 | 5, 7, 9, 11 in `results/t3/e3_3_rmax.csv`, reported against dimensionality. |
| `se_shape` | `ellipse` | **Measured** — E3.2 | Ellipse, rect, cross in `results/t3/e3_2_se_shape.csv`. A disk is isotropic and apple defects have no characteristic orientation; the sweep found the choice makes almost no difference either way. |
| `tophat_radius` | 9 | **Inherited** | One step above `r_max`, so the top hat responds to structures the granulometry has already passed over. Not measured. |
| `blemish_method` | `bth_otsu` | **Measured** — E3.4, **and deliberately not the winner** | `results/t3/e3_4_segmentation.csv` shows hue deviation scoring 0.7412 against black top hat's 0.7163. That is 2.49 points **left on the table on purpose**: hue deviation reads the same colour evidence T1 is built on, so adopting it would make T3 non-disjoint from T1 and no result could be attributed to morphology. The number is reported as the measured cost of keeping the descriptor families independent, not as a configuration to change. |
| `min_blemish_area` | 15 px | **Inherited** | Noise floor for connected components. Not measured. |
| `exclude_poles` | `false` | **Reasoned** | Moallem et al. (2017) note the stem cavity and calyx are dark concavities a naive detector counts as damage. The guard is implemented and left **off** for the main benchmark so it does not become an uncontrolled variable; its effect is to be reported separately. This is a known open problem: mean blemish coverage is 17.2% on fruit graded **Ripe**, which is not credible. |
| Background fill = mean intensity inside the mask | — | **Reasoned** | A zero background creates an artificial cliff at the fruit boundary, and every opening and closing responds to that cliff instead of to the peel. |
| Interior mask eroded by `r_max + 1` | — | **Reasoned** | The second half of the same boundary guard. All spectrum and top-hat statistics accumulate over the interior only. |
| `mm_per_px` | 1.0 | — | **No calibration stage exists.** Blemish areas are therefore in **pixels**, not mm², and every table that reports them says so. Spatial calibration from a reference object is scoped and not built. |

---

## 5. The enhancements

| Choice | Value | Status | Why |
| :-- | :-- | :-- | :-- |
| E1 PCA variance | 0.95 | **Inherited** | Standardise, then PCA to 95% variance, then the shared SVM. Standardisation precedes PCA because PCA is scale-sensitive and the three blocks are on very different numeric scales. The 0.95 itself is not swept. |
| E2 `MIN_SUBREGION_PX` | 40 | **Reasoned** | Below this a sub-region is described as a zero block rather than by forcing an extractor onto a handful of pixels. Clean apples are common in the Unripe class, so this is a normal path, not an error. |
| E2 block order | `[T1(healthy) │ T1(blemished) │ T2(healthy) │ T2(blemished)]` | **Reasoned** | Fixed so the ablation can select an arm by column rather than by re-extracting, which keeps every arm on identical images. `enhance.e2_block_columns` is the single definition and the tests assert it against what `_regional_vector` actually emits. |
| E2 ablation control = **rotate** the blemish mask | — | **Reasoned** | Scattering an equal number of random pixels would also destroy the spatial coherency T1's block B measures, so beating that control would show only that coherent regions beat speckle. Rotating about the fruit centroid preserves area, shape, component count and coherency, and varies only *where* the split falls — which is the actual null hypothesis. |
| E3 weights | cross-validated macro F1 per technique | **Reasoned** | Kittler et al. (1998): a technique that separates classes better on held-out folds is trusted more. Recomputed inside each fold from that fold's training rows. |
| E3 uses `CalibratedClassifierCV` | — | **Reasoned** | Soft voting only means anything if the three probability vectors are on a common scale, and an uncalibrated SVM decision function is not a probability. Used here and nowhere else. |

---

## 6. Experimental protocol

| Choice | Value | Status | Why |
| :-- | :-- | :-- | :-- |
| `seed` | 42 | **Inherited** | One constant, read from the config by every driver, threaded into every RNG, k-means, GrabCut seed, split and estimator. |
| Split | 80:20 stratified | **Brief** | Fixed by the specification. |
| Folds | 5, stratified, drawn over **source images** | **Reasoned** | Splitting the augmented matrix row-wise would put a flipped copy of a validation image into the training fold. Folds are drawn over source images; training folds carry originals and variants, validation folds carry originals only. |
| Augmentation in the **sub-experiment** runners | **off** by default | **Reasoned** | `run_e1/e2/t3_experiments.py` default to no augmentation, matching each other, because these sweeps tune a descriptor against real images. `--augment` is accepted; `--no-augment` is accepted too so the documented smoke commands run verbatim. The Phase 5 driver defaults augmentation **on**, which is why its row counts are four times larger. |
| Sub-experiment arms committed all-or-nothing | — | **Reasoned** | A row enters a sweep only when every arm has described it. A marginal mask can be describable at N=3 and not at N=6, and without this the arms would be scored on subtly different image sets — a difference in accuracy would then be a difference in which apples each arm was shown. |
| Per-class metrics carry a `source` column | `test` / `cv_out_of_fold` | **Reasoned** | The six main configurations predict the held-out test split. An ablation arm never does — it is only ever cross-validated — so it is scored on pooled out-of-fold validation predictions instead. Both provenances are in one file and neither is silently mixed with the other. |
| Results tables are **tracked**, PNGs are not | — | **Reasoned** | A teammate who clones the repository must be able to read the numbers the report quotes without re-running a two-hour pipeline; the CSVs total under half a megabyte. `run_config.json` and `run_metadata.json` are tracked with them, because a table of accuracies is not interpretable without the seed, the partition and the dataset audit verdict that produced it. |

---

## 7. The one that is not a choice

The primary dataset **fails its own suitability audit**. A classifier trained on
these images with the fruit masked out reaches 0.7396 against a three-class
chance level of 0.3333: the backgrounds predict the labels. The audit also finds
45 colour-confirmed near-duplicate pairs, none yet removed.

Nothing in this file mitigates that, and no parameter here could. Every absolute
accuracy in `results/` carries that ceiling, and every `run_config.json` records
the verdict alongside the numbers so a table cannot be read without it. Relative
comparisons between arms of one sweep are unaffected, because every arm sees the
same images — which is the only reason the sweeps above are worth reading at all.

---

## References

```
Gonzalez, R. C., & Woods, R. E. (2018). Digital image processing (4th ed.). Pearson.
Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural features for image
    classification. IEEE SMC, 3(6), 610 to 621.
Kittler, J., Hatef, M., Duin, R. P. W., & Matas, J. (1998). On combining classifiers.
    IEEE TPAMI, 20(3), 226 to 239.
Maragos, P. (1989). Pattern spectrum and multiscale shape representation. IEEE TPAMI,
    11(7), 701 to 716.
Moallem, P., Serajoddin, A., & Pourghassem, H. (2017). Computer vision-based apple
    grading for golden delicious apples based on surface features. Information
    Processing in Agriculture, 4(1), 33 to 40.
Otsu, N. (1979). A threshold selection method from gray level histograms. IEEE SMC, 9(1).
Rother, C., Kolmogorov, V., & Blake, A. (2004). GrabCut: interactive foreground
    extraction using iterated graph cuts. ACM TOG, 23(3), 309 to 314.
Stricker, M., & Orengo, M. (1995). Similarity of color images. SPIE 2420, 381 to 392.
```
