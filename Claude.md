# Technique 3 Implementation Spec: Multiscale Morphological Descriptors

**Project:** FreshSight (BMDS2133 Image Processing, Mode A, Topic 2)
**Owner:** Tang Yue Hann (2414352), Member 3
**Status:** Replaces the previous LBP based design. LBP is dropped entirely.

---

## 1. Read this first

This spec covers **one branch only**: the T3 feature extractor and its internal experiments.

**Do not modify** the shared harness, T1 (dominant colour), T2 (GLCM), the SVM
configuration, the data partition, or the evaluation module. Mode A requires that the
descriptor family is the only variable. Any change outside `t3_morphological.py` breaks
the comparison and must be raised before it is made.

If the shared harness does not exist yet in the repo, stub it against the contract in
Section 3 and flag it. Do not invent a different preprocessing pipeline.

---

## 2. Why this design

The previous T3 used uniform LBP at (P=8,R=1) and (P=16,R=2) plus hue deviation blemish
detection. Two problems:

1. LBP is not covered in the BMDS2133 syllabus, so the technique had no teaching basis.
2. Hue deviation reuses the colour evidence that T1 already exploits, which weakens
   attribution in a controlled comparison.

The replacement is built entirely from Chapter 8 morphology plus Chapter 9 region
statistics: granulometry, top hat transforms, morphological gradient, and connected
component descriptors. It reads surface **geometry**, not colour and not intensity
statistics, so it stays disjoint from T1 and T2.

---

## 3. Upstream contract (what the harness gives you)

The harness has already done all of this. Do not repeat any of it.

| Step | Detail |
|---|---|
| Resize | 224 x 224 pixels |
| Denoise | Gaussian, 5 x 5 kernel |
| Illumination | CLAHE on L* channel of CIE L*a*b*, clip 2.0, tiles 8 x 8, recombined to BGR |
| Segmentation | HSV value channel, Otsu threshold, closing with 5 x 5 SE, largest connected component retained |
| Outputs | `image_bgr` (uint8 HxWx3), `mask` (uint8 HxW, 0 or 255), `bbox` (x, y, w, h), `contour`, `mm_per_px` (float, defaults to 1.0) |

`mm_per_px` comes from the calibration stage. Blemish areas must be reported in mm^2
when it is not 1.0, and in pixels with an explicit note when it is.

---

## 4. Output contract

```python
def extract(image_bgr, mask, mm_per_px=1.0, config=None):
    """
    Returns
    -------
    features : np.ndarray, shape (36,), dtype float64, no NaN, no inf
    aux : dict
        {
          "blemish_mask": np.ndarray uint8 HxW (0 or 255, full image frame),
          "blemish_ratio_pct": float,
          "labels": np.ndarray int32 HxW,
          "n_blemish": int,
          "areas_mm2": list[float],
        }
    """
```

The signature must match T1 and T2 so the harness can swap branches without changes.
`aux["blemish_mask"]` is consumed by Enhancement 2 for regional weighting and by the
dashboard, so it must be returned in the **full image frame**, not the cropped frame.

---

## 5. Feature vector, 36 dimensions

Order is fixed. Do not reorder, the fusion code in Enhancement 1 indexes by position.

| Idx | Name | Block |
|---|---|---|
| 0 to 6 | `gran_open_r1` .. `gran_open_r7` | A. Pattern spectrum |
| 7 to 13 | `gran_close_r1` .. `gran_close_r7` | A. Pattern spectrum |
| 14 | `gran_mean_size` | B. Spectrum moments |
| 15 | `gran_entropy` | B. Spectrum moments |
| 16 | `gran_peak_scale` | B. Spectrum moments |
| 17 | `gran_spread` | B. Spectrum moments |
| 18 to 21 | `bth_mean`, `bth_std`, `bth_max`, `bth_p95` | C. Top hat statistics |
| 22 to 25 | `wth_mean`, `wth_std`, `wth_max`, `wth_p95` | C. Top hat statistics |
| 26 to 27 | `mgrad_mean`, `mgrad_std` | D. Roughness |
| 28 | `blemish_ratio` | E. Blemish geometry |
| 29 | `blemish_count` | E. Blemish geometry |
| 30 | `blemish_area_mean` | E. Blemish geometry |
| 31 | `blemish_area_max` | E. Blemish geometry |
| 32 | `blemish_ecc_mean` | E. Blemish geometry |
| 33 | `blemish_solidity_mean` | E. Blemish geometry |
| 34 | `blemish_extent_mean` | E. Blemish geometry |
| 35 | `blemish_eqdiam_max` | E. Blemish geometry |

Expose the names as a module constant `FEATURE_NAMES` so the dashboard and the ablation
study can label columns without hardcoding.

---

## 6. Region preparation

1. Convert `image_bgr` to greyscale.
2. Crop greyscale and mask to `bbox`.
3. **Background fill.** Set pixels outside the mask to the mean intensity of the pixels
   inside the mask. Do not leave them at 0. A zero background creates an artificial cliff
   at the fruit boundary, and every opening and closing will respond to that cliff instead
   of to the peel.
4. **Interior mask.** Erode the cropped mask with a disk of radius `r_max + 1`
   (default 8). All statistics in Sections 7 to 9 are accumulated over the interior mask
   only, never over the full mask. This is the second half of the boundary guard.
5. If the interior mask has fewer than 100 pixels, return a zero vector and set
   `aux["degenerate"] = True`. Log the filename. Do not raise.

Structuring elements are **disks** throughout, built with
`cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2r+1, 2r+1))`. A disk is isotropic, and
apple defects have no characteristic orientation.

---

## 7. Stage 1: granulometric pattern spectrum

Let `f` be the prepared greyscale, `S_r` the disk of radius r.

**Opening spectrum**, r = 1 to 7:

```
g_0 = f
g_r = opening(f, S_r)
PS_open[r] = sum(g_{r-1}) - sum(g_r)        # summed over interior mask only
PS_open    = PS_open / (sum(PS_open) + 1e-8)
```

**Closing spectrum**, r = 1 to 7, same structure with `h_r = closing(f, S_r)` and
`PS_close[r] = sum(h_r) - sum(h_{r-1})`.

Both are non negative by the monotonicity of openings and closings. If any bin comes out
negative, the interior mask is wrong. Assert it.

**Moments.** Build `q[r] = (PS_open[r] + PS_close[r]) / 2` for r = 1 to 7, then:

```
gran_mean_size = sum(r * q[r])
gran_entropy   = -sum(q[r] * log2(q[r] + 1e-12))
gran_peak_scale= argmax(q) + 1
gran_spread    = sqrt(sum((r - gran_mean_size)^2 * q[r]))
```

Reference: Maragos (1989), pattern spectrum and multiscale shape representation.

---

## 8. Stage 2: top hat responses and roughness

Disk radius 9 for both transforms.

```
BTH = closing(f, S_9) - f        # black top hat, dark structures on bright peel
WTH = f - opening(f, S_9)        # white top hat, bright structures
MG  = dilate(f, S_1) - erode(f, S_1)   # morphological gradient
```

Use `cv2.morphologyEx` with `MORPH_BLACKHAT`, `MORPH_TOPHAT`, `MORPH_GRADIENT` rather
than composing manually. Compute in `float32` to avoid uint8 clipping on the difference.

From BTH and WTH take mean, standard deviation, maximum, and 95th percentile over the
interior mask. From MG take mean and standard deviation. Normalise all eight response
statistics by 255.0 so they sit on a comparable scale to the spectrum bins before the
harness standardiser runs.

---

## 9. Stage 3: blemish region analysis

1. Threshold BTH with Otsu (`cv2.threshold` with `THRESH_BINARY + THRESH_OTSU` on the
   uint8 rescaled BTH).
2. AND with the interior mask.
3. Morphological opening with a 3 x 3 SE to drop isolated pixels.
4. `skimage.measure.label` then `regionprops`.
5. Discard components smaller than `min_blemish_area` (default 15 px) as noise.

Descriptors:

```
blemish_ratio       = 100 * blemish_px / fruit_mask_px      # per report formula
blemish_count       = number of surviving components
blemish_area_mean   = mean area, in mm^2 if mm_per_px != 1.0
blemish_area_max    = max area, same units
blemish_ecc_mean    = mean eccentricity
blemish_solidity_mean = mean solidity
blemish_extent_mean = mean extent
blemish_eqdiam_max  = equivalent diameter of the largest component
```

Note `blemish_ratio` uses the **full fruit mask** as denominator (that is the published
formula), while detection runs on the interior mask. Keep the two separate.

**Optional guard, off by default.** Moallem et al. (2017) report that the stem cavity and
calyx are dark concavities that naive defect detectors count as blemishes. Implement
`config["exclude_poles"]` which removes components whose centroid falls within 15% of the
mask centroid along the major axis extremes. Leave it `False` for the main benchmark and
report its effect separately, otherwise it becomes an uncontrolled variable.

Finally, paste the cropped blemish mask back into a full frame array before returning.

---

## 10. Internal comparative experiments

Each writes a CSV to `results/t3/`. All tuning uses 5 fold stratified CV on the
**training partition only**. Never touch the test split.

| ID | Variable | Sweep | Output file |
|---|---|---|---|
| E3.1 | Multiscale contribution | full 36 vector vs Block C+D only (10 dims) vs Block A+B only (18 dims) | `e3_1_ablation.csv` |
| E3.2 | SE shape | `MORPH_ELLIPSE`, `MORPH_RECT`, `MORPH_CROSS` | `e3_2_se_shape.csv` |
| E3.3 | Max granulometric radius | r_max in {5, 7, 9, 11} | `e3_3_rmax.csv` |
| E3.4 | Blemish segmentation | black top hat + Otsu, fixed intensity threshold, hue deviation baseline | `e3_4_segmentation.csv` |

E3.4 needs the hue deviation method implemented purely as a comparator, so the report can
justify dropping it. Score it on mean absolute blemish ratio error against manually
annotated masks, not on classification accuracy.

E3.3 changes the vector length. Report accuracy against dimensionality, do not silently
pad or truncate.

---

## 11. Repository layout

```
src/
  harness/            # DO NOT EDIT
  techniques/
    t1_dcd.py         # DO NOT EDIT
    t2_glcm.py        # DO NOT EDIT
    t3_morphological.py   # your file
  experiments/
    run_t3_experiments.py
results/t3/
tests/test_t3.py
```

Keep everything for T3 inside `t3_morphological.py` plus its experiment runner. No shared
utility edits.

---

## 12. Reproducibility

- Seed everything from a single `SEED` constant read from the harness config.
- No randomness in this technique, so `extract()` must be deterministic. The test suite
  asserts that two calls on the same input return byte identical vectors.
- Log the config dict used for each experiment run alongside the CSV.

---

## 13. Acceptance criteria

- [ ] `extract()` returns exactly 36 finite floats for every image in the primary dataset
- [ ] `FEATURE_NAMES` has length 36 and matches the order in Section 5
- [ ] All spectrum bins are non negative and each spectrum sums to 1.0 within 1e-6
- [ ] `blemish_mask` is returned in the full image frame with the same shape as `mask`
- [ ] Feature extraction plus inference stays under 5 seconds per image
- [ ] Degenerate masks return a zero vector and a flag rather than raising
- [ ] Standalone accuracy target is 80% or better; report the number honestly either way
- [ ] All four experiment CSVs are produced and committed

---

## 14. Known pitfalls

1. **Boundary response.** The single most likely bug. If the spectrum is dominated by
   bin r=1 or the values look identical across all classes, the interior mask erosion is
   missing or the background fill is still zero.
2. **uint8 overflow.** `closing(f) - f` on uint8 wraps around. Cast to float32 first.
3. **Radius vs kernel size.** `getStructuringElement` takes a size, not a radius. A disk
   of radius r is size `2r+1`.
4. **Empty blemish mask.** Perfectly clean apples exist in the Unripe class. All eight
   Block E features must default to 0.0, not NaN.
5. **Scale leakage.** Standardisation is fitted on the training fold only, inside the
   harness. Do not standardise inside `extract()`.

---

## 15. References for the docstrings

```
Bai, X., Zhou, F., & Xue, B. (2012). Image enhancement using multi scale image features
    extracted by top hat transform. Optics & Laser Technology, 44(2), 328 to 336.
Maragos, P. (1989). Pattern spectrum and multiscale shape representation. IEEE TPAMI,
    11(7), 701 to 716.
Matheron, G. (1975). Random sets and integral geometry. Wiley.
Otsu, N. (1979). A threshold selection method from gray level histograms. IEEE SMC, 9(1).
Serra, J. (1982). Image analysis and mathematical morphology. Academic Press.
Soille, P. (2004). Morphological image analysis (2nd ed.). Springer.
```