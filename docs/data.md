# Data — what we have, in numbers

A consolidated note on the IBL Brain-Wide Map data we use, layered from "everything public" down to the specific neural sequences fed into our GLM / SVCA / CCA / pCCA pipeline. The point of this document is to make the **scale** of the data concrete: where it's deep, where it's wide, where it's sparse, and which of those constrains our results.

---

## 1. Layered accounting — funnel from public IBL to our analyses

```
Public IBL Alyx (anything accessible with password='international')
   13,637 sessions total
        ↓ filter: has at least one Neuropixels probe insertion
   1,376 sessions
        ↓ filter: in the curated 2023_12 BWM release
       459 sessions   ← what bwm_query() gives us
        ↓ filter: passed strict-QC unit filter (min_qc=1.0, min_units_sessions=(5,2))
       454 sessions, 62,990 good-QC units across 210 Beryl regions
        ↓ filter: top 10 sessions per ROI (VISp, CB, MO, CA1) ∪ V1+CB pair sessions
        41 unique sessions, 4,286 neurons   ← what we fit GLMs on
        ↓ filter: simultaneous V1+CB recording (≥5 units each region)
         3 V1+CB pair sessions, 1 informative-strong + 1 informative-mixed
```

The funnel is **13,637 → 1,376 → 459 → 41 → 3**. The bottleneck is the last step: simultaneous V1+CB Neuropixels recordings are scarce because BWM probe placements are independent across sessions.

**Important null findings on data expansion**:

- The live IBL `Brainwidemap` tag = the frozen 2023_12 CSV. Identity. Zero new sessions since the freeze.
- The Wang–Druckmann 2026 MAP dataset (DANDI:000363) is public and has cerebellum + ALM, but **does not contain V1 recordings** — useful for a different study, not ours.
- 917 probe-bearing sessions exist outside the BWM curation. They didn't pass IBL's QC for the paper. Could in principle be queried, but each requires individual metadata inspection (~30 s each on Alyx) and most failed for reasons that would also affect us.

**The free real-estate move** that's still on the table: expand region definitions from "VISp / MOp+MOs only" to "all visual cortex / all motor cortex." Inside the existing 459 BWM sessions:

- V1↔CB pair sessions: **3 → 10** (at min_units=5)
- V1↔M1 pair sessions: **0 → 9** ← entirely new
- CB↔M1 pair sessions: **1 → 3**

---

## 2. What's in a single session

Concrete numbers from the focal session `41431f53` (`CSH_ZAD_022`, Zador lab, ~88 min):

### Spikes (per probe)

| Probe | Total spikes | Good-QC units | Spike-time range |
|---|---|---|---|
| `2781c08e…` | **20,242,331 spikes** | **177 units** | 0–5268 s |
| `a9c9df46…` | **2,510,913 spikes** | **112 units** | 0–5268 s |

~22.7 million spike events across 289 sorted units in this session. Each spike has timestamp (sub-ms), cluster ID, amplitude (μV), and depth along the probe shank (μm).

### Trials table

```
trials: 570 rows × 20 columns
```

One row per trial. Twenty timestamp/outcome columns: stim onset, first movement, response, feedback, choice (-1/0/1), contrast (left/right), block prior (`probabilityLeft`), reward volume, intervals.

### Wheel

```
wheel: 5,255,570 rows × 4 cols  ← interpolated to 1 kHz
columns: times, position (rad), velocity (rad/s), acceleration (rad/s²)
```

### Cameras (left only on this session — older recording rig)

- Pose: `(317,056 frames × 34 columns)` — 11 body parts × (x, y, likelihood)
- Motion energy: `(317,056 × 2)` — single scalar per frame inside whisker-pad ROI
- Frame rate: 60 Hz

### Disk footprint per session

- Spike sorting per probe: 150–300 MB
- Pose markers (per camera): 30–80 MB
- Motion energy: 5–20 MB
- Wheel: 5–20 MB
- Trials: <1 MB
- **Total ALF (what we use)**: ~300 MB – 2 GB
- Raw video (one camera, optional): **~3 GB**
- Total session including raw video for all cameras: 8–15 GB

We've downloaded ~7 GB of ALF data across 41 sessions, plus 3 GB of one camera mp4 for the focal session.

---

## 3. Long sequences — the time axis is huge

After binning at 20 ms, every session is **one continuous time axis**:

```
Per session:
  recording duration              ~88 min  (5,268 s)
  bin size                        20 ms
  bins per session                T = 262,779
  spike sampling                  sub-millisecond
```

Per-region matrices on this time grid:

| Region | Units | Shape | Total entries |
|---|---|---|---|
| CB on probe 1 | 46 | (46, 263k) | ~12 M |
| V1 on probe 2 | 63 | (63, 263k) | ~17 M |
| **Per session** | 109 | — | **~29 M** |

Across **41 sessions**: 4,286 neurons × ~263k bins each ≈ **~1 billion spike-count entries** in the data fit by the GLM.

Each entry is a small integer (most are 0; rate-modulated bins go up to ~10 spikes per 20 ms for high-firing units). Storage: int16 mostly fits.

### Total scale across all data streams

| Scale | Per session | × 41 sessions |
|---|---|---|
| Spike events (sub-ms) | ~22 M | ~900 M |
| 20 ms bins per neuron | 263,000 | — |
| Total binned spike entries | ~30 M | **~1.0 B** |
| Behavior covariates (20 ms × 8 cols) | ~2 M | ~80 M |
| Trials (event rows) | 570 | ~24,000 |
| Camera frames (60 Hz, leftCamera) | ~317,000 | ~13 M (per camera) |
| Wheel samples (1 kHz) | ~5.3 M | ~217 M |

---

## 4. Why these long sequences matter for the methods

### SVCA / CCA / pCCA

T = 263,000 is what makes the cross-validated canonical correlations meaningful:

- **Phase-shuffle null is narrow.** The wider the time series, the tighter the null bound, so even small canonical correlations (ρ ≈ 0.05) rise statistically above chance.
- **Train/test splits have ~210k / ~53k bins each** — plenty of held-out time to honestly evaluate without leakage. The 5-fold KFold on this T converges to stable mean ρ.
- **Per-fold residualization for pCCA** fits ~210k samples × 2 confound columns onto each region's 8 SVCA scores — well-conditioned, no overfitting risk.

### GLM

Each per-neuron Ridge fit solves a `(263,000, 60)` linear system, 5 times (one per CV fold), then 8 more times for leave-one-group-out. Vectorizing across all neurons in a region (the `fit_region` function in `src/decomp/glm/fit.py`) replaces 100+ small solves with a single batched solve. That's how a 109-neuron session fits in 12 seconds.

### What the long T does NOT fix

The bottleneck on **SVCA reliability** is *not* the time axis. It's the population width.

- T = 263k is enormous; the time-side estimator has plenty of samples.
- But splitting 63 V1 units into 31 + 32 cell halves limits the cross-half cell-similarity estimate. With small cell halves, the off-diagonal of the `(F_train, F_test)` cross-covariance is noisy, and Stringer's reliability ρ_SVCA is bounded above mechanically.
- This is why most components in fig02 sit below 0.5: not weak biology, just small populations.
- More T does not help SVCA. Only more neurons per region do.

So:

- **Depth (T = 263k)**: fine. Makes CCA / pCCA statistics tight.
- **Width (~50 units per region per session)**: the limiting factor on SVCA reliability and downstream confidence.

This is the dataset constraint we cannot fix without different recording experiments. Targeted V1+CB Neuropixels with 200+ units per region would dramatically tighten fig02 and reduce the across-session variance in fig04 — but that's outside course-project scope.

---

## 5. The shape of "the data" at each pipeline stage

A schematic of what each pipeline component sees:

```
spikes (sub-ms)
    ↓ bin at 20 ms
spike-count matrix per region
    Shape: (n_units, T) per ROI per session
    Type: int16
    GLM Y                  : (T, n_units) used per region
    SVCA X                 : (n_units, T) used per region
    CCA scores             : (T, K=8) PCA-projected per region

behavior covariates (1 kHz / 60 Hz / event-aligned)
    ↓ interpolate to 20 ms grid
covariate matrix
    Shape: (T, p ≈ 8) — wheel velocity/acceleration, ME, pupil, lick rate
    Type: float64
    GLM design matrix      : (T, p × n_basis) after raised-cosine expansion
    pCCA confound Z        : (T, 2) — wheel velocity, pupil

trials (event-aligned)
    Shape: (570 trials, 20 columns)
    GLM design matrix      : event indicators × raised-cosine basis kernels
    pCCA / SVCA / CCA      : not directly used (only used for trial-aligned PSTHs)

clusters (static metadata)
    Shape: 109 sorted units × 25 columns of metadata
    Used as: filter to ROI based on Beryl atlas region
```

---

## 6. Pipeline I/O sizes at a glance

Per session at 20 ms binning:

| Stage | Input shape | Output shape | Cost |
|---|---|---|---|
| Bin spikes | (~22M spike events) | (n_units, 263k) per ROI | <1 s |
| Build design matrix | trials + covariates | (263k, ~60) | <1 s |
| Per-region GLM (vectorized) | (263k, 60) X, (263k, n_units) Y | (n_units,) full_R²; (n_units,) ΔR² per group | ~12 s for 109 units |
| SVCA per region | (n_units, 263k) | (k_components,) reliabilities + (263k, K) scores | ~2 s |
| CCA + pCCA per region pair | (263k, 8) × 2 | (8,) ρ_CCA, ρ_pCCA + null | ~80 s with 100 phase-shuffle surrogates |

Total per session for the V1+CB pair pipeline: **~2 minutes** of pure compute, modest CPU. Most of the wall-clock time we spent on the 41-session run was network I/O (downloading ALF), not compute.

---

## 7. Practical implications

- **Do not worry about "more time."** T is already 263k bins per session. Method noise comes from elsewhere (small populations).
- **Adding more sessions sharpens fig01 (per-region ΔR²) but not fig03/04.** The pair-session bottleneck is the recording-overlap structure of BWM, not anything we can fix with longer sessions or more time bins.
- **The data is dense in time, sparse in cells.** This shapes which methods work: linear projections + cross-validation across time = great. High-d population geometry where each component is small-N estimated = noisy. Pick methods accordingly.
- **The disk budget is realistic for course-project scope.** ~10–15 GB of ALF data covers our 41-session universe. Add ~3 GB per camera per session if we ever want raw video. We've used ~7 GB of ALF + 3 GB of one mp4.

If we ever wanted to upgrade to research-paper scope, the data argument would be: targeted dual-probe V1+CB experiments (or a non-IBL public dataset that emerges) is the only path. The pCCA pipeline as written would scale immediately to richer data — the methods are not the bottleneck; the recording geometry is.

---

## 8. Post-MVP expansion (added 2026-05-06)

The MVP analyzed 3 V1+CB pair sessions under strict ROI definitions (`VISp` only for V1, `MOp+MOs` for M1). The post-MVP expansion widens these definitions and adds two new region pairs to test whether the V1↔CB result is specific or generalizes.

### Expanded ROI definitions

| ROI key | Beryl labels | Used in |
|---|---|---|
| `VIS` | VISp, VISa, VISam, VISl, VISli, VISpl, VISpm, VISpor, VISrl, VISal | post-MVP pair pairs |
| `VISp` | VISp only | MVP figures (preserved for comparison) |
| `CB` | LING, CENT2, CENT3, CUL4 5, DEC, FOTU, PYR, UVU, NOD, SIM, ANcr1, ANcr2, PRM, COPY, PFL, FL, FN, IP, DN, VeCB | both |
| `MO` | MOp, MOs | post-MVP pair pairs |
| `MO_WIDE` | MOp, MOs, SSp-bfd, SSp-ll, SSp-m, SSp-n, SSp-tr, SSp-ul, SSp-un | available but not used in default run |
| `CA1` | CA1 | per-region GLM pool only |

### Per-pair coverage matrix on 2023_12 freeze (`min_units_sessions=(5, 2)`)

| Pair | min_units=10 | min_units=5 | (was MVP, strict) |
|---|---|---|---|
| V1↔CB (VIS expansion) | 6 | **10** | 1 / 3 (strict VISp) |
| V1↔M1 (VIS + MOp+MOs) | 3 | **4** | 0 |
| V1↔M1 wide (VIS + MO_WIDE) | 5 | 8 | 0 |
| CB↔M1 (MOp+MOs) | 1 | 1 | 1 |
| CB↔M1 wide (MO_WIDE) | 2 | 3 | 1 |
| 3-way V1+CB+M1 | **0** | **0** | 0 (impossible on this freeze) |

### Implication for the project

- **V1↔CB sample triples** (3 → 10), enabling per-session distribution claims rather than n=2 anchor-session claims.
- **V1↔M1 unlocks** (0 → 4) at strict M1 definition, providing the comparison anchor that lets us test whether V1↔CB is specifically interesting or whether all cortical pairs preserve shared structure under wheel+pupil partialling.
- **CB↔M1 stays at n=1** strict — informative but not statistically meaningful. The CB↔M1 wide variant adds 2 more sessions if needed.
- **Within-session 3-region partial CCA `CCA(V1, CB | M1)` is impossible.** Zero sessions in BWM 2023_12 have simultaneous V1+CB+M1 Neuropixels coverage, even at the widest definitions and lowest QC. The "is V1↔CB shared structure just an echo of motor cortex's command" hypothesis cannot be tested within-session on this dataset.
- **Updated post-expansion funnel**: 13,637 → 1,376 → 459 → **51 sessions actually used** (vs 41 in MVP), 4,886 GLM neurons (vs 4,286), 15 pair sessions across 3 pair types (vs 3 across 1 pair type).

See `docs/expansion_analysis.md` for the full post-expansion result interpretation.
