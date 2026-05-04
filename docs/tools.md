# Establishing Understanding between Cortical Regions via GLM, SVCA, CCA, and pCCA

This note explains, in math and in plain language, why each step of our pipeline is the
right tool for **the specific question** we're asking on the IBL Brain-Wide Map data:

> Are the movement signals in V1 and cerebellum *the same kind of signal*, or *distinct
> region-specific computations that just look similar at the decoding level*?

Each section: what the method is mathematically, why it is necessary, what it would tell us
on our data, and what it cannot decide on its own.

---

## 0. The setup

Per session $s$, after binning at $\Delta t = 20$ ms and aligning behavior covariates:

- $Y^{(R)}_s \in \mathbb{R}^{n_R \times T}$ — spike-count matrix for region $R \in \{\mathrm{V1},\mathrm{CB},\mathrm{M1},\mathrm{CA1}\}$, with $n_R$ good-QC units and $T$ bins ($T \approx 2.6 \times 10^5$ for an hour-long session).
- $X_s \in \mathbb{R}^{T \times p}$ — design matrix of behavior + task regressors, expanded with raised-cosine basis kernels (see §1).
- $Z_s(t) = (\text{wheel}(t),\, \text{pupil}(t)) \in \mathbb{R}^{T \times 2}$ — the two confounds we will partial out.

The question asks about the **representational geometry** of movement signals across regions. Geometry $\neq$ information. Two regions can carry the same scalar fact (running speed) in completely different population codes, and a decoder will be happy with both. We need methods that probe geometry, not just decoding accuracy.

---

## 1. GLM — Per-Neuron Variance Partition

### Intuition first

Think of one neuron's spike train as **the sum of many overlapping echoes**. When the mouse sees a stimulus, the neuron fires a little bump 80 ms later. When the mouse first moves its paw, another bump. When the wheel speeds up, the firing rate ramps. These bumps and ramps overlap in time — at any given 20 ms bin, the neuron's spike count is the sum of every recent event's lingering influence plus its instantaneous coupling to ongoing variables (wheel velocity, motion energy, pupil) plus noise.

A GLM is the formal version of *"untangle these overlapping echoes."* We give the model a menu of behaviors and task variables, expand each one into a small set of basis shapes that look like plausible neural responses (raised cosines — bumps with different peak times and widths), and ask: **which combination of these basis shapes, with what weights, best predicts this neuron's spike count?** The "best fit" is the linear combination $\hat{\boldsymbol{\beta}}$.

The trick that makes the GLM useful for *our* question is what comes next. Once we have the full model fit, we ask: *if I deleted the entire "movement" group of regressors and refit, how much worse does prediction get on held-out time?* That gap — full-model R² minus drop-group R² — is **ΔR², the unique contribution of movement-related variables to this neuron**. Doing this for every neuron, every region, gives us a per-region distribution of "how much is this neuron's variance about movement, after we've already let stimulus and choice and feedback compete for the same variance."

The literature prediction is an ordering: **movement variance per neuron should be highest in M1 (motor command), then cerebellum (forward model + locomotion speed), then V1 (arousal-driven gain modulation), then CA1 (theta-gated locomotion).** If our pipeline recovers that ordering, we know the design matrix is honest and we can trust everything downstream.

What this *cannot* tell us is what the movement signal **looks like as a population**. Two regions can land at the same per-neuron ΔR² medians while encoding movement in completely different population codes. ΔR² is a scalar per neuron. The geometry-vs-decoding distinction is invisible to it. That is what the next three steps are for.

### Mathematical form

For neuron $i$ in region $R$, with binned spike count $y_i(t)$:

$$
y_i(t) \;=\; \beta_{i,0} \;+\; \sum_{g \in \mathcal{G}} X_g(t)\,\boldsymbol{\beta}_{i,g} \;+\; \varepsilon_i(t),
\qquad
\hat{\boldsymbol{\beta}}_i \;=\; \arg\min_{\boldsymbol{\beta}} \;\|y_i - X\boldsymbol{\beta}\|_2^2 \;+\; \alpha\|\boldsymbol{\beta}\|_2^2 .
$$

The kernel groups $\mathcal{G}$ are
$\{\text{stim}, \text{movement}, \text{feedback}, \text{choice}, \text{prior}, \text{wheel}, \text{me\_face}, \text{pupil}\}$.
Each event-aligned regressor is the convolution of the event train with a bank of
$n_b{=}5$ raised-cosine basis kernels:

$$
b_k(\tau) \;=\; \tfrac{1}{2}\bigl(\cos\!\bigl(\tfrac{\pi (\log(\tau + \tau_0) - c_k)}{2 w}\bigr) + 1\bigr)
\quad \text{clipped to } [-\pi,\pi].
$$

A continuous regressor $u(t)$ contributes columns $(u \star b_k)(t)$. The design matrix $X$
ends up with $p \approx 60$ columns: enough to fit a smooth temporal response without
overfitting at 20 ms resolution.

For each kernel group $g$, the **leave-one-group-out CV ΔR²** is

$$
\Delta R^2_{i,g} \;=\;
R^2\bigl(y_i,\, \hat y_i^{\mathrm{full}}\bigr)\Big|_{\text{test}}
\;-\;
R^2\bigl(y_i,\, \hat y_i^{\setminus g}\bigr)\Big|_{\text{test}},
\qquad
\text{averaged over 5 KFold splits.}
$$

This is the IBL BWM 2025 paper's variance-partition formula verbatim
(`neurencoding.SequentialSelector(direction="backward", full_scores=True)` in their stack;
implemented from scratch and vectorized in ours so we can fit all neurons in a region in
one matrix solve).

### Why GLM is necessary

A naive decoder ("can I read movement out of V1?") gives a single scalar per region. It
*cannot* attribute that decodability to specific behavior variables vs others, and it
*cannot* tell us whether a region encodes movement in a clean low-d code or as a
faint contribution to noisy heterogeneous responses. The GLM gives a per-neuron, per-kernel
**unique contribution**: $\Delta R^2_{i,\text{movement}}$ is the variance of neuron $i$'s
spike count that is *only* explainable by movement-related regressors after wheel,
motion energy, and licks have been allowed to compete with stimulus, choice, feedback,
and prior. No simpler estimator yields this number.

### What it tells us on this dataset

The literature prior (Wang–Druckmann 2026; IBL 2025 Extended Data Fig. 13): movement
encoding is **structured across regions**, with stronger encoding closer to the motor
periphery. Concretely, per-region medians should rank

$$
\widetilde{\Delta R^2_\text{movement}}(\mathrm{M1})
\;\geq\;
\widetilde{\Delta R^2_\text{movement}}(\mathrm{CB})
\;\geq\;
\widetilde{\Delta R^2_\text{movement}}(\mathrm{V1})
\;\geq\;
\widetilde{\Delta R^2_\text{movement}}(\mathrm{CA1}) .
$$

This is our **literature-reproduction figure** (`fig01_glm_dr2_per_region`). If this
ordering does not emerge from our pipeline, the design matrix is wrong and nothing
downstream is trustworthy. If it does, we have validated the data path end-to-end and
can interpret the cross-region results that follow.

### What GLM cannot decide

GLM operates per neuron and ignores population geometry. Two regions can land at identical
$\Delta R^2_\text{movement}$ medians with **completely different population codes** —
e.g. CB carrying movement in a low-d running-speed axis vs V1 carrying it in a
high-dimensional gain-modulated state space. ΔR² is silent on which case we are in.

---

## 2. SVCA — What Dimensions Are Real Signal Within a Region

### Intuition first

CCA (the next stage) wants two **clean low-dimensional summaries**, one per region. A naive way to get one would be: PCA on V1's `(63 units × 262k bins)` activity matrix, keep the top 8 PCs. But that's dangerous — the leading PCs of *any* matrix exist whether or not they represent reproducible structure. With 63 noisy spike trains and 260,000 bins, even pure noise produces "leading components" that look interpretive. We would be doing high-precision arithmetic on noise.

SVCA is the version of PCA that **certifies its own components**. The trick: split the neurons in V1 randomly into two halves, A and B. Split time randomly into halves $\mathcal{T}_\text{tr}$ and $\mathcal{T}_\text{te}$. On the *training* time block, find the directions of maximal cross-covariance between A and B — these are the components where A's activity and B's activity are *coupled*. Now project the *held-out* time block onto those directions. If the same component is also coupled on the held-out time, it's real signal. If A and B's projections diverge on the held-out time, the component was overfit noise.

Concretely: for each candidate component $k$, compute the correlation of A's projection $s_A^{(k)}(t)$ with B's projection $s_B^{(k)}(t)$ on held-out time. The closer this is to 1, the more reliably this dimension is shared across two random halves of the same region — i.e. it's not a single-cell idiosyncrasy or a noise artifact, it's a population mode. This number is the **reliability** $\rho^\mathrm{SVCA}_k$.

Why this matters for our question: when we later ask *"do V1 and CB share a subspace?"*, the answer is meaningful only if we trust that V1's subspace and CB's subspace are themselves trustworthy summaries of each region's activity. SVCA is the formal trust certificate. We use the reliability spectrum as a per-region quality figure and project the full population onto its top-K PCs (K=8) as the **denoised state** $S_R(t) \in \mathbb{R}^{T \times K}$ that CCA will operate on.

What SVCA *cannot* tell us is whether V1's 8 reliable dimensions and CB's 8 reliable dimensions point in compatible directions. SVCA is purely intra-regional — within V1, within CB. The cross-regional alignment question is what CCA addresses.

### Mathematical form

Let $Y^{(R)} \in \mathbb{R}^{n \times T}$ (region $R$ activity matrix, mean-centered). Randomly partition cells $\{1,\ldots,n\} = \mathcal{F}_\text{tr} \sqcup \mathcal{F}_\text{te}$ and time $\{1,\ldots,T\} = \mathcal{T}_\text{tr} \sqcup \mathcal{T}_\text{te}$.

Compute the cell × cell cross-covariance on the training time block:

$$
C \;=\; Y^{(R)}_{\mathcal{F}_\text{tr},\,\mathcal{T}_\text{tr}}\, Y^{(R)\top}_{\mathcal{F}_\text{te},\,\mathcal{T}_\text{tr}}
\quad\in\quad \mathbb{R}^{|\mathcal{F}_\text{tr}|\times|\mathcal{F}_\text{te}|}.
$$

PCA-decompose $C \approx U \Sigma V^\top$. Project the **held-out time block** with these loadings:

$$
s_\text{tr}^{(k)}(t) \;=\; u_k^\top\, Y^{(R)}_{\mathcal{F}_\text{tr},\,\mathcal{T}_\text{te}}, \qquad
s_\text{te}^{(k)}(t) \;=\; v_k^\top\, Y^{(R)}_{\mathcal{F}_\text{te},\,\mathcal{T}_\text{te}}.
$$

The Stringer 2019 **reliability** of component $k$ is

$$
\rho^{\mathrm{SVCA}}_k \;=\;
\frac{\langle s_\text{tr}^{(k)}\,s_\text{te}^{(k)}\rangle_t}
     {\frac{1}{2}\bigl(\langle (s_\text{tr}^{(k)})^2\rangle_t + \langle (s_\text{te}^{(k)})^2\rangle_t\bigr)}.
$$

A reliability close to 1 means: this component is reproducibly present in *both* random halves of the population on *held-out* time. Close to 0 means it is noise.

For downstream CCA, we need a $(T, K)$ score time series per region. We take the top-$K$ principal components of the full $Y^{(R)}$ (whitened, $K=8$) as the region's denoised state $S_R(t) \in \mathbb{R}^{T \times K}$, and report the SVCA reliability spectrum as a diagnostic.

### Why SVCA is necessary

A vanilla PCA of any $(n, T)$ matrix produces "leading components" whether or not the neurons share any real structure. With $T$ bins and $n \approx 50$ units, the top PCs of a pure-noise matrix already look impressively coherent. SVCA's cross-half certificate ($\rho^\mathrm{SVCA}_k > 0.5$, say) is the formal version of the question *"would I get this same component if I had measured a different random half of the population, on different time?"* — exactly the claim CCA needs as input.

### What it tells us on this dataset

Per-region reliability spectra are written to `fig02_svca_reliability`. With ~50–100 QC units per region per session, we typically see 1–3 components clearly above 0.5 and a slow decay through the next several. The fixed $K=8$ choice is a pragmatic upper bound for the CCA stage: enough headroom to detect higher-d shared structure if it exists, small enough that the cross-region SVD remains well-conditioned.

### What SVCA cannot decide

SVCA is **intra-regional**. It says nothing about whether V1's reliable components and CB's reliable components are *aligned*, or whether the shared structure (if any) is arousal-driven. It supplies trustworthy coordinates; the comparative work happens next.

---

## 3. CCA — Is There a Shared Subspace Between Regions?

### Intuition first

Here is the failure mode of decoding-based answers. Suppose you train a classifier on V1 spike rates that predicts "running vs not running" with 95% accuracy. You train the same classifier on CB and also get 95%. You conclude: "movement is in both." But this tells you *nothing* about whether the two regions encode movement using the same features. V1 might use a single arousal-driven gain axis; CB might use a high-d forward-model state that *happens* to also be linearly decodable for running. Both can hit 95% decoding while being **representationally orthogonal**.

CCA asks the strictly stronger question: *given V1's reliable population state $S_\mathrm{V1}(t)$ and CB's reliable population state $S_\mathrm{CB}(t)$, is there a direction $a$ in V1-space and a direction $b$ in CB-space such that the time series $a^\top S_\mathrm{V1}(t)$ and $b^\top S_\mathrm{CB}(t)$ rise and fall together?* If yes, V1 and CB are not just both informative about movement — their codes share an axis. If no, they are coupled only at the decoding level, not at the geometry level.

The math finds the most-correlated axis $(a_1, b_1)$ with correlation $\rho_1$, then the next-most-correlated axis orthogonal to the first $(a_2, b_2)$ with $\rho_2$, and so on. $\rho_1 = 1$ means V1 and CB have a perfectly aligned direction. $\rho_1 = 0$ means they have no shared linear structure at all. We measure $\rho_k$ on **held-out time folds** (not the time used to find $a_k, b_k$) so the answer can't be inflated by overfitting.

We also need a null. Long timeseries with autocorrelation can produce spurious correlations — slow drift in two regions of the brain *will* covary just from shared arousal slow waves, even after SVCA. The **phase-shuffle null** addresses this: randomize the Fourier phases of each channel of $S_\mathrm{V1}$ independently (preserving its power spectrum, breaking its temporal alignment with $S_\mathrm{CB}$), then redo the CCA. Repeat 200 times. Anything above the 99th percentile of these null correlations is real cross-region coupling, not autocorrelation drift.

What CCA *still cannot tell us* is the most important thing. Suppose we find $\rho_1 = 0.34$ above null, $\rho_2 = 0.31$ above null, and so on. There is shared structure. But why is it there?

- Maybe both V1 and CB are reading off a single global "running speed × arousal" state. That global state is itself basically a 1- or 2-dimensional function of wheel speed and pupil diameter. Both regions then show high $\rho_k$ trivially, because they both contain a copy of the same global drive. Same kind of signal, copied.
- Or maybe V1 and CB share representational structure that goes *beyond* what wheel speed and pupil can explain. Different signals — but coupled.

CCA doesn't distinguish these. They give the same answer. The next step is what distinguishes them.

### Mathematical form

Given the SVCA score time series $S_\mathrm{V1}(t) \in \mathbb{R}^{T \times K}$ and $S_\mathrm{CB}(t) \in \mathbb{R}^{T \times K}$, both centered:

$$
C_{xx} = \tfrac{1}{T} S_\mathrm{V1}^\top S_\mathrm{V1} + \lambda I, \quad
C_{yy} = \tfrac{1}{T} S_\mathrm{CB}^\top S_\mathrm{CB} + \lambda I, \quad
C_{xy} = \tfrac{1}{T} S_\mathrm{V1}^\top S_\mathrm{CB}.
$$

Whiten and SVD:

$$
\boxed{\;
\widetilde{C} \;=\; C_{xx}^{-1/2}\, C_{xy}\, C_{yy}^{-1/2}
\;=\; U\,\Sigma\,V^\top
\;}
$$

The diagonal entries $\sigma_1 \geq \sigma_2 \geq \cdots$ are the **canonical correlations** $\rho_k$. The canonical directions in V1 and CB are $a_k = C_{xx}^{-1/2} u_k$ and $b_k = C_{yy}^{-1/2} v_k$. By construction, $a_k$ and $b_k$ are the linear combinations of V1 and CB scores that maximize the Pearson correlation of their projections, subject to orthogonality of earlier pairs:

$$
\rho_k \;=\; \max_{a,\,b} \;\mathrm{corr}\bigl(a^\top S_\mathrm{V1}(t),\; b^\top S_\mathrm{CB}(t)\bigr)
\quad \text{s.t.} \quad a^\top C_{xx} a_j = b^\top C_{yy} b_j = 0,\; j<k.
$$

We compute everything on training folds, project held-out time onto the learned $(a_k, b_k)$, and report the **5-fold cross-validated** Pearson $\rho_k$ on the test fold. A **phase-shuffle null** randomizes the Fourier phases of each $S_R$ channel independently (preserving its power spectrum, breaking temporal alignment across regions); 200 such surrogates give us the 99th-percentile null bound per component.

### Why CCA is necessary — *and why decoding alone fails*

A decoder asks: *"is movement information present in this code?"* CCA asks the strictly stronger geometric question: *"is movement information represented along the same axes in the two codes?"*

Concretely: imagine V1 carries running speed in direction $\hat a$, and CB carries running speed in some other direction $\hat b$ that has zero overlap with $\hat a$ when both are expressed in a common neural coordinate (which they aren't, but the comparison makes sense after whitening and SVCA's principal-component change of basis). A linear decoder reads each off perfectly. CCA returns $\rho_k = 0$ for all $k$. The two regions share *information* but not *representation*.

Conversely, if both regions carry running speed along directions that *do* correspond through the cross-covariance, CCA picks up a leading $\rho_1$ near 1. The phase-shuffle null is critical: with $T \approx 10^5$ and slow drift, spurious correlations from autocorrelated noise can be substantial. The null preserves each channel's autocorrelation while breaking cross-region temporal alignment, so anything above the null is genuinely coupled, not just both-slowly-drifting.

### What it tells us on this dataset

`fig03_cca_canonical_correlations` shows the per-component $\rho_k$ for the 1–3 V1+CB pair sessions in BWM 2023_12, with the 99% null bound. On the strong session (`CSH_ZAD_022`, 63 V1 + 46 CB):

| $k$ | $\rho_k$ | null$_{99}$ | above null? |
|---|---|---|---|
| 1 | 0.344 | 0.324 | ✓ (just) |
| 2 | 0.306 | 0.149 | ✓ strongly |
| 3 | 0.061 | 0.031 | ✓ |
| 4 | 0.048 | 0.016 | ✓ |
| 5 | 0.033 | 0.012 | ✓ |
| 6 | 0.013 | 0.010 | ✓ marginal |
| 7+ | $\leq 0.004$ | $\sim 0.009$ | ✗ |

So ~5–6 dimensions of V1 activity and CB activity co-vary on held-out time more than phase-shuffled surrogates do. *Some* shared subspace exists.

### What CCA cannot decide

CCA does not distinguish two qualitatively different reasons for shared structure:

(a) **Both regions are simply tracking a global low-d arousal/locomotion state** (running speed × pupil ≈ 1- or 2-D). A common drive into both populations would look exactly like a high $\rho_1$ between V1 and CB scores.

(b) **The regions are doing distinct movement-related computations that share more than the global drive.**

These are exactly the two hypotheses in the project's research question, and CCA cannot adjudicate between them. We need to *control for* the global drive.

---

## 4. pCCA — Is the Shared Structure Reducible to Global Arousal?

### Intuition first

This is the step the entire project hinges on. Everything before it has been setup; this is the actual experimental contrast.

Imagine you're trying to figure out whether two students cheat by comparing answer patterns. You find they agree on 80% of test questions — high "shared structure". But both students are also taking the same lectures, doing the same homework, learning from the same teacher. Of course they agree on a lot. The real test of cheating is: *if I control for everything they have in common from the lectures, do they STILL agree on the remaining questions?* If yes — cheating. If no — they're just both well-taught.

Partial CCA is the same idea applied to V1 and CB. Both regions are bathed in a global arousal/locomotion drive that is itself approximately a function of wheel speed and pupil diameter (these are our two confounds, $Z = (\text{wheel}, \text{pupil})$). Of course V1 and CB will look correlated when the mouse is running fast with dilated pupils vs sitting still with constricted pupils. The question is: *what's left after we remove the part of V1 activity and CB activity that is linearly predictable from $Z$?*

Mechanically: regress $Z$ onto $S_\mathrm{V1}$ to get the part of V1 explainable by wheel + pupil, and subtract it. Same for CB. Now run CCA on the **residuals**. The canonical correlations on the residuals are called *partial canonical correlations*, $\rho_k^\mathrm{pCCA}$. They measure shared structure between V1 and CB **that is not attributable to wheel or pupil**.

The key diagnostic is the **survival ratio**, $\rho^\mathrm{pCCA}_k / \rho^\mathrm{CCA}_k$:

- If the V1–CB shared structure was *just* the global arousal drive, removing wheel and pupil should make most of it vanish. $\rho^\mathrm{pCCA}_k \to 0$. Survival ratio $\to 0$. Conclusion: same kind of signal, inherited copy of one global state. This is the "movement signal is one thing showing up everywhere" hypothesis.
- If V1 and CB share representational structure *beyond* what wheel and pupil can account for, the residuals retain shared canonical correlations. $\rho^\mathrm{pCCA}_k \approx \rho^\mathrm{CCA}_k$. Survival ratio $\to 1$. Conclusion: distinct computations sharing more than global arousal. This is the "different signals that happen to share substructure" hypothesis.

Two technical subtleties matter for honesty.

First, **train/test discipline.** When we fit "the part of V1 explainable by Z," we fit that on training data only and apply the fit to held-out test data. If we instead fit on all data and then did CCA with cross-validation, the residualization itself would have already seen the test set, and even truly-unrelated $Z$ would shrink test-set correlations spuriously. This would inflate apparent collapse and bias the test toward $H_0$. Fitting $Z\to S$ only on the training fold avoids this leakage.

Second, **what's in $Z$ matters.** We chose wheel velocity and pupil because they are the two cleanest, lab-standard scalars for locomotion and arousal. If the global drive were nonlinear in these scalars (e.g. depends on $\dot{Z}$ or on $Z^2$), our pCCA might underestimate it. A higher-dimensional $Z$ — including motion energy, body camera markers, lick rate — would be a stronger test. We're using the conservative, minimal pair because it makes the result *interpretable*: "after wheel and pupil are removed." A failure to collapse with this minimal $Z$ is a stronger argument against the global-state hypothesis than a failure with a kitchen-sink $Z$ would have been, because the minimal $Z$ is exactly what people mean when they say "global arousal."

Preliminary on the strong session (`CSH_ZAD_022`, 63 V1 + 46 CB units): the survival ratio is essentially 1 across the components that rise above null. Wheel and pupil do not absorb the V1–CB shared subspace. That is direct evidence against the "one-global-signal" hypothesis on this session, and toward "distinct computations that share more than global arousal."

### Mathematical form

Same as CCA, but residualize first. With confounds $Z(t) \in \mathbb{R}^{T \times 2}$ (wheel velocity, pupil), and Frisch–Waugh–Lovell semantics:

$$
S_\mathrm{V1}^{\,r}(t) \;=\; S_\mathrm{V1}(t) \;-\; Z(t)\,\hat{B}_\mathrm{V1}, \qquad
S_\mathrm{CB}^{\,r}(t) \;=\; S_\mathrm{CB}(t) \;-\; Z(t)\,\hat{B}_\mathrm{CB},
$$

where $\hat{B}_R = (Z^\top Z)^{-1} Z^\top S_R$ are fit **on the training fold only**, and the residualization is applied to the held-out test fold. Then perform CCA on the residuals:

$$
\widetilde{C}^{\,\mathrm{partial}} \;=\;
(C_{xx}^{r})^{-1/2} \, C_{xy}^{r} \, (C_{yy}^{r})^{-1/2} \;=\; U' \Sigma' V'^\top, \qquad
\rho_k^{\mathrm{pCCA}} \;=\; \sigma'_k.
$$

The headline diagnostic is the **survival ratio**:

$$
\boxed{\;
\mathrm{survival}_k
\;=\; \frac{\rho_k^{\mathrm{pCCA}}}{\rho_k^{\mathrm{CCA}}}
\;}
$$

aggregated over significant components.

### Why pCCA is the experimental contrast that decides our question

This is the step the project hinges on. The two competing hypotheses make sharp, opposite predictions about the survival ratio:

**$H_0$: V1 and CB inherit the same global low-d arousal/locomotion state.** Then the global drive is approximately a function of $Z = (\text{wheel}, \text{pupil})$. Residualizing on $Z$ removes that drive from both populations. After residualization, $S_\mathrm{V1}^{\,r}$ and $S_\mathrm{CB}^{\,r}$ should be uncorrelated except for noise. Prediction: $\rho_k^{\mathrm{pCCA}} \to 0$, **survival ratio $\to 0$**.

**$H_1$: V1 and CB are doing distinct movement-related computations that share representational structure beyond global arousal.** Then there exists shared variance that wheel + pupil cannot account for. After residualization, this variance is still expressed in the canonical pairs $(a_k, b_k)$. Prediction: $\rho_k^{\mathrm{pCCA}} \approx \rho_k^{\mathrm{CCA}}$, **survival ratio $\to 1$**.

A survival of 0.5 would be intermediate evidence. The cross-validated residualization is what makes this rigorous: fit $Z \to S_R$ only on training folds, apply to test — otherwise leakage inflates $\rho^{\mathrm{pCCA}}$ and makes $H_0$ look false even when it isn't.

### What it tells us on this dataset

Preliminary on the strong session (`CSH_ZAD_022`):

| $k$ | $\rho_k^\mathrm{CCA}$ | $\rho_k^\mathrm{pCCA}$ | survival |
|---|---|---|---|
| 1 | 0.344 | 0.338 | 0.98 |
| 2 | 0.306 | 0.303 | 0.99 |
| 3 | 0.061 | 0.056 | 0.92 |
| 4 | 0.048 | 0.048 | 1.00 |
| 5 | 0.033 | 0.033 | 1.00 |

Survival ratios are essentially 1 across the significant components. **Wheel and pupil do not absorb the V1–CB shared structure.** That is direct evidence against $H_0$ on this session, and toward $H_1$. fig04_pcca_vs_cca makes this graphical: a nearly flat survival-ratio bar means the shared subspace is not a global-arousal echo. A bar collapsed near zero would have meant the opposite.

This figure is the project's **answer figure**.

### Caveats

- **Linearity.** pCCA assumes the global state enters $S_R$ linearly through $Z$. A nonlinear arousal signal would leave residual variance even under $H_0$. We can test this in a follow-up by adding $Z^2$, $Z \cdot \dot{Z}$, or kernelized residualization.
- **What's in $Z$.** Wheel speed and pupil are reasonable proxies for locomotion + arousal on the IBL task, but not exhaustive — for instance, fine whisker movement or face movements that drive V1 gain are not in $Z$. A higher-dimensional $Z$ that includes motion energy and DLC body parts would be a stronger test.
- **Statistical power.** With only 1 strong V1+CB session in BWM 2023_12 and 2 borderline ones, the across-session variability of the survival ratio is the limiting factor, not within-session estimation noise.

---

## 5. The whole pipeline as one chain of intuitions

Before the table, here's the project in five sentences:

1. We measure spike trains in V1 and cerebellum simultaneously while the mouse runs, licks, sees stimuli, and makes choices.
2. We use a GLM per neuron to confirm that the data behave the way the literature says they should — movement variance is structured across regions, with M1 ≥ CB ≥ V1 ≥ CA1. This is the data-trust check.
3. We use SVCA per region to extract a clean, low-dimensional, reliability-certified summary of each region's population activity over time. These are coordinates we can compare across regions without fear of fitting noise.
4. We use CCA between V1's and CB's coordinates to ask whether their codes share any linear axes — *not* whether they're both informative, but whether they're *aligned*. Decoding can't answer this; CCA can.
5. We use pCCA to ask whether that alignment survives partialling out wheel velocity and pupil diameter — i.e. whether the V1–CB shared structure is just both regions reading the same global running/arousal drive (in which case it disappears) or something else (in which case it persists). The survival ratio is the answer.

## 6. Why these four, in this order

| Stage | Question answered | Question opened |
|---|---|---|
| GLM | Does each region carry movement variance, in the proportions the literature reports? | What population structure does that variance live in? |
| SVCA | What is the reliable low-d population state per region? | Do the reliable states align across regions? |
| CCA | Is there a shared subspace between V1 and CB at all? | Is the shared subspace just an arousal echo? |
| pCCA | Does the shared subspace survive partialling out wheel + pupil? | (Decides $H_0$ vs $H_1$.) |

Each stage asks a question the previous stage cannot answer, and supplies a clean input to the next. Removing any of them weakens the argument: skip the GLM and we cannot verify the dataset behaves as the literature says; skip SVCA and CCA inherits per-neuron noise; skip CCA and pCCA has nothing to subtract from; skip pCCA and we cannot decide between the two hypotheses.

The ΔR² figure (fig01) is the sanity check. The survival-ratio figure (fig04) is the answer.
