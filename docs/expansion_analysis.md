# Post-MVP Expansion — V1↔CB / V1↔M1 / CB↔M1 Pair Analysis

> **MVP record**: see `docs/mvp_conclusion.md` (frozen). This document describes the post-MVP expansion run that adds V1↔M1 and CB↔M1 as comparison anchors and triples the V1↔CB sample size by widening "V1" to all dorsal visual cortex.

## Bottom line

The MVP claim survives at 3.3× the sample size — V1↔CB shared subspace does not collapse under wheel + pupil partialling on any of the 10 V1+CB sessions in BWM 2023_12 (median survival 0.97). The new V1↔M1 (n=4) and CB↔M1 (n=1) comparison anchors land at the same survival ratios (0.98 and 0.98). Read at the medians, this is **Branch (A): generalized cortical preservation** — the pattern the MVP read as something specific about V1↔CB is generic to cortical region pairs in this dataset.

But the reading needs an immediate qualification, because the **input quality is not uniform across pairs**. Cerebellum on the V1+CB pair sessions has high SVCA reliability (median ρ₁ = 0.76, max 0.86 — the most reliable region anywhere in the run), while V1 and M1 on the V1+M1 pair sessions both sit near ρ₁ ≈ 0.20. The raw canonical correlations follow this gradient: V1↔CB has median ΣρCCA ≈ 0.80, V1↔M1 has ≈ 0.44, CB↔M1 has 0.62. So when we say "all three pairs preserve at survival ≈ 0.97," the V1↔M1 part of that claim is preservation of substantially smaller canonical correlations on noisier population coordinates. Branch (A) is correct as a directional summary, but it isn't quite "the same finding three times."

| Item | MVP | Expansion |
|---|---|---|
| Total neurons fit (GLM) | 4,286 | **4,886** |
| Unique sessions analyzed | 41 | **51** |
| V1↔CB pair sessions | 3 (1 strong, 1 mixed, 1 unusable) | **10** |
| V1↔M1 pair sessions | 0 (no such sessions exist with strict VISp + MOp/s) | **4** |
| CB↔M1 pair sessions | 1 | **1** |
| 3-region V1+CB+M1 sessions | 0 | 0 (impossible on this freeze) |
| Bin size | 20 ms | 20 ms |

The argument unfolds as a single sweep across all three pairs simultaneously: a per-region GLM check (still works at the new scale), a cross-pair canonical-correlation comparison (where the raw signal magnitudes diverge), a cross-pair reliability comparison (where the input quality diverges), and a cross-pair survival comparison (where the headline result converges).

---

## The dataset still behaves the way the literature says

Before trusting any cross-region comparison at the larger sample, the per-neuron Ridge GLM has to keep reproducing the brain-wide-encoding ranking established by Wang–Druckmann 2026 / IBL 2025: movement variance should rank from the motor periphery inward (M1 ≥ CB > V1 > CA1).

<img src="figures/expansion/glm_dr2.png" width="100%" alt="Per-region GLM ΔR²">

It does. Across 4,886 neurons (vs 4,286 in the MVP), the medians and upper tails of ΔR²_movement still rank CB ≈ MO > VISp > CA1, with CB and MO essentially tied and showing long upper tails reaching ΔR² ≈ 0.05–0.12. V1's movement signal stays small, mostly contained below 0.025; CA1 sits near zero. The Stimulus and Choice panels behave as expected. Adding sessions and widening regions did not change the ranking, so the pipeline is operating consistently at the new scale and everything below inherits trust from this passing.

---

## Canonical correlations diverge between pairs — but the partialling barely moves them

The cleanest cross-pair comparison is the canonical-correlation curve itself: ρ_k as a function of component index k, with the phase-shuffle 99% null as a floor. Plotting all three pairs on the same axes (median across sessions, IQR shaded) gives the most honest visual of how the raw shared structure differs *before* partialling, and what changes once wheel + pupil is regressed out.

<img src="figures/expansion/correlations.png" width="100%" alt="Cross-pair canonical correlations: raw vs partialled">

Two patterns are immediately visible. First, the raw ρ_k^CCA panel (left) shows V1↔CB on top across components — its leading canonical correlation peaks near 0.36, with a long tail above the null through k = 4. CB↔M1 sits in the middle (ρ_1 ≈ 0.30, also tailing through k = 4). V1↔M1 is the lowest-magnitude pair (ρ_1 ≈ 0.22), with components 3+ already inside the phase-shuffle null. So even without partialling, the pairs are not extracting equally large shared variance — V1 couples more strongly with CB than with M1, and CB couples about equally with both partners.

Second — and this is the headline — the partialled ρ_k^pCCA panel (right) is nearly indistinguishable from the raw panel. The lines drop a bit (V1↔CB peak from 0.36 to 0.32; V1↔M1 from 0.22 to 0.20; CB↔M1 from 0.30 to 0.28), but their shape, ordering, and component-by-component magnitude all survive intact. **The structure that wheel + pupil could plausibly explain is a small fraction of the shared variance on every pair.** The IQR ribbons widen a little under partialling, but no pair's median curve gets pushed into the null band that wasn't already inside it. If the null hypothesis were correct (V1, CB, and M1 share movement structure only via a globally coupled wheel/pupil drive), the right panel would look like the gray null band itself; instead it looks like the left panel.

A subtler observation: the raw-magnitude *gradient* across pairs (V1↔CB > CB↔M1 > V1↔M1) does not align with anatomical-connection strength. Cerebellum and motor cortex are directly coupled through the cerebellothalamocortical loop; V1 and cerebellum are coupled only indirectly through pontine and brainstem nuclei. Yet V1↔CB has the largest raw canonical correlations. This is at least partly a reliability story rather than a connectivity story — see the next section.

---

## Reliability is the asymmetry the MVP couldn't see

The MVP saw a single CB recording and called it "high reliability" without much else to compare against. With 15 pair-sessions of SVCA scores in hand, the asymmetry is structural, not anecdotal.

<img src="figures/expansion/reliability.png" width="100%" alt="Cross-pair SVCA reliability">

On the V1↔CB sessions (n = 10), the CB box (red, second from left) sits visibly above every other region-context in the figure: median ρ₁ = 0.76, IQR 0.46–0.81, with several sessions clearing 0.8 and three clearing Stringer's 0.5 reliability threshold for at least one component. The V1 box on the same sessions (blue, leftmost) has median ρ₁ = 0.24 and only one outlier session above 0.5. So *within the V1↔CB pair*, the CCA is being computed between a very noisy V1 score and a relatively clean CB score — an asymmetric pairing that produces meaningful canonical correlations because the CB side carries the structure.

On the V1↔M1 sessions (n = 4), both regions sit in the [0.07, 0.45] reliability range with medians ≈ 0.20–0.25 — uniformly noisy on both sides. This is the worst input quality of any pair in the run. On the CB↔M1 single session, CB drops to ρ₁ = 0.49 (just below threshold; lower than CB on the V1↔CB sessions, because this is a different mouse / different probe placement) and M1 sits at ρ₁ = 0.28.

Two consequences for the cross-pair comparison fall out of this. First, the raw-correlation gradient from the previous figure (V1↔CB > CB↔M1 > V1↔M1) tracks *cleanest-region reliability* almost perfectly: the pair with the most reliable region (CB on V1+CB sessions) has the highest ρ_CCA, the pair with the least reliable regions (V1↔M1) has the lowest. This is consistent with the canonical correlations being limited by the noise floor of the worse-resolved view, with the better-resolved view providing the structure that gets aligned. Second, when we read survival ratios as "98% of shared variance survives partialling," the V1↔M1 version of that claim is about a smaller absolute amount of variance computed from noisier coordinates — directionally consistent with the V1↔CB result, but evidentially weaker.

---

## Survival converges across pairs — even where the inputs diverge

The headline figure of the expansion puts the raw vs partialled magnitudes (left panel) and the survival ratios (right panel) side by side, all three pairs in the same axes. The left panel restates the canonical-correlation finding above in summed-Σρ_k form; the right panel is the actual answer.

<img src="figures/expansion/survival.png" width="100%" alt="Cross-pair survival of shared subspace">

Left panel (raw vs partialled ΣρCCA): the V1↔CB pair (left group) shows the largest absolute shared variance — ΣρCCA medians around 0.80 — and the partialled boxes (terracotta) sit just below the raw boxes (gray). The MVP-anchor session, ringed in gold, sits inside the V1↔CB distribution rather than at an extreme. The V1↔M1 pair (middle) has roughly half the raw shared variance of V1↔CB (ΣρCCA ≈ 0.44), but the same gray-vs-terracotta gap pattern: partialling barely moves the boxes. The single CB↔M1 session (right) sits between, with ΣρCCA ≈ 0.62 and again nearly zero gap.

Right panel (survival ratios): all three pairs cluster at median ≈ 0.97–0.98. V1↔CB has the widest visible spread (range 0.85–1.00 across n = 10), V1↔M1 sits tighter at [0.93, 1.00] across n = 4, and CB↔M1 is a single point at 0.98. No pair drops near zero anywhere in the figure — Branch (A) is unambiguous on this dataset under wheel + pupil as the operationalization of "global state."

The cross-pair table makes the convergence and the underlying-input divergence sit next to each other:

| Pair | n | Median ΣρCCA | Median ρ₁ region A | Median ρ₁ region B | Median survival |
|---|---|---|---|---|---|
| V1 ↔ CB | 10 | 0.80 | V1: 0.24 | **CB: 0.76** | 0.97 |
| V1 ↔ M1 | 4 | 0.44 | V1: 0.25 | M1: 0.20 | 0.98 |
| CB ↔ M1 | 1 | 0.62 | CB: 0.49 | M1: 0.28 | 0.98 |

V1↔CB has the highest raw shared variance and the most asymmetric reliability (a clean CB side feeding a noisy V1 side); V1↔M1 has roughly half that raw signal on uniformly noisy inputs; CB↔M1 sits in between with one strong session of moderate reliability. The survival ratios converge to ≈ 0.97 not because the underlying signals are equally strong, but because **the partial of each pair's CCA against wheel + pupil removes the same small fraction of shared variance everywhere** — small in absolute terms on weak-input pairs, small in proportional terms on strong-input pairs. The MVP-anchor session (`41431f53`) returns survival = 0.98 in this rerun, exactly the MVP-reported value, confirming the new code reproduces the original computation.

---

## What we now believe — and what changed from the MVP

The chain of evidence shifts the interpretation. The MVP read its V1↔CB survival as evidence for something specific about how V1 and CB are coupled — distinct movement-related computations sharing structure beyond global arousal. The expansion shows that the survival-under-partialling pattern is not unique to V1↔CB: V1↔M1 and CB↔M1 show the same headline numbers. The MVP claim is therefore generic to cortical region pairs in BWM, not specific to V1's relationship with cerebellum.

The cleaner reading from the expansion is two-part. First, on the V1↔CB pair specifically (which has the most data, the highest raw canonical correlations, and the cleanest CB-side population coordinates), there is real shared linear structure that does not collapse under wheel + pupil partialling — the MVP got this right at the larger sample. Second, the same survival metric on V1↔M1 and CB↔M1 reproduces the directional pattern (preservation rather than collapse) but with weaker raw inputs — these are confirming evidence for Branch (A) as a *general* property, not as additional evidence for the V1↔CB-specific story.

This shifts the natural follow-up question. The MVP's conclusion section asked things like "is the V1↔CB shared signal an inherited copy of arousal" — a question the MVP correctly answered no and the expansion confirms. The new question is structural: what is the shared cortical state that all these pairs are participating in, why does wheel + pupil fail to remove it, and is it the same shared state across all pairs or different shared states that all happen to survive Z?

## What does survival ≈ 0.97 actually mean, biologically?

A survival ratio of 0.97 says that 97% of the linear shared variance between two regions on held-out time is variance that wheel velocity and pupil diameter cannot predict. Concretely, the pipeline extracts each region's leading population dimensions (the K=8 SVCA-projected scores), finds via CCA the directions in V1's K-d state and CB's K-d state that maximally co-fluctuate, and then asks how much of that co-fluctuation can be predicted by wheel + pupil acting as a common drive on each region. Survival = 0.97 means the answer to that second question is "almost none." Wheel and pupil do have predictive power in their own right — running and arousal genuinely modulate firing across all our regions, as fig01 confirms — but the V1↔CB shared *axes* of fluctuation are not the axes that wheel+pupil drive.

That has a sharp neuroscientific implication: whatever common signal V1 and CB share, it is not the part of their activity that gets driven by ongoing locomotion or arousal in a globally coupled way. **The shared signal lives orthogonally to the wheel/pupil-driven gain modulation.** This is a stronger statement than the MVP made, because we now know the same is true of V1↔M1 and CB↔M1 — the orthogonality is structural across cortical region pairs, not specific to one circuit. A useful intuition: if wheel+pupil were a 1- or 2-D state that all regions inherited identically, it would inflate ρ_CCA but be removed by pCCA, and survival would tend to 0. We see survival ≈ 1, so the global drive *exists in each region* (regression onto Z removes some variance) but it is not the dominant axis of cross-region coupling. The cross-region coupling lives in different dimensions.

## What this would have looked like under H₀

To make Branch (A) interpretable, it helps to picture the figures we *would* have produced if the alternative hypothesis were true. Under H₀ — V1, CB, and M1 share movement structure only via a common global arousal/locomotion drive — ρ_CCA would still be high on the leading components (because both regions encode the global drive, which co-fluctuates by definition), but ρ_pCCA would drop to near zero on those same components (residualizing out wheel+pupil would remove the entire shared latent). Survival would land near 0 on the strong sessions and stay near 0 across all pairs. The right panel of `correlations.png` would look like the gray null band, and the cross-pair survival boxplot would collapse against the bottom of the y-axis. What we observe is the opposite: every pair clusters at ~0.97, the partialled and raw curves are nearly indistinguishable. The H₀ counterfactual is decisively rejected on this dataset for every pair we tested.

A more nuanced counterfactual is **H₀′: the shared subspace is a motor-command echo from M1.** Under H₀′, V1↔CB CCA correlations would be inflated by both regions inheriting M1's drive, and `CCA(V1, CB | M1)` would collapse those correlations. We could not test H₀′ within-session (zero V1+CB+M1 sessions exist in BWM 2023_12) — it remains an open alternative that this run cannot adjudicate.

## Alternative interpretations of Branch (A)

Branch (A) — "all cortical pairs preserve shared structure under wheel+pupil partialling" — is the empirical pattern. There are at least four mechanistic stories that could produce it, and the analysis as presented does not distinguish them:

1. **Pervasive cortico-cortical and cortico-cerebellar loops carrying non-arousal signals.** The brain's interregional connectivity is dense; many regions exchange task-related, prediction-error, and motor-context signals constantly. Under this view, the shared subspace we see is real coupling — actual communication beyond arousal — and Branch (A) is the prediction we'd make a priori for any cortical pair.
2. **A higher-dimensional global state that wheel+pupil only weakly proxies.** "Global arousal" is bigger than wheel and pupil. Whisking, body posture, breathing, brain-state oscillations (beta/gamma envelope), pupillary dilation derivative — any of these could be driving all regions in lockstep but only weakly captured by the 2-D Z. In that case, the surviving 0.97 is "shared global state we didn't measure," not "shared computation."
3. **A common upstream sensory or motor drive.** Visual and motor input enter the brain at distinct loci but propagate through dense networks; thalamocortical drive is shared across many cortical areas. The shared subspace could be the time course of that thalamic input, projected into each region.
4. **Methodological artefact: the SVCA scores are still partly arousal-correlated.** SVCA selects the K=8 most reliable population components; if those leading components are themselves dominated by arousal axes (which Stringer 2019 shows for mouse cortex), then the CCA we compute is largely between two arousal-projected views, and partialling Z removes only the arousal portion that *Z captures*. The residual might be the part of arousal Z misses, not a separate computation.

These are not mutually exclusive; the actual answer is probably (1) + (2) + maybe (3) for any given pair. Branch (A) on this dataset is consistent with all four stories and incompatible with the simplest H₀ ("Z fully captures the shared drive"). Discriminating between (1)–(4) requires the next-step analyses listed at the end of this document.

## Comparison to the published literature

Branch (A) sits comfortably within the brain-wide-coupling literature rather than contradicting it. **Stringer et al. 2019 (*Science*)** showed that spontaneous activity in mouse V1 reliably encodes a high-dimensional behavioral latent, with the leading PCs arousal-aligned but most of the population variance unexplained by arousal. Our result extends that within-V1 finding to cross-region: the non-arousal-driven structure is shared across cortical regions, not just present in each. **Musall et al. 2019 (*Nat Neurosci*)** found cortex-wide widefield activity dominated by uninstructed movements, with residual neuronal differences across regions becoming visible only after movement was accounted for. Wheel and pupil are coarse proxies for those uninstructed movements; the surviving cross-region shared structure may be exactly the residual Musall pointed to. **Wang, Kurgyis & Druckmann 2026 (*Nat Neurosci*)** report brain-wide movement signals as structured rather than uniform, with stronger encoding closer to the motor periphery — exactly the per-region ranking our fig01 reproduces. Where Wang's claim is per-region, ours adds a cross-region claim: even at the population-axis level, the shared movement-related structure is preserved across pairs after global state is removed. **Niell & Stryker 2010 (*Neuron*)** established that locomotion gain-modulates V1 visually evoked responses without changing tuning, framing V1's "movement signal" primarily as multiplicative gain. Our result says that gain has structure that aligns with cerebellum's and motor cortex's population states in directions wheel+pupil don't capture, suggesting the gain modulation is itself part of a coordinated cortical state rather than an isolated V1 phenomenon. Finally, **Semedo et al. 2019 (*Neuron*)** introduced the "communication subspace" — cross-region neural coupling lives in a low-dimensional subspace distinct from each region's dominant within-region modes. Branch (A) is qualitatively consistent with that: the cross-region shared subspace we identify is robust across pairs and distinct from the wheel/pupil-driven global state.

The integrated picture: Branch (A) is essentially what the brain-wide-coupling literature would predict. Diffuse cortical coupling exists, persists after subtracting the obvious global state proxies, and shows up between pairs that aren't even directly anatomically connected (V1↔CB has only indirect routes via thalamus and brainstem). The MVP V1↔CB result was a special case of the same broad phenomenon Stringer / Musall / Semedo had already documented at different angles.

## Reframing the project's question

The original course-project question — *are V1 and CB doing the same kind of computation, or different things that look similar at the decoding level?* — was binary: same vs different. The MVP gave a directional answer (different, because pCCA didn't collapse). The expansion now reveals that the question itself was too narrow. The cleaner reframing is: *is there a shared low-dimensional cross-region structure in the BWM dataset that survives partialling out wheel + pupil?* Yes, on every pair we tested. The follow-up: *what is in that shared structure, and what would partial it out?* That's a forward-looking research question, answerable with richer Z, with predicts-vs-follows timing, with experimental perturbations. The much harder question — now visible because we asked the simpler one — is whether different cortical pairs are sharing the *same* underlying low-d structure, or whether V1↔CB and V1↔M1 share *different* things that happen to both survive Z. The current pipeline cannot tell, because the canonical pairs are extracted independently per pair-session; a pooled / aligned analysis (Pang & Sahani's dSCA, or principal-angles between per-pair canonical subspaces) could.

So the MVP answered the original question (favoring the "different computations" side, weakly), and the expansion clarifies that the answered question was the wrong one — V1↔CB isn't special, the property is more general. **The expansion's real contribution is showing that the next interesting question is about the structure of cortical coupling itself, not about V1 specifically.**

---

## What the next experiment would be

The most productive follow-up extensions are, in priority order:

1. **Richer Z** (~1–2 hr). Add motion energy (whisker + body), DLC paw + snout, and lick rate to Z and re-fit pCCA on all 15 pair sessions. If survival drops materially, the wheel + pupil operationalization of global state was too narrow and Branch (A) is conditional on Z choice. If it stays high under richer Z, Branch (A) is robust.

2. **Bootstrap CIs on per-pair survival ratios** (~1 hr). Block bootstrap with 50-bin blocks, 1000 samples per session. Adds error bars to the cross-pair distribution figure and lets us answer "is the V1↔CB IQR genuinely wider than V1↔M1, or is it the n = 10 vs n = 4 sample-size effect?"

3. **Cross-mouse subspace alignment via CKA / principal angles** (~3–4 hr). Use all 35 V1 + 86 CB + 58 M1 + 94 CA1 single-region sessions (not the 15 within-session pair recordings) to compute a 4×4 representational similarity matrix. This is a different scientific claim — shape similarity across mice, not within-session coupling — and is the natural next analysis since within-session 3-region pCCA is impossible on this freeze.

4. **Predicts-vs-follows timing analysis** (Wang 2026 trick) on the 10 V1↔CB and 4 V1↔M1 pair sessions. Cross-correlation lags between region SVCA scores and behavior tell us which region leads; gives mechanism, not just structure.
