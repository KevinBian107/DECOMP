# DECOMP

V1-vs-cerebellum movement signal decomposition on the IBL Brain-Wide Map (2025 release).

A course project investigating whether the brain-wide movement signature in mouse neural
activity is **one global low-dimensional state inherited everywhere**, or **distinct
movement-related computations that just look similar at the decoding level**.

## Question

Stringer et al. (2019) and the IBL Brain-Wide Map (2025) both report that movement variables
are decodable from neurons essentially everywhere in the mouse brain. A V1 neuron firing
during running and a cerebellar neuron firing during running are both "movement-correlated"
— but they almost certainly reflect different computations. V1's signal is some mix of
efference copy, arousal-driven gain, and retinal slip; cerebellum's is forward-model prediction
error and proprioceptive integration; M1's is the command itself; hippocampus's is
locomotion-modulated theta gating.

Decoding accuracy alone cannot answer this. Only **shared-subspace structure** can.

## Approach (MVP)

Per-session analysis of 3–5 IBL BWM sessions with simultaneous coverage of V1 (VISp),
cerebellum (CB cortex + nuclei, Beryl-atlas union), motor cortex (MOp/MOs), and CA1:

1. **Per-region encoding GLM** with movement / stimulus / choice kernels (Ridge on 20 ms binned
   counts; `neurencoding.linear.LinearGLM`). ΔR² per kernel via leave-one-regressor-out
   cross-validation. Reproduces the IBL BWM 2025 / Wang–Druckmann 2026 finding that
   movement variance is structured across the brain, with stronger encoding closer to the motor
   periphery.
2. **Within-region SVCA** (Stringer et al. 2019; `MouseLand/neuropop`) extracts the reliable
   low-dimensional movement-related subspace per region.
3. **Pairwise CCA** between SVCA score time series for the 6 region pairs, with phase-shuffle
   nulls.
4. **Pairwise partial CCA**, partialling out wheel speed + pupil diameter — the answer step.
   If shared canonical correlations collapse after partialling out global arousal, V1 and CB
   were inheriting from a common low-dimensional state. If shared structure survives,
   movement-related computations across regions are not just an inherited copy.

## Install

```bash
conda env create -f environment.yaml
conda activate decomp
```

This installs the IBL stack (`ONE-api`, `ibllib`, `iblatlas`, `brainbox`), the BWM analysis
repo (`paper-brain-wide-map`, where `bwm_query` lives), the GLM toolkit (`neurencoding`),
the SVCA / RRR toolkit (`MouseLand/neuropop`), and `pyrcca` for regularized CCA.

ONE public-mode access uses the well-known token:
```python
from one.api import ONE
one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international')
```
The cache lives at `~/Downloads/ONE/` by default; override with `cache_dir=` if disk is tight.

## Run the MVP

```bash
python run_all.py                       # full pipeline → outputs/
python run_all.py --rerun-from svca     # skip earlier stages from cache
python run_all.py --eids EID1 EID2      # restrict to specific sessions
```

Outputs:
- `outputs/fig01_glm_dr2_per_region.{png,pdf}` — per-region ΔR² for movement / stimulus / choice
- `outputs/fig02_svca_reliability.{png,pdf}` — reliable-component spectrum per region
- `outputs/fig03_cca_canonical_correlations.{png,pdf}` — CCA canonical correlations vs null
- `outputs/fig04_pcca_vs_cca.{png,pdf}` — **answer figure**: shared variance survival after
  partialling out wheel + pupil
- `outputs/summary.json` — all per-pair metrics + provenance

For interactive exploration, see `notebooks/exploration.ipynb`.

## Layout

```
src/decomp/
├── data/        # IBL data access, session selection, spike/covariate binning
├── glm/         # Per-neuron Ridge GLM with raised-cosine kernels + ΔR²
├── svca/        # Within-region SVCA via neuropop
├── cca/         # Pairwise CCA + partial CCA + nulls
├── viz/         # Matplotlib figure factory
└── pipeline/    # Stage orchestrators
```

The `decomp` package is pip-installable in editable mode (handled by `environment.yaml`).

## References

Primary literature for the question and the methods are tracked in
`scratch/2026-05-04-v1-cb-movement-decomposition/README.md`. Key papers:

- IBL et al. *Nature* 2025 — A brain-wide map of neural activity during complex behaviour.
- Stringer et al. *Science* 2019 — Spontaneous behaviors drive multidimensional brainwide activity.
- Wang, Kurgyis et al. *Nat Neurosci* 2026 — Brain-wide analysis reveals movement encoding
  structured across and within brain areas.
- Musall et al. *Nat Neurosci* 2019 — Single-trial neural dynamics dominated by movements.
- Semedo et al. *Neuron* 2019 — Cortical areas interact through a communication subspace.
- Pang & Sahani *NeurIPS* 2020 — Demixed shared component analysis.

## License

MIT — academic course project. IBL data is governed by the IBL data-sharing agreement.
