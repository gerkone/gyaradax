# quasilinear package review (2026-06-11)

Three-way audit: this package against its QuaLiKiz/TGLF basis (primary
sources), against TORAX's established QLKNN/TGLFNN transport-model
machinery, and against the legacy calibration dataset
(`data/aggregated_scan.npz`, 1367 rows). Actionable fixes applied same day
are marked [FIXED]; the rest are ranked work items.

## 1. The rule as implemented (saturation.py)

```
kperp2_eff(ky) = <k_perp^2>_{|phi|^2}              (eigenmode-weighted, s+kx)
sat_amp(ky)    = sigmoid(20*gamma) * relu(gamma) / max(kperp2_eff, floor)
w(kx,ky)       = flux_kxy / max(phi2_kxy, eps)     (normalization-invariant)
Q_QL           = cn(features) * sum_kx sum_{ky != 0} w * sat_amp
```

Strengths (keep): the flux/|phi|^2 weight is renormalization- and
amplitude-invariant (kills the exp(2*gamma*T) artifacts that broke the
torch prototype); <k_perp^2> uses the actual eigenmode rather than
QuaLiKiz's heuristic k_r,NL; fully differentiable JAX path; the
gyroBohm conversion layer in the fork (gyaradax_normalization.py) is
verified against GKW/QLKNN conventions.

## 2. Findings vs the literature (QuaLiKiz / TGLF / QL-validity papers)

Established QL anatomy is weight x intensity per (ky, mode, channel,
field), with the weight taken from the linear eigenmode (robust: ~5%
RMS on Q_i when fed the measured NL spectrum) and ALL the physics risk
in the intensity model. Against that:

- The amplitude gamma^1/<k_perp^2> is used as an INTENSITY, which is
  dimensionally a diffusivity; the standard mixing-length intensity is
  (gamma/<k_perp^2>)^2. cn is therefore dimensionful and absorbs the
  mismatch — no guaranteed rho* scaling outside the calibration box.
  The gamma-exponent was chosen by Spearman on 50 sims (MODEL.md).
- Applied per-ky with no spectral shape S(ky), no Delta-ky measure, and
  the kx sum is a sum of per-kx RATIOS (scales ~nkx for ballooning
  modes). QuaLiKiz: single peak amplitude at argmax(gamma/<k_perp^2>)
  (~ kthrho 0.2) + k^3 / k^-3 envelope. Consequence measured in the
  plugin-grid probe: X scales with the box; only the pinned plugin grid
  + a flat ratio (0.24 +- 0.02 to the reference grid, ES and EM) makes
  the calibration valid.
- No ExB shear model (quench or spectral shift). TGLF SAT0's Waltz
  quench is gamma_net = max(gamma - alpha_E*gamma_E, 0); the solver now
  has rotation terms (vcor/uprim, 2026-06-11), so a quench on the real
  gamma is cheap.
- Dominant IVP mode only: misses subdominant branches (MTM under
  ITG/KBM), causes branch-jump discontinuities; TGLF keeps >= 2 modes
  per ky. The ncv_eigensolve knob exists but is not wired in.
- EM channel: flutter flux is summed into the ES flux with the SAME
  electrostatic amplitude. Literature warning (Pueschel PoP 2008): the
  linear flutter/ES split scales ~beta where the nonlinear one scales
  ~beta^2, and the Rechester-Rosenbluth prefactor is regime-dependent
  (0.625 ITG / 0.37 ITG-TEM / 0.46 KBM). Above the KBM threshold and
  near the non-zonal transition the QL rule (not the weights) breaks.
  The beta ladder of the hi-fi campaign is the direct test.
- Validation checklist worth adopting from the literature (Casati 2009,
  Staebler 2021): Kubo number < ~0.2; cross-phase comparison vs NL;
  weight-only test against the measured NL spectrum; QL/NL overage
  CONSTANT (~1.4-1.6) across an R/L_T scan before fitting cn; spectrum
  peak at kthrho ~ 0.2; low-shear stress test; EM beta-ordering check;
  stationarity guard on the NL reference data (KBM regime runs that
  never saturate must be excluded — observed first-hand in the beta=1%
  ladder runs).

## 3. Findings vs TORAX's QLKNN/TGLFNN machinery

Good-citizen checklist (full version in the audit): the plugin HAS the
quasilinear base-class integration, single-point GB conversion, input
clipping, NaN guards, jit-clean caches/static fields. It LACKED:

- [FIXED] DV_effective / An_min hardcoded -> now config-exposed.
- [FIXED] no tiny-flux->exact-zero clamp (QLKNN does this to keep the
  solver from waking zero-flux modes) -> q_i < 1e-8 snaps to 0.
- [FIXED] q_i could go negative (cn head sign, below) -> clipped to
  [0, 1e3].
- [FIXED] fast_ion_stabilization config flag silently ignored -> raises.
- Placeholder channels: qe := qi (wrong whenever L_Te != L_Ti and for
  TEM regimes; TORAX divides by R/L_Te so chi_e is fictitious), pfe := 0
  (D_e/V_e dead). Needs the kinetic-electron tier.
- No smag-alpha trio (smag - alpha/2 correction, q<1 sawtooth proxy,
  avoid_big_negative_s) — cheap, QLKNN-essential per the references.
- No rotation/ExB quench inputs (see above).
- No collisionality, Ti/Te, Zeff features (collisionless single-ion
  solve; out of MVP scope but should be declared as validity domain).

## 4. Findings vs the calibration dataset (data/aggregated_scan.npz)

1367 rows: 10 GKW batches + 301 gkw_raw + 99 gyaradax LHS; ES only,
GKW part has eps pinned at 0.19; Y from short-tail NL averages (the
+-20% tail-noise diagnosed in the resolution sweep applies).

- The bundled v1 head (linear-space affine in shat,q,rlt,rln; advertised
  r2_test 0.82 on its curated 835-row subset) goes NEGATIVE on 41% of
  the plugin's input clip box (e.g. rlt >~ 22 at typical backgrounds):
  negative cn -> negative chi into TORAX. [FIXED at the consumer: the
  fork now prefers the bundled strictly-positive polynomial_log head and
  clips q_i >= 0.]
- Honest re-evaluation on the FULL dataset: no head generalizes.
  r2_test on a fresh split: scalar 0.52, polynomial -0.23,
  polynomial_log -16.9; in-sample on the full positive set every head
  including the scalar explains <~ 6% of variance. The legacy dataset
  is too heterogeneous/noisy to support a feature head. Refit candidate
  saved as data/cn_iter_hybrid_v2_full1367.pkl (scalar 0.2027) — NOT
  promoted.
- Conclusion: the planned hi-fi campaign (uniform protocol, long tails,
  threshold-relative sampling, eps+beta coverage) is not an
  enhancement, it is a prerequisite for any feature-dependent head.

## 5. Fixes applied (2026-06-11)

| where | change |
|---|---|
| saturation.py | sat_amp uses relu(gamma): stable modes contribute exactly 0 (was: small negative leak through sigmoid*gamma) |
| saturation.py | zonal mask by krho==0, not ky index 0 |
| integrals.py | calculate_em_fluxes 5D branch raises on bpar (was: silent drop of the compressional flux) |
| fork gyaradax_ql_transport_model.py | head preference -> polynomial_log (positivity); q_i clipped to [0, 1e3]; tiny-flux (<1e-8) snapped to 0; DV_effective/An_min config-exposed; fast_ion_stabilization raises |

NOTE: the relu gate changes the X definition for near-marginal modes
(rule version bump); cn must be refit on data generated with the same
rule — scheduled with the hi-fi campaign, which regenerates X anyway.

## 6. Ranked work plan

1. (with the hi-fi campaign) Refit cn on protocol-uniform data; promote
   a positive-definite head; record the rule version in the pickle.
2. Spectral-measure fix: per-ky intensity-weighted ratio
   (sum_kx flux)/(sum_kx phi2) + explicit Delta-ky weights -> X becomes
   grid-robust; requires refit (free during the campaign).
3. Kinetic-electron tier: real q_e and pfe (harvest path already
   produces them); unlocks DV_effective/An_min and TEM regimes.
4. ExB quench using the implemented rotation terms (Waltz rule on the
   computed gamma); add gamma_E plumbing from TORAX.
5. smag-alpha trio in the plugin input prep (parity with QLKNN).
6. EM saturation: validate the equal-amplitude flutter assumption
   against the beta ladder (Pueschel beta vs beta^2 ordering test);
   regime-tagged flutter prefactor if it fails.
7. Subdominant modes: wire ncv_eigensolve, >= 2 modes per ky.
8. Intensity-model upgrade study: gamma^2/<k_perp^2>^2 vs gamma^1 vs
   QLK max-rule + S(ky), evaluated on the campaign dataset.
9. Hygiene: deduplicate ql_flux_diagnostics; remove dead ds parameter;
   _get_cn_head lru staleness; harvest() bare except.

## 7. Measured baseline and fix impact (2026-06-11, addendum)

Dataset-level evaluation (data/aggregated_scan.npz; held-out split 1/5):

| configuration | rows | R2_test | logRMS | neg preds |
|---|---|---|---|---|
| old rule, v1 scalar (basic) | 930 | 0.085 | 2.51 | 0% |
| old rule, v1 polynomial (linear) | 930 | 0.050 | 2.54 | 0% |
| old rule, refit polynomial | 930 | 0.080 | 2.74 | **8.6%** |
| new rule (relu), refit scalar | 1101 | 0.818 | — | 0% |
| new rule (relu), refit polynomial | 1101 | **0.873** | — | 0% |

Decomposition (the number that matters): on the 859 rows positive under
BOTH rules, old and new are IDENTICAL (scalar R2 0.732 vs 0.733,
Spearman 0.936 both) — the relu fix does not touch healthy rows. The
old headline R2 ~ 0.09 was a contaminated tail: ~71 rows whose X was
corrupted by negative stable-mode contributions (excluded by the fix)
plus 242 rows whose X had been driven NEGATIVE by the same mechanism
(rescued by the fix; sampled GKW states showed the stable-mode
contamination reaching 100-200% of X at p90/max). Conclusion: the QL X
carries real predictive signal (rank correlation 0.94 with the NL
flux); the legacy fits were poisoned by the rule bug, not by the
physics. New-rule X for the legacy set saved to
data/aggregated_scan_xnew.npz; any interim refit must use it.

Caveats that stand: log-space errors remain large (small-flux regime
poorly captured — a dataset-protocol limit, addressed by the campaign's
long-tail Y protocol), and the application-level three-way
(basic/linear/log heads through full TORAX iterhybrid runs vs QLKNN,
incl. wall-clock under the converged X protocol) is running — head
promotion decisions wait for it.

## 8. Saturation-variant study (2026-06-11, addendum)

TGLF/QuaLiKiz-style sophistications recomputed per-row on the 1268
legacy GKW _Lin states (data/sat_variants.npz; eval
data/sat_variants_eval.json), heads refit per variant, common held-out
split:

| variant | spearman | R2_scalar | R2_poly | logRMS |
|---|---|---|---|---|
| current rule (gamma^1, relu) | 0.959 | 0.818 | 0.873 | 8.8 |
| kx-intensity-weighted | 0.961 | 0.798 | 0.834 | 8.4 |
| Delta-ky measure | 0.926 | 0.554 | 0.880 | 9.0 |
| gamma^2 intensity | 0.956 | 0.214 | 0.768 | 6.4 |
| QLK k^3/k^-3 envelope | 0.721 | 0.136 | 0.284 | 9.3 |
| QLK peak rule | 0.841 | 0.161 | 0.284 | 8.5 |

Verdict: on this (single-grid, eps-pinned, ES) slice the canonical QLK
spectral shapes lose badly; the empirically-chosen per-ky gamma^1 rule
stands. The kx-intensity-weighted measure TIES the current rule while
removing the ~nkx grid-scaling pathology — adopted as the DEFAULT
(saturation.py kx_ratio_sum=False; legacy behavior switchable). gamma^2
gives the best log-space (relative) accuracy at the cost of absolute
big-flux skill — revisit on campaign data. Caveats: the dataset cannot
reward grid-/regime-robustness (where the literature rules earn their
keep), and its Y noise penalizes all variants equally.

RULE VERSION: with relu (sec. 5) + intensity-weighted kx, the X
definition is now v2. The hi-fi campaign and all future fits use v2;
the legacy npz carries v1 columns (and v2-recomputable via
data/sat_variants.npz 'X_kxweighted').

## 9. Component study beyond saturation (2026-06-11, addendum)

Same protocol as sec. 8 (1268 legacy GKW pairs, common held-out split;
data/component_study.npz + _eval.json). One component varied at a time
against the rule-v2 baseline (spearman / R2_scalar / R2_poly / logRMS):

| component | variant | result | verdict |
|---|---|---|---|
| <k_perp^2> | eigenmode-weighted (current) | .961/.798/.834/8.4 | KEEP — validated |
| | bare ktheta^2 | .915/.307/.830/8.2 | reject (head must re-learn ballooning width) |
| | QLK analytic (1+shat^2<theta^2>) | .911/.091/.764/7.4 | reject — eigenmode average measurably beats the analytic form |
| gate | hard step / soft sigmoid-5 | ~tie | keep sigmoid-20 |
| | threshold gamma>0.05 | .961/.795/.866/7.4 | small in-sample win, but would zero near-threshold points the campaign's 1.1x rung needs — not adopted; revisit post-campaign |
| resonance | Lorentzian gamma^2/(gamma^2+omega^2) | .909/.325/.722/7.3 | reject — converged IVP frequencies carry no usable broadening signal in this form |
| Y definition | mean last 240 (current) | .961/.798/.834 | |
| | geometric mean last 240 | .962/.819/.877 | ADOPT for legacy refits (burst-robust target); campaign attacks the same noise via long tails |
| | mean last 480 / median 240 | mixed | — |
| head form | scalar / poly1 | .798 / .834 | |
| | ridge_poly2 | R2 .879, 0% neg | campaign-refit candidate |
| | gp_log | R2 .875, 0% neg | campaign-refit candidate |
| | gbm | R2 .800, logRMS 7.27 (best relative) | candidate where relative accuracy matters |

Net: the current v2 rule survives every challenge it could be tested on;
the eigenmode-weighted <k_perp^2> is the single most load-bearing design
choice (quantified for the first time). Adoptions: geometric-mean Y for
legacy refits; ridge_poly2/gp_log shortlisted as campaign head forms.
sklearn added to the env for the advanced heads.
