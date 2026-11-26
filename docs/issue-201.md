# Issue 201 electromagnetic instability

The native GKW and Gyaradax parallel dissipation stencil drives a grid-scale mode in the low-\(k_y\) electromagnetic `3559` case. An independent decaying control validates the CGYRO–Gyaradax field, time and wavenumber conversions at zero ballooning angle. **We reject the first attempted fix, characteristic parallel damping.** It suppresses the fast artifact but turns a growing slow mode into a decaying one and distorts its electrostatic field profile.

This is a local lab investigation. The issue bundle and prior analysis are restricted. Keep this branch local.

## What the translation control validates

We initialized native FP64 Gyaradax from a CGYRO distribution at CGYRO time 1. The control used circular Miller geometry, kinetic electrons, two fields, zero gradients and zero ballooning angle. We applied the prescribed time factor (√2/3) and absolute field factors (40.7120 for φ and 28.7878 for A∥), then advanced Gyaradax independently. The errors below are complex field differences on the mapped grid, normalized by the CGYRO field norm, with no amplitude or phase fit. [CGYRO deck](/system/user/publicwork/volkmann/stable-cgyro.vqmUXb/cg-refined/history/input.cgyro) · [Gyaradax configuration](/system/user/publicwork/volkmann/stable-cgyro.vqmUXb/cg-refined/fine.config.json) · [Comparison driver](/system/user/publicwork/volkmann/stable-cgyro.vqmUXb/probe.py)

| CGYRO time | Relative error in φ | Relative error in A∥ |
|---|---:|---:|
| 1 (restart) | 0.08% | 0.59% |
| 2 | 1.75% | 1.82% |
| 3 | 4.67% | 2.26% |

At time 3 the potential norms had fallen to 50.4% and 50.5% of the CGYRO restart norm in Gyaradax and CGYRO, respectively. The mapped perpendicular wavenumber agreed with CGYRO’s central-harmonic geometry and the circular reconstruction across eight harmonics to 1.5 parts per million. An independent driven case checked the frequency sign, with converted CGYRO ω = +0.8911 and Gyaradax ω = +0.9026. [Field history](/system/user/publicwork/volkmann/stable-cgyro.vqmUXb/comparison.csv) · [Geometry](/system/user/publicwork/volkmann/stable-cgyro.vqmUXb/geometry.json) · [Driven CGYRO log](/system/user/publicwork/galletti/git/cgyro_study/em/cgyro-compare/runs/phys/v_k0.5_b0.004_nf2_xi96/out.cgyro.freq) · [Gyaradax eigenvalues](/system/user/publicwork/galletti/git/cgyro_study/em/cgyro-compare/gyx/eig.jsonl)

The control validates the early trajectory. By time 20 the potential norms were about 1.3% of the restart norm, and the relative errors reached 34% in φ and 133% in A∥. Doubling CGYRO’s velocity resolution alone changed its final A∥ by 92% of the refined norm. A matched early-time restart at the finite ballooning angle of `3559` remains open. [Decay curves](/system/user/publicwork/volkmann/stable-cgyro.vqmUXb/stable-comparison.png)

## Why the fast mode grows

At 32 parallel points per turn, GKW and Gyaradax reproduce the same fast mode, with growth γ ≈ +2.607 and frequency ω ≈ −56.61 in GKW units. Stock CGYRO selects a slow mode on the matched three-turn chain. Three controls locate the source of the fast growth. [GKW reference](/system/user/galletti/git/gkw-issue201/run/test_3559_off/time.dat) · [Gyaradax reference](/system/user/publicwork/volkmann/issue201.I85RLq/issue201/growth-summary.txt) · [CGYRO rates](/system/user/publicwork/volkmann/issue201-code-comparison.iZz2N9/cgyro-rates.csv)

- **Free-energy budget.** The interior fourth-difference stencil damps electrons and ions at different speeds. This disrupts cancellation between their charge moments. On the fast mode the stencil contributes −0.345 through the distribution and +2.639 through the electrostatic field, supplying +2.294 of the +2.607 growth. The first and last two stencil rows contribute negligibly on this mode. [Energy budget](/system/user/publicwork/volkmann/issue201-energy.8GiJIX/driven-0.002.csv) · [Interior and end-row work](/system/user/publicwork/volkmann/issue201-parallel-balance.stY1l8/interior.csv)
- **Matched restart.** Removing only the stencil-generated charge from the charge-neutral part of the distribution takes the free-energy amplitude from 1 to 0.00036 after 16 time units. The native restart reaches 1.3 × 10¹⁸. Halving the timestep reproduces the decay. Separate gradient controls show that the density drive through φ sustains the phase relation that lets the stencil do positive work. [Restart comparison](/system/user/publicwork/volkmann/issue201-restart.mnteGy/summary.txt) · [Gradient controls](/system/user/publicwork/volkmann/issue201-phase.ilapQ3/phase-0.002.csv) · [Phase budget](/system/user/publicwork/volkmann/issue201-phase.ilapQ3/original-phase-budget.csv)
- **Parallel refinement.** Tracking the same eigenvector at fixed velocity resolution gives γ = +0.254 at 38 points per turn and −0.132 at 39, where the electric stencil work changes sign. Swapping only the dissipation’s inverse-grid-spacing factor between those two grids reverses stability on both. [Tracked mode](/system/user/publicwork/volkmann/issue201-refinement.GUo30C/refinement.csv) · [Fixed-grid controls](/system/user/publicwork/volkmann/issue201-crossover.sX8Lfm/fixed-grid.csv)

## Why characteristic damping failed

For the three-turn `3559` comparison we matched the signed outboard-midplane radial wavenumber with CGYRO `PX0=-0.2273901`. This reduced the relative difference in the perpendicular wavenumber profiles from 52.4% to 1.35%. The table compares the CGYRO mode to the same Gyaradax slow branch followed continuously from native to characteristic damping. To express CGYRO rates in GKW units on this deck, multiply by \(1.9732662/\sqrt{2}\) and reverse the frequency sign. Each overlap is the magnitude of the normalized complex inner product of field profiles on the common angular interval. [Matched deck](/system/user/publicwork/volkmann/issue201-code-comparison.iZz2N9/mapped_nr3/input.cgyro) · [Geometry comparison](/system/user/publicwork/volkmann/issue201-code-comparison.iZz2N9/geometry.py) · [Branch continuation](/system/user/publicwork/volkmann/issue201-characteristic.h2ZR2P/iter-32-slow.csv)

| Three-turn slow mode | γ | ω | φ overlap with CGYRO | A∥ overlap with CGYRO |
|---|---:|---:|---:|---:|
| Stock CGYRO | +0.14609 | −0.39447 | 1 | 1 |
| Native Gyaradax | +0.07561 | −0.49788 | 0.967 | 0.979 |
| Gyaradax with characteristic damping | −0.05204 | −0.38097 | 0.180 | 0.986 |

The modified frequency moves closer to CGYRO and A∥ retains its shape, but the growth reverses sign and the φ overlap falls from 0.967 to 0.180. A separate eigensolve near the CGYRO rate also returned three decaying modes with φ overlaps of only 0.18–0.33. CGYRO reached time convergence on the matched grid. Its slow growth rate still depends on chain length, and the velocity quadratures and geometry models differ. [Mode measurements](/system/user/publicwork/volkmann/issue201-code-comparison.iZz2N9/modes.csv) · [Targeted eigensolve](/system/user/publicwork/volkmann/issue201-code-comparison.iZz2N9/target.log) · [Field profiles](/system/user/publicwork/volkmann/issue201-code-comparison.iZz2N9/comparison.png) · [Chain-length scan](/system/user/publicwork/volkmann/issue201-code-comparison.iZz2N9/cgyro-rates.csv)

Two other controls constrain a repair. Changing only the stencil’s charge response also stabilized both fast modes and reversed slow-mode growth. Adding a sixth difference at coefficient 0.25 suppressed both measured fast modes while shifting native slow-mode growth by 0.114% on the original mesh. Its coefficient and end treatment were empirical, and the combined stencil still admits positive free-energy work. [Charge-response control](/system/user/publicwork/volkmann/issue201-parallel-balance.stY1l8/charge-slow.csv) · [Sixth-difference scan](/system/user/publicwork/volkmann/issue201-energy-stable.a64ES6/other-branches-32.csv) · [Boundary work test](/system/user/publicwork/volkmann/issue201-energy-stable.a64ES6/boundary-nullspace.csv)

The next comparison is a matched finite-angle restart, followed by velocity-grid and chain-length convergence of the slow reference. Judge a replacement parallel flux and its end treatment against both the disappearance of the fast branches and the slow mode’s growth, frequency and complex field profiles. The independent `base4_01` deck provides a cross-check before adopting a solver change. [Cross-deck restart](/system/user/publicwork/volkmann/issue201-energy-stable.a64ES6/base4-1-long.csv)

## Reproduction on the lab hosts

The recovered `3559` input is a June 2026 port of the original attachment. It uses circular geometry, kinetic electrons, `kthrho=0.07`, `CHIN=1.8`, `beta=0.0035`, and parallel dissipation `disp_par=2` with `idisp=2`. It evolves the electrostatic potential and parallel vector potential. There is one radial harmonic over three poloidal turns, with 32 points per turn and a 32 × 8 velocity grid. The original rotation and parallel flow gradients were dropped in the surviving port. The bundle also has the separate `base4_01` reproducer for a follow-up regression check. [Bundle README](/system/user/publicwork/volkmann/issue201.I85RLq/issue201/README.md) · [Input provenance](/system/user/publicwork/volkmann/issue201.I85RLq/issue201/SOURCES.md)

### GKW

On a lab host with the GKW build and its MPI runtime available, run the recovered deck in a fresh directory. The wrapper derives eight ranks from the input. Use the double-precision build from `785a215` with conservative parallel dissipation off. The expected late-time rate is γ ≈ 2.60712 and ω ≈ −56.6134 in GKW units.

```bash
issue_bundle=/system/user/publicwork/volkmann/issue201.I85RLq/issue201
issue_runs=$(mktemp -d /system/user/publicwork/volkmann/issue201-repro.XXXXXX)
gkw_bin=/system/user/galletti/git/gkw-issue201/run/gkw_scorpion-DP_785a215-dirty.x
issue_python=/system/apps/userenv/volkmann/jax_env/bin/python

mkdir "$issue_runs/gkw-3559"
cp "$issue_bundle/gkw/3559/input.dat" "$issue_runs/gkw-3559/"
bash "$issue_bundle/scripts/run_gkw.sh" "$issue_runs/gkw-3559" "$gkw_bin"
"$issue_python" "$issue_bundle/scripts/gkw_growth.py" "$issue_runs/gkw-3559"
```

The completed reference reached `t=60`. A later launch on `wasp` used `NTIME=1` and `NAVERAGE=50` and completed its requested interval at `t=0.10`. Open MPI also printed fabric warnings. Use the completed reference for comparison and check that a fresh run reaches late time before reading its rates. [Reference output](/system/user/galletti/git/gkw-issue201/run/test_3559_off/time.dat) · [Wasp MPI log](/system/user/publicwork/volkmann/issue201-gkw-wasp.Sampe0/3559/gkw.log)

### Gyaradax

Use the bundled `src-3559` snapshot (`95da48b` plus the GKW import patch) to reproduce the recorded GPU run. The patch is needed on that base to retain the finite ballooning angle and match GKW's zero perpendicular dissipation for the single-mode input. The bundled YAML already points to the exported GKW grid and specifies `chin: 1.8`, `disp_x: 0`, `disp_y: 0`, and the electromagnetic fields. The investigation branch starts from current `master`. Port and verify those import settings on this branch before using it for solver comparisons. [GPU config](/system/user/publicwork/volkmann/issue201.I85RLq/issue201/3559.yaml) · [Import patch](/system/user/publicwork/volkmann/issue201.I85RLq/issue201/gyaradax/gkw_import_fix.patch)

Continue in the same shell. Check `nvidia-smi` and set `issue_gpu` to a free GPU index. The recorded run used JAX 0.10.2 in FP64 on a H200 on `wasp`.

```bash
CUDA_VISIBLE_DEVICES="${issue_gpu:?set issue_gpu to a free GPU index}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$issue_bundle/src-3559" \
  "$issue_python" -B -P -m gyaradax.cli run "$issue_bundle/3559.yaml" \
  --backend jax --dp --from-scratch --n-steps 8000 \
  --output-dir "$issue_runs/gyaradax-3559"

"$issue_python" "$issue_bundle/scripts/gyx_growth.py" "$issue_runs/gyaradax-3559"
```

Expect γ ≈ 2.60700 at `t=16` with `dt=0.002`. The recorded ω ≈ −56.61302 came from the potential overlap phase over one additional RK4 step. The growth helper reports γ only. [GPU measurements and frequency method](/system/user/publicwork/volkmann/issue201.I85RLq/issue201/growth-summary.txt)

### CGYRO

On `wasp`, use the lab's GPU-enabled GACODE environment and one MPI rank per run. Copy the two recorded decks into clean run directories and keep the GPU idle before starting. With the GPU-enabled `cgyro` launcher on `PATH`, run the following in the same shell.

```bash
issue_cgyro=/system/user/publicwork/volkmann/issue201-3559-cgyro.k1JmWL

for nr in 4 16; do
  mkdir "$issue_runs/cgyro-nr$nr"
  cp "$issue_cgyro/gpu_nr$nr/input.cgyro" "$issue_runs/cgyro-nr$nr/"
  CUDA_VISIBLE_DEVICES="${issue_gpu:?set issue_gpu to a free GPU index}" \
    cgyro -e "$issue_runs/cgyro-nr$nr" -n 1 > "$issue_runs/cgyro-nr$nr/run.log" 2>&1
done

"$issue_python" \
  /system/user/publicwork/galletti/git/cgyro_study/em/cgyro-compare/scripts/freq_summary.py \
  "$issue_runs/cgyro-nr4" "$issue_runs/cgyro-nr16"
```

Both decks use `N_THETA=32`, `BOX_SIZE=1`, two fields, and `PX0=0.2864789`. Only `N_RADIAL` differs. With `BOX_SIZE=1` it gives the number of linked poloidal turns. Four and sixteen test sensitivity to chain length. Stock CGYRO also supports three turns for this linear deck. The matched three-turn deck used for the slow-mode test also adjusts `PX0` as described above. [Four-mode deck](/system/user/publicwork/volkmann/issue201-3559-cgyro.k1JmWL/gpu_nr4/input.cgyro) · [Sixteen-mode deck](/system/user/publicwork/volkmann/issue201-3559-cgyro.k1JmWL/gpu_nr16/input.cgyro)
