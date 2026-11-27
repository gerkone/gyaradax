# Issue 201 on the three-turn 3559 case

GKW and Gyaradax both develop a fast, grid-scale electromagnetic mode on this case. Stock CGYRO selects a slowly growing mode. A charge correction to Gyaradax's parallel numerical dissipation suppresses the two identified fast branches. The slow growth rate remains about half the matched CGYRO value. This note records the linear experiment. The companion handover describes the skin-gated variant, production-grid tests and nonlinear transport results. [Companion investigation](../investigations/issue-201/README.md)

## How the parallel stencil drives the instability

The native fourth difference damps electrons and ions at different rates. Their perturbed gyroaveraged charges can initially cancel and then acquire a net charge under damping. Quasineutrality turns that charge into a potential perturbation. On the primary fast eigenmode at 32 points per turn, the distribution contribution to the amplitude growth rate from the stencil is −0.345 and the electrostatic contribution is +2.639. Together they supply +2.294 of the observed +2.607. The two outer rows at each end contribute negligibly to this mode. Tracking the native branch at fixed velocity resolution gives growth +0.254 at 38 points per turn and −0.132 at 39.

## A charge constraint on the centered rows

Write $\rho f$ for the local gyroaveraged charge moment in the field solve and $u_s=(Z_s/T_s)F_{Ms}J_{0s}$. Then $e=u/\rho u$ carries unit charge and $Pf=e\rho f$ is the orthogonal projection onto that response in the kinetic quadrature norm. For $f_n=(I-P)f$, the native damping $D$ produces charge $\rho D f_n$. A correction must carry the opposite charge. Its minimum-norm value is $-e\rho D f_n$, since any other correction with the same charge adds an orthogonal neutral component. The resulting operator is

$$
D_c = D-PD(I-P), \qquad \rho D_c(I-P)=0, \qquad D_cP=DP .
$$

The input projection uses the local geometry at each stencil neighbor. The output projection uses the receiving cell. The prototype applies this constraint on nonzonal centered five-point rows and retains the one-sided open-end updates. This is an ungated linear diagnostic. The companion investigation found sizable changes in nonlinear fluxes for ungated F2 variants with its end treatment.

For a constant-coefficient periodic interior, let $D=-A$ for a parallel Fourier harmonic, where $A$ is a nonnegative damping multiplier. Quasineutrality gives the energy conjugate $rPf+(I-P)f$ with $r\ge1$. Setting $X=A^{1/2}Pf$ and $Y=A^{1/2}(I-P)f$ gives the corrected damping work

$$
-r\|X\|^2-\|Y\|^2-\mathrm{Re}\langle Y,X\rangle
= -\|Y+X/2\|^2-(r-1/4)\|X\|^2\le0 .
$$

Variable geometry and the open ends remain outside this proof.

The checkpoint driver applies the five-point stencil directly to the neutral distribution. For the 3559 damping weight (`idisp=2`), the parallel-velocity sum can be taken before gathering the five neighbors. A separate factored implementation agreed with the direct correction to about $10^{-15}$ on the checkpoint and a random complex state. Median warmed FP64 RK4 times on one H200 were 44.4 µs native, 62.6 µs factored and 2,461 µs for the direct prototype, for one linear harmonic. Production nonlinear cost remains unmeasured.

## What the matched runs establish

All rates below use GKW time units and frequency sign, $\lambda=\gamma+i\omega$. The CGYRO calculation matches the signed midplane radial wavenumber and the three-turn domain. Converting its rates uses the factor $1.9732662/\sqrt{2}$ and reverses the frequency sign. Field overlaps are normalized complex inner products in magnitude, evaluated separately for each field on the common angular interval.

| Slow mode | Points per turn | Growth γ | Frequency ω | φ overlap with CGYRO | A∥ overlap |
|---|---:|---:|---:|---:|---:|
| CGYRO reference | 32 | 0.14609 | −0.39447 | 1 | 1 |
| Native Gyaradax | 32 | 0.07561 | −0.49788 | 0.967 | 0.979 |
| Corrected Gyaradax | 32 | 0.07748 | −0.49081 | 0.969 | 0.988 |
| Corrected Gyaradax | 48 | 0.07544 | −0.49675 | 0.973 | 0.988 |

Two different continuations distinguish suppression of the fast artifact from the slow-rate discrepancy.

1. **Introduce the charge correction at fixed native damping.** The two fast branches start at +2.607 and +1.922. Both decay by 6% correction on the coarse grid. At full strength, a restart from the original fast checkpoint selects the slow mode with γ = 0.07749 and ω = −0.49077 at time 128. The slow mode remains growing on both grids. Earlier characteristic damping reversed its growth (−0.05204) and reduced its φ overlap to 0.180. An empirical sixth difference suppressed both fast branches with a 0.114% shift in slow growth on the original mesh, but admitted positive free-energy work. [Branch continuations and direct eigenpair residuals](../investigations/issue-201/results/3559-charge-branches.csv)
2. **Scale the entire corrected parallel dissipation.** On the 32-point grid the same slow branch gives γ = 0.07748 at nominal strength, 0.06041 at half and 0.03640 at one quarter. It enters decay at weak damping and reaches +0.00014 at zero. Adjacent kinetic eigenvector overlaps exceed 0.97 throughout. Reducing this coefficient widens the growth-rate discrepancy with the stated CGYRO reference. [Dissipation scan](../investigations/issue-201/results/3559-dissipation-scan.csv)

The potential and vector potential have similar profiles across codes, but the slow rate depends on the three-turn boundary conditions and the velocity quadratures differ. A zero-angle decaying control agreed in both complex fields to within 2% one CGYRO time unit after restart. On the actual finite-angle checkpoint the transfer recovered φ and A∥ to within 0.8% on the finer Gyaradax velocity grid. Instantaneous field derivatives were much more sensitive to transfer and CGYRO timestep. The late-time growth and frequency give the more robust comparison. Saturated transport and quantitative slow-mode agreement remain open.

## Sensitivity of the CGYRO reference

Starting each run from scratch, change only the pitch count (`N_XI`), the energy count (`N_ENERGY`), or both in the matched three-turn deck. The energy cutoff, spatial grid, timestep and geometry are fixed. All four runs met CGYRO’s linear convergence criterion (`FREQ_TOL=1e-6`). A fit to the complex potential over the last 30% of each run reproduced the tabulated growth and frequency within 0.000004 in GKW units.

| Pitch × energy points | Growth γ | Frequency ω |
|---|---:|---:|
| 32 × 8 | 0.14609 | −0.39447 |
| 64 × 8 | 0.14557 | −0.39607 |
| 32 × 16 | 0.14774 | −0.39539 |
| 64 × 16 | 0.14732 | −0.39701 |

The largest growth shift from the baseline was 0.00165, compared with the 0.06861 gap to corrected Gyaradax. All three refined potential profiles had complex overlap above 0.9997 with the baseline CGYRO profile. This checks sensitivity to CGYRO velocity resolution at fixed cutoff. Gyaradax velocity-space convergence still requires its own test. [Rates, field fits and overlaps](../investigations/issue-201/results/3559-cgyro-velocity.csv) · [Analysis script](../investigations/issue-201/scripts/cgyro_velocity_scan.py)

## Where the slow mode gains and loses energy

For a discrete eigenmode with free energy $W$, a term $L_j g$ contributes $\mathrm{Re}\langle f+u\phi,L_j g\rangle_{\rm kin}/(2W)$ to amplitude growth. On the corrected Gyaradax eigenstate the terms close to the independently measured $\gamma$ on both parallel grids. The transport column combines parallel streaming, trapping and the parallel electric-field term.

| Points per turn | Gradient drive | Parallel transport | Corrected parallel dissipation | Net growth | Estimated spatial end flux |
|---:|---:|---:|---:|---:|---:|
| 32 | +0.60392 | −0.50953 | −0.01691 | +0.07748 | −0.49401 |
| 48 | +0.54408 | −0.46433 | −0.00431 | +0.07544 | −0.44872 |

The final column estimates the signed energy flux from the two open ends using linearly extrapolated endpoint traces of the streaming speed and the quadrature-weighted nonadiabatic distribution. It estimates part of the transport column and is excluded from the sum. Using the nearest cell values instead gives −0.49319 and −0.44838. Gradient drive and outward transport are much larger than the net growth. Comparing those two terms with CGYRO on its own eigenstate would distinguish weaker drive from stronger loss. [Signed work and both end-flux estimates](../investigations/issue-201/results/3559-slow-budget.csv)

## Reproducing the linear checkpoint test

From the repository root, with the FP64 JAX environment and a local copy of the restricted 3559 inputs, run

```bash
python -m scripts.issue201_charge_filter "$config" "$checkpoint" --grid "$exported_gkw_grid" \
    --dt .002 --steps 8000 32000 64000
```

Use the saved fast-mode checkpoint (step 8000) and the matching 32-point-per-turn GKW grid. The output gives elapsed time, complex potential growth and frequency, the relative eigenmode residual in φ, and its norm. Add `--native` for the control. Run `python -m pytest -q tests/unit/test_issue201_charge_filter.py` for the synthetic linked-chain test. [Checkpoint driver](../scripts/issue201_charge_filter.py) · [Regression test](../tests/unit/test_issue201_charge_filter.py) · [Lab input locations](../investigations/issue-201/README.md#11-material-that-stays-on-the-lab-hosts)
