# Issue 201: handover of the gyaradax-side investigation

Written 2026-11-27 for whoever picks issue 201 up. It records everything tried on the gyaradax / GKW / CGYRO side between 2026-11-13 and 2026-11-27, the current candidate fix (the skin-gated neutral dissipation), the configurations and the results. The [matched three-turn linear study](../../docs/issue-201.md) derives the centered-row charge correction and compares its tracked slow branch with CGYRO.

Conventions. Units are GKW units ($v_{th,ref}/R$, GKW sign, $\omega<0$ electron direction) unless a row says $c_s/a$ (CGYRO units and sign, used for the Waltz and i201 cases). "Production grid" means the grid a production run would use; everything else is a test grid and is labelled. "Plain" means the stock GKW/gyaradax parallel dissipation. "Fast mode" means the grid-scale numerical mode of issue 201: the largest growth among modes whose $\phi$ centroid along $s$ is above 0.3 of the Nyquist wavenumber.

## 1. Standing in one paragraph

Issue 201 is a numerical mode driven by the work that $\phi$ does on the charge created by the parallel hyper-dissipation, at the $k_\perp$ minimum of low-$k_y$ chains with finite $\theta_0$, once $A_\parallel$ is on and the electron skin term dominates $k_\perp^2$. Removing that charge everywhere (forms F1 and F2 below) removes the mode but changes the physics at production resolution (nonlinear kinetic fluxes moved by 10 to 72 %), so the maintainer rejected it. The current candidate removes the charge only where the inductive electron response dominates, $\alpha_A = K/k_\perp^2 > \alpha_c$ (the "skin gate"). It removes the fast mode on every deck tried, leaves electrostatic runs bitwise unchanged, and keeps physical linear eigenvalues on production grids within 2 % (step gate, $\alpha_c$ 100 to 300) or about 1 % (smooth gate, $\alpha_c$ 30 to 100). The first electromagnetic nonlinear runs (2026-11-27, Cyclone, $\beta=0.005$) show the plain scheme blowing up with kinetic electrons at the issue-201 location while both gates run, and no measurable change with adiabatic electrons. The code is a patch in `patch/`, not merged. Open: a nonlinear kinetic EM case where master survives, the choice between step and smooth gate, and the 1 % target on deck 3559.

## 2. The issue and the decks

GKW issue 201 is an electromagnetic numerical instability at low $k_\theta\rho$ and finite $\theta_0$. Two decks from the original thread reproduce it in GKW develop (785a215). Both come from the restricted issue bundle; the files stay on the lab hosts (paths in section 11).

- **Deck 3559**: circular geometry, $q=3.1962$, $\hat s=2.357$, $\epsilon=0.2534$, CHIN 1.8 (finite $\theta_0$; GKW `kxrh` $-0.304404$, used as `kx0` in gyaradax YAML runs), $k_\theta\rho=0.07$, NX 1, nperiod 2 (3 poloidal turns), 32 points per turn, $32\times8$ velocity points (vpmax 3, mumax 4.5). Ions $m=1$, $T=1$, $R/L_T=6.2$, $R/L_n=5.4$; electrons $m=2.7232\times10^{-4}$, $T=0.9$, $R/L_T=7.9$, $R/L_n=5.4$. $\beta=0.0035$, $A_\parallel$ on, $B_\parallel$ off, collisionless, $\beta'=0$, `disp_par` 2, `disp_vp` 0, `idisp` 2, `disp_x`/`disp_y` zeroed by GKW without mode_box (gyaradax must set them to 0 explicitly; the old importer kept them and gave 1.16 instead of 2.61). Rotation and `uprim` of the original were dropped in the port.
- **Deck base4_01**: ITER hybrid scenario, $q=3/2$ surface, Miller geometry, $\beta_{ref}=0.0122$, $k_\theta\rho=0.075$, collisional, 36 points per turn, $48\times16$ velocity points. gyaradax runs a reduced version (`base4red`: $32\times8$, collisions off because gyaradax supports $Z_{eff}=1$ only; `nlapar` and `beta` set by hand because the GKW importer drops them).
- **i201**: a Miller-circle analogue of deck 3559 built for CGYRO comparisons (KY 0.0618, BETAE_UNIT 0.00295, PX0 0.2865, $N_\theta$ 32), run in gyaradax at $(48, 64\times12)$ on 31 turns.
- **Waltz decks** (ours, in `configs/waltz_linear/`): Waltz standard case ($q=2$, $\hat s=1$, $\epsilon=1/6$, $R/L_T=9$, $R/L_n=3$, kinetic electrons, Miller circle), $k_\theta\rho$ 0.05, 0.1, 0.25, 0.5 at $\beta$ 0.001 and 0.005, translated from `waltz_lin.cgyro`. Waltz $k_\theta\rho=0.05$, $\beta=0.005$ on 21 turns is the "production-type deck" below.
- **Kinetic validation decks** (electrostatic, `configs/kinetic_linear/`, the same physics as `configs/kinetic/nonlinear/*.yaml`): itg_collisionless, itg_mid_flux, tem_low_flux, tem_high_flux. Used to check that a remedy leaves the physics alone.

Reproductions of the fast mode:

| code, route | value |
|---|---|
| GKW develop 785a215, deck 3559 | $2.60712 - 56.6134i$ |
| gyaradax, GKW run dir, exact `ds` | $2.607035 - 56.613099i$ |
| gyaradax, YAML with `kx0` $-0.3044$ | $2.6071 - 56.6135i$ |
| gyaradax, CHIN 0 ($\theta_0=0$) | no fast mode ($0.0893 - 0.470i$) |
| GKW, base4_01 full | $2.75874 - 6.94744i$ |
| GKW, base4_01 reduced $32\times8$ | $2.79146 - 6.77309i$; gyaradax $2.791$ |
| physical mode of deck 3559, GKW at 64 per turn, 3 turns | $0.0784 - 0.500i$ (gyaradax $0.0764$) |

The physical mode of deck 3559 is not converged in domain length: GKW at 64 per turn gives $0.0784-0.50i$ on 3 turns, $0.0581-0.36i$ on 9 and $0.0441-0.322i$ on 15 ($|\phi|$ at the chain ends 0.4 to 0.6 of its maximum). The fast mode is the same to every printed digit on 3 and 9 turns. Deck values below are values on the deck's 3 turns, used as a reference for the numerics, not the growth rate of the case.

Parameter dependence of the plain scheme on deck 3559 (gyaradax eigen-solves, GKW initial value where marked):

| change | fast mode | physical mode |
|---|---|---|
| base (`disp_par` 2, 32 per turn, $\beta=0.0035$) | $2.6117 - 56.61i$ | $0.0756 - 0.498i$ |
| `disp_par` 0.5 / 1 / 4 / 8 | $10.71$ / $8.11$ / none / none | $0.0357$ / $0.0593$ / $0.0848$ / $0.0905$ |
| `disp_par` 0 | GKW unconverged (3 to 7.5), gyaradax $0.22 + 243.7i$ | |
| 24 / 48 / 64 per turn | $5.73$ / none / none | $0.0733$ / $0.0774$ / $0.0764$ |
| $\beta$ 0 / 0.001 / 0.01 / 0.02 | none / none / $4.74$ / GKW $5.48$ | |
| $k_\theta\rho$ 0.035 | 17 to 22 | |
| $k_\theta\rho \geq 0.14$ ($\beta\le0.01$) | none | |
| `idisp` 1 | $10.13$ (worse) | |
| $n_{v\parallel}$ 64 / $n_\mu$ 16 / 5 or 7 turns | $2.608$ / $2.654$ / none on 5 and 7 turns at 32 per turn | |
| time step | unchanged (eigenvalue of the space-discrete operator) | |

Resolution looks like the cure on deck 3559 but is not: on the production-type Waltz deck ($k_\theta\rho=0.05$, $\beta=0.005$, 21 turns, `disp_par` 1) the leading growth is 0.95 to 1.0 at 16, 32 and 48 points per turn and still 0.54 at 64, against about 0.03 for the physical mode; `disp_par` 4 at 16 per turn leaves 0.33. The frequency follows the grid ($-3.8$, $-7.2$, $-11.3$, $-14.0$ at 16 to 64 per turn). A threshold $\beta_c \approx 0.35 (k_\theta\rho)^2$ holds only at $k_\theta\rho$ 0.035 to 0.07 on deck 3559.

## 3. Mechanism

Full derivation in `MATH.md` and Section 3 of `writeup/skin_gated_dissipation.pdf`. With the free energy $W = \tfrac12\langle f,f\rangle + \tfrac12\mathrm{Re}(\phi^*\rho_f) + \tfrac12\mathrm{Re}(A_\parallel^* j_f)$, any right-hand side gives $dW/dt = \mathrm{Re}\langle H, \partial g/\partial t\rangle$ with $H = f + (Z/T) J_0\phi F_M$ (checked to $10^{-11}$ on every eigenvector). The dissipation contributes

$$\mathrm{Re}\langle H,\mathcal D f\rangle = \underbrace{\mathrm{Re}\langle f,\mathcal D f\rangle}_{\text{damping}} + \underbrace{\mathrm{Re}(\phi^*\rho_{\mathcal D})}_{\text{field work}},\qquad \rho_{\mathcal D} = \sum_s Z_s n_s \{J_0 \mathcal D_s f_s\}_v .$$

The second term has no sign. Budget of the leading fast mode on the 3559 physics (5 turns, 24 per turn, $8\times2$ velocity points, test grid, $\lambda = 3.62 - 49.6i$): electron field work $+6.87$, ion field work $-3.26$, damping $-0.32$, gradient drive $+0.33$; streaming and Term VII cancel; 79 % of $W$ in $\phi$, 20 % in $A_\parallel$.

What follows from it, all measured:

- The dissipation removes electron gyrocentre density in phase with $\phi$. At the $k_\perp$ minimum the polarisation $\propto k_\perp^2\rho^2$ is small, so $\phi$ responds strongly. The odd-even mode is damped about 60 times faster for electrons than for ions at `idisp` 2, which breaks the charge cancellation.
- $A_\parallel$ enables the mode but does not drive it. Below $\beta_c$ electrons cancel the charge by streaming. Above it the skin term $K$ dominates $k_\perp^2$ at the minimum and the charge accumulates.
- Projections that remove only the current leave the mode unchanged. Every projection that removes the density removes it.
- The same charge drives a mode on a closed loop with no ends and no gradients ($0.016 - 3.04i$ at 16 per turn), so open ends are not the cause.
- The time integrator, the field solve and the velocity resolution are not the cause. The scheme without dissipation is not stable either.
- CGYRO is protected by its per-species number projection of the upwind dissipation (since 2016; eb8884c1 moved it to the output side and added a current projection, which is not what protects it: both builds agree to three digits on deck 3559's analogue). With the projection removed (two-line change in `cgyro_upwind.F90` on the pre-eb8884c1 build) CGYRO grows at 9.3 (GKW units) with a similar $\beta$ dependence (0.03, 0.71, 9.3, 12.6, 16.0 at $\beta$ $10^{-4}$, 0.002, 0.0035, 0.005, 0.01), gone at $N_\theta\ge64$. With `UP_THETA` 0 it grows at about 5. The unprojected CGYRO mode has a different frequency and also appears at $\theta_0=0$, so it is not shown to be the same mode.

## 4. Everything tried

"E n" / "END n" = plain dissipation rows kept within n points of an open chain end. Values are the fast mode / the physical mode on deck 3559 (32 per turn, `disp_par` 2) unless stated.

| approach | where | result | verdict |
|---|---|---|---|
| finer parallel grid | deck 3559; Waltz 0.05; i201; base4_01 | removed at 48 to 64 per turn on 3559 (4x cost), at 108 on base4_01 (9x); not removed on Waltz 0.05 up to 64 (0.54 left) nor on i201 up to 128 | not a fix |
| stronger `disp_par` | same | removed at 4 on 3559 (physical +12 %), at 8 on base4_01 (dt 5x smaller, $\omega$ +5 %); Waltz 0.05 not cleanly up to 8 (+18 % at 4) | not a fix |
| weaker `disp_par`, `disp_par` 0, `idisp` 1 | deck 3559 | worse (10.7 at 0.5, unstable at 0, 10.1 with `idisp` 1) | no |
| June fix, input-side projection of $J_0$ density and current, $k_y$-gated at 0.15 (GKW branch `issue201-conservative-dissipation`; gyaradax `disp_par_conserve` 2) | 3559, base4_01 | removes the fast mode; leaves $0.246 - 0.757i$ on 3559 (3.3x the physical value), 0.26 at 64 per turn against 0.076 plain; base4_01 $2.79 \to 0.038$ | rejected: wrong physical mode; within 3 points of an open end 41 to 45 % of the output density survives; the $k_y$ cut changes results |
| `disp_par_conserve` 1 (current only, input) / current only, output | 3559 | $2.5955$, unchanged | current plays no role |
| `disp_par_conserve` 3 ($\lvert v\rvert$-weighted density, input) | 3559 | removed; physical 0.109 (E4 0.078) | not pursued |
| input $J_0$ density (CGYRO 2016 form, "mode 4") | 3559; Waltz NL | removed; physical $0.246$ (E4 0.111, 64 per turn E8 0.0761); moves the physical mode by O(1) on 3-turn chains without END; Waltz NL ($\beta=0.005$) runs away with and without it | rejected |
| output density + current per species (CGYRO eb8884c1 form, "mode 5") | 3559; ES TEM deck | removed; physical 0.080 with E4; frees an odd-even mode in $s$ on an electrostatic TEM deck (0.240 against the TEM's 0.469) | rejected (per-species form) |
| output density, ions only / electrons only | 3559 | 7.92 (worse) / 0.55 left | no |
| dissipation on $g$ instead of $f$ | 3559 | 6.65 (worse) | no |
| dissipation on $H$ (sign-definite damping) | 3559 | removes the field work; spectral radius 560 to 91045, unaffordable with explicit RK4 | rejected |
| two-sided $(I-P)\mathcal D(I-P)$ (provably dissipative) | 3559, 16 to 24 per turn | physical mode lower by 3x or more | rejected |
| projection including the $B_\parallel$ moment | 3559 | physical growth $-40$ % | rejected |
| **F1**: $(I-P)\mathcal D f$, total-charge projector, output side, END rows | all decks | removes the fast mode everywhere; linear EM physical modes within $10^{-4}$ of F2 | rejected with F2 (overdamps kinetic decks, below) |
| **F2**: $\mathcal D f - P\mathcal D(I-P)f$ everywhere (`disp_par_neutral: true`, END 4) | all decks, NL kinetic validation | linear EM fine; NL kinetic decks shifted by $-72$ % to $+21$ % (section 7) | **rejected by the maintainer 2026-11-26** |
| END 2 vs END 4 | 3559 one-turn chains | identical shifts | END 4 kept |
| **skin gate, step** ($\alpha_c$ 30, 100, 300, 1000) | all decks | removes the fast mode for $\alpha_c$ 30 to 300; physical within 2 % at 100 to 300 | candidate |
| **skin gate, smooth** ($\alpha_A/(\alpha_A+\alpha_c)$) | all decks | physical within 1 % on Waltz at $\alpha_c$ 30 to 100; deck 3559 $\omega$ $-1.1$ to $-1.2$ % | candidate |
| F1 correction inside the gate | all decks | equals F2 inside the gate within 0.15 % | no gain |
| time high-pass of the removed charge (one extra state variable) | none | proposed to get below the 0.6 % floor of any weight along $s$ | untested |
| characteristic parallel damping; sixth difference at 0.25 | see `docs/issue-201.md` | characteristic damping rejected there; sixth difference suppresses both fast modes with a 0.114 % shift but is empirical | other line of work |

## 5. The skin-gated neutral dissipation

Definition. With $u_s = (Z_s/T_s) J_0 F_M$, the gyro-averaged charge $\rho[x] = \sum_s Z_s n_s\{J_0 x_s\}_v$ and the total-charge projector $P x = u\,\rho[x]/\rho[u]$ at each $(s, k_x, k_y)$:

$$\mathcal D^{SG} f = \mathcal D f - M\,P\,\mathcal D\,(I-P)f,\qquad \alpha_A = \frac{K}{k_\perp^2} \approx \frac{1}{(k_\perp d_e)^2},$$

with $K$ = `ampere_dia` $- k_\perp^2$ (the skin term of Ampère's law), the step gate $M = [\alpha_A > \alpha_c]$ or the smooth gate $M = \alpha_A/(\alpha_A+\alpha_c)$, and plain rows at the 4 points next to an open chain end. $\alpha_c=100$ means the correction acts where $k_\perp d_e < 0.1$; 300 means $k_\perp d_e < 0.058$.

Properties. Without $A_\parallel$, $K=0$ and no tables are built: electrostatic runs are bitwise the plain scheme (unit test). With $A_\parallel$, $\alpha_A$ is large only near the $k_\perp$ minimum of low-$k_y$ chains at finite $\theta_0$. One explicit parameter, no cut in $k_y$, $\beta$ or resolution, default off.

Gated points on the $k_y$ chain:

| deck | max $\alpha_A$ | points | $\alpha_c$ 100 | 300 | 1000 |
|---|---|---|---|---|---|
| i201, 31 turns at 48 per turn | 1204 | 1488 | 23 | 10 | 2 |
| Waltz $k_\theta\rho=0.05$, $\beta=0.005$, 16 per turn | 7306 | 336 | 40 | 18 | 12 |
| Waltz 0.1, $\beta=0.005$ | 1838 | 992 | 34 | 24 | 12 |
| Waltz 0.1, $\beta=0.001$ | 368 | 992 | 20 | 8 | 0 |
| Waltz 0.25, $\beta=0.005$ | 294 | 992 | 18 | 0 | 0 |
| Waltz 0.25 $\beta=0.001$; 0.5 $\beta=0.005$ | 59; 74 | 992 | 0 | 0 | 0 |
| electrostatic decks | | | 0 | 0 | 0 |

Implementation (`patch/disp_par_neutral.patch`, applies on this branch and on master 5a31042):

- `gyaradax/core/params.py`: options `solver.disp_par_neutral` (bool), `disp_par_neutral_end` (int, default 2; use 4), `disp_par_neutral_alpha` (float, 0 = everywhere = F2), `disp_par_neutral_variant` (string, `smooth` for the smooth gate, `f1` for the F1 correction; experimental).
- `gyaradax/operators/precompute.py`: `_disp_neutral_tables` builds `dneu_w` (velocity weights $Z n B\,$`intmu intvp`$\,J_0$), `dneu_ein` ($u/\rho[u]$), `dneu_eout` ($e_{in}$ with END rows zeroed, times the gate) and the stencil table; the gate is computed from `apar_diag` and `kperp_sq`.
- `gyaradax/backends/_jax.py`: `disp_neutral_term(f) = -e_out rho[D[f - e_in rho[f]]]`, added to the linear RHS. `_cuda.py` adds the same JAX term on top of the CUDA kernel. `backends/__init__.py` raises for multi-device runs (the term is not sharded).
- `tests/unit/test_disp_par_neutral.py`: default off, the dissipation's charge is removed (adiabatic and kinetic), the charge-carrying part is untouched, END width, the alpha limit (no tables for electrostatic runs), sharded runs refuse, fast mode removed on a test chain.
- Cost: one more parallel stencil and two moments per RHS in EM runs, about 2x in the present JAX form on a B300; a fused CUDA kernel would be cheaper. In GKW it fits as a second pass in `calc_linear_terms`, with the gate read from `ampere_dia` and $k_\perp^2$.

YAML:

```yaml
solver:
  disp_par_neutral: true
  disp_par_neutral_end: 4
  disp_par_neutral_alpha: 100.0      # step gate; 300 is the other candidate
  disp_par_neutral_variant: smooth   # optional: smooth gate alpha_A / (alpha_A + alpha_c)
```

## 6. Linear results on production grids

fp64 Krylov-Schur eigen-solves of the RK4 step operator (`scripts/gyx_eig_p4.py`), records in `results/ACC.jsonl`, table in `results/acc_table.txt` (tags: `pl` plain, `aN` step gate at N, `a0` F2 everywhere, `sN` smooth gate at N, `f1aN` F1 correction with step gate, `f1a0` F1 everywhere, `pl_nsN` plain at N points per turn, `pl_dpN` plain at `disp_par` N).

Fast mode:

| case, grid | plain | $\alpha_c$ 1000 | 300 | 100 | 30 |
|---|---|---|---|---|---|
| deck 3559, 32 per turn, $32\times8$ | 2.607 | 0.78 left | removed | removed | removed |
| i201, $(48, 64\times12)$, 31 turns ($c_s/a$) | 5.60 | removed | removed | removed | removed |
| base4_01 reduced, $(36, 32\times8)$, collisions off | 2.791 | removed | removed | removed | removed |
| Waltz 0.05, 16 per turn (test grid) | 0.376 | 0.031 left | removed | removed | removed |
| Waltz 0.1, $\beta=0.005$ (grid mode) | 0.212 | removed | removed | removed | removed |

On coarse grids the margin at 300 is gone: an i201 chain at 16 per turn (test grid, $32\times8$, `disp_par` 1) kept a fast mode at 0.94 with $\alpha_c=300$.

Physical modes, growth-rate shift against plain on the same grid ("untouched" = no gated point, bitwise plain):

| case (plain value) | 1000 | 300 | 100 | 30 |
|---|---|---|---|---|
| deck 3559, 32 per turn (0.0756) | | +0.5 % | +1.2 % | +4.4 % |
| against its 64-per-turn value (0.0764) | | $-0.5$ % | +0.1 % | +3.3 % |
| i201 (CGYRO $0.0148+0.0506i$) | 0.0158 | 0.0155 | 0.0156 | 0.0156 |
| base4_01 reduced (marginal, $-0.0001+0.259i$) | $-0.0003$ | $-0.0003$ | $-0.0003$ | $-0.0001$ |
| Waltz 0.1, $\beta=0.005$ ($0.0597-0.112i$) | $-5.1$ % | $-1.5$ % | $-0.1$ % | $-0.6$ % |
| Waltz 0.1, $\beta=0.001$ ($0.0744-0.056i$) | untouched | +0.3 % | +0.8 % | $-0.1$ % |
| Waltz 0.25, $\beta=0.005$ ($0.1813-0.281i$) | untouched | untouched | $-1.9$ % | $-0.1$ % |
| Waltz 0.25, $\beta=0.001$ (0.2398) | untouched | untouched | untouched | $-0.02$ % |
| electrostatic decks | untouched | untouched | untouched | untouched |

Worst shift in growth or frequency of the leading physical mode, 32 per turn, Waltz points $k_\theta\rho$ 0.1, 0.25, 0.5 at $\beta$ 0.001 and 0.005:

| form | worst on Waltz | deck 3559 $\gamma$ / $\omega$ | fast mode |
|---|---|---|---|
| step 300 | 1.5 % | +0.5 % / $-1.0$ % | removed |
| step 100 | 1.9 % | +1.2 % / $-1.3$ % | removed |
| step 30 | 2.4 % | +4.4 % / $-1.2$ % | removed |
| smooth 300 | 1.3 % | +0.6 % / $-0.9$ % | removed |
| smooth 100 | 1.0 % | +0.3 % / $-1.1$ % | removed |
| smooth 30 | 0.75 % | +0.6 % / $-1.2$ % | removed |
| smooth 1 | 0.67 % | +2.9 % / $-1.4$ % | removed |
| F1 correction, step 100 | 1.9 % | +1.0 % / $-1.3$ % | removed |

About 0.6 % is a floor for any weight along $s$: the physical modes carry the same term at the $k_\perp$ minimum as the fast mode. The step-gate shift is not monotonic in $\alpha_c$ (it depends on which few points sit just inside the edge). On 16-per-turn test grids physical modes move by 6 to 27 %, where plain is itself 4 to 6 % off its resolved value. On Waltz 0.05 at 16 per turn every form puts the second physical mode at 0.013 to 0.015 against plain 0.0201 (resolved 0.0207); at 32 per turn 0.0206 to 0.0208.

## 7. Nonlinear results

### F2 everywhere on the kinetic validation decks (2026-11-25/26; reason for the rejection)

Production grid $2\times32\times12\times18\times192\times32$ (18 per turn), option against plain with the same code, driver and GPU type, fluxes over the last 30 % of each run. Electrostatic decks, so issue 201 does not occur; the test is whether the form leaves the physics alone.

| deck | $\hat s$ | F2 against plain | linear-phase growth plain / F2 |
|---|---|---|---|
| itg_collisionless | 1.69 | $Q_i$ +9.5 % ± 3.5 %, $Q_e$ +21.5 % ± 3.8 % | 0.145 / 0.142 |
| tem_high_flux | 0.28 | $Q_i$ $-20$ %, $Q_e$ $-18$ %, $\Gamma$ $-17$ % | about equal |
| tem_low_flux | 2.67 | $Q_i$ $-72$ %, $Q_e$ $-54$ %, $\Gamma$ $-68$ % | 0.151 / 0.076 |
| itg_mid_flux | 3.74 | not saturated by $t=350$ | 0.090 / 0.02 to 0.04 |

Spread of the plain controls against stored references: $-4.5$ % to +18 % in $Q_i$. With adiabatic electrons F2 is close to plain (+1 to +2 % high-shear ITG, $-8$ % ± 3 % base ITG). Mechanism, measured on single-turn chains at the production grid: at 18 per turn the direct damping of the plain dissipation on the physical modes is as large as their growth, and 30 to 70 % of it comes back as electron field work (tem_low_flux: damping $-0.336$, field work $+0.232$; itg_mid_flux $-0.174$ / $+0.066$; itg_collisionless $-0.055$ / $+0.018$). F1 and F2 remove that return, so they overdamp. Same with and without collisions; largest at high shear where the mode is narrower than the grid spacing. Any form that removes the field work at every point has this problem; CGYRO pays for it the same way (27 % change in an adiabatic ITG ion heat flux between $N_\theta$ 16 and 32).

### Skin gate against master, electromagnetic (2026-11-27, in progress)

Configs in `configs/cbc_em_nonlinear/`: Cyclone parameters ($q=1.4$, $\hat s=0.78$, $\epsilon=0.19$, $R/L_T=6.9$, $R/L_n=2.2$, $T_e=T_i$), $\beta=0.005$, $A_\parallel$ only, collisionless, circular geometry, CUDA backend, mixed precision, adaptive dt, noise start. The case is a choice made for this test; no production EM deck exists. Variants: `master` (option off, bitwise master), `smooth100`, `step300` (END 4). Fluxes are $(\Gamma, Q, \Pi)$ per species in gyaradax units.

Kinetic electrons, kinetic production grid (18, $32\times12$, $192\times32$, `krhomax` 1.5, `ikxspace` 5):

- `master` blew up by $t\approx1.0$: dt fell from 0.0026 to $7\times10^{-6}$, growth 30 to 800 at every $k_y$, $Q$ about $10^4$. The amplitude sits at $k_y$ index 1 ($k_\theta\rho=0.048$) and $k_x$ indices $\pm2$ around zero ($\theta_0\approx\pm2.5$): the issue-201 location. Stopped by hand.
- `smooth100` and `step300` run at constant dt 0.0026. At $t\approx55$ to 59 they were still growing towards saturation ($Q_i$ about 128, $Q_e$ about 53 to 55 averaged over the last 30 %, both rising), so no statement about the saturated level yet. Stopped at about 11:00 on 2026-11-27 to free GPUs; to be resumed from their snapshots.

Adiabatic electrons, adiabatic production grid (16, $32\times8$, $85\times32$, `krhomax` 1.4), $t=0$ to 600, mean over $t$ 420 to 600:

| variant | $Q_i$ | half-window means |
|---|---|---|
| master | 21.19 | 22.02, 20.43 |
| step300 | 21.19 (bitwise master: no gated point, $\alpha_A$ from the ion current only) | identical |
| smooth100 | 20.89 ($-1.4$ %) | 18.94, 22.68 |

The smooth gate's change is inside the scatter of the window. A longer window or several realisations are needed for a tighter statement.

## 8. Decisions taken by the maintainer

- F2 (and F1) are rejected as fixes for issue 201 because they change the physics on the kinetic decks (2026-11-26). They are not banding fixes either; banding (s-Nyquist banding in kinetic runs) is out of scope for issue 201.
- A remedy must remove the fast mode without changing physical results: target 1 % on physical linear eigenvalues at production resolution; 2 % was accepted as the best available ("if 2 % is the best we have, then so be it").
- No hidden, result-changing thresholds: the gate is one explicit option, default off, named in the docs.
- Non-bitwise numerics changes need long-run nonlinear statistics against master, adiabatic and kinetic, on production grids.
- Not decided yet: smooth gate at $\alpha_c$ 30 to 100 (about 1 %, touches every EM mode a little) or step gate at 300 (1.5 %, leaves $k_\theta\rho \gtrsim 0.2$ bitwise plain).

## 9. Open items, in suggested order

1. A kinetic EM nonlinear case where master survives, to compare the gates against master (candidates: the same Cyclone case at $\beta=0.001$; $\beta=0.005$ with a larger $k_\theta\rho_{min}$, e.g. `nky` 16; a finer parallel grid, 2x cost). Finish the two gated kinetic runs above.
2. Choose the form (step 300 or smooth 30 to 100).
3. If 1 % on deck 3559 is required: a correction that separates the modes in time (high-pass of the removed charge, one extra state variable). Untested.
4. base4_01 with its collisions ($Z_{eff}>1$ is not supported in gyaradax).
5. CUDA kernel for the term and multi-device support (it currently raises for sharded runs).
6. The GKW side: implement as a second pass in `calc_linear_terms` and report to the GKW maintainers.
7. The matched finite-$\theta_0$ restart against CGYRO and the convergence of the slow reference (see `docs/issue-201.md`).
8. The GKW importer drops `nlapar` and `beta` for some decks (seen on base4_01), and the import fixes for EM and CFL on branch `fix/gkw-import-em-cfl` are not merged.
9. Analytical check in the gate's region: inertial/kinetic Alfvén wave across $k_\perp d_e$, plain against both gates (being set up in `notebooks/analytical_extended.ipynb`, not on this branch).

## 10. Reproducing

```bash
git apply investigations/issue-201/patch/disp_par_neutral.patch     # on this branch or master 5a31042
pytest tests/unit/test_disp_par_neutral.py                         # on a free GPU or CPU

# linear eigen-solve of one config (fp64 Krylov-Schur on the RK4 step operator of the ky row)
# env: NALPHA = alpha_c with END NEND (unset = plain), NVARIANT=smooth for the smooth gate, NS / NKX / DISP_PAR overrides, KX0 / PX0 shift
NALPHA=100 NEND=4 python investigations/issue-201/scripts/gyx_eig_p4.py \
  investigations/issue-201/configs/waltz_linear/waltz_k0.25_b0.005/prod.yaml my_results.jsonl w025b5_a100
ACC=my_results.jsonl python investigations/issue-201/scripts/acc_table.py

# nonlinear run
GYX_CODE=$PWD PY=python bash investigations/issue-201/scripts/launch.sh 0 \
  investigations/issue-201/configs/cbc_em_nonlinear/cbc_em_kin_b005_smooth100.yaml runs/cbc_em_kin_b005_smooth100
```

The Waltz configs carry a `.units.json` sidecar with the CGYRO unit factors used by the eigen-solve script. `gyx_eig_p4.py` was used with JAX x64 on B300 and H100; one solve takes minutes to an hour.

## 11. Material that stays on the lab hosts

Not in this repository because it is restricted (issue bundle, UKAEA-related notes) or large. Paths on the lab file system:

- The recovered issue bundle is under `/system/user/publicwork/volkmann/issue201.I85RLq/issue201` on the lab hosts. Access to restricted inputs is managed separately.
- gyaradax configs of the restricted decks: `/restricteddata/ukaea/gyrokinetics/cgyro/gyaradax_cgyro_em/gapcheck/gyx/cfg/E/d3559/g0.yaml`, `E/base4red/g0.yaml`, `C/i201_px0/g0.yaml`; CGYRO decks in `gapcheck/decks/`.
- Full measured log of the investigation, sections 1 to 19 with every number and its source file: `/restricteddata/ukaea/gyrokinetics/cgyro/gyaradax_cgyro_em/advisor/ISSUE201.md` (restricted; quotes UKAEA run notes). Advisor report `advisor/REPORT.md`, campaign results `gapcheck/RESULTS.md` (section 6), status `gapcheck/STATUS.md`.
- Write-up sources with the deck listings: `advisor/writeup/` (same directory).
- Nonlinear runs and their snapshots: `/restricteddata/ukaea/gyrokinetics/cgyro/gyaradax_cgyro_em/nl_skin/` (`runs/`, `logs/`, `STATUS.md`); F2 nonlinear runs: `gapcheck/gyx/nlk/`.
- The patch's source worktree: `/restricteddata/ukaea/gyrokinetics/cgyro/gyaradax_cgyro_em/worktrees/gyaradax-f1` (uncommitted).
- GKW issue-201 runs: `/system/user/galletti/git/gkw-issue201/run/`. Reference code builds (GENE 3.1, CGYRO, GKW) with READMEs: `/system/user/publicwork/galletti/gkcodes/`.

## 12. Files in this folder

| path | what |
|---|---|
| `README.md` | this handover |
| `MATH.md` | GKW, gyaradax and CGYRO equations in one notation, with the discretisation choices that matter here |
| `writeup/skin_gated_dissipation.pdf`, `.tex`, `fig_*.pdf` | the write-up (11 pages), restricted deck listings removed |
| `patch/disp_par_neutral.patch` | the skin gate and F1/F2 options in gyaradax, with docs and tests |
| `configs/waltz_linear/` | Waltz linear decks used in the acceptance solves (`prod` 32 per turn; `p16`/`p32` for $k_\theta\rho=0.05$), CGYRO source deck |
| `configs/kinetic_linear/` | the four kinetic validation decks as used in the linear checks |
| `configs/cbc_em_nonlinear/` | the six nonlinear EM configs of section 7 |
| `results/ACC.jsonl`, `results/acc_table.txt` | acceptance eigen-solves (145 records) and their table |
| `scripts/` | eigen-solve (`gyx_eig_p4.py`, `ks.py`), table (`acc_table.py`), nonlinear launcher and stop watcher |
