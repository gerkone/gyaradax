# gyaradax (GKW) and CGYRO in one notation (advisor, 2026-11-23)

Equations each code (GKW, gyaradax, CGYRO) actually advances, in one notation, with the discretisation choices that matter for issue 201. Written 2026-11-23/24 during the lab investigation; references to lab notes (CGYRO_NOTES, SYNTHESIS, SKEPTIC) point to files that stay on the lab hosts.

## 0. Sources and notation

Source keys:
- [gyx file:line]: gyaradax master 5a31042, `/system/user/galletti/git/gyaradax/gyaradax/...`.
- [cc file:line]: worktree gyaradax-cgyroconv (95da48b + uncommitted disp_par_conserve 4/5, end mask, order-6 stencils), `/system/user/publicwork/galletti/git/gyaradax-cgyroconv/gyaradax/...`.
- [gkw file:line]: GKW develop 785a215, `/system/user/galletti/git/gyaradax/gkw_ref/src/...` (identical to `git show 785a215:src/...` in gkw-issue201 for linear_terms, exp_integration, fields, mode, dist, velocitygrid, non_linear_terms, init, control).
- [cg file:line]: CGYRO 925525eb, `/system/user/publicwork/galletti/git/gacode-b300-cur/cgyro/src/...`; [cgpre file:line]: CGYRO 6357db30, `/system/user/publicwork/galletti/git/gacode-preupwind/cgyro/src/...`.
- [JCP eq. N, §]: Candy, Belli, Bravenec, J. Comput. Phys. 324 (2016) 73, OSTI accepted manuscript 1512805 (equation numbers of the manuscript; text extracted with pdftotext).

Notation (per species s; I drop s where it is clear): charge Z, mass m, temperature T, density n; Maxwellian F (GKW `fmaxwl`, CGYRO f0); parallel velocity v_par, magnetic moment mu (GKW) or pitch xi = v_par/v and speed v (CGYRO); gyro-averages J0 = J0(k_perp v_perp/Omega) and the B_par factor; fields phi, A (= A_par), B (= B_par).

- f: the perturbed gyrocentre distribution (carries the gyrocentre density and current; what both codes' Maxwell equations integrate once written in f).
- chi: generalised potential, chi = J0 phi - v_par J0 A + (B_par term). GKW: chi = <phi> + (2 mu T/Z) <B> - 2 v_R v_par <A> [gyx docs/numerics/index.md:38-40]; CGYRO: Psi = sum_i jvec_c(i) field(i), jvec_c = (J0, -xi v vth J0, B_par factor) [cg cgyro_init_arrays.F90:78-93].
- g: the mixed variable, g = f + (Z/T) v_par J0 A F (normalisation factors aside). GKW/gyaradax evolve g [gyx docs/numerics/index.md:17; gyx operators/fields.py:15-37, g2f_factor = -2 Z v_R v_par J0 F/T at gyx operators/precompute.py:612-614; gkw linear_terms.f90:4557-4619 g2f_correct]. CGYRO evolves h = H - (Z/T) Psi F (stored as h/F) [cg cgyro_step_gk.f90:10-22; JCP eq. 16, 19, 52]. With Psi = J0 phi - v_par J0 A + ..., h = f + (Z/T) v_par J0 A F: CGYRO's h is GKW's g (electromagnetic) and f (electrostatic).
- H: the non-adiabatic part, H = h + (Z/T) Psi F = f + (Z/T)(J0 phi + B_par term) F [cg cgyro_rhs.F90:105-108; JCP eq. 19].

Normalisations differ (GKW: R_ref, v_thref = sqrt(2T/m); CGYRO: a, c_s, B_unit). Conversion factors are in `notes/CGYRO_NOTES.md` §3; I re-derived gamma_GKW = (RMAJ/sqrt 2) gamma_C from the time units and found no error there. Below I write equations in one dimensional-free form and keep code factors out.

## 1. The gyrokinetic equation both codes solve

In the common notation, linear part, one species:

dg/dt = - v_par b.grad H_s - (mirror) - i omega_d H - i omega_* chi F + N[chi, g] + C + D_par + D_perp,   H_s = f + (Z/T) J0 phi F (+ B_par term)

How each code groups it:

| piece | GKW / gyaradax | CGYRO |
|---|---|---|
| time-stepped variable | g [gyx rk4.py:401-487] | h [cg cgyro_step_gk.f90:24-74] |
| streaming | Term I on f plus Term VII on (Z/T) F d(J0 phi)/ds plus Term X for B_par; each with its own stencil (code differs: Term X reuses Term VII's table s_t7, _jax.py:478) [gyx backends/_jax.py:437-441, 469-478; gkw linear_terms.f90:415-561 (I), 2721-3011 (VII)] | one centred stencil on H [cg cgyro_rhs.F90:258-286 (cderiv on cap_h at 278-279); JCP eq. 81, 83] |
| mirror (trapping) | Term IV on f, explicit, 4th-order central difference in v_par with zero outside the grid [gyx _jax.py:443-449; docs/numerics/index.md:86] | (1 - xi^2) d/dxi on H with the Legendre collocation matrix, implicit inside the collision step by default [cg cgyro_rhs.F90:489-570 explicit path only with EXPLICIT_TRAP_FLAG 1; JCP §3.9 eq. 125, 128] |
| magnetic drift | Term II on f plus Term VIII on (Z/T) J0 phi F, i.e. on H (Term VIII uses phi only) [gyx _jax.py:451-466] | omega_cap_h on H [cg cgyro_init_arrays.F90:355-380] |
| gradient drive | Term V on chi (incl. A) [gyx _jax.py:456-457] | omega_s on Psi [cg cgyro_init_arrays.F90:390-396; cgyro_rhs.F90:117-118] |
| nonlinearity | [chi, g], 3/2 dealiased pseudo-spectral [gyx docs/numerics/index.md:92-94] | [Psi, h] (bracket of f0 + h with Psi) [JCP eq. 16, 121-122] |
| collisions | on f, explicit stencil [gyx docs/numerics/index.md:113-116] | on H, Crank-Nicolson with field update [cg cgyro_step_collision.F90:361-388; JCP eq. 128-130] |

Both codes apply the linear operator to f, not to g: gyaradax forms f = g_to_f(g) before the linear RHS [gyx backends/ops.py:161-189; rk4.py:447-457], as GKW does (fdis_tmp = f via matg2f) [gkw exp_integration.F90:800-812]. CGYRO's streaming acts on H and its drifts on H; H contains no A part, so the A_par time derivative never appears explicitly in either code.

## 2. Field equations (as solved)

GKW / gyaradax, from g at every RK stage [gyx operators/fields.py:89-130]:
- Ampere: [k_perp^2 + beta sum_s Z^2 n/m Gamma_num] A = beta sum_s Z n v_R int J0 v_par g, with Gamma_num = sum 2 B intmu intvp J0^2 v_par^2 F on the actual grid [gyx operators/ampere.py:13-46, 99-129; gkw linear_terms.f90:3198-3254 ampere_int, 3704-3796 ampere_dia, the Feb 2023 comment at 3773-3776]. In f this is exactly k_perp^2 A = beta sum Z n v_R int J0 v_par f (the discrete skin term cancels the discrete g-to-f term).
- Poisson (quasineutrality): sum_s Z n int J0 f = - sum_s (Z^2 n/T)(Gamma_0(b) - 1) phi with the analytic Gamma_0 = I0(b) e^-b [gyx operators/poisson.py:303-337, 371-372]; computed from f, which equals the result from g by v_par parity on a symmetric grid [gyx fields.py:62-70].
- B_par: closed-form 2x2 coupled with Poisson [gyx operators/ampere.py:162-240; docs/numerics/index.md:103].
- Adiabatic electrons respond to phi - <phi>_FS for the zonal mode [gyx docs/numerics/index.md:100].

CGYRO, from h at every RK stage (explicit step) [cg cgyro_field.F90:337-493 (cgyro_field_c), 784-1005 (coefficients)]:
- Poisson: phi = [k_perp^2 lambda_D^2 n_e/T_e + sum_v w (Z^2 n/T)(1 - J0^2)]^-1 sum_v w Z n J0 h (gcoef(1); discrete polarisation with the moment weights).
- Ampere: A = [-2 k_perp^2 rho^2 n_e T_e/beta_e,unit - sum_v w (Z^2 n/T)(J0 v_par)^2]^-1 sum_v w Z n (-v_par J0) h (gcoef(2); discrete skin term with the moment weights).
- B_par: 2x2 with Poisson (pb11..pb22) for N_FIELD 3.
- The H-form (no polarisation, no skin term) is used only inside the implicit collision step [cg cgyro_field.F90:248-303].
- Zonal adiabatic electrons: the theta-coupled matrix xzf subtracts the flux-surface average [cg cgyro_field.F90:1013-1030].

Same equations; the only discrete difference is the polarisation term (analytic Gamma_0 in GKW/gyaradax, discrete 1 - J0^2 sum in CGYRO). Both make the Ampere skin term the discrete partner of their g-to-f (h-to-H) map.

## 3. The Ampere cancellation and which variable is dissipated

In the g (= h) form, Ampere's source is the current of g, which at low k_perp and finite beta is almost entirely cancelled by the skin term: k_perp^2 A = beta [j(g) - S A], S = sum Z^2 n/m Gamma_num. The physical current j(f) = j(g) - S A is a small difference of two large numbers when k_perp^2 << beta S (at kthrho 0.07, beta 0.0035: k_perp^2 about 0.005 against beta S about 13). Two consequences:
- The quadrature of S must equal the quadrature of the g-to-f term exactly. GKW (since Feb 2023), gyaradax and CGYRO all do this [gkw linear_terms.f90:3766-3776; gyx ampere.py:13-46; cg cgyro_field.F90:842-865, 878-887].
- A dissipation that acts on g (= h) damps j(g), and through Ampere damps A at a rate amplified by S/k_perp^2. CGYRO's paper states that dissipating h "performs poorly at moderate to high beta" and dissipates g_a = h_a - z (v_par/c_s) A instead [JCP §3.4 eq. 82, citing Dannert and Jenko 2004]; the code uses g = h + (Z/T) jvec_c(2) A = f [cg cgyro_rhs.F90:270-276]. GKW/gyaradax dissipate f (the linear operator acts on f, §1). So both codes dissipate the same object, the gyrocentre f. The instability agent's "g" variant (dissipation on g in gyaradax) is worse (6.65 vs 2.61 on the issue-201 deck, `ISSUE201.md` §6), consistent with this.

## 4. Parallel grid and streaming discretisation

GKW / gyaradax:
- s grid: cell centred, uniform, ns points over 2 nperiod - 1 turns [gyx geometry/grids.py:16-27]; no point at s = 0 for even ns.
- Term I: upar D1[f]/Delta_s with upar = -ffun vthrat v_par, ffun(s) = b.grad s evaluated at the output point [gyx precompute.py:383; gkw linear_terms.f90:479]. D1 interior is the 4th-order centred (1, -8, 0, 8, -1)/12; the two points nearest an open end use GKW's one-sided or truncated rows chosen by the sign of upar [gyx operators/stencils.py:26-48; precompute.py:205-232; gkw linear_terms.f90:4251-4548 differential_scheme].
- Term VII: -(Z/T) ffun vthrat v_par F D1[J0 phi]/Delta_s with its own upwind-class table and no dissipation [gyx precompute.py:400, 220-228; _jax.py:437-441].
- Any centred D1 has the symbol 2i sum c_k sin(k x), zero at the s-Nyquist wavenumber (s-banding study); the streaming and Term VII do not act on the odd-even mode.

CGYRO:
- theta grid: N_THETA points from -pi, uniform in theta_eq; with CONSTANT_STREAM_FLAG 1 (default) b.grad theta_eq is constant [cg cgyro_equilibrium.F90:79-118]; theta = 0 is a grid point for even N_THETA.
- Streaming: -omega_stream(theta) xi v cderiv[H], cderiv the (2 NUP_THETA + 1)-point centred derivative, 6th order for NUP_THETA 3 (default) [cg cgyro_rhs.F90:246-286; cgyro_init_arrays.F90:472-537 advect_schemes; JCP eq. 83-84]. omega_stream = sqrt(2) vth/(q R g_theta) [cg cgyro_equilibrium.F90:225]. xi v is constant along theta (velocity grid is theta-independent).
- Same Nyquist blindness of the centred stencil.

## 5. Parallel dissipation (the operator at the centre of issue 201)

GKW / gyaradax (plain):
D_GKW[f](s) = disp_par c(v) ffun(s) vthrat (1/Delta_s) sum_k d4_k f(s + k),   d4 = (-1, 4, -6, 4, -1)/12 in the interior,
with c = |v_par| for idisp 1 and c = v_rms of the v_par grid for idisp 2 (default) [gyx precompute.py:385-397; gkw linear_terms.f90:571-694, idisp at 638-647]. GKW applies it to the zonal mode with order_of_the_zf_scheme [gkw linear_terms.f90:658-665]. The end rows: no dissipation on the two outflow-end points, truncated row on the second point of the inflow end, and (12, -24)/12 (a zero-ghost second difference) on the inflow end point [gyx stencils.py:50-72]. Interior damping rate of the odd-even mode: (16/12) disp_par c ffun vthrat/Delta_s, about 57 times larger for electrons than for ions at idisp 2 (vthrat_e/vthrat_i = sqrt(m_i T_e/(m_e T_i))). Strength scales as Delta_s^3 (4th derivative).
Options in gyaradax: master disp_par_conserve 1 (input-side, current only), 2 (input-side, density and current), 3 (input-side, |v_par|-weighted density); kycut gate [gyx precompute.py:506-552]; cgyroconv adds 4 (CGYRO before eb8884c1: input-side J0 density of the stencil input, |v_par| weight at idisp 1), 5 (CGYRO after eb8884c1: output-side J0 density and, with A_par, J0 v_par current), disp_par_conserve_end, and order-6 interior rows [cc operators/precompute.py:205-227 (order 6), 246-319 (_disp_conserve_tables); cc backends/_jax.py:81-120 (parallel_stencil, disp_post_term)].

CGYRO since eb8884c1 (current):
D_C[f](theta) = - UP_THETA |omega_stream(theta)| (I - P_n - P_j) S[|xi| v f/F],
S = uderiv, the (2 NUP_THETA + 1)-point filter, 6th derivative for NUP_THETA 3 [cg cgyro_rhs.F90:258-287 (operand |xi| v (h + (Z/T) jvec_c(2) A) = |xi| v f/F at 270-276), 305-399 (moments), 401-464 (projection and scaling, 452); cgyro_init_arrays.F90:203-275 (upfac_num)]. P_n x = J0 <J0 x>/<J0^2>, P_j x = J0 v_par <J0 v_par x>/<J0^2 v_par^2> per species and (theta, kx, n), <> the w_exi sum; P_j only for N_FIELD > 1; RES_WEIGHT_POWER is read and ignored [cg cgyro_init_arrays.F90:208-209]. Output density and current are zero at every theta point. Strength scales as Delta_theta^5.

CGYRO before eb8884c1 (2023 to 2026-08-05):
D_old[f] = - UP_THETA |omega_stream| S[ |xi| v f/F - J0 (|xi| v)^p <J0 |xi| v f/F> / <J0^2 (|xi| v)^p> ], p = RES_WEIGHT_POWER = 1 [cgpre cgyro_upwind.F90:1-16, 54-60, 146-147; cgpre cgyro_init_arrays.F90:203-262]. The stencil input has zero J0 density; the output density is the commutator residual sum_d S_d sum_v w [J0(theta_i) - J0(theta_{i+d})] x(theta_{i+d}), negligible for electrons (k_perp rho_e << 1). The current is not projected.

Paper: Eq. 88 projects the J0 density of |v_par| g with weights J0 |v_par| on the input [JCP eq. 88, §3.4], without J0 on A in g [JCP eq. 82]; "By conservative, we mean conservation of gyrocenter number with respect to the numerical dissipation" [JCP §3.4]. The paper's argument for it is accuracy at k_perp rho_s < 0.1 and finite beta; it does not give an energy argument.

## 6. Parallel boundary conditions and chains

GKW / gyaradax: a ky != 0 mode leaving the s domain continues as (kx + Delta kx, ky), Delta kx proportional to shat ky; chains of linked kx modes; a chain that leaves the kx grid has an open end (ixplus/ixminus = -1) (code differs: GKW marks it 0, mode.f90:778, 795), taps beyond the end read zero, the end rows of §4/§5 apply; ky = 0 is periodic per kx [gyx geometry/topology.py:18-138; gkw mode.f90:757-820 (kgrid connections), linear_terms.f90:511 (vpar_grd_df calls connect_parallel)]. GKW resets 'zero' to 'open' [gkw control.f90:745-750].

CGYRO: crossing theta = +-pi moves the radial index to modulo(ir +- n BOX_SIZE sign(q s), N_RADIAL) with the phase exp(+-2 pi i n k_theta_base rmin) [cg cgyro_rhs.F90:223-231, 258-268; cgyro_init_arrays.F90:324-330]. Every chain is a closed loop that wraps around the kx grid. The paper derives zero end conditions from the periodicity condition and then makes p cyclic, "quasi-periodic boundary conditions" [JCP eq. 78-80, §3.3]. The radial spectral upwind (§7) damps the high-|p| part of every loop, so in practice the loop is cut by an absorbing layer at the kx edge [JCP §3.5: "an effective outgoing-wave boundary condition for ballooning modes"].

Consequence: when n BOX_SIZE is comparable to N_RADIAL, a CGYRO mode links to aliased kx (n BOX_SIZE >= N_RADIAL links p to itself after one turn), while the gyaradax mode is a short open chain. Both differ from the physical infinite chain. The input06 box (BOX_SIZE 7, N_RADIAL 64: n >= 10 have no physical neighbour) and the R3 box are in this regime for most toroidal modes (CGYRO_NOTES "input06 box").

## 7. Radial and binormal dissipation

gyaradax: hyper = -(disp_x (kx/kx_max)^4 + disp_y (ky/ky_max)^4) times f, constant in velocity and s [gyx precompute.py:158-170; _jax.py:488]. GKW applies its perpendicular hyperdissipation only with mode_box and zeroes disp_x/disp_y otherwise [gkw linear_terms.f90:283-287; mode.f90:368-372].

CGYRO: -(N_RADIAL/L) UP_RADIAL P^(n)(2 pi (p + PX0)/N_RADIAL) (|omega_rdrift| e (1 + xi^2) + |omega_cdrift_r xi| v + ...) times h, and the same in the binormal direction with UP_ALPHA (default 0) [cg cgyro_init_arrays.F90:359-388, 434-470; JCP eq. 89-95]. It is velocity and theta dependent (proportional to the local radial drift speed), acts on every n including n = 0 (the zonal p modes), and acts on h, which in EM runs contains the A_par part (h = g, not f).

## 8. Velocity space

gyaradax: v_par uniform cell centres on [-vpar_max, vpar_max]; mu from v_perp uniform cells on (0, vpar_max), weights 2 pi v_perp Delta v_perp (midpoint) [gyx geometry/grids.py:29-35]; F(s, v_par, mu) depends on s through B(s). GKW: same v_par grid [gkw velocitygrid.f90:192-198], mu uniform in mu or in v_perp [gkw velocitygrid.f90:170-185]. v_par dissipation disp_vp on the mirror term with idisp-dependent coefficient [gkw linear_terms.f90:838-951].

CGYRO: xi on Gauss-Legendre nodes (weights halved, N_XI even), energy on Gauss nodes in v for weight v^ALPHA_POLY e^-v^2 on [0, sqrt(E_MAX)] with the Maxwellian tail lumped on the last node; w_exi = w_e w_xi carries the Maxwellian; the grid does not depend on theta [cg cgyro_init_manager.F90:75-126; shared/math/half_hermite.f90:784-865 per CGYRO_NOTES]. No velocity-space dissipation.

## 9. Time integration

gyaradax: explicit RK4 with a field solve per stage, dt from GKW's linear estimate (0.95/t_max) [gyx rk4.py:445-487; solver/cfl.py:88; docs/numerics/index.md:118-127]. GKW: RK4 explicit (EXP) and implicit options; METHOD IMP forces idisp 1 [gkw linear_terms.f90:247].

CGYRO: collisionless part by RK4 (DELTA_T_METHOD 0) or Cash-Karp 5(4) sub-steps inside DELTA_T (1), field solve per stage [cg cgyro_step_gk.f90:24-74; cgyro_kernel.F90:50-56]; then one Crank-Nicolson step for collisions and trapping with the fields written through H [cg cgyro_kernel.F90:69-70; cgyro_step_collision.F90:361-388; JCP eq. 123-130]. One RK step then one collision step is Lie splitting; the paper's "second order except for the first and last timesteps" [JCP §3.9] is the standard equivalence of repeated Lie steps with Strang splitting up to an O(dt) end transformation, so the paper's statement and CGYRO_NOTES' "formally first order" describe the same scheme from two sides. For linear eigenvalues from an initial-value run, the splitting error is an O(dt [A, B]) shift of the eigenvalue; it vanishes without collisions and without implicit trapping effects.

## 10. Discrete free energy (derived here, checked numerically)

For gyaradax's discrete system (symmetric v_par grid, Ampere and Poisson as in §2), define per ky chain
W = <f, f>/2 + Re(phi* rho_f)/2 + Re(A* jt_f)/2,
with <a, b> = sum_s sum_{v, mu, s-points} (bn intmu intvp Delta_s n) (T/F) a* b, rho_f = sum_s Z n int J0 f (= P phi by Poisson, P > 0), jt_f = 2 sum_s Z n v_R int J0 v_par f (= 2 k_perp^2 A/beta by Ampere in f). Each term is non-negative. Then for any dg/dt
dW/dt = Re<H, dg/dt>,  H = f + (Z/T) J0 phi F.
Proof: dW/dt = Re<f, df/dt> + P Re(phi* dphi/dt) + (2 k_perp^2/beta) Re(A* dA/dt); df/dt = dg/dt - (2Z/T) v_R v_par J0 F dA/dt; the A terms cancel because <f, (2Z/T) v_R v_par J0 F> = jt_f*; P dphi/dt = d rho_f/dt = d rho_g/dt by parity. Numerically: the sum of the per-term contributions Re<H, term>/(2W) equals Re(lambda) of each eigenvector to 1e-11 (`advisor/results/dense2.jsonl`).

For a dissipation term D[f] in dg/dt:
Re<H, D f> = Re<f, D f> + Re(phi* rho_D),  rho_D = sum_s Z n int J0 D f.
- The first term is the intended damping (non-positive when D is symmetric negative semi-definite in <,>; GKW's interior D4 nearly is, its open-end rows are not).
- The second term is work done by the charge that the dissipation creates. It has no sign.
- The current moment of D does not enter: its effect on A and its effect on f cancel inside W. The same holds for CGYRO's h equation (H here is CGYRO's H without the A part, and the identity uses only the field equations).
- With B_par, H carries the B_par term and W the B_par energy, so the same algebra adds Re(B_par* M_D), M_D the moment of D f that sources B_par (the perpendicular-pressure moment with the J1 gyro-average). Derived, not checked numerically. Neither CGYRO's projection nor any gyaradax mode removes M_D.
- With an output-side projection that removes the J0 density per species (CGYRO new, gyaradax mode 5, "outn"), or only the total charge (one constraint per point), rho_D = 0 and the dissipation can no longer do field work.
- With an input-side projection (CGYRO old, mode 4) rho_D is the commutator residual of §5, plus, in gyaradax, the truncated end rows near open chain ends.
- The dissipation-free part of the scheme is not conservative either: the separately discretised streaming (centred in s, coefficient ffun(s)) and mirror (centred in v_par) terms do not cancel in <,> at the grid scale, and on my test grids the undriven, dissipation-free operator has hundreds of weakly growing grid-scale electron modes (max 0.24-0.30 R/v_th on 16-24 points per turn, 8 x 2 velocity points; ISSUE201.md §8). A conservative discretisation would need the s and v_par differences to be adjoint in the weight bn/F, which neither GKW nor CGYRO is shown to have.

## 11. Where each code's dissipation acts

| operator | GKW / gyaradax | CGYRO |
|---|---|---|
| parallel | f; 4th derivative times Delta_s^3; coefficient v_rms (idisp 2) or |v_par| (idisp 1); no projection by default; open-end rows | f/F times |v_par|; 6th derivative times Delta_theta^5; projected (density, + current since eb8884c1); closed loops |
| v_par | f (disp_vp, mirror-scaled) | none |
| radial | f, disp_x (kx/kx_max)^4, constant | h, spectral upwind of order 6, coefficient |omega_rdrift|, all n |
| binormal | f, disp_y (ky/ky_max)^4 | h, UP_ALPHA default 0 |
| collisions | f, explicit | H, implicit (Crank-Nicolson) |
| time | RK4 (no dissipation) | RK4 or Cash-Karp; implicit trapping |

## 12. Where earlier notes differ from what I read

- CGYRO_NOTES §1.2 says the paper's upwind acts on "GKW's f" and the code on "f/F0 with J0 on A": correct. The statement "In EM runs this [radial upwind] acts on g, not f" (§2.4) is correct and matters: CGYRO's radial dissipation is not on the same object as its parallel dissipation.
- CGYRO_NOTES §2.5/§7.2 "paper second order vs code first order": not a disagreement (§9).
- CGYRO_NOTES §5.1 "In EM runs the old scheme therefore dissipates parallel current numerically and the new one does not": correct as a statement about the operator; the free-energy argument (§10) shows the current moment does not by itself feed or drain W, so the input06 Q_e rise after eb8884c1 is a change of dissipation strength on the current-carrying part of f, not the removal of a spurious source.
- The eb8884c1 commit comment (quoted in CGYRO_NOTES §2.3) calls the residual of the input projection "the spurious grid-scale source behind the slow finite-beta instability" and says restoring current conservation removes "the spurious A_par drive". The first part agrees with §10 (density residual feeds W); the second does not follow from the free-energy balance.
