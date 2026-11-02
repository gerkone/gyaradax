# Numerics and GKW notes

gyaradax discretises the local δf gyrokinetic equation the way GKW does
([Peeters et al. 2009](https://doi.org/10.1016/j.cpc.2009.07.001); the GKW manual,
chapters "Practise" and "Implementation"). This page summarises the equations as
they are solved, the numerical choices, and where gyaradax departs from GKW. Code
references point to the functions that implement each piece.

## Normalisation and coordinates

Lengths are normalised to $R_\mathrm{ref}$, velocities to
$v_{\mathrm{th,ref}} = \sqrt{2T_\mathrm{ref}/m_\mathrm{ref}}$, time to
$R_\mathrm{ref}/v_{\mathrm{th,ref}}$, and the fields carry the usual powers of
$\rho_* = \rho_\mathrm{ref}/R_\mathrm{ref}$
($\phi \sim \rho_* T_\mathrm{ref}/e$, $A_\parallel \sim \rho_*^2 B_\mathrm{ref} R_\mathrm{ref}$,
$B_{1\parallel} \sim \rho_* B_\mathrm{ref}$). Species have mass $m_R$, charge $Z$,
temperature $T_R$, density $n_R$ relative to the reference, and thermal-velocity
ratio $v_R = \sqrt{T_R/m_R}$ (`mas`, `signz`, `tmp`, `de`, `vthrat`).

The flux tube uses field-aligned coordinates $(\psi, \zeta, s)$: $s$ along the
field line, $\psi$ radial, $\zeta$ binormal. The perpendicular directions are
Fourier modes $(k_x, k_y)$, the parallel one a finite-difference grid. The
geometry enters through the metric, $B(s)$ and the drift tensors
$\mathcal{D}, \mathcal{E}, \mathcal{F}, \mathcal{G}, \mathcal{H}$, computed
analytically for s-α, circular ([Lapillonne et al. 2009](https://doi.org/10.1063/1.3096710))
and Miller ([Miller et al. 1998](https://doi.org/10.1063/1.872666)) surfaces, or read
from a GKW `geom.dat` ({mod}`gyaradax.geometry`).

## The equation

Each species evolves $g$, which equals the perturbed distribution $f$ in
electrostatic runs and the mixed variable
$\hat g = \hat f + \tfrac{2Z}{T_R} v_R v_\parallel \langle\hat A_\parallel\rangle F_M$
with $A_\parallel$ (GKW `g2f_correct`; {func}`~gyaradax.operators.fields.g_to_f`).
Evolving $g$ removes $\partial A_\parallel/\partial t$ from the equation, so it
stays explicit. With GKW's term numbering,

$$
\frac{\partial \hat g}{\partial t} = \mathrm{I} + \mathrm{II} + \mathrm{III} + \mathrm{IV}
+ \mathrm{V} + \mathrm{VII} + \mathrm{VIII} + \mathrm{X} + \mathrm{XI} + D(\hat f) + C(\hat f),
$$

| Term | Physics | Normalised form (local limit) |
|---|---|---|
| I | parallel streaming | $-v_R v_\parallel \mathcal{F}\,\partial_s \hat f$ |
| II | magnetic drift | $-\tfrac{i}{Z}\big[T_R E_D \mathcal{D}^\alpha + T_R v_\parallel^2\beta'\mathcal{E}^{\psi\alpha} + 2 m_R v_R v_\parallel \Omega \mathcal{H}^\alpha\big] k_\alpha \hat f$ |
| III | ExB nonlinearity | $\mathcal{E}^{\psi\zeta}\big(\partial_\zeta\chi\,\partial_\psi g - \partial_\zeta g\,\partial_\psi\chi\big)$ |
| IV | mirror force | $v_R\,\mu B\,\mathcal{G}\,\partial_{v_\parallel}\hat f$ |
| V | gradient drive | $i k_\alpha \hat\chi\,\mathcal{E}^{\alpha\psi}\big[\tfrac{1}{L_n} + E_T \tfrac{1}{L_T} + \tfrac{2 v_\parallel}{v_R}\tfrac{R B_t}{B} u'\big] F_M$ |
| VII | parallel field drive | $-\tfrac{Z}{T_R} v_R v_\parallel \mathcal{F}\,\partial_s \langle\hat\phi\rangle F_M$ |
| VIII | drift field drive | $-i\big[E_D\mathcal{D}^\alpha + \beta' v_\parallel^2 \mathcal{E}^{\psi\alpha} + \tfrac{2 m_R v_R}{T_R} v_\parallel\Omega\mathcal{H}^\alpha\big] k_\alpha \langle\hat\phi\rangle F_M$ |
| X | $B_\parallel$ mirror | $-2 v_R v_\parallel \mu F_M \mathcal{F}\,\partial_s \langle\hat B_{1\parallel}\rangle$ |
| XI | $B_\parallel$ drift | $-\tfrac{i}{Z} F_M\, 2 T_R \mu\big(E_D\mathcal{D}^\beta + v_\parallel^2\beta'\mathcal{E}^{\psi\beta}\big)k_\beta \langle\hat B_{1\parallel}\rangle$ |

with $E_D = v_\parallel^2 + \mu B$, $E_T = v_\parallel^2 + 2\mu B - 3/2$,
$\langle\cdot\rangle = J_0(k_\perp\rho)$ the gyro-average and the generalised
potential

$$
\hat\chi = \langle\hat\phi\rangle + \frac{2\mu T_R}{Z}\langle\hat B_{1\parallel}\rangle - 2 v_R v_\parallel \langle\hat A_\parallel\rangle .
$$

The linear terms act on the physical $\hat f$ (GKW converts $g \to f$ before the
linear operator), the nonlinearity on $g$. Term VI (neoclassical) is not
included. Term VIII and Term VII use $\langle\phi\rangle$ only, not $\chi$, as in
GKW. $\Omega$ is the Coriolis rotation `vcor`; centrifugal terms are not
implemented. $D$ is the numerical dissipation and $C$ the collision operator.

## Phase-space grid

| Coordinate | Grid | Integration weight |
|---|---|---|
| $v_\parallel$ | `nvpar` cell centres, uniform on $[-v_\mathrm{max}, v_\mathrm{max}]$ | $\Delta v_\parallel$ |
| $\mu$ | `nmu` points, uniform in $v_\perp$ on $(0, v_\mathrm{max})$, $\mu = v_\perp^2/2$ | $2\pi v_\perp \Delta v_\perp$ |
| $s$ | `ns` cell centres over $2\,$`nperiod`$-1$ poloidal turns | $\Delta s$ |
| $k_x$ | `nkx` modes centred on 0, spacing from the shear | spectral |
| $k_y$ | `nky` modes $0 \le k_y \le k_{y,\max}$ (half spectrum) | Parseval factor 2 for $k_y \neq 0$ |

Arrays are dense: `(nvpar, nmu, ns, nkx, nky)` for adiabatic electrons and
`(nsp, nvpar, nmu, ns, nkx, nky)` with kinetic electrons. All static coefficients
(Bessel functions, Maxwellians, drifts, stencil tables, field-solve weights) are
built once in {func}`~gyaradax.operators.precompute.linear_precompute`.

## Parallel derivatives and boundary conditions

Terms I and VII use fourth-order finite differences along $s$
(GKW `differential_scheme`, {mod}`gyaradax.operators.stencils`). Streaming carries
a fourth-derivative upwind dissipation scaled by $|v|$ and $\Delta s^3$, so that
the coefficient `disp_par` of order one damps grid-scale oscillations:

$$
v\,\partial_s g \;\to\; v_i\,\frac{g_{i-2} - 8g_{i-1} + 8g_{i+1} - g_{i+2}}{12\Delta s}
\;-\; D\,|v_i|\,\frac{-g_{i-2} + 4g_{i-1} - 6g_i + 4g_{i+1} - g_{i+2}}{12\Delta s}.
$$

Term VII has no dissipation. At the ends of a connected field line the stencils
drop to the one-sided second-order (boundary cell) and third-order (next cell)
forms of GKW's open boundary condition, upwinded by the sign of the advection
velocity, so the solution flows off the grid without reflections. Each grid
point has a stencil class (`pos_par_grid_class`, $-2 \dots 2$) for its distance
to an open end, and the stencil coefficients are stored per class
(`s_upar_tab`, `s_t7_tab`, `s_disp_par_tab`), not as full 6D arrays.

After one poloidal turn a mode $(k_x, k_y)$ continues as
$(k_x + \Delta k_x, k_y)$ with $\Delta k_x \propto 2\pi\hat s\, k_y$; modes
`ikxspace` apart on the kx grid are therefore connected into chains
({mod}`gyaradax.geometry.topology`). The parallel stencil follows the chain across
the ends of the $s$ grid; the zonal mode $k_y = 0$ is periodic in $s$. A chain
leaving the kx grid is an open end.

## Velocity derivatives

The mirror term IV uses the fourth-order central difference in $v_\parallel$
with zero values outside the grid, and a fourth-derivative dissipation with
coefficient `disp_vp` (`idisp` selects GKW's velocity scaling). Perpendicular
hyper-dissipation is spectral, `disp_x` $(k_x/k_{x,\max})^4$ + `disp_y`
$(k_y/k_{y,\max})^4$ (second order for negative coefficients).

`disp_par_conserve` (off by default, not in GKW) is a conservative variant of
the parallel dissipation in the style of CGYRO
([Candy et al. 2016](https://doi.org/10.1016/j.jcp.2016.07.039)): it removes from
the dissipated distribution the components that source the field equations,
which suppresses a low-$k_y$ numerical instability of electromagnetic runs
reported for GKW (GKW issue 201).

## Nonlinear term

Term III is pseudospectral, as in GKW `non_linear_terms.F90`. The derivatives
$ik_x\hat\chi$, $ik_y\hat\chi$, $ik_x\hat g$, $ik_y\hat g$ are transformed to a
zero-padded real-space grid of `mrad` x `mphi` points (3/2 rule, lengths as in
{doc}`../performance/index`), multiplied and transformed back, and the
retained modes are kept. Two implementation details differ from GKW without
changing the result beyond round-off:

- the CUDA backend (and the JAX backend with `use_z2z`) packs the two
  derivatives of one field into one complex inverse FFT
  ($Z = \partial_x u + i\,\partial_y u$), with a Hermitian symmetrisation of the
  $k_y = 0$ column, so each stage needs two inverse transforms instead of four;
- with `mixed_precision` (the default) the transforms run in FP32 while the
  linear terms, the fields and the accumulated RHS stay FP64.

The potential $\chi$ is separable in $v_\parallel$
($\chi = A(\mu, s) + v_\parallel B(\mu, s)$), so only $2\,n_\mu n_s$ potential
planes per species are transformed.

## Field equations

The fields are solved at every RK stage, pointwise in $(s, k_x, k_y)$:

- **Quasineutrality.** $\sum_s Z_s n_s \int J_0 \hat f_s\,\mathrm{d}^3v
  = -\sum_s \tfrac{Z_s^2 n_s}{T_s}(\Gamma_0(b_s) - 1)\,\hat\phi$ (+ adiabatic
  electrons), with $\Gamma_0(b) = I_0(b)e^{-b}$. Adiabatic electrons respond to
  $\phi - \langle\phi\rangle_\mathrm{FS}$, which adds a flux-surface-average
  correction to the zonal mode (GKW `poisson_zf`). With kinetic electrons all
  species enter the sums and the denominator of the $(k_x, k_y) = (0, 0)$ mode is
  set to one, as in GKW ({mod}`gyaradax.operators.poisson`).
- **Ampère's law** for $A_\parallel$: the parallel current of $g$ divided by
  $k_\perp^2 + \beta\sum_s \tfrac{Z_s^2 n_s}{m_s}\Gamma_\mathrm{num}$, where, as in
  GKW `ampere_dia`, $\Gamma_\mathrm{num}$ is the discrete velocity moment
  $\sum 2B J_0^2 v_\parallel^2 F_M$ on the actual grid rather than the analytic
  $\Gamma_0$; it then cancels the discrete g-to-f correction exactly
  ({mod}`gyaradax.operators.ampere`).
- **$B_\parallel$** (perpendicular Ampère) is coupled to quasineutrality through
  $\Gamma_0 - \Gamma_1$; the 2x2 system is solved in closed form at every point,
  with the modified Bessel function $\hat J_1 = 2J_1(x)/x$ for the gyro-average.

## Collisions

The linearised Fokker-Planck operator of GKW (`collision_differential_numu`,
GKW manual chapter "Collisions"): pitch-angle scattering, energy scattering and
friction between every pair of species, in the conservative flux form on the
$(v_\parallel, v_\perp)$ grid. Each point gets a 9-point stencil, precomputed in
{func}`~gyaradax.operators.collisions.precompute_collisions`; at $\mu = 0$ the
ghost point is the mirror image of the first $v_\perp$ cell, as in GKW. Optional
momentum and energy conservation follow GKW's `cons_type = 'Xu'` on the
self-collision part. Frequencies come from `coll_freq` or from Coulomb
logarithms at the reference values `rref`, `nref`, `tref`.

## Time integration

Explicit fourth-order Runge-Kutta (GKW `rk4`, {func}`~gyaradax.solver.rk4.gkstep_single`),
with each stage a full field solve and RHS evaluation, and the steps of a block
fused into one `jax.lax.scan`.

- **Timestep.** The linear bound follows GKW's von Neumann estimate
  (`get_estimated_timestep`): $\Delta t = 0.95/t_\max$ with
  $t_\max = \max(t_1/2.4,\; t_2,\; t_4/2.4,\; 40)$, where $t_1$ comes from
  streaming, trapping and, with kinetic electrons, the field (electrostatic or
  shear-Alfvén) frequency, $t_2$ from the collision stencil and $t_4$ from the
  fourth-order dissipation. A fixed `dt` larger than this bound is reduced to it.
  With `adaptive_dt`, the nonlinear bound
  $2/\max|\nabla\chi|$ (largest ExB velocity of the four stages, scaled by
  `fac_dtim_nl`) is taken as well, and dt grows by at most 5 % per step.
- **Linear runs** renormalise every `naverage` steps. The growth rate of each
  $k_y$ is $\gamma = \ln(A_\mathrm{end}/A_\mathrm{start})/\Delta t_\mathrm{window}$
  with the amplitude $A(k_y)^2 = \Delta s \sum_s\sum_{k_x\in\mathrm{chain}} |\phi|^2$
  over the chain through $k_x = 0$ (GKW `diagnos_growth_freq`). The linear
  operator is also available to an eigenvalue solver
  ({func}`~gyaradax.solver.eigenvalue.eigensolve`, GKW `eiv_integration`), which
  finds subdominant and damped modes.

## Diagnostics

Fluxes, spectra and growth rates are computed from the physical $f$ (GKW
`get_f_from_g`). The particle, heat and momentum fluxes per species are the
velocity moments of $f$ against the ExB velocity of $\phi$, with the Parseval
factor for $k_y > 0$; the electromagnetic (flutter) fluxes of $A_\parallel$ and
$B_\parallel$ are written separately to `fluxes_em.npz`. The spectra are
$\sum_{s,k_x}|\phi|^2$ and $\sum_{s,k_y}|\phi|^2$, field-line averaged.

## Differences from GKW

Implemented as in GKW: the terms above, electrostatic and electromagnetic
($A_\parallel$, $B_\parallel$) fields, adiabatic and kinetic electrons, the
linearised collision operator, the Coriolis drift and the $u'$ drive, s-α,
circular, Miller and `geom.dat` geometry, the CFL estimates, the initial
conditions (`finit`) and the GKW input format.

Choices that differ:

| | GKW | gyaradax |
|---|---|---|
| time integration | explicit RK4, implicit and semi-implicit schemes, RK-Chebyshev | explicit RK4 only |
| data layout | sparse matrix of the linear operator, MPI decomposition over species, $s$ and velocity space | dense arrays, operator applied as stencils; sharding over species, $v_\parallel$, $\mu$ |
| nonlinear FFTs | FP64, four inverse transforms | FP32 by default; two packed inverse transforms on CUDA |
| linear normalisation | one global factor by default | every connected kx chain of every $k_y$ separately |
| parallel dissipation | plain upwind | plain upwind, optionally conservative (`disp_par_conserve`) |
| collision CFL | largest single matrix element | summed stencil diagonal (about 1.5x tighter) |
| EM fluxes | separate files for $A_\parallel$ and $B_\parallel$ | one summed triple per species |
| geometry default of `compute_geometry` | s-α | circular (`compute_geometry_from_input` keeps GKW's s-α) |

Not implemented: neoclassical corrections (Term VI), centrifugal effects and ExB
shear, energetic-particle species, global profiles, Krook sources, the Arakawa
form of the parallel terms, and Fourier, MXH and CHEASE geometries. In the
collision operator: `zeff` rescaling, `lorentz`, `freq_input`, `cons_type = 'Lin'`
and `selfcollcon = .false.` (refused on import).
