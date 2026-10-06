# gyaradax — gyrokinetic solver in JAX

A JAX reimplementation of the GKW Fortran gyrokinetic code for local flux-tube
simulations. Supports both adiabatic and kinetic electron configurations.

## 1. physics overview

gyaradax solves the gyrokinetic Vlasov-Poisson system in the
local (flux-tube) limit (this section describes the electrostatic core;
the electromagnetic extension is in §10). The code evolves the perturbed gyrocenter distribution
function $\delta f_s$ for each kinetic species $s$ in a 5D phase space
$(v_\parallel, \mu, s, k_x, k_y)$, where the perpendicular coordinates are
Fourier-decomposed.

The fundamental equation is the collisionless gyrokinetic equation:

$$
\frac{\partial \delta f_s}{\partial t} + v_\parallel \nabla_\parallel \delta f_s
+ \mathbf{v}_d \cdot \nabla \delta f_s + \mathbf{v}_E \cdot \nabla \delta f_s
+ \dot{v}_\parallel \frac{\partial \delta f_s}{\partial v_\parallel}
= -(\mathbf{v}_E + \mathbf{v}_d) \cdot \nabla F_{M,s} - v_\parallel \nabla_\parallel \langle\phi\rangle_s \frac{\partial F_{M,s}}{\partial v_\parallel}
$$

where $\langle\phi\rangle_s = J_0(k_\perp \rho_s) \phi$ is the gyro-averaged
potential and $F_{M,s}$ is the background Maxwellian.

### 1.1 normalization

All quantities are normalized to reference values at the magnetic axis:

| quantity | normalization |
|----------|--------------|
| length | $R_{ref}$ (major radius) |
| velocity | $v_{th,ref} = \sqrt{2T_{ref}/m_{ref}}$ |
| time | $R_{ref} / v_{th,ref}$ |
| potential | $T_{ref} / e$ |
| magnetic field | $B_{ref}$ |
| distribution | $n_{ref} / v_{th,ref}^3$ |

Species parameters are normalized relative to the reference species:
- $\hat{m}_s = m_s / m_{ref}$, $\hat{Z}_s = Z_s / Z_{ref}$
- $\hat{T}_s = T_s / T_{ref}$, $\hat{n}_s = n_s / n_{ref}$
- $v_{th,s}/v_{th,ref} = \sqrt{\hat{T}_s / \hat{m}_s}$ (stored as `vthrat`)

### 1.2 field-aligned coordinates and geometry

The gyrokinetic equation is not solved on a physical $(R, Z, \phi)$ grid.
Instead it operates in a field-aligned coordinate system $(\psi, \zeta, s)$
that follows the magnetic field lines. Here $s$ runs along the field line
(parallel direction), $\psi$ labels the flux surface (radial direction), and
$\zeta$ is the field-line label within a surface (binormal direction).

The *geometry* encodes how this abstract coordinate system maps to physical
space. It provides:
- the **covariant metric tensor** $g_{ij}$, needed for $k_\perp^2$ and
  perpendicular gradients
- the **magnetic field strength** $B(s)$, needed for the mirror force,
  gyro-averaging, and drift velocities
- a set of **derived drift tensors** (D, E, H, I) that enter the
  gyrokinetic equation as advection coefficients

gyaradax supports two geometry paths: loading precomputed GKW files via
`load_geometry()`, or computing everything analytically from equilibrium
parameters via `compute_geometry()`. See section 9 for the circular model
formulas.

**Equilibrium parameters.** The safety factor $q$ controls field-line winding;
the magnetic shear $\hat{s} = (r/q)\,dq/dr$ drives spectral mode connectivity
(adjacent $k_x$ modes couple with shift $\Delta k_x = 2\pi\hat{s} k_y$);
the inverse aspect ratio $\varepsilon = r/R_0$ sets the strength of toroidal
effects (trapped particles, ballooning).

### 1.3 phase space coordinates

| coordinate | symbol | grid | range |
|-----------|--------|------|-------|
| parallel velocity | $v_\parallel$ | uniform | $[-v_{max}, v_{max}]$, typically $\pm 3 v_{th}$ |
| magnetic moment | $\mu$ | uniform in $v_\perp$ | $\mu = v_\perp^2/2$, weights $2\pi v_\perp \Delta v_\perp$ |
| field-line coordinate | $s$ | uniform | $[-0.5, 0.5]$ for `nperiod=1` |
| radial wavenumber | $k_x$ | discrete | centered FFT grid, from mode connectivity |
| binormal wavenumber | $k_y$ | uniform | $[0, k_{y,max}]$ |

The standard grid is `(nvpar=32, nmu=8, ns=16, nkx=85, nky=32)`.

### 1.4 species model

**Adiabatic electrons** (`adiabatic_electrons=True`): only ions are evolved
kinetically. Electrons respond instantaneously via the Boltzmann relation
$\delta n_e = n_e e\phi / T_e$, entering the quasineutrality equation as a
diagonal correction.

**Kinetic electrons** (`adiabatic_electrons=False`): both ions and electrons
are evolved as independent kinetic species. The distribution function gains a
leading species axis: `(nsp, nvpar, nmu, ns, nkx, nky)`. All RHS terms are
computed per-species with species-dependent mass, charge, temperature, and
thermal velocity. The species couple only through the shared potential $\phi$
from quasineutrality.

## 2. equations

### 2.1 RHS terms

The time derivative of $\delta f_s$ is a sum of seven terms:

**Term I — parallel streaming:**
$$-v_R v_{\parallel,s} \frac{\partial \delta f_s}{\partial s}$$
where $v_R = v_{th,s}/v_{th,ref}$. Uses 4th-order upwinded finite differences
along the field line with open boundary conditions via `mode_label` connectivity.

**Term II — magnetic drift advection:**
$$-i(k_x v_{d,x} + k_y v_{d,y}) \delta f_s$$
where $v_d \propto (v_\parallel^2 + \mu B) / Z_s$ is the curvature + grad-B drift.

**Term III — nonlinear ExB advection:**
$$\mathbf{v}_E \cdot \nabla \delta f_s = \{J_0 \phi, \delta f_s\}$$
Evaluated pseudospectrally using 2D FFTs with 3/2-rule dealiasing. The Poisson
bracket is computed in real space and transformed back.

**Term IV — trapping (mirror force):**
$$v_{th,s} \mu B g(s) \frac{\partial \delta f_s}{\partial v_\parallel}$$
where $g(s) = -B^{-1} \partial B / \partial s$. Uses 4th-order centered stencils
in $v_\parallel$.

**Term V — equilibrium drive:**
$$i k_y E_\alpha J_0 \phi \left[\frac{R}{L_n} + \frac{R}{L_T}\left(\frac{E}{T_s} - \frac{3}{2}\right)\right] F_{M,s}$$
with $E = v_\parallel^2 + 2\mu B$ the particle energy.

**Term VII — parallel field drive (Landau damping):**
$$-\frac{Z_s}{T_s} v_{th,s} v_\parallel F_{M,s} \frac{\partial (J_0 \phi)}{\partial s}$$

**Term VIII — drift field drive:**
Included in the drive term assembly alongside Term V:
$$-\frac{Z_s}{T_s} (k_x v_{d,x} + k_y v_{d,y}) F_{M,s} J_0 \phi$$

### 2.2 dissipation

- **parallel dissipation**: 4th-order damping on the streaming operator,
  coefficient `disp_par`, using upwinded 4th-derivative stencils.
  Optional conservative projection (`disp_par_conserve=2`,
  `disp_par_conserve_kycut=0.15`, default off): for modes with
  $0 < k_\theta\rho \le$ kycut the dissipated distribution is projected
  orthogonal to the Poisson/Ampère field-solve moments, curing the GKW
  issue #201 EM low-ky numerical instability — plain dissipation corrupts
  a physical KBM at $k_\theta\rho=0.1$, $\beta=1\%$ by +73%; the projection
  restores it to 1.1% of the dissipation-free value. See
  `instabilities/REPORT.md` (added 2026-06-11)
- **velocity dissipation**: 4th-order smoothing in $v_\parallel$,
  coefficient `disp_vp`
- **perpendicular hyper-dissipation**: spectral damping
  $(k_x/k_{x,max})^4 + (k_y/k_{y,max})^4$, coefficients `disp_x`, `disp_y`.
  $k_{x,max}$/$k_{y,max}$ are sourced from the geometry dict in
  `precompute.py`, not from `GKParams` — the params defaults (1.0)
  silently misnormalized the damping for direct-constructed params
  (corrected 2026-06-11)

### 2.3 field equation (quasineutrality)

The electrostatic potential $\phi$ is obtained from the quasineutrality condition.

**Adiabatic electrons:**
$$\phi(s, k_x, k_y) = -\frac{\sum_{v_\parallel, \mu} Z_i n_i J_0 B \Delta v_\parallel \Delta\mu \cdot \delta f_i}
{Z_i^2 n_i (\Gamma_0^i - 1)/T_i - Z_e n_e / T_e}$$

The adiabatic electron term $Z_e n_e / T_e$ appears in the denominator. For the
zonal mode ($k_y = 0$), a flux-surface-averaged correction is applied when
`zonal_adiabatic=True`.

**Kinetic electrons:**
$$\phi(s, k_x, k_y) = -\frac{\sum_s \sum_{v_\parallel, \mu} Z_s n_s J_0^s B \Delta v_\parallel \Delta\mu \cdot \delta f_s}
{\sum_s Z_s^2 n_s (\Gamma_0^s - 1) / T_s}$$

The sum runs over all kinetic species. $\Gamma_0^s = I_0(b_s) e^{-b_s}$ with
$b_s = \frac{1}{2}(m_s v_{th,s} k_\perp / Z_s B)^2$. For the zonal mode the
denominator is set to 1. No flux-surface averaging is needed.

### 2.4 transport fluxes

The heat flux for species $s$ is:

$$Q_s = \text{Im} \sum_{s, k_x, k_y, v_\parallel, \mu} P_{k_y} \Delta s \cdot k_y E_\alpha \left(v_\parallel^2 + 2\mu B\right) \delta f_s (J_0 \phi)^* B \Delta\mu \Delta v_\parallel \cdot d^2 X$$

where $P_{k_y}$ is the Parseval factor (1 for $k_y=0$, 2 otherwise)
and $d^2 X$ is the velocity-space volume element.

## 3. numerical methods

### 3.1 time integration

Explicit Runge-Kutta 4th order (RK4). Each small timestep requires 4 RHS
evaluations, each involving a full phi solve + linear terms + (optionally)
nonlinear FFTs.

The large-step cadence `naverage` groups small steps for diagnostic output.
In linear mode, per-$k_y$ normalization is applied at large-step boundaries.

**CFL-adaptive timestep** (`adaptive_dt=True`, default for kinetic electrons):
the timestep is adjusted each step to satisfy CFL constraints derived from
von Neumann stability analysis, matching GKW's `get_estimated_timestep`
(`matdat.F90:1356-1512`).

The analysis separates RHS terms by derivative order and applies
RK4-specific stability factors:

1. **ideriv=1 — first-derivative terms** (streaming, trapping):
   $$t_{max,1} = \max\!\left(\frac{|u_\parallel|_\infty \cdot c_{D1}}{\Delta s},\;
   \frac{|u_{trap}|_\infty \cdot c_{V1}}{\Delta v_\parallel}\right)$$
   where $c_{D1} = 2$ and $c_{V1} = 2/3$ are the maximum finite-difference
   stencil coefficients (boundary row for the parallel 4th-order upwinded
   scheme; interior central stencil for velocity).

2. **Field CFL — electrostatic mode frequency** (kinetic electrons only,
   `time_est_field` in `matdat.F90:1859-1940`):
   $$t_{max,\text{field}} = \frac{1}{\min_s\left[2\pi q\,\Delta s\, B(s)
   \sqrt{m_{ir}\, k_{\perp,\min}^2\, m_{er}}\right]}$$
   where $m_{ir} = \sum_\text{ion} m_s n_s$, $m_{er} = m_e / n_e$, and
   $k_{\perp,\min}^2 = k_{y,1}^2 g_{\zeta\zeta}(s)$.  For kinetic electrons
   this is typically the **dominant constraint** ($t_{max,\text{field}} \approx 3.4
   \times t_{max,1}$ for the standard kinetic grid).

3. **ideriv=4 — fourth-derivative dissipation** (parallel and velocity):
   $$t_{max,4} = \max\!\left(\frac{\nu_\parallel\, |u_\parallel|_\infty \cdot c_{D4}}{\Delta s},\;
   \frac{\nu_v\, |u_{trap}|_\infty \cdot c_{V4}}{\Delta v_\parallel}\right)$$

4. **Nonlinear ExB CFL**: $\Delta t_{NL} = \sigma \times 2 / \max_\text{NL}$
   with $\max_\text{NL} = \max(\max|\partial_y\phi|\cdot m_\text{rad}^2 m_\text{phi}/L_x,\,
   \max|\partial_x\phi|\cdot m_\text{rad} m_\text{phi}^2/L_y) + 2 v_{th,\max} v_{pmax} \cdot (\text{same for } A_\parallel)$,
   computed from the dealiased real-space potential gradients. The factors
   $m_\text{rad}^2 m_\text{phi}/L_x$ (y-branch) and $m_\text{rad} m_\text{phi}^2/L_y$
   (x-branch) match GKW's FFTW-unnormalized scaling
   (`non_linear_terms.F90:1538`); since gyaradax uses
   `jnp.fft.irfft2(norm="backward")` which applies $1/N$ on the inverse,
   an extra $N \cdot l_\text{inv}$ factor is needed per branch.
   Safety factor $\sigma = 0.95$ by default (`cfl_safety` parameter,
   matching GKW's `fac_dtim_est`). The 2026-04-18 EM benchmarks in
   §10.13 were run with `cfl_safety=0.5`, which explains the 0.53×
   mean dt reported there (corrected 2026-06-11).

The combined constraint for RK4 (`meth=2` in GKW):
$$t_{max} = \max\!\left(\frac{\max(t_{max,1},\, t_{max,\text{field}})}{2.4},\;
\frac{t_{max,4}}{2.4},\; 40\right)$$
$$\Delta t_\text{lin} = \frac{f_\text{dtim}}{t_{max}}, \qquad f_\text{dtim} = 0.95$$

The factor 2.4 is the RK4 stability boundary; the floor of 40 prevents
unreasonably large $\Delta t$ when linear terms are weak
(`matdat.F90:1507`).  The effective timestep is
$\Delta t = \min(\Delta t_{NL}, \Delta t_\text{lin}, \Delta t_\text{input})$.
Uses one-step lag: each step's dt is estimated from the previous step's $\phi$.

### 3.2 spatial discretization

**Parallel (s):** 4th-order finite differences with 9-point stencils. Open
boundary conditions use the spectral mode connectivity from `mode_label`:
adjacent $k_x$ modes connect across the $s$ boundary via magnetic shear.
Upwinding is selected based on the sign of $v_\parallel$.

**Parallel velocity ($v_\parallel$):** 4th-order centered stencils with
zero-padding at the boundaries.

**Perpendicular ($k_x, k_y$):** pseudospectral. The nonlinear term uses 2D
real-to-complex FFTs with 3/2-rule zero-padding for dealiasing.

### 3.3 precomputation

Species-dependent coefficients (Bessel functions, Maxwellians, drift velocities,
fused stencils) are precomputed once in `linear_precompute` and reused across
all RK4 stages and `jax.lax.scan` steps. For kinetic electrons, these arrays
gain a leading species dimension and are vmapped over during the RHS evaluation.

Fused stencils combine the streaming velocity with the upwinded
finite-difference coefficients, avoiding per-step branching on the sign of
$v_\parallel$. They are stored per stencil class and shift (`s_upar_tab`,
`s_t7_tab`, indexed by `par_stencil_class`) rather than as full 9 x 6D arrays.

## 4. code architecture

### 4.1 modules

(table refreshed 2026-06-11: precompute/CFL/field-solve split out of
`solver.py`; `geometry.py` is now the `geometry/` package)

| module | purpose |
|--------|---------|
| `solver.py` | RK4 integrator (`gkstep_single`, `gksolve`), per-ky normalization; linear/nonlinear RHS dispatch via `backends/` |
| `backends/` | `SolverOps` with the JAX and CUDA implementations of the field solve, linear RHS and Poisson bracket (§10.14) |
| `precompute.py` | one-time precomputation: stencils, species coefficients, EM weights, dissipation arrays |
| `cfl.py` | adaptive CFL timestep estimation (nonlinear + von Neumann + field) |
| `fields.py` | field solve dispatch (`_compute_fields`), g↔f transforms |
| `integrals.py` | phi solvers (adiabatic + kinetic), flux calculations (ES + EM) |
| `params.py` | `GKParams` dataclass, config/input.dat loading |
| `geometry/` | geometry package: circular (Lapillonne), s-alpha, Miller, tensors, grids, mode connectivity |
| `stencils.py` | finite difference coefficient tables |
| `collisions.py` | Fokker-Planck collision stencil precompute + apply |
| `quasilinear/` | quasilinear flux rule (saturation), calibration, linear pipeline |
| `utils.py` | K-dump loading, checkpoint save/load, diagnostics, GKW file-loading (`load_geometry`, `parse_input_dat`) |
| `simulate.py` | high-level simulation runner from YAML config |
| `cli.py` | `gyaradax run / bench / info` console script (docs/CLI.md) |
| `sharding.py` | device mesh, sharded precompute and init, `shard_map` helpers for the kernels |
| `eigenvalue.py` | linear eigenvalue solver (GKW `eiv_integration`) |
| `diag.py` | spectral diagnostics, 1D projections, nonlinear term analysis |
| `jax_config.py` | centralized JAX configuration and device initialization |
| `plot_utils.py` | publication-quality visualization |

### 4.2 key interfaces

```python
# standalone geometry (no GKW files needed)
geometry = compute_geometry(q=7.73, shat=2.14, eps=0.19, ns=16, nkx=85, nky=32, nvpar=32, nmu=8)

# or load from GKW files
geometry = load_geometry("/path/to/gkw_run")

# single/multi-step solver
next_df, (phi, fluxes), state = gksolve(df, geometry, params, state, n_steps)

# phi (adiabatic/kinetic based on df.ndim)
phi = calculate_phi(geometry, df, params=params, pre=pre)

# phi + fluxes
phi, fluxes = get_integrals(df, geometry, params=params)
# fluxes is (pflux, eflux, vflux) for adiabatic, (nsp, 3) array for kinetic

# per-species kinetic fluxes
per_sp_fluxes = calculate_fluxes_kinetic(geometry, df, phi)  # (nsp, 3)
```

### 4.3 multi-species implementation

When `adiabatic_electrons=False`, the solver:

1. `linear_precompute`: computes per-species coefficients with shape
   `(nsp, nvpar, nmu, ns, nkx, nky)` from geometry arrays.

2. `_compute_phi`: calls the unified `calculate_phi` which dispatches to
   `_phi_kinetic`, summing the Poisson integral over all species.

3. `ops.linear_rhs_from_g` / `ops.linear_rhs`: backend handles 5D/6D dispatch
   internally. JAX backend uses `jax.vmap` over species for 6D; the CUDA backend
   runs all species in one kernel launch (per-species `signz`/`tmp` are buffers).
   Each species gets its own precomputed coefficients.

4. `ops.nonlinear_term_iii`: backend handles 5D/6D dispatch. JAX backend vmaps
   over species with per-species Bessel; the CUDA backend puts all species in
   one cuFFT batch (see §10.14).

The adiabatic path is completely untouched — branching is via Python `if/else`
on `params.adiabatic_electrons` (a static pytree field resolved at trace time).

## 5. GKW Fortran reference

### 5.0 running GKW

The GKW binary is at `/system/user/publicwork/galletti/gkw.x`. Run it with
MPI from a directory containing `input.dat`:

```bash
cd /path/to/run_dir   # must contain input.dat
/usr/lib64/openmpi/bin/mpirun -np 64 /system/user/publicwork/galletti/gkw.x
```

GKW creates output files (`time.dat`, `fluxes.dat`, `FDS`, K-dumps, etc.)
in the same directory. Notes:
- Do not include `ndump_ts` or `keep_dumps` in `input.dat` (unsupported
  by this binary version).
- Reference input files are in `gkw_ref/benchmarks/`.
- Benchmark cases from the manual are in `gkw_ref/benchmarks/{cyclone,
  zonal_flow, ETG, beta, geom_miller, ...}/`.

### 5.1 source code mapping

| physics | Fortran file | key subroutine |
|---------|-------------|----------------|
| main loop | `gkw.f90` | program `gkw` |
| RK4 integration | `exp_integration.F90` | `rk4`, `calculate_rhs` |
| linear terms | `linear_terms.f90` | `calc_linear_terms`, `vpar_grd_df`, `ve_grad_fm`, `vpar_grd_phi` |
| field solver | `fields.F90` | `calculate_fields` |
| Poisson integral | `linear_terms.f90` | `poisson_int` |
| Poisson diagonal | `linear_terms.f90` | `poisson_dia` |
| zonal correction | `linear_terms.f90` | `poisson_zf` |
| nonlinear terms | `non_linear_terms.F90` | `add_non_linear_terms_spectral` |
| species setup | `components.f90` | `components_input_species` |
| Gamma function | `functions.f90` | `gamma_gkw` |
| geometry | `geom.f90` | `geom_circ`, `calc_geom_tensors` |
| CFL estimation | `matdat.F90`, `non_linear_terms.F90` | `get_estimated_timestep` |

### 5.2 manual references

The GKW manual (`gkw_ref/manual/`) contains:

- `theory.tex`: full gyrokinetic equation derivation and ordering
- `practise.tex`: discretized equations, Poisson splitting, boundary conditions
- `implementation.tex`: code structure and term-by-term mapping
- `diagnostics.tex`: output file conventions
- `buildandrun.tex`: input options and run configuration
- `collisions.tex`: collision operator (implemented — see §11)
- `rotation.tex`: centrifugal and Coriolis effects (Coriolis drift `vcor`
  + `uprim` drive implemented; centrifugal not — corrected 2026-06-11)
- `neoclassics.tex`: neoclassical corrections (not implemented)

## 6. reference data

### 6.1 adiabatic baselines

Located at `/restricteddata/ukaea/gyrokinetics/raw/iteration_{N}`:
- iterations 8, 13, 131, 200 (nonlinear)
- iterations 8, 13, 200 with `_Lin` suffix (linear)
- grid: `(32, 8, 16, 85, 32)`, `dt=0.01`, `naverage=40`
- single species (ions), adiabatic electrons, `zonal_adiabatic=True`

### 6.2 kinetic electron baselines

Located at `/restricteddata/ukaea/gyrokinetics/raw/kinetic_electrons/`:

| case | suffix | electron R/LT | ion R/LT |
|------|--------|--------------|----------|
| low drive | `half_rlt` | 3.45 | 5.394 |
| medium | `ntsks128` | 6.9 | 5.394 |
| high drive | `double_rlt` | 13.8 | 5.394 |

Common: grid `(32, 8, 16, 85, 32)`, 2 species (ion + electron),
`dt_actual=2.132e-3` (CFL-adapted from `dt_input=4e-3`), `naverage=100`,
`non_linear=True`, `zonal_adiabatic=False`.

K-dump binary format: `(2_re_im, nvpar, nmu, ns, nkx, nky, nspecies)` Fortran
order. Species is the outermost (slowest) index.

`fluxes.dat`: 6 columns = `[pflux_i, eflux_i, vflux_i, pflux_e, eflux_e, vflux_e]`.

## 7. differences from GKW / missing physics

### 7.1 implemented

- electrostatic gyrokinetics
- electromagnetic $A_\parallel$ (shear Alfvén, Ampere's law, mixed variable $g$)
- electromagnetic $B_\parallel$ (magnetic compression, coupled 2×2 Poisson-Bpar solve)
- adiabatic and kinetic electron models
- linearized Fokker-Planck collision operator (pitch-angle, energy diffusion,
  friction; inter-species pairs, Coulomb-log path, Xu-style momentum/energy
  conservation corrections — see §11; corrected 2026-06-11, no longer MVP-only)
- toroidal rotation: Coriolis drift (`vcor`) + `uprim` drive (added
  2026-06-11; no centrifugal terms; the uprim drive is ~25% strong vs
  GKW — known refinement item, see `instabilities/LOG.md`)
- all 7 linear RHS terms (I, II, III, IV, V, VII, VIII) plus EM terms X and XI
- nonlinear ExB advection (pseudospectral, spectral Poisson bracket)
- 4th-order parallel and velocity dissipation
- perpendicular hyper-dissipation
- RK4 explicit time integration
- per-$k_y$ normalization (linear mode)
- CFL-adaptive timestep (nonlinear ExB + linear parallel streaming + EM Alfvén)
- standalone circular geometry computation (no precomputed GKW files needed)
- Miller flux-surface parametrisation (elongation κ, triangularity δ, squareness ζ, skappa/sdelta/ssquare, Zmil, dRmil, dZmil) — ports GKW `geom_miller` with Simpson flux-surface integrals; matches GKW `geom.dat` to ≤1e-5 max rel-error

### 7.2 not implemented

(rows for collision conservation corrections and inter-species collisions
removed 2026-06-11 — both are now implemented, see §11)

| feature | GKW module | notes |
|---------|-----------|-------|
| neoclassical | `neoclassics.f90` | equilibrium corrections to $F_M$ |
| centrifugal rotation, ExB shear | `rotation.f90` | centrifugal (`cfen`, `cf_trap`/`cf_drift`) and toroidal ExB shear; Coriolis (`vcor`) + `uprim` are implemented (corrected 2026-06-11) |
| energetic particles | `components.f90` | `types='EP'`, `types='alpha'` |
| implicit integration | `imp_integration.F90` | for stiff parallel streaming |
| RK-Chebyshev | `exp_integration.F90` | for diffusion-dominated regimes |
| real-space nonlinear | `non_linear_terms.F90` | Arakawa bracket variant |
| global effects | `global.f90` | radial profile variation |
| source terms | various | Krook operator, external sources |
| general geometry (Fourier / MXH / chease) | `geom.f90` | only `s-alpha`, `circ` (Lapillonne), and `miller` supported |

### 7.3 growth rate convention

gyaradax matches the GKW growth rate definition (`diagnos_growth_freq.f90`).

**Amplitude.** Both codes compute per-$k_y$ amplitude as
$A(k_y) = \sqrt{\Delta s \sum_s \sum_{k_x \in \text{chain}} |\phi(s, k_x, k_y)|^2}$,
where the $k_x$ sum runs only over the connected mode chain containing $k_x = 0$
(determined by `mode_label`). See `solver.py:mode_amplitude`.

**Growth rate.** Computed as $\gamma = \ln(A_\text{end} / A_\text{start}) / \Delta t_\text{window}$
over each `naverage` window. In linear mode, per-$k_y$ normalization resets the
amplitude to $\approx 1$ at each window boundary, so $A_\text{start} = 1$. In
nonlinear mode (no normalization), $A_\text{start}$ is set to the amplitude at
the previous window boundary, giving the instantaneous growth rate between
consecutive windows. See `solver.py:advance_state`.

## 8. validation results

### 8.1 adiabatic solver

| test | window | metric | result |
|------|--------|--------|--------|
| linear 80 steps | DM2→FDS | `rel_l2(df)` | `8.9e-6` |
| nonlinear 120 steps × 4 iters | 100→101 | `rel_l2(df_subset)` | `< 1e-3` |
| heat flux parity | 100→101 | `rel_err(eflux)` | `3.8e-6` |

### 8.2 kinetic electron solver

| test | case | metric | result |
|------|------|--------|--------|
| trajectory 300 steps | half_rlt | `rel_l2(df, ion)` | `8.4e-7` |
| trajectory 300 steps | half_rlt | `rel_l2(df, electron)` | `2.0e-6` |
| trajectory 300 steps | ntsks128 | `rel_l2(df, ion)` | verified |
| trajectory 300 steps | double_rlt | `rel_l2(df, ion)` | verified |
| per-species flux | all 3 cases × 2 dumps | `rtol(eflux)` | `< 1e-2` |
| CFL vs GKW dtim | all 3 cases | `ratio(dt_est, dtim)` | `0.3 – 3.0` |
| adaptive CFL 20 steps | all 3 cases | finiteness (dt=0.004) | pass |
| adiabatic fallback | 4 iterations | shapes + finiteness | pass |
| **nl_em_apar** (β=0.001, 30k steps) | ion eflux | `ratio(gyra, GKW)` | **1.04** |
| **nl_em_apar** (β=0.001, 30k steps) | elec eflux | `ratio(gyra, GKW)` | **1.02** |
| **nl_em_apar** (β=0.001) | φ(ky) Pearson lin/log | rel L2 log | **1.000 / 0.999, 0.034** |
| **nl_em_beta01** (β=0.01, 420k steps adaptive) | ion eflux | `ratio(gyra, GKW)` | **0.97** |
| **nl_em_beta01** (β=0.01) | elec eflux | `ratio(gyra, GKW)` | **0.95** |
| **nl_em_beta01** (β=0.01) | adaptive dt range | `min / mean / max` | `0.00038 / 0.00107 / 0.00297` (GKW: `0.00064 / 0.00202 / 0.00432`) |

### 8.3 analytical benchmarks

Two analytical benchmarks are verified and included as unit tests in
`tests/unit/test_gk_cases.py`. Figures in `notebooks/analytical_benchmarks.ipynb`.

#### 8.3.1 Rosenbluth-Hinton zonal flow test

Uses the GKW benchmark parameters from `gkw_ref/benchmarks/zonal_flow/zonal01`:
q=1.3, shat=0.1592, eps=0.05, s-alpha geometry, ns=128, nvpar=128, nmu=16,
krhomax=0.025, ikxspace=1, disp_par=0.01, dt=0.01, finit='zonal'.

The phi solve (`_phi_adiabatic`) satisfies quasineutrality to machine precision
(rel err 2.2e-15). The Gamma0 uses `i0e(b)` for stability (matches GKW `expbessi0`).

| test | metric | result |
|------|--------|--------|
| residual at t>80 | `sqrt(mean(kxspec/kxspec_0))` | **0.0711** |
| Xiao-Catto target | analytical | **0.0711** |
| match | relative error | **< 0.1%** |
| eps scan (5 values) | residual vs eps | traces analytical curve |

**Key requirements:** `disp_par > 0` (damps velocity-space recurrence),
`drive_scale=1.0` (Term VIII needed for GAM), `disp_x=disp_y=0` (no spurious
hyper-dissipation). Use `non_linear=False` with large naverage to avoid
per-ky normalization without computing the NL FFT.

#### 8.3.2 Cyclone Base Case linear ITG

Uses the GKW benchmark parameters from `gkw_ref/benchmarks/cyclone/linear`:
q=1.4, shat=0.78, eps=0.19, rlt=6.9, rln=2.2, **s-alpha geometry**, ns=160,
nvpar=64, nmu=16, nperiod=5, disp_par=1.0, dt=0.003, naverage=100.

| test | metric | result |
|------|--------|--------|
| gamma at kt=0.5 | growth rate | **0.179** |
| GKW/GS2 reference | — | **0.18** |
| match | relative error | **< 1%** |
| kt scan (8 values) | gamma spectrum shape | matches GKW |
| R/LT scan (5 values) | gamma vs gradient | matches GKW |

**Key requirements:**
- **s-alpha geometry** (circ gives ~50% higher growth rates)
- **ns=160, nperiod=5** (low ns underresolves Landau damping → lifted spectrum)
- **naverage ≥ 10** (naverage=1 gives spurious negative growth from mode phase
  rotation within a single step)

## 9. circular geometry model (`geometry/` package, formerly `geometry.py`)

Formulas translated from `gkw_ref/src/geom.f90` (`geom_circ` lines 1444-1616,
`calc_geom_tensors` lines 3487-3634). `compute_geometry()` produces the full
geometry dict from equilibrium parameters; `simulate()` uses it automatically
when `data_dir` is absent from the YAML config.

### 9.1 magnetic field and poloidal angle

The field-line coordinate $s$ maps to poloidal angle $\theta$ via
$\theta + \varepsilon \sin\theta = 2\pi s$, solved by 10 fixed-point iterations
(convergence $\sim \varepsilon^{10}$). The magnetic field strength is:

$$B(s) = \frac{\delta}{1 + \varepsilon\cos\theta}, \qquad
\delta = \sqrt{1 + \frac{\varepsilon^2}{q^2(1-\varepsilon^2)}}$$

### 9.2 metric tensor

In $(\psi, \zeta, s)$ coordinates, $g_{\psi\psi} = 1$ and:
- $g_{\psi\zeta} = d\zeta/d\varepsilon$: shear coupling, computed with
  branch-tracked `atan` (`geom.f90` lines 1492-1511)
- $g_{\psi s} = \sin\theta / (2\pi)$
- $g_{\zeta\zeta}$, $g_{\zeta s}$, $g_{ss}$: standard circular formulae

### 9.3 jacobian transform

All field derivatives ($dB/d\psi$, $dR/d\psi$, $dZ/d\psi$) are computed in
$(\psi, \theta)$ space then transformed to $(\psi, s)$:

$$f_\psi^{(s)} = f_\psi^{(\theta)} - \frac{\sin\theta}{1+\varepsilon\cos\theta} f_\theta, \qquad
f_s = \frac{2\pi}{1+\varepsilon\cos\theta} f_\theta$$

The radial derivative of $B$ in $(\psi, \theta)$ uses the finite-$\varepsilon$
formula from `geom.f90` line 1528:

$$\partial_\psi B\big|_\theta = B\left(\frac{-\cos\theta}{1+\varepsilon\cos\theta}
+ \frac{\varepsilon(1-\hat{s}+\varepsilon^2/(1-\varepsilon^2))}
{\varepsilon^2 + q^2(1-\varepsilon^2)}\right)$$

### 9.4 drift tensors

**E-tensor** (ExB): antisymmetric cofactors of metric rows 0 and 1,
scaled by $\pi \cdot dp_f/d\psi / B^2$ where
$dp_f/d\psi = \varepsilon / (q\sqrt{1-\varepsilon^2})$.

**D-tensor** (curvature + $\nabla B$):
$D_j = (-2 E_{j,\psi}\,\partial_\psi B - 2 E_{j,s}\,\partial_s B) / B$

**H-tensor** (Coriolis): $H_j = -\sigma_B(g_{j,\psi}\,\partial_\psi Z + g_{j,s}\,\partial_s Z)/B$
with finite-$\varepsilon$ correction $H_s \mathrel{+}= \sigma_B b_{ups}^2 (\partial_s Z)/B^2$.

**I-tensor** (centrifugal): $I_j = 2R(E_{j,\psi}\,\partial_\psi R + E_{j,s}\,\partial_s R)$

### 9.5 validation

All arrays verified against 7 GKW trajectories (4 adiabatic, 3 kinetic):

| field | max relative error |
|-------|--------------------|
| `bn`, `ffun`, `bt_frac`, `rfun` | $< 5 \times 10^{-6}$ |
| `gfun`, `efun`, `little_g` | $< 2 \times 10^{-5}$ |
| `dfun`, `hfun`, `ifun` (eps component) | $< 10^{-4}$ |
| `dfun`, `hfun`, `ifun` (zeta component) | $< 2 \times 10^{-3}$ |
| velocity / wavenumber grids | $< 10^{-6}$ |

The zeta-direction tensors (`D_zeta`, `H_zeta`, `I_zeta`) have ~0.1% model-level
error originating from the finite-$\varepsilon$ correction in `_dzetadeps` (the
branch-tracked atan for $d\zeta/d\varepsilon$). This is an inherent approximation
in the Lapillonne circular model, not numerical error. The radial (eps) components
are unaffected.

68 tests in `tests/unit/test_analytic_geometry.py`.


## 10. electromagnetic formulation

Extension of the electrostatic solver to include the parallel vector
potential $A_\parallel$ (shear Alfvén physics) and the parallel magnetic
field perturbation $B_{1\parallel}$ (magnetic compression). Derived from
the GKW Fortran source (`fields.F90`, `linear_terms.f90`) and the GKW
manual (`theory.tex`, `practise.tex`).

### 10.1 mixed variable (g vs f)

GKW evolves the **mixed variable** $\hat{g}$, not the physical
perturbation $\delta\hat{f}$ directly. The relation is:

$$\hat{g}_s = \delta\hat{f}_s + \frac{2 Z_s}{T_{R,s}}\,v_{R,s}\,v_\parallel\,
\langle\hat{A}_\parallel\rangle_s\,F_{M,s}$$

where $\langle\hat{A}_\parallel\rangle_s = J_0(k_\perp\rho_s)\,\hat{A}_\parallel$
is the gyro-averaged vector potential and $v_{R,s} = v_{th,s}/v_{th,ref}$
(`vthrat` in code).

**g-to-f transform** (from `g2f_correct` in `linear_terms.f90:4587`):

$$\delta\hat{f}_s = \hat{g}_s - \frac{2 Z_s}{T_{R,s}}\,v_{R,s}\,v_\parallel\,
J_0(k_\perp\rho_s)\,\hat{A}_\parallel\,F_{M,s}$$

The g2f matrix element in GKW is:
```
mat_elem = -2.0 * signz(is) * vthrat(is) * vpgr(i,j,k,is) * J0 * fmaxwl / tmp(ix,is)
```

**Why the mixed variable?** Evolving $g$ instead of $f$ avoids a stiff
$\partial A_\parallel/\partial t$ cancellation problem that would otherwise
require implicit time stepping. With $g$, the time derivative of the
$A_\parallel$ coupling is absorbed into the field equation.

When `nlapar=False`, $g = \delta f$ (identity transform, no EM correction).

The g2f transform is controlled by the `lg2f_correction` flag in GKW
(`linear_term_switches` namelist). When True (default when `nlapar=True`),
the correction matrix `matg2f` is applied.

**How GKW applies g2f** (`exp_integration.F90:800–912`):

1. **Field solve** (`calculate_fields`): fields are **zeroed** first, then
   `mat_poisson * fdis` computes the Poisson/Ampere integrals from $g$
   alone. Since `matg2f` maps `iapar → ifdis` and $A_\parallel = 0$
   before the solve, the g2f entries contribute nothing. The field solve
   uses the **bare Ampere denominator** — no self-consistent g2f correction.

2. **g→f conversion**: after the field solve, `fdis_tmp(i) = g(i) +
   matg2f%mat(i) * apar` converts the distribution from $g$ to $f$.

3. **Linear RHS**: `mat * fdis_tmp` applies all linear terms (I–VIII)
   to $f$ (the physical distribution), not $g$.

4. **Nonlinear terms**: use $g$ (not $f$), per the comment at line 875:
   "distribution g = f + Z v∥ A∥ etc., rather than f".

### 10.2 modified potential χ

The electromagnetic ExB drift uses the generalized potential $\chi$
instead of $\phi$ alone:

$$\hat{\chi} = \langle\hat{\phi}\rangle
+ \frac{2\mu T_{R,s}}{Z_s}\,\langle\hat{B}_{1\parallel}\rangle
- 2\,v_{R,s}\,v_\parallel\,\langle\hat{A}_\parallel\rangle$$

The ExB velocity becomes $\mathbf{v}_\chi = (\mathbf{b}\times\nabla\chi)/B_0$.
This affects the nonlinear Term III (Poisson bracket uses $\chi$ instead
of just $\phi$) and the drive terms (V, VIII).

### 10.3 Ampere's law for $A_\parallel$

The parallel component of Ampere's law in normalized GKW form
(`theory.tex` eq. 401–405, `ampere_int` + `ampere_dia` in `linear_terms.f90`):

$$\left[k_{\perp,N}^2 + \beta_\text{ref}\sum_s
\frac{Z_s^2\,n_{R,s}}{m_{R,s}}\,e^{-\mathcal{E}_s/T_{R,s}}\,
\Gamma_0(b_s)\right]\hat{A}_\parallel
= \beta_\text{ref}\sum_s Z_s\,v_{R,s}\,n_{R,s}\;
2\pi B_N\int v_\parallel\,J_0(k_\perp\rho_s)\,\hat{g}_s\,
\mathrm{d}v_\parallel\,\mathrm{d}\mu$$

where:
- $k_{\perp,N}^2 = k_\perp^2\rho_\text{ref}^2$ (`krloc**2` in code)
- $\beta_\text{ref} = 2\mu_0 n_\text{ref} T_\text{ref}/B_\text{ref}^2$
- $\Gamma_0(b_s) = I_0(b_s)\,e^{-b_s}$ with $b_s = k_\perp^2\rho_s^2/2$
- $\mathcal{E}_s$ is the centrifugal energy correction (`cfen` in code)
- The RHS integrates $v_\parallel J_0 \hat{g}$ over velocity space (the parallel current)

**LHS (diagonal)** from `ampere_dia` (`linear_terms.f90:3753–3789`):
```
mat_elem = -krloc^2
dum = sum_sp[ -veta * signz^2 * de * gamma_num / mas ]
  where gamma_num = sum_{j,k}[ 2*bn*intmu*intvp * J0^2 * vpgr^2 * fmaxwl ]
elem%val = -1.0 / (mat_elem + dum)
```

**RHS (integral)** from `ampere_int` (`linear_terms.f90:3246`):
```
elem%val = signz * de * veta * intvp * intmu * vthrat * bn * vpgr * J0
```

**Key detail:** The Ampere equation is diagonal in $(k_x, k_y)$ space
(no parallel coupling), so it reduces to a pointwise division at each
$(s, k_x, k_y)$ grid point. This makes the solve trivial — no matrix
inversion needed beyond the precomputed inverse denominator.

**Bare denominator (no g2f self-consistency):** GKW's `calculate_fields`
zeros all field entries before the `mat_poisson` multiply. Since the g2f
matrix maps `iapar → ifdis`, it produces zero contribution (apar is zero
at that point). The effective Ampere solve is simply:

$$A_\parallel = \frac{\text{numerator}(g)}{\text{diag}(k_\perp^2 + \beta\sum\ldots)}$$

There is **no** self-consistent g2f correction to the denominator. A
naive self-consistent solve would replace `diag` with `diag − g2f_correction`,
where $g2f\_correction = -\text{diag\_em}$ analytically, effectively
doubling the EM part of the denominator and halving $A_\parallel$. This
is incorrect for matching GKW.

**Numerical denominator:** GKW uses a numerically computed $\Gamma_\text{num}$
(`ampere_dia:3768–3777`) rather than the analytical $\Gamma_0(b)$:
```
gamma_num = sum_{j,k}[ 2*bn*intmu*intvp * J0^2 * vpgr^2 * fmaxwl ]
```
This integral sums $2 B\,J_0^2\,v_\parallel^2\,F_M$ over velocity space.
It matches the analytical $\Gamma_0$ to $<0.1\%$ at the waltz_linear grid
resolution but eliminates discretization-dependent discrepancies.

**Zonal mode (ky=0):** When $k_\perp \approx 0$, $\Gamma_0 \to 1$ and
$J_0 \to 1$. The denominator simplifies but remains well-defined.

### 10.4 $B_{1\parallel}$ equation (perpendicular Ampere)

The perpendicular component of Ampere's law gives the magnetic
compression equation (`theory.tex` eq. 412–418):

$$\left[1 + \beta_\text{ref}\sum_s
\frac{T_{R,s}\,n_{R,s}}{B_N^2}\,e^{-\mathcal{E}_s/T_{R,s}}\,
\bigl(\Gamma_0(b_s) - \Gamma_1(b_s)\bigr)\right]\hat{B}_{1\parallel}$$
$$= -\beta_\text{ref}\sum_s\left[
2\pi B_N\,T_{R,s}\,n_{R,s}\int\mu\,\hat{J}_1(k_\perp\rho_s)\,
\hat{g}_s\,\mathrm{d}v_\parallel\,\mathrm{d}\mu
+ e^{-\mathcal{E}_s/T_{R,s}}\,
\bigl(\Gamma_0 - \Gamma_1\bigr)\,
\frac{Z_s\,n_{R,s}}{2B_N}\,\hat{\phi}\right]$$

where:
- $\Gamma_1(b_s) = I_1(b_s)\,e^{-b_s}$ (modified Bessel of first kind, order 1)
- $\hat{J}_1 = 2J_1(k_\perp\rho_s)/(k_\perp\rho_s)$ is the **modified J1**
  (`mod_besselj1_gkw` in code)
- The $\hat{\phi}$ coupling makes the B_par equation coupled to Poisson

**Coupling structure:** When `nlbpar=True`, the Poisson equation and
B_par equation are solved as a coupled 2×2 system at each $(s,k_x,k_y)$.
The coupling is mediated by $(\Gamma_0 - \Gamma_1)$ terms. GKW decouples
them using intermediate coefficients:

From `poisson_dia` (`linear_terms.f90:3446–3512`):
```
F_sp1 = sum_sp[ signz^2 * de * (gamma - 1) / tmp ]
F_sp2 = sum_sp[ signz * veta * de * gamma_diff / (2*bn) ]
B_sp1 = sum_sp[ signz * de * gamma_diff / bn ]
B_sp2 = sum_sp[ tmp * de * veta * gamma_diff / bn^2 ]
  where gamma_diff = (Gamma_0 - Gamma_1) * exp(-cfen)

diagonal = F_sp1 * (1 + B_sp2) - F_sp2 * B_sp1
elem%val = -1.0 / diagonal
```

### 10.5 modified RHS terms with EM

The standard 8-term RHS is modified as follows when `nlapar=True`:

| term | ES formula | EM modification | GKW code |
|------|-----------|-----------------|----------|
| I (parallel streaming) | $-v_R v_\parallel \partial_s \delta f$ | acts on $f$ not $g$ (via g2f) | `vpar_grd_df` |
| II (magnetic drift) | $-i\,\mathbf{k}\cdot\mathbf{v}_d\,\delta f$ | acts on $f$ not $g$ (via g2f) | `vdgradf` |
| III (nonlinear ExB) | $\{\langle\phi\rangle, \delta f\}$ | bracket uses $\chi$ instead of $\phi$; acts on $g$ | `calculate_nonlinear` |
| IV (trapping) | $v_{th}\mu B g(s)\,\partial_{v_\parallel}\delta f$ | acts on $f$ not $g$ (via g2f) | `dfdvp_trap` |
| V (equilibrium drive) | $i k_y E_\alpha J_0\phi(\ldots)F_M$ | add $-2 v_{R,s} v_\parallel$ factor coupling to $A_\parallel$ | `ve_grad_fm:2452` |
| VII (parallel field drive) | $-\frac{Z}{T}v_{th}v_\parallel F_M\partial_s(J_0\phi)$ | add $\nabla_\parallel(J_0 A_\parallel)$ with rhostar effects | `vpar_grd_phi:2957` |
| VIII (drift field drive) | $-\frac{Z}{T}\mathbf{k}\cdot\mathbf{v}_d F_M J_0\phi$ | add $-2 v_{R,s} v_\parallel$ factor coupling to $A_\parallel$ | `vd_grad_phi_fm` |

**g2f in kinetic terms:** GKW converts $g \to f$ via `matg2f` before
the linear RHS multiply (`exp_integration.F90:805`). Terms I, II, IV,
and dissipation act on $f$, not $g$. Confirmed by running GKW with
`lg2f_correction=.false.`: the 1-step mode shape changes by 1.1%.
gyaradax matches this: `g_to_f` is applied before `linear_rhs`.

**Term VII uses $J_0\phi$ only, not $\chi$:** GKW's Term VII has
`elem%itloc = iphi` — it reads from $\phi$, not $A_\parallel$. The
EM $A_\parallel$ correction to Term VII (lines 2957–3004) is only active
when `rhostar_linear > 0`. gyaradax separates `gyro_phi` (for Term VII)
from `gyro_chi` (for drive terms V, VIII, XI).

**New terms when `nlbpar=True`:**

| term | formula | description |
|------|---------|-------------|
| X | $-2 v_R v_\parallel \mu F_M \mathcal{F}\,\partial_s\langle\hat{B}_{1\parallel}\rangle$ | mirror force from $B_{1\parallel}$ perturbation |
| XI | $-\frac{i}{Z}F_M\,2T_R\mu\,(\text{drift})\,k\,\langle\hat{B}_{1\parallel}\rangle$ | drift coupling to $B_{1\parallel}$ |

**EM coefficient in Terms V and VIII** (`linear_terms.f90:2452`):
```
elem2%val = -2.0 * vthrat(is) * vpgr(i,j,k,is) * [ES_coefficient]
```
This multiplies the electrostatic drive by $-2 v_{R,s} v_\parallel$ and
couples to $A_\parallel$ instead of $\phi$.

**EM coefficient in Term VII** (`linear_terms.f90:2959`):
```
dum = -2 * tmp / vthrat / mas * vpgr * (term5+term9) / signz
```
This creates $\nabla_\parallel(J_0 A_\parallel)$ using the same parallel
stencil infrastructure as $\nabla_\parallel(J_0\phi)$.

### 10.6 normalization

Field normalizations from `practise.tex`:

$$\phi = \rho_*\frac{T_\text{ref}}{e}\,\phi_N, \qquad
A_\parallel = B_\text{ref}R_\text{ref}\rho_*^2\,A_{\parallel,N}, \qquad
B_{1\parallel} = B_\text{ref}\rho_*\,B_{1\parallel,N}$$

where $\rho_* = \rho_\text{ref}/R_\text{ref}$ is the normalized
gyroradius. Note that $A_\parallel$ scales as $\rho_*^2$ (one order
higher in $\rho_*$ than $\phi$), reflecting the subsidiary ordering
of the parallel vector potential in the gyrokinetic expansion.

### 10.7 Alfvén CFL constraint

When `nlapar=True` with kinetic electrons, the shear Alfvén wave
introduces a tight CFL constraint. From `matdat.F90:1918`:

$$\Delta t_\text{Alfvén} = 2\pi q\,\Delta s\,B(s)\,
\sqrt{m_{ir}\,(v_{\eta} + k_{\perp,\min}^2\,m_{er})}$$

where:
- $m_{ir} = \sum_\text{ion} m_s n_s$ (ion inertial mass)
- $m_{er} = m_e/n_e$ (electron mass/density ratio)
- $v_\eta$ = `veta` (plasma $\beta$ at radial point)
- $k_{\perp,\min}^2 = k_{y,1}^2 g_{\zeta\zeta}(s)$ (smallest nonzero ky mode)

The timestep is $\Delta t_\text{max} = 1/\Delta t_\text{Alfvén}$, minimized
over all $s$ grid points. This constraint is only active when
`adiabatic_electrons=False` (kinetic electrons required for Alfvén CFL).

### 10.8 Bessel functions for EM

| function | definition | usage | code |
|----------|-----------|-------|------|
| $J_0(k_\perp\rho_s)$ | Bessel first kind, order 0 | gyro-averaging of $\phi$ and $A_\parallel$ | `besselj0_gkw` |
| $\hat{J}_1 = 2J_1(x)/x$ | modified Bessel, order 1 | $B_{1\parallel}$ gyro-averaging | `mod_besselj1_gkw` |
| $\Gamma_0(b) = I_0(b)e^{-b}$ | modified Bessel envelope | Poisson and Ampere diagonals | `gamma_gkw` |
| $\Gamma_1(b) = I_1(b)e^{-b}$ | modified Bessel envelope, order 1 | $B_{1\parallel}$ coupling | `gamma1_gkw` |

where $b_s = k_\perp^2\rho_s^2/2$ is the Bessel argument.

Limits for $k_\perp\rho \to 0$: $J_0 \to 1$, $\hat{J}_1 \to 1$,
$\Gamma_0 \to 1$, $\Gamma_1 \to 0$, $\Gamma_0 - \Gamma_1 \to 1$.

### 10.9 GKW control flags

| flag | namelist | default | description |
|------|---------|---------|-------------|
| `nlapar` | `control` | `.false.` | enable $A_\parallel$ field variable |
| `nlbpar` | `control` | `.false.` | enable $B_{1\parallel}$ field variable |
| `lampere` | `linear_term_switches` | `.true.` | enable Ampere coupling in linear RHS |
| `lbpar` | `linear_term_switches` | `.true.` | enable $B_\parallel$ coupling in linear RHS |
| `lg2f_correction` | `linear_term_switches` | `.true.` | enable g-to-f transform |
| `beta_ref` | `spcgeneral` | `0.0` | reference plasma beta |

Auto-downgrade: if `beta_ref ≈ 0` and `nlapar=True`, GKW warns and
sets `nlapar=False`, `nlbpar=False` (`components.f90:848–853`).

Adiabatic electrons can coexist with `nlapar=True` (test case:
`adiabat_apar`), but the Alfvén CFL constraint is only active with
kinetic electrons.

### 10.10 GKW EM reference test cases

Available in `gkw_ref/tests/standard/`:

| test case | nlapar | nlbpar | beta | adiab. e⁻ | species | grid (s×μ×v×modes) | np |
|-----------|--------|--------|------|-----------|---------|-------------------|-----|
| `bpar_waltz_linear` | T | T | 0.01 | F | 2 | 112×8×32×1 | 16 |
| `adiabat_apar` | T | F | 0.234 | **T** | 3 | 45×8×16×1 | 12 |
| `non_spectral_apar_noampere` | T | F | 0.003 | F | 2 | 8×4×16×1 | 16 |
| `kin_nl_bpar` | T | T | 0.002 | F | 3 | 12×4×8×11 | 24 |
| `slab_itg` | F | F | 3e-6 | T | 2 | 11×4×12×1 | 4 |

### 10.11 GKW → gyaradax variable mapping (EM)

| GKW Fortran | gyaradax | shape | description |
|-------------|----------|-------|-------------|
| `fdis(iapar,...)` | `apar` | `(ns, nkx, nky)` | parallel vector potential |
| `fdis(ibpar,...)` | `bpar` | `(ns, nkx, nky)` | parallel magnetic perturbation |
| `matg2f` | `g2f_factor` | `(nv, nmu, ns, nkx, nky)` | g-to-f correction matrix element |
| `gamma_gkw` | `gamma` / `phi_gamma` | `(ns, nkx, nky)` | $\Gamma_0 = I_0(b)e^{-b}$ |
| `gamma1_gkw` | `gamma1` | `(ns, nkx, nky)` | $\Gamma_1 = I_1(b)e^{-b}$ |
| `mod_besselj1_gkw` | `j1_hat` | `(nv, nmu, ns, nkx, nky)` | $\hat{J}_1 = 2J_1/x$ |
| `krloc**2` | `kperp_sq` | `(ns, nkx, nky)` | $k_\perp^2\rho_\text{ref}^2$ |
| `veta` | `beta` (param) | scalar | reference $\beta$ |
| `vpgr` | `vpar_grid` | `(nv,)` | parallel velocity grid |
| `ampere_int` weight | `apar_weight` | `(nsp, nv, nmu, ns, nkx, nky)` | Ampere numerator weight |
| `ampere_dia` inverse | `apar_diag` | `(ns, nkx, nky)` | Ampere denominator (precomputed inverse) |

### 10.12 EM validation results

Test case: `bpar_waltz_linear` (kinetic 2-species, beta=0.01, 112×8×32×1).
Both codes start from the same evolved ES distribution (GKW FDS file).

**20k-step distribution correlation (dt=0.001, t=20.0):**

| case | ion | electron |
|------|-----|----------|
| ES (beta=0) | 99.64% | 99.30% |
| A_par only | 99.28% | 98.96% |
| A_par + B_par | 99.36% | 98.98% |

EM parity matches ES at all timescales (100–20000 steps). Fluxes
computed from the same distribution match GKW to machine precision
after parseval and flux-surface-average corrections.

**Linear EM γ parity update (2026-06-11,
`instabilities/em_suite_report.json`):** across a β = 0.001–0.01
ladder at $k_\theta\rho = 0.4$ (through the KBM transition; both
apar-only and apar+bpar batches) gyaradax matches GKW to 0.02–0.1%
(e.g. β=0.01 KBM: γ = 0.9236 vs GKW 0.9238). With rotation
(vcor + uprim) on: 0.3%. In the low-ky band ($k_\theta\rho = 0.1$,
β=0.01) with the issue-#201 cure active in both codes: 0.9%.

### 10.13 EM gotchas from GKW benchmarking

Collected from the CBC NL-EM vs GKW benchmark (see `docs/em_debug_report.md`
for full numbers).

**CFL terms add as max-frequencies, not multiplicatively.** The Alfvén CFL
lives in `tmax_field`; the parallel streaming CFL lives in `tmax1`. GKW
takes the max — no extra $(1 + \beta v_{th,e}^2)^2$ multiplier on top of
the streaming bound. For CBC kinetic electrons at $\beta=0.001$ the naive
squaring mistake is 22× in $\Delta t$ because $v_{th,e} \approx 60.6$ and
$(1 + 0.001 \cdot 60.6^2)^2 \approx 21.8$. Invisible at low $\beta$ with
adiabatic electrons; catastrophic once electrons go kinetic.

**Diagnostics use $f$, not $g$.** GKW's `diagnos_fluxes_vspace.F90:444`
applies `get_f_from_g()` before every flux, field, and k-spectrum. `pflux`
and `eflux` are unchanged by skipping the transform (the g→f correction
$-(2Z/T) v_\parallel v_R J_0 A_\parallel F_M$ is odd in $v_\parallel$; the
flux integrands are even) but `vflux` and phi-based spectra quietly
differ. `gksolve` applies `g_to_f` before the final `get_integrals` to
match GKW's convention even for the flux channels that are invariant by
parity.

**Per-code `geom_type` defaults differ.** GKW defaults to `s-alpha` when
`input.dat` doesn't set `geom_type`; gyaradax's Python `compute_geometry`
defaults to `circ` (Lapillonne) — though `compute_geometry_from_input`
now mirrors GKW's `s-alpha` default for input.dat-driven runs. These geometries differ by ~50% in linear γ at finite ε
(§8.3.2). Set the geometry explicitly on both sides when benchmarking —
never rely on the default.

**Constant `pred/ref` ratio across windows ≠ physics bug.** In an
exponentially growing linear phase, a stable `pred/ref ≈ const` signals
an initialization or normalization difference, not a growth-rate
difference. Compare log-space slopes of |flux| vs window index instead
of absolute magnitudes. GKW's default `amp_init` is `1.0e-3`; gyaradax
had inherited `1.0e-4` in the YAML loader — a clean "constant ratio"
signature.

**Sub-percent linear perturbations can shift NL saturation by tens of
percent.** At CBC $\beta=0.001$ the $B_\parallel$ contribution to RHS is
~0.08% of the total, yet including vs excluding it changes saturated
flux by ~46% in gyaradax and ~12% in GKW. Both codes are correct; both
codes are sensitive. Validating $B_\parallel$-related formulas requires
stateless, hand-rolled comparisons at fixed fields — not saturated-flux
regression.

**Zonal-vs-drift saturation balance as the first NL suspect.** Once
matched-geometry runs produce (ky, kx) spectrum Pearson ≥ 0.99 but the
absolute flux amplitudes still disagree, the next thing to look at is
the zonal-flow vs drift-wave weight in the saturated state. The CBC
apar-only benchmark has GKW drift-wave-dominant (zonal/drift = 0.78)
while gyaradax is zonal-dominant (zonal/drift = 2.4). Zonal flows do
not transport heat, so the ratio sets the overall amplitude. Tracing
back, the linear γ(ky) peak is shifted from ky=0.7 (GKW) to ky=0.5
(gyaradax) with a 20× under-drive at ky=0.1 — a linear spectrum shift
that the NL mode coupling amplifies into zonal vs drift rebalancing.

**Re-audit 2026-04-18 — CFL bugs fixed, flux parity achieved.**
A four-axis line-by-line audit against GKW source (RHS / field solve /
mode connectivity / CFL) found five concrete bugs, all since fixed:

1. **NL A_∥ CFL missing `2·vthrat_s`** (`solver.py:251-256`). GKW bakes
   `2·vthrat(is)` into `a_apar` at `non_linear_terms.F90:1241`;
   gyaradax used only `vpmax`. Fixed via `pre["vthrat_max"]`.
2. **Collisions in wrong CFL bucket** (`solver.py:320-333`). Moved
   from `tmax4/2.4` to an independent `tmax2` bucket (undivided),
   matching GKW `matdat.F90:1498` for `meth=2`.
3. **`tmax_field` missing `min(2π·lxinv, ky_min²·g_yy)`**
   (`solver.py:817` vs `matdat.F90:1911-1914`). Added.
4. **`calculate_em_fluxes` missing `em_vflux`** + wrong-sign 5D path
   (`integrals.py:672`). Now returns `(em_pflux, em_eflux, em_vflux)`
   and 5D path matches GKW `diagnos_fluxes_vspace.F90:464` with
   `-2·vthrat·vpgr`.
5. **NL CFL FFT-normalisation mismatch (the big one).** GKW uses FFTW
   c2r unnormalized (`|ar| = N·|∂phi|_real` with `N = mrad·mphi`);
   gyaradax uses `jnp.fft.irfft2(norm="backward")` which returns
   `|∂phi|_real` directly. GKW's formula `max|ar|·mrad·lxinv` scales
   as `mrad²·mphi·lxinv·|∂phi|`; gyaradax was scaling as `mrad·|∂phi|`
   — under-conservative by factor `N·lxinv ≈ 15`. Fix: added
   `pre["nl_lxinv"]`, `pre["nl_lyinv"]` and multiplied
   `estimate_nl_timestep` and `gkstep_single::_max_grad_inline` by
   `ycorr = mrad²·mphi·lxinv` (y-branch) and
   `xcorr = mrad·mphi²·lyinv` (x-branch), matching GKW exactly.

What is NOT a bug (audited clean): linear RHS formulas for all terms
I–VIII+X, field solves for φ/A_∥/coupled-B_∥, parallel boundary/mode
connectivity.

**Post-fix validation (2026-04-18).** All EM unit tests (67) pass.

| case | metric | pre-fix | post-fix | GKW |
|------|--------|---------|----------|-----|
| nl_em_apar (β=0.001) | ion eflux | 76.6 | **49.5** | 47.8 |
| nl_em_apar (β=0.001) | elec eflux | 32.2 | **22.1** | 21.6 |
| nl_em_apar (β=0.001) | φ(ky) Pearson | 0.984 | **1.000** | — |
| nl_em_beta01 10× (β=0.01) | ion eflux | 113* | **124** | 128 |
| nl_em_beta01 10× (β=0.01) | elec eflux | 61* | **67** | 70 |
| nl_em_beta01 10× (β=0.01) | dt adaptive range | 0.001 capped | **[0.00038, 0.00297]** | [0.00064, 0.00432] |

*: with `dt=0.001` hard cap (pre-fix blew up otherwise).

Both codes now adapt dt similarly through the linear-to-NL crossover
at β=0.01 (gyra mean dt is 0.53× GKW's, consistent with the
`cfl_safety=0.5` used in that benchmark vs GKW's `fac_dtim_est=0.95`;
the `cfl_safety` default is 0.95, matching GKW). The previously
reported "1.5× over at β=0.001, 0.87× under at β=0.01" flux gap was
a CFL artefact of the FFT-normalisation mismatch, not a physics
issue. The `configs/nl_em_beta01.yaml` `dt=0.001` cap is removed.

**The §10.13 §7 "linear γ(ky) peak shift" narrative remains
unverified**: it was inferred from NL saturation spectra and never
confirmed with a controlled single-mode γ scan. With the CFL fix
closing most of the flux gap, that narrative is likely wrong.

Full investigation trail in `docs/em_debug_report.md` §11.

### 10.14 CUDA backend (electrostatic and electromagnetic)

The CUDA backend (`gyaradax/backends/_cuda.py`, `backends/cuda_kernels/`) covers the same
operator set as the JAX backend: A_par, B_par, adiabatic + A_par, conservative parallel
dissipation and collisions (the collision term is the JAX `collision_rhs`, added to the kernel
output). All species run in one launch of each kernel.

**Linear RHS** (`linear_rhs_fused.cu`). One block per (species, v, mu, kx), one thread per
(s, ky); when ns * nky > 1024 a separate instantiation splits ky into tiles of 1024 / ns. The EM
variants read the mixed variable g and form f = g + g2f A_par on the fly (centre, parallel and
vpar neighbours), so the solver's g -> f pass disappears (`linear_rhs_from_g`). B_par enters
through psi = J0 phi + bpar_chi B_par, which carries terms VII + X and VIII + XI; term V drives
on chi = psi - 2 v_R v_par J0 A_par. The fused parallel stencils are read from per-(species, v,
[mu,] s) tables indexed by the stencil class `clip(pos_par_grid_class + 2, 0, 4)` and the shift
(`s_upar_tab`, `s_t7_tab`, `s_disp_par_tab`), so the kernel neither streams a 9 x 6D coefficient
array nor divides by `sgr_dist`. The EM variants are capped at 64 registers to keep 1024 threads
resident per SM; vpar shards read their edge neighbours from halo buffers (§13.4).

**Field solve** (`field_moments.cu`, kinetic EM only). A_par, phi and B_par numerators are
velocity moments sum_{sp,v,mu} w f; the kernel computes up to two per pass over g, with
f = g + g2f A_par formed in-kernel, using chunked partial sums (deterministic). ES and adiabatic
EM keep the JAX field solve.

**Poisson bracket.** chi is separable in v_par, chi = A(sp, mu, s) + vfac(sp, v) B(sp, mu, s)
with A = J0 phi + bpar_chi B_par, B = J0 A_par and vfac = -2 vthrat vpar, so only 2 nsp nmu ns
potential planes are transformed and chi is formed in real space.
- v5 (`cufft_graph_bracket_*.cu`): batched 2D z2z 2-for-1 inverse + 2D R2C forward with cuFFT
  LTO callbacks. On power-of-two dealiased planes with large batches cuFFT silently skips the
  FP32 C2C load callback (all-zero bracket on e.g. 9 x 5 mode boxes); a plan-time NaN probe
  detects this and switches to an explicit pack kernel, which is also kept whenever it is
  >= 10 % faster (both paths are bit-identical).
- v6 (`cufft_bracket_v6.cu`, needs cuFFTDx from `nvidia-mathdx`): the 2D transforms are split
  into 1D passes. A cuFFT column pass along kx on the 2 nky - 1 retained ky columns only
  (layout [mrad][planes][2 nky - 1]); cuFFTDx row kernels for the potential rows and for df row
  pairs (inverse row FFT, bracket against the real-space potential row, 2-for-1 forward row
  FFT, nky modes kept); a cuFFT column pass on the nky kept columns and an unpack to
  [b_df, nkx, nky]. This removes the zero ky band from the column passes and the full
  real-space round trips between them. Row kernels are instantiated for
  mphi in {16, 32, 36, 40, 64, 96, 128, 144, 192}; other grids, odd plane counts or a build
  without cuFFTDx run v5. `GYARADAX_BRACKET=v5` forces v5.

Performance, memory use, multi-GPU sharding and the CUDA vs JAX parity checks are collected
in §13.

## 11. linearized Fokker-Planck collision operator

Port of GKW's `collision_differential_numu` (`collisionop.f90:1547-2228`)
to JAX. Handles three operator pieces, each independently toggleable:
**pitch-angle scattering** $D_{\theta\theta}$, **energy diffusion**
$D_{vv}$, and **friction** $F_v$. Discretization matches GKW's
conservative flux form on the uniform-$v_\perp$ $\mu$ grid that
gyaradax already uses (see §1.3).

### 11.1 scope

(scope bullets corrected 2026-06-11: earlier MVP-only restrictions —
self-collisions only, freq_override-only, no conservation corrections —
no longer apply and contradicted the bullets above them)

- Adiabatic electrons + single kinetic ion (original MVP) **and**
  kinetic-electron / multi-kinetic-species configurations. In the
  kinetic case `precompute_collisions` vmaps the 9-point stencil over
  species axis, giving shape `(nsp, 9, nv, nmu, ns)`. Each target
  stencil sums over **all background species** (kinetic + adiabatic),
  each pair contributing the full Fokker-Planck operator with
  $\Gamma^{a/b}$, thermal-velocity ratio `vtb = v·vthrat_a/vthrat_b`,
  and mass-ratio friction — inter-species pairs are included
  (linearized test-particle form: no field-particle back-reaction).
- Both `freq_override=True` (scalar `coll_freq` → `Γ^{a/b} =
  Z_a²·Z_b²·coll_freq·de_b·(L_ab/L_ref)/T_a²`) and
  `freq_override=False` (Coulomb-log path via `rref, tref, nref` →
  `Γ^{a/b} = 6.5141e-5·rref·nref/tref²·de_b·Z_a²·Z_b²·L_ab/T_a²`;
  `_coulomb_log_pair` covers e-e, e-i, i-e, and i-i pairs).
- Optional **Xu-style momentum and energy conservation corrections**
  (`coll_mom_conservation`, `coll_ene_conservation`), added via
  `conservation_correction` as a scalar rebalance on top of the base
  operator RHS. Drives `Δp, ΔE → 0` to machine precision.
- `mass_conserve=True` (zero outward flux at $v_\parallel = \pm v_{par,\max}$
  and $v_\perp = v_{\perp,\max}$).
- JAX backend only; no CUDA fused kernel.

### 11.2 operator and discretization

In $(v_\parallel, v_\perp)$ the full operator has the flux form

$$C(f) = \partial_{v_\parallel}\bigl(A\,\partial_{v_\parallel} f + B\,\partial_{v_\perp} f\bigr)
     + \partial_{v_\perp}\bigl(B\,\partial_{v_\parallel} f + C\,\partial_{v_\perp} f\bigr)
     + \partial_{v_\parallel}(G_\parallel f) + \partial_{v_\perp}(G_\perp f)$$

with coefficients assembled from $D_{\theta\theta}$, $D_{vv}$, $F_v$
(manual eqs. 81-109). Velocity-dependent $D$, $F$ are the error-function
formulas in `caldthth`, `caldvv`, `calfv` (`collisionop.f90:697-870`).

Discretely each grid point gets a **9-point $(v_\parallel, v_\perp)$ stencil**
— self, four axis neighbors, and four diagonal corners — precomputed once
in `gyaradax/collisions.py:precompute_collisions` and stored in
`GKPre["coll_stencil"]` with shape `(9, nv, nmu, ns)`. At RHS time
`collision_rhs(df, stencil)` applies the stencil with zero-padded
boundaries. The boundary mass-conserve flux-zeroing is baked into the
precomputed coefficients.

### 11.3 config and plumbing

YAML section `collisions:` and GKW namelist `&collisions` both map to
`GKParams` fields:

```
collisions            master switch (default False)
coll_pitch_angle      D_theta_theta (default True)
coll_en_scatter       D_vv            (default True)
coll_friction         F_v             (default True)
coll_freq             scalar collision frequency for freq_override mode
coll_freq_override    True: scalar coll_freq; False: Coulomb-log path (default True)
coll_mass_conserve    zero boundary flux (default True)
coll_mom_conservation Xu momentum conservation correction (default False)
coll_ene_conservation Xu energy conservation correction (default False)
```

(flag list corrected 2026-06-11: `coll_freq_override`/`coll_mass_conserve`
are no longer MVP-locked, and the conservation flags exist)

All are static pytree fields (resolved at trace time). When
`collisions=False` the compile-time branch in `_linear_rhs_core` drops
out entirely, so existing non-collisional runs are unaffected.

### 11.4 validation

Unit tests in `tests/unit/test_collisions.py`:

| test | what it checks |
|------|----------------|
| `test_full_operator_preserves_maxwellian` | full operator residual on $F_M$ is below $10^{-2}$ (FDT cancellation) |
| `test_pitch_angle_preserves_isotropic_function` | $C_{\text{pitch}}(v^2) \approx 0$ in the interior |
| `test_perturbation_relaxes_to_maxwellian` | a $v_\parallel$-perturbed Maxwellian decays under the operator |
| `test_xu_conservation_zeroes_deltas` | with mom/ene conservation ON, $\Delta p, \Delta E \to 0$ to machine precision |
| `test_coulomb_log_path_runs_and_scales` | freq_override=False yields gamma_pref $=6.5\!\times\!10^{-5} L_\text{ii}$ |
| `test_kinetic_produces_per_species_stencil` | 6D path yields `(nsp, 9, nv, nmu, ns)` and per-species residuals stay small |
| `test_disabled_gives_zero_stencil` | `collisions=False` emits no stencil |

Trajectory parity test in `tests/unit/test_gk_cases.py::test_adiabat_collisions_weak_1step_parity` checks 1-step FDS parity to rel L2 < $10^{-4}$ (measured 1.75e-5).

GKW parity (weak case, `coll_freq=1e-4`, $50\times4\times16$, nperiod=3,
kthrho=0.5, s-alpha, normalization disabled on both codes):

| horizon | rel L2 $\|df\|$ | rel $L_\infty$ | eflux ratio (parseval-corrected) |
|---------|-----------------|-----------------|----------------------------------|
| 1 step | $1.75\times 10^{-5}$ | $1.15\times 10^{-4}$ | **1.0000** |
| 1000 steps | $1.30\times 10^{-3}$ | $1.47\times 10^{-3}$ | 0.9973 |
| 20000 steps | $4.26\times 10^{-2}$ | $4.27\times 10^{-2}$ | 0.9241 |

The 1-step error is **identical** to the no-collisions baseline
(`adiabat_collisions_weak_1step_nocoll`, same 1.75e-5), so the
collision operator itself matches GKW to machine precision — the
remaining 1.75e-5 is pre-existing parallel-stencil/drive-term float
ordering drift. Long-horizon drift at 20k steps is amplified by
exponential ITG growth with normalization disabled (expected).

Run via `python scripts/validate_collisions.py`.

### 11.5 gotchas

- **Parseval convention for single non-zonal mode — FIXED (corrected
  2026-06-11).** gyaradax used to hardcode `parseval[0]=1`, assuming the
  first ky index is the zonal (ky=0) mode, so a `mode_box=False, nmod=1,
  kthrho≠0` run got fluxes 2× smaller than GKW's (the validation script
  compensated by multiplying by 2). The parseval factor is now
  wavenumber-based, `where(|krho| < eps, 1, 2)`, in
  `gyaradax/geometry/{assembly,geom,loaded}.py` — no compensation needed.
- **Normalization timing.** GKW's `normalize_per_toroidal_mode` is
  default false and `normalized` default true (single global factor);
  gyaradax normalizes per-ky. For parity validation the cleanest path
  is to set `normalized=.false.` in the GKW input and force
  `naverage` large in gyaradax, so both run without normalization.
- **Coordinate singularity at $\mu=0$.** Individual operator pieces
  (pitch-only, energy-only, friction-only) have O(1) discretization
  error at the lowest $v_\perp$ grid cell due to the $1/v_\perp$ factor
  in the operator. The *full* operator cancels these to $O(\Delta v^2)$
  thanks to the FDT balance $F_v = 2v\,D_{vv}/T$ — this is why the
  `test_full_operator_preserves_maxwellian` threshold (1e-2) is much
  tighter than the per-term isolation would suggest.
- **CFL contribution.** The collision stencil adds a spectral-radius
  bound to `tmax4` via `pre["coll_stencil"][0]` (diagonal). At
  `coll_freq ≤ 1` this is never the limiting constraint; above
  `coll_freq ~ 10` it can dominate velocity dissipation.

## 12. bugs and limitations

Known open issues and limitations of the current implementation,
grouped by severity/impact.

### 12.1 numerical / solver

- **Adaptive CFL one-step lag (minor).** Each step's $\Delta t$ is
  estimated from the previous step's $\phi$/$A_\parallel$. Fine in
  practice for smoothly evolving states; post-2026-04-18 FFT-norm
  fix, the estimator binds correctly at high β and reproduces GKW's
  adaptive dt trajectory. A potential "pre-step clamp" refinement
  was designed but not needed in practice.
- **`circ` vs `s-alpha` at extreme ky.** Linear gyaradax runs with
  `s-alpha` blow up at ky=0.1 (over-drive) and ky=1.0 (under-damped).
  NL runs use `circ` for this reason. Root cause likely in the
  finite-ε correction to drift tensors or in missing high-ky
  dissipation. (Update 2026-06-11: at finite β the low-ky blow-up is
  the GKW issue #201 parallel-dissipation instability — present in
  *both* codes, β-gated, geometry-independent — and is cured by
  `disp_par_conserve=2, disp_par_conserve_kycut=0.15`; see
  `instabilities/REPORT.md`. The electrostatic high-ky observation
  above remains unexplained.)

### 12.2 electromagnetic

- **Term VIII potential fix (2026-05-08).** GKW's `vd_grad_phi_fm`
  (Term VIII, curvature drift × F_M) acts on `phi` only, not on
  `chi = phi − 2·vthrat·vpar·Apar`. The GKW source has
  `elem%itloc = iphi_ga` with no second `iapar_ga` element for this
  term (linear_terms.f90 lines 2572–2611). Gyaradax incorrectly
  used `gyro_chi` for both Term V and Term VIII. After the fix
  (Term V → chi, Term VIII → phi):
  - ES γ = 0.611 vs GKW 0.589 (+4%, acceptable — needs ~12k steps to converge)
  - EM γ = 0.939 vs GKW 1.132 (−17% at the time; superseded — see next bullet)
  - EM−ES Δγ = 0.328 vs GKW 0.543 — KBM NOW ACTIVATES at physical mass ✓
  All 25 EM unit tests and all 67 non-sharding tests pass after the fix.

- **KBM at physical mass ratio — RESOLVED (corrected 2026-06-11).**
  Superseded by current measurement
  (`instabilities/em_suite_report.json`): linear EM γ matches GKW to
  0.02–0.1% across a β = 0.001–0.01 ladder at kθρ=0.4 through the KBM
  transition (β=0.01: γ = 0.9236 vs GKW 0.9238), and to 0.3% with
  rotation on. The "shared velocity grid" root-cause narrative recorded
  below was wrong on its own terms: GKW uses the *same* shared vpgr
  grid for all species (`velocitygrid.f90:197`) — per-species velocity
  grids are not how GKW handles kinetic electrons, and no grid change
  was needed. (The 2026-05-08 GKW reference of γ = 1.132 came from a
  differently-configured Waltz setup; the current matched-config suite
  gives GKW γ = 0.9238 on the same box.) Historical (superseded)
  analysis:

  The kinetic-electron Alfvénic (KBM) mode at β=0.01 Waltz benchmark
  partially activated after the Term VIII fix:
  GKW gave γ_EM = 1.132 (above the ES γ_ES = 0.589), while
  gyaradax gave γ_EM ≈ 0.939 (−17% vs GKW, EM−ES Δγ = 0.328 vs 0.543).
  The gap was attributed to the shared vpgr ∈ [−3, 3] v_thi grid
  truncating the electron Alfvénic resonance, with per-species
  velocity grids proposed as the fix — refuted above.

  **Convergence note**: both ES and EM growth rates require ~12000 steps
  (t ≈ 12) to converge at these parameters. Measurements at ≤4000 steps
  give transient values (ES: 0.51, EM: 0.94) that underestimate the true
  asymptotic rates. Use block=1000 and ≥15k total steps for diagnostics.

  **Streaming-on-g hypothesis (2026-05-08, ruled out).** GKW's
  `exp_integration.F90:802-814` explicitly converts g→f before
  computing linear terms (`fdis_tmp = g + matg2f * Apar = f`).
  All GKW terms act on f, not g. Gyaradax correctly converts g→f
  before `linear_rhs`. The 17% gap is NOT from streaming-on-g;
  the shared velocity grid explanation stands.

  **Alfvén wave test**: direct ω ∝ 1/√β measurement in flux-tube
  s-alpha geometry is not feasible — toroidal curvature drives
  ITG-like modes even at rlt=rln=0 (Term VIII depends on kdotvd
  which is non-zero in curved geometry). The `em_analytical.ipynb`
  notebook uses a γ vs β sweep instead.

  **What works**: kinetic ES (+4% vs GKW with 15k steps),
  NL EM at β=0.001 (ITG-dominated, small EM corrections),
  linear EM with m_i/m_e=100 (full KBM, Δγ ≈ +0.46) — and, as of
  2026-06-11, linear EM at physical mass ratio to 0.0–0.1% (see the
  correction at the top of this bullet).

- **apar-only NL flux (β=0.001, CBC)** — matches GKW within 4% on
  both species' eflux after the FFT-norm fix (2026-04-18). Previously
  reported "1.5× over" was a CFL artefact. φ(ky,kx) Pearson = 1.00/0.97.
- **β=0.01 apar-only NL** — matches GKW within 3-5% on eflux after
  the FFT-norm fix. Adaptive dt works without a cap and tracks GKW's
  dt trajectory (gyra mean 0.53× GKW, consistent with
  `cfl_safety=0.5` vs GKW's `fac_dtim_est=0.95`).
- **Full EM (apar+bpar) NL flux 0.5–0.7× GKW at CBC.** All
  $B_\parallel$ formulas (coupled solve, chi factor, Term X) verified
  to match GKW to machine precision by isolated tests at fixed
  fields. May also be a CFL-related artefact now that the
  normalisation is fixed — NL re-benchmark still pending. (Linear
  apar+bpar γ now matches GKW to 0.0–0.1% across the β ladder,
  see §10.12 update.)
- **Linear γ(ky) peak shift vs GKW** (old, unverified). Never
  confirmed via a controlled single-mode γ scan; with the CFL fix
  closing the flux gap, the narrative is likely wrong. Re-run if
  the issue resurfaces.
- **Adiabatic + apar absolute amplitude at high β.** The
  `em_adiabat_apar` benchmark (β=0.234) has gyaradax ion eflux
  ~O(90×) smaller than GKW at the same window. Sign and exponential
  growth are correct. Contributing factors: high-β normalization
  convention and the Boltzmann-electron's implied flux contribution
  that GKW reports as 6 columns (both species) while gyaradax
  reports only the kinetic-ion 3 columns.

### 12.3 diagnostics / I/O

- **`save_dumps` fluxes are per-species but not per-kx/ky.** The
  saved `fluxes.npz` array is `(nsp, 3)` (time-collapsed). Spectra
  are saved separately as `kyspec.npz` / `kxspec.npz` from the same
  final state. There is no per-(kx, ky) flux decomposition *output*;
  in-memory, `calculate_fluxes(..., reduce=False)` returns per-(kx, ky)
  flux fields (used by the quasilinear pipeline).
- **EM fluxes are a separate file.** `fluxes_em.npz` sits alongside
  `fluxes.npz`; it carries `(pflux_em, eflux_em, vflux_em)` shaped like
  the ES output — `em_vflux` was added in the 2026-04-18 audit (§10.13
  item 4), matching GKW's `diagnos_fluxes_vspace.F90:464` (corrected
  2026-06-11; an earlier note here claimed the vflux slot was zero).
- **B_par flux is kinetic-only.** `calculate_em_fluxes` implements the
  compressional B_par flux in the 6D (kinetic) branch only; the 5D
  (adiabatic) branch raises on `bpar` instead of silently dropping it
  (guard added 2026-06-11, `quasilinear/REVIEW.md` §5).

### 12.4 scope / missing features (intentional)

See §7.2 for the full not-implemented list. The most commonly
requested gaps (updated 2026-06-11 — collisions, Coriolis rotation and
Miller geometry are now implemented): centrifugal rotation terms,
neoclassical corrections, global/radial-varying profiles, implicit
time stepping.

## 13. performance and HPC

Performance work on the solver step (2026-10): the CUDA backend covers the electromagnetic
operators, both backends need about a third of the memory on large grids, and multi-GPU runs keep
df distributed. This section collects the design choices, measurements and the verification
method; §10.14 describes the CUDA operators. Timings are wall time per `gksolve` step unless noted
(H100 NVL on cayman, B300 on scorpion); kernel times are CUPTI times from `jax.profiler`.
`gyaradax bench` reproduces step timings.

### 13.1 where the step time goes

An RK4 step evaluates four times: the field solve (velocity moments of g giving phi, A_par,
B_par), the linear RHS (parallel and vpar stencils, drifts, drives, dissipation) and the Poisson
bracket (pseudospectral, FP32 FFTs in mixed precision). On a production grid (H100, CUDA,
2 x 32 x 8 x 16 x 85 x 32, mixed precision) one evaluation costs ~1.3 ms for the linear RHS,
~2.8-3.0 ms for the bracket and ~0.5 ms for the EM field moments; the rest is XLA elementwise
work (RK combinations, CFL). The linear kernel is FP64 instruction bound: ~60 ps per grid point on
H100 and ~250 ps on B300, whose FP64 rate is much lower, so the CUDA lead over JAX shrinks on B300
(3.0x against 3.9x on the large grid) and H100 is the reference platform for kernel timings.

### 13.2 single-GPU kernels

Per evaluation, H100, 2 x 32 x 8 x 16 x 85 x 32, mixed precision (master = before this work):

| component | JAX | CUDA, master | CUDA, now | change |
|---|---|---|---|---|
| linear RHS, ES | 5.34 ms | 1.78 ms | 1.27 ms | per-class stencil tables, species in one launch |
| linear RHS, A_par | - | not supported | 1.36 ms | f = g + g2f A_par formed in-kernel |
| bracket, ES | 10.2 ms | 5.0 ms | 2.8 ms | tuned v5 (4.46 ms), then v6 |
| bracket, A_par | - | not supported | 3.0 ms | separable chi, two potential planes per (sp, mu, s) |
| bracket, FP64 | - | - | 3.32 ms | v6 (v5: 5.96 ms) |
| EM field solve | 1.03 ms | - | 0.50 ms | fused `field_moments` kernel |

- **Linear RHS.** The parallel stencils come from small per-class tables instead of a streamed
  9 x 6D coefficient array (no `sgr_dist` division either); all species run in one launch (master
  looped over species); EM variants form f from g in-kernel, removing the solver's g -> f pass.
  Register use decides occupancy: the EM variants are capped at 64 registers, and the
  compile-time-sized ES variant must stay at 64 (a 76-register build halves occupancy, 2.08 against
  1.27 ms). ns * nky > 1024 runs a ky-tiled instantiation; vpar shards use HALO instantiations of
  the same sized kernels (§13.4).
- **Field moments.** Up to two velocity moments per pass over g, with g -> f in-kernel and
  chunked (deterministic) partial sums.
- **Bracket v5.** The EM potential is separable in vpar, so only the potentials A and B are
  transformed. On power-of-two dealiased planes with large batches cuFFT silently skips the FP32
  load callback (master returned an all-zero bracket on e.g. 9 x 5 mode boxes); a plan-time NaN
  probe detects it and switches to an explicit pack kernel, which is also kept when >= 10 % faster.
- **Bracket v6.** The 2D transforms become 1D passes: cuFFT along kx on the 2 nky - 1 retained ky
  columns only, cuFFTDx row kernels fusing the inverse row FFT, the bracket and the forward row FFT,
  and a final column pass on the nky kept columns. This removes the zero ky band from the column
  FFTs and the real-space round trips between passes.
- **Tried, not kept.** A vpar-fastest block order (more stack spills, EM linear 1.43 -> 1.98 ms);
  staged v6 row kernels (slower); hand-written FFTs (cuFFT / cuFFTDx are used instead).

Wall time per `gksolve` step on H100 NVL (mixed precision unless noted; JAX on its default R2C
path; master CUDA had no EM support). Medium is 2 x 64 x 16 x 16 x 85 x 32, large
2 x 64 x 16 x 32 x 85 x 64; entries marked * needed the platform allocator or a 0.95 memory
fraction to avoid BFC fragmentation, and master JAX does not fit the large grid on H100. The large
A_par + B_par case does not fit one 94 GB H100 for either backend: the eager setup (init +
`linear_precompute`, ~38 GiB peak) leaves XLA's pool fragmented, and with preallocation the CUDA
bracket workspaces no longer fit outside the pool; it runs sharded or on B300 (rows below, where
the FP64-bound linear kernel narrows the CUDA lead).

| case | grid | master JAX | master CUDA | JAX | CUDA | CUDA / JAX |
|---|---|---|---|---|---|---|
| ES adiabatic | 32 x 8 x 16 x 85 x 32 | 33.0 ms | 19.4 ms | 33.9 ms | 11.3 ms | 3.0x |
| ES kinetic | 2 x 32 x 8 x 16 x 85 x 32 | 64.4 ms | 39.9 ms | 66.2 ms | 24.4 ms | 2.7x |
| kinetic A_par | 2 x 32 x 8 x 16 x 85 x 32 | 101.0 ms | - | 103.9 ms | 31.2 ms | 3.3x |
| kinetic A_par + B_par | 2 x 32 x 8 x 16 x 85 x 32 | 104.8 ms | - | 105.6 ms | 31.6 ms | 3.3x |
| adiabatic + A_par | 32 x 8 x 16 x 85 x 32 | 49.4 ms | - | 51.1 ms | 13.5 ms | 3.8x |
| waltz beta = 0.01 | 2 x 32 x 8 x 16 x 55 x 8 | 24.0 ms | - | 24.1 ms | 8.4 ms | 2.9x |
| waltz beta = 0.01, FP64 | 2 x 32 x 8 x 16 x 55 x 8 | 31.8 ms | - | 31.7 ms | 8.6 ms | 3.7x |
| waltz A_par + B_par | 2 x 32 x 8 x 16 x 55 x 8 | 24.0 ms | - | 25.6 ms | 8.6 ms | 3.0x |
| CBC A_par (small) | nl_em_apar | 3.5 ms | - | 3.5 ms | 1.8 ms | 2.0x |
| waltz, linear | 2 x 32 x 8 x 16 x 55 x 8 | 4.4 ms | - | 3.6 ms | 2.0 ms | 1.8x |
| ES kinetic | medium | 271.7 ms* | - | 239.8 ms | 93.7 ms | 2.6x |
| waltz beta = 0.01 | medium | 448.9 ms* | - | 428.1 ms* | 103.1 ms | 4.2x |
| waltz beta = 0.01 | large | - | - | 1613.3 ms* | 416.1 ms | 3.9x |
| waltz beta = 0.01 | large, B300 | 1737 ms* | - | 1737 ms | 579 ms | 3.0x |
| waltz A_par + B_par | large, B300 | out of memory | - | 1832 ms | 593 ms | 3.1x |

### 13.3 memory

The fused parallel stencils are kept only as the per-class tables of
§10.14 (the 9 x 6D `s_total_*` arrays were 5.6 df of device memory, and the JAX species vmap
materialised a transposed copy of the 4.5 df `s_total_t7` on top). `gkstep_single` forms the RK4
update as a running sum in the association of the closed form (prev + dt/6 k1 + dt/3 k2 + dt/3 k3
+ dt/6 k4), behind `optimization_barrier`s, so each stage's k (linear and bracket outputs) is
freed before the next stage. The JAX bracket loops over species, then over vpar chunks, once one
full-batch real-space intermediate would exceed `_NL_CHUNK_BYTES` (1 GiB); cuFFT results do not
depend on the batch size, so the looped and batched paths agree bitwise (checked on 20-step
trajectories). The bracket workspaces of the CUDA backend (v6: ~13 GiB on the grid below) are
allocated outside XLA's pool. On production grids the table gathers and barriers cost the JAX path
1-3 % per step against master (min of interleaved runs on H100).

Peak device memory of one compiled `gksolve` (XLA memory analysis, 2 x 64 x 16 x 32 x 85 x 64,
mixed precision, A_par; one df is 5.3 GiB):

| | precompute | temporaries | peak |
|---|---|---|---|
| JAX, master | 36.2 GiB | 121.3 GiB | 168.0 GiB |
| JAX, now | 10.8 GiB | 41.3 GiB | 62.7 GiB |
| CUDA, before these changes | 36.2 GiB (22.8 unused) | 47.8 GiB | 89.3 GiB |
| CUDA, now | 10.8 GiB | 21.3 GiB | 40.0 GiB (+ ~13 GiB bracket workspace) |

- **Allocator.** The CLI runs with `XLA_PYTHON_CLIENT_PREALLOCATE=false`; the pool then grows in
  separate regions, and the eager setup (init + `linear_precompute`, up to ~38 GiB on the large
  grid) can leave it fragmented, so a 26-41 GiB temporary allocation fails although the compiled
  peak fits. `--mem-fraction 0.95` or `XLA_PYTHON_CLIENT_ALLOCATOR=platform` avoid it, and so does
  sharding.
- **Bracket workspaces.** The CUDA bracket allocates its buffers outside XLA's pool (v6 on the
  large grid: 8.6 GiB + 4.2 GiB plus cuFFT work areas); with preallocation they must fit in the
  remaining 25 %.

### 13.4 multi-GPU sharding

GSPMD cannot partition the opaque FFI kernels, nor the bracket's FFT pipeline: it all-gathered
every CUDA kernel operand (~5 df per step), and on the JAX path the bracket's FFT intermediates
(~3.4 GiB per step for a 55 MiB df, master included). `gksolve` passes the mesh to `create_ops`,
and the bracket (both backends), the CUDA linear and field-moment kernels and the JAX vpar stencil
run on the local (sp, vp, mu) blocks through `sharding.velocity_map` (`shard_map`); only the field
moments are all-reduced (a few MiB per step). Sharding vpar exchanges the two vpar planes next to
each shard edge per stage (`sharding.vpar_halo_planes`): df, plus F_M and the g2f factor for the
in-kernel g -> f. The CUDA kernel reads them from separate halo buffers, the JAX stencil from a
halo-extended block; planes past the grid ends are zero, so interior points are computed exactly
as on one device. The sharded precompute keeps scalar entries (`dvp`, `sgr_dist`) concrete, which
the CUDA kernels need. Multi-GPU runs need NCCL (`nvidia-nccl-cu13`, in the `cuda13` extra).

- **Which axis.** Species and mu shards need nothing from their neighbours; vpar shards receive
  4 nmu ns nkx nky complex values per device and stage (178 MB on the large grid, ~0.1 ms over
  NVLink against a ~290 ms step). Measured: mu and species sharding 292 and 298 ms on the large grid
  (two quiet B300); vpar 356 ms with the first, copying halo, and on par with mu after the halo
  buffers (medium grid, two loaded B300: 207-272 against 261-270 ms for CUDA, 765-768 against
  729-764 ms for JAX, with equal or lower memory per device). In practice the axes are equivalent;
  `--n-gpus N` fills species, then mu, then vpar, which need no halo and balance exactly, and keeps
  vpar as the reserve that allows up to nsp nmu nvpar / 2 GPUs.
- **Correctness.** Sharded runs match single-device runs at round-off, not bitwise: the field
  moments are summed per shard and then across shards. FP64: 2e-16 in df over 100 adiabatic steps
  (vpar), 1e-15 to 1e-14 over 3 EM steps (every axis); mixed precision: ~1e-9 in df after 40-100
  steps (FP32 bracket round-off amplified by the nonlinear dynamics).

On the 2 x 32 x 8 x 16 x 55 x 8 waltz A_par + B_par case on two B300 the step went from 19.1 to
11.6 ms (CUDA, mu = 2) and from 55.5 to 31.9 ms (JAX, mu = 2).

Large grid (2 x 64 x 16 x 32 x 85 x 64, A_par + B_par, mixed precision) on two B300:

| backend | mesh | step | peak per device |
|---|---|---|---|
| CUDA | one device | 593 ms | - |
| CUDA | mu = 2 | 292 ms | 25.6 GiB |
| CUDA | sp = 2 | 298 ms | 25.6 GiB |
| JAX | one device | 1832 ms | - |
| JAX | mu = 2 | 1298 ms | 52.7 GiB |

### 13.5 behaviour and verification

The JAX backend is bitwise identical to master: df, phi, dt, state and the flux
diagnostics over 20-step trajectories (fixed and adaptive dt, FP64 and mixed precision; ES, A_par,
B_par, conservative dissipation, collisions). Across separate processes the flux diagnostic can
differ in the last bit for master itself (XLA autotuning picks reduction configurations from
timings). With `GYARADAX_BRACKET=v5` the CUDA ES results are bit-identical to the original CUDA
backend; v6 changes the bracket at round-off level (mixed precision ~1e-7, FP64 ~1e-16 per
evaluation), the same level at which the JAX and CUDA backends differ.

CUDA vs JAX parity per evaluation (rel. L2, H100; the same on B300). Grids are the configs'
production grids; kinetic rows give the worse species.

| case | linear (FP64) | bracket FP64 | bracket mixed |
|---|---|---|---|
| ES adiabatic (iteration_13) | 1.5e-16 | 1.1e-15 | 6.2e-7 |
| ES kinetic | 1.5e-16 | 1.6e-15 | 8.4e-7 |
| kinetic A_par | 1.5e-16 | 1.6e-15 | 8.5e-7 |
| kinetic A_par + B_par | 2.0e-16 | 1.6e-15 | 8.2e-7 |
| kinetic B_par only | 1.9e-16 | 1.6e-15 | 8.0e-7 |
| waltz beta = 0.01 (A_par) | 3.0e-16 | 2.2e-15 | 1.3e-6 |
| waltz A_par + B_par | 4.0e-16 | 2.1e-15 | 1.2e-6 |
| CBC A_par (small) | 3.2e-16 | 1.7e-15 | 9.1e-7 |
| adiabatic + A_par | 1.5e-16 | 1.1e-15 | 6.2e-7 |
| waltz, conservative dissipation | 3.0e-16 | 2.2e-15 | 1.3e-6 |
| ES adiabatic, conservative dissipation | 1.5e-16 | 1.1e-15 | 6.2e-7 |

Method: 20-step `gksolve` trajectories (fixed and adaptive dt, FP64 and mixed precision) of nine
configurations per backend, compared bitwise against a frozen copy of the previous code, with XLA
autotuning off (`--xla_gpu_autotune_level=0`) for the flux diagnostics; per-kernel register and
stack use (`cuobjdump --dump-resource-usage`) compared across builds; CUDA vs JAX parity per term;
timings as the minimum over interleaved repeats when the GPU is shared.

### 13.6 recommendations

- Production runs: CUDA backend (`--backend cuda`), mixed precision, H100-class GPUs for the
  FP64-heavy linear kernel.
- Gradients and bitwise references: JAX backend.
- Large grids: shard (`--n-gpus N`); close to the memory limit on one GPU, `--mem-fraction 0.95`.
- `GYARADAX_BRACKET=v5` reproduces the v5 bracket bitwise (and the master CUDA ES results).
- `gyaradax info` reports whether the CUDA library, the v6 bracket and NCCL are available.
- `gyaradax run ... --telemetry` records per-block timings, dt and device memory in the run
  directory (`telemetry.jsonl`); `--profile` adds a profiler trace of one block and its GPU kernel
  breakdown (`profile_summary.json`). Both are off by default (docs/CLI.md).
