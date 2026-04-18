# gyaradax — gyrokinetic solver in JAX

A JAX reimplementation of the GKW Fortran gyrokinetic code for local flux-tube
simulations. Supports both adiabatic and kinetic electron configurations.

## 1. physics overview

gyaradax solves the electrostatic gyrokinetic Vlasov-Poisson system in the
local (flux-tube) limit. The code evolves the perturbed gyrocenter distribution
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
  coefficient `disp_par`, using upwinded 4th-derivative stencils
- **velocity dissipation**: 4th-order smoothing in $v_\parallel$,
  coefficient `disp_vp`
- **perpendicular hyper-dissipation**: spectral damping
  $(k_x/k_{x,max})^4 + (k_y/k_{y,max})^4$, coefficients `disp_x`, `disp_y`

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

$$Q_s = \text{Im} \sum_{s, k_x, k_y, v_\parallel, \mu} P_{k_y} \Delta s \cdot k_y E_\alpha \left(v_\parallel^2 + 2\mu B\right) \delta f_s (J_0 \phi)^* B \Delta\mu \Delta v_\parallel \Delta s \cdot d^2 X$$

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

4. **Nonlinear ExB CFL**: $\Delta t_{NL} = \sigma \times 2 / \max|\nabla\phi|$,
   computed from the dealiased real-space potential gradient. Safety factor
   $\sigma = 0.95$ by default (`cfl_safety` parameter).

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

Fused stencils (`s_total_upar`, `s_total_t7`) combine the streaming velocity
with the upwinded finite-difference coefficients into a single array, avoiding
per-step branching on the sign of $v_\parallel$.

## 4. code architecture

### 4.1 modules

| module | purpose |
|--------|---------|
| `solver.py` | RK4 integrator, linear RHS, nonlinear term III, precomputation, CFL |
| `integrals.py` | phi solvers (adiabatic + kinetic), flux calculations |
| `params.py` | `GKParams` dataclass, config/input.dat loading |
| `geometry.py` | analytic circular geometry + mode connectivity (primary path) |
| `stencils.py` | finite difference coefficient tables |
| `utils.py` | K-dump loading, checkpoint save/load, diagnostics, GKW file-loading (`load_geometry`, `parse_input_dat`) |
| `simulate.py` | high-level simulation runner from YAML config |
| `diag.py` | spectral diagnostics, 1D projections, nonlinear term analysis |
| `bootstrap.py` | centralized JAX configuration and device initialization |
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

3. `ops.linear_rhs`: backend handles 5D/6D dispatch internally. JAX backend
   uses `jax.vmap` over species for 6D; CUDA backend flattens species dimension
   for uniform params. Each species gets its own precomputed coefficients.

4. `ops.nonlinear_term_iii`: backend handles 5D/6D dispatch. JAX backend vmaps
   over species with per-species Bessel; CUDA backend raises NotImplementedError
   for 6D (kinetic electrons not yet supported).

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
- `collisions.tex`: collision operator (not implemented in gyaradax)
- `rotation.tex`: centrifugal and Coriolis effects (not implemented)
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
- linearized Fokker-Planck collision operator (pitch-angle, energy diffusion, friction; MVP scope — see §12)
- all 7 linear RHS terms (I, II, III, IV, V, VII, VIII) plus EM terms X and XI
- nonlinear ExB advection (pseudospectral, spectral Poisson bracket)
- 4th-order parallel and velocity dissipation
- perpendicular hyper-dissipation
- RK4 explicit time integration
- per-$k_y$ normalization (linear mode)
- CFL-adaptive timestep (nonlinear ExB + linear parallel streaming + EM Alfvén)
- standalone circular geometry computation (no precomputed GKW files needed)

### 7.2 not implemented

| feature | GKW module | notes |
|---------|-----------|-------|
| collision conservation corrections | `collisionop.f90` | `mom_conservation`, `ene_conservation` (base operator is in §11) |
| inter-species collisions | `collisionop.f90` | only self-collisions in the kinetic-electron MVP |
| neoclassical | `neoclassics.f90` | equilibrium corrections to $F_M$ |
| rotation | `rotation.f90` | centrifugal (`cfen`), Coriolis, toroidal shear |
| energetic particles | `components.f90` | `types='EP'`, `types='alpha'` |
| implicit integration | `imp_integration.F90` | for stiff parallel streaming |
| RK-Chebyshev | `exp_integration.F90` | for diffusion-dominated regimes |
| real-space nonlinear | `non_linear_terms.F90` | Arakawa bracket variant |
| global effects | `global.f90` | radial profile variation |
| source terms | various | Krook operator, external sources |
| Miller / general geometry | `geom.f90` | Lapillonne circular and s-alpha are the only supported models |

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

## 9. circular geometry model (`geometry.py`)

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
`input.dat` doesn't set `geom_type`; gyaradax defaults to `circ`
(Lapillonne). These geometries differ by ~50% in linear γ at finite ε
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

## 11. linearized Fokker-Planck collision operator

Port of GKW's `collision_differential_numu` (`collisionop.f90:1547-2228`)
to JAX. Handles three operator pieces, each independently toggleable:
**pitch-angle scattering** $D_{\theta\theta}$, **energy diffusion**
$D_{vv}$, and **friction** $F_v$. Discretization matches GKW's
conservative flux form on the uniform-$v_\perp$ $\mu$ grid that
gyaradax already uses (see §1.3).

### 11.1 scope

- Adiabatic electrons + single kinetic ion (original MVP) **and**
  kinetic-electron / multi-kinetic-species configurations. In the
  kinetic case `precompute_collisions` vmaps the 9-point stencil over
  species axis, giving shape `(nsp, 9, nv, nmu, ns)`. Each species
  uses its own prefactor; **no species-species coupling** — operator
  is self-collision only.
- Both `freq_override=True` (scalar `coll_freq` → `gamma_pref =
  coll_freq·de/tmp²`) and `freq_override=False` (Coulomb-log path
  via `rref, tref, nref` → `gamma_pref = 6.5141e-5·rref·nref/tref²·
  de·Z⁴·L_ii/tmp²`; ion-ion only).
- Optional **Xu-style momentum and energy conservation corrections**
  (`coll_mom_conservation`, `coll_ene_conservation`), added via
  `conservation_correction` as a scalar rebalance on top of the base
  operator RHS. Drives `Δp, ΔE → 0` to machine precision.
- `mass_conserve=True` (zero outward flux at velocity boundaries).
- `freq_override=True` only: the species-pair Coulomb-log machinery is
  collapsed to a single scalar `coll_freq`, giving $\Gamma^{a/a} =
  \mathrm{coll\_freq} \cdot n_a / T_a^2$. Reference density/temperature
  path (`rref/tref/nref`) not yet used.
- `mass_conserve=True` (zero outward flux at $v_\parallel = \pm v_{par,\max}$
  and $v_\perp = v_{\perp,\max}$).
- Momentum/energy conservation corrections not implemented — the base
  operator already conserves particles but not like-particle
  momentum/energy (manual §7).
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
coll_freq_override    must be True in MVP
coll_mass_conserve    must be True in MVP
```

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

- **Parseval convention for single non-zonal mode.** gyaradax hardcodes
  `parseval[0]=1` assuming the first ky index is the zonal (ky=0) mode.
  For runs with `mode_box=False, nmod=1, kthrho≠0` (like the validation
  case), the single mode is non-zonal but still gets parseval=1, so
  fluxes are 2× smaller than GKW's. The validation script multiplies
  gyaradax fluxes by 2 for a fair comparison. Fix is a geometry-level
  change (`geometry.py:557,760`) — not MVP-blocking.
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

### 11.1 numerical / solver

- **Adaptive CFL one-step lag.** Each step's $\Delta t$ is estimated
  from the previous step's $\phi$. Fine in practice for smoothly
  evolving states; can miss sharp NL bursts.
- **`estimate_nl_timestep` under-triggers.** On the standard kinetic
  reference runs, GKW's NL CFL reduces $\Delta t$ from 2.13e-3 to
  1.01e-3 mid-run while gyaradax's estimator returns `dt_input` for
  all steps. The computed $\max|\nabla\phi|$ is too small. Contributes
  to 1.5–3.4× flux trajectory divergence by window 3.
- **`circ` vs `s-alpha` at extreme ky.** Linear gyaradax runs with
  `s-alpha` blow up at ky=0.1 (over-drive) and ky=1.0 (under-damped).
  NL runs use `circ` for this reason. Root cause likely in the
  finite-ε correction to drift tensors or in missing high-ky
  dissipation.

### 11.2 electromagnetic

- **apar-only NL flux magnitude ~1.5× GKW at CBC.** Matched geometry,
  Pearson ≥ 0.99 on both (ky, kx) spectra. Species ratio matches GKW
  to 5%. The residual amplitude gap is within the window-phase
  fluctuation band (~60% of mean) for a single-run comparison.
- **Full EM (apar+bpar) NL flux 0.5–0.7× GKW at CBC.** All
  $B_\parallel$ formulas (coupled solve, chi factor, Term X) verified
  to match GKW to machine precision by isolated tests at fixed
  fields. The discrepancy is NL-dynamics amplification of a ~0.08%
  RHS perturbation — not a bug in any one formula but a sensitivity
  difference between codes, likely in upwinding direction during
  bursts or subtle phase relationships.
- **Linear γ(ky) peak shift vs GKW.** Clean linear scan at CBC
  kinetic EM: gyaradax peak at ky=0.5 (γ=0.59), GKW peak at ky=0.7
  (γ=0.55). At ky=0.1, gyaradax γ=0.005 vs GKW 0.095 (20× under-drive).
  Suspects: low-ky EM drive (J0/Γ at small b), k_perp definition,
  high-ky dissipation.
- **Adiabatic + apar absolute amplitude at high β.** The
  `em_adiabat_apar` benchmark (β=0.234) has gyaradax ion eflux
  ~O(90×) smaller than GKW at the same window. Sign and exponential
  growth are correct. Contributing factors: high-β normalization
  convention and the Boltzmann-electron's implied flux contribution
  that GKW reports as 6 columns (both species) while gyaradax
  reports only the kinetic-ion 3 columns.

### 11.3 diagnostics / I/O

- **`save_dumps` fluxes are per-species but not per-kx/ky.** The
  saved `fluxes.npz` array is `(nsp, 3)` (time-collapsed). Spectra
  are saved separately as `kyspec.npz` / `kxspec.npz` from the same
  final state. There is no per-(kx, ky) flux decomposition output.
- **EM fluxes are a separate file.** `fluxes_em.npz` sits alongside
  `fluxes.npz`; it carries `(pflux_em, eflux_em, 0)` shaped like the
  ES output. `vflux_em` is not tracked (GKW's
  `calculate_em_fluxes` does not produce it either).

### 11.4 scope / missing features (intentional)

See §7.2 for the full not-implemented list. The most commonly
requested gaps: collisions, rotation/Coriolis, Miller geometry,
global/radial-varying profiles, implicit time stepping.
