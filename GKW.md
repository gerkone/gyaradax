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

where $P_{k_y}$ is the Parseval factor (1 for $k_y=0$, $2N_{ky}$ otherwise)
and $d^2 X$ is the velocity-space volume element.

## 3. numerical methods

### 3.1 time integration

Explicit Runge-Kutta 4th order (RK4). Each small timestep requires 4 RHS
evaluations, each involving a full phi solve + linear terms + (optionally)
nonlinear FFTs.

The large-step cadence `naverage` groups small steps for diagnostic output.
In linear mode, per-$k_y$ normalization is applied at large-step boundaries.

**CFL-adaptive timestep** (optional, `adaptive_dt=True`): the timestep is
adjusted each step based on the maximum real-space ExB velocity gradient:
$\Delta t = \sigma \times 2 / \max|\nabla\phi|$, clamped to the input `dt`.
Safety factor $\sigma = 0.95$ by default.

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
| `geometry.py` | load geometry from GKW `geom.dat` and `input.dat` |
| `analytic_geometry.py` | compute geometry analytically from equilibrium parameters (no GKW files) |
| `stencils.py` | finite difference coefficient tables |
| `utils.py` | K-dump loading, checkpoint save/load, diagnostics |
| `simulate.py` | high-level simulation runner from YAML config |
| `plot_utils.py` | publication-quality visualization |

### 4.2 key interfaces

```python
# standalone geometry (no GKW files needed)
geometry = compute_geometry(q=7.73, shat=2.14, eps=0.19, ns=16, nkx=85, nky=32, nvpar=32, nmu=8)

# or load from GKW files
geometry = load_geometry("/path/to/gkw_run")

# single/multi-step solver
next_df, (phi, fluxes), state = gksolve(df, geometry, params, state, n_steps)

# phi only (adiabatic)
phi = calculate_phi(geom_tensors(geometry, params=params), df)

# phi only (kinetic)
phi = calculate_phi_kinetic(geometry, df)

# phi + fluxes
phi, (pflux, eflux, vflux) = get_integrals(df, geometry, params=params)

# per-species kinetic fluxes
per_sp_fluxes = calculate_fluxes_kinetic(geometry, df, phi)  # (nsp, 3)
```

### 4.3 multi-species implementation

When `adiabatic_electrons=False`, the solver:

1. `linear_precompute`: computes per-species coefficients with shape
   `(nsp, nvpar, nmu, ns, nkx, nky)` from geometry arrays.

2. `_compute_phi`: calls `calculate_phi_kinetic` which sums the Poisson
   integral over all species.

3. `_compute_linear_rhs`: uses `jax.vmap` over the species axis. Each species
   gets its own precomputed coefficients; all share the same $\phi$.

4. `_compute_nonlinear_rhs`: vmaps `nonlinear_term_iii` over species, each
   with its own Bessel function for gyro-averaging.

The adiabatic path is completely untouched — branching is via Python `if/else`
on `params.adiabatic_electrons` (a static pytree field resolved at trace time).

## 5. GKW Fortran reference

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

- electrostatic gyrokinetics (no $A_\parallel$, no $B_\parallel$)
- adiabatic and kinetic electron models
- all 7 linear RHS terms (I, II, III, IV, V, VII, VIII)
- nonlinear ExB advection (pseudospectral, spectral Poisson bracket)
- 4th-order parallel and velocity dissipation
- perpendicular hyper-dissipation
- RK4 explicit time integration
- per-$k_y$ normalization (linear mode)
- CFL-adaptive timestep (optional)
- standalone circular geometry computation (no precomputed GKW files needed)

### 7.2 not implemented

| feature | GKW module | notes |
|---------|-----------|-------|
| electromagnetic ($A_\parallel$) | `fields.F90`, `ampere_*` | shear Alfvén physics |
| compressional ($B_\parallel$) | `fields.F90`, `bpar_*` | magnetic compression |
| collisions | `collisions.f90` | Lenard-Bernstein, Lorentz, full FP |
| neoclassical | `neoclassics.f90` | equilibrium corrections to $F_M$ |
| rotation | `rotation.f90` | centrifugal (`cfen`), Coriolis, toroidal shear |
| energetic particles | `components.f90` | `types='EP'`, `types='alpha'` |
| implicit integration | `imp_integration.F90` | for stiff parallel streaming |
| RK-Chebyshev | `exp_integration.F90` | for diffusion-dominated regimes |
| real-space nonlinear | `non_linear_terms.F90` | Arakawa bracket variant |
| global effects | `global.f90` | radial profile variation |
| source terms | various | Krook operator, external sources |

### 7.3 known limitations

- adaptive CFL uses one-step lag (current step uses previous step's CFL estimate)
  data comes from the geometry dict
- no multi-species output in `save_dumps` (fluxes are summed over species)

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
| adiabatic fallback | 4 iterations | shapes + finiteness | pass |

## 9. circular geometry model (`analytic_geometry.py`)

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
| `dfun`, `hfun`, `ifun` (all components) | $< 2 \times 10^{-5}$ |
| velocity / wavenumber grids | $< 10^{-6}$ |

68 tests in `tests/unit/test_analytic_geometry.py`.

