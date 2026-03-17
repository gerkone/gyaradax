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

### 1.2 phase space coordinates

| coordinate | symbol | grid | range |
|-----------|--------|------|-------|
| parallel velocity | $v_\parallel$ | uniform | $[-v_{max}, v_{max}]$, typically $\pm 3 v_{th}$ |
| magnetic moment | $\mu$ | Gauss-Laguerre | $[0, \infty)$, 8 quadrature points |
| field-line coordinate | $s$ | uniform | $[-0.5, 0.5]$ in units of $2\pi q$ |
| radial wavenumber | $k_x$ | discrete | from mode connectivity |
| binormal wavenumber | $k_y$ | uniform | $[0, k_{y,max}]$ |

The standard grid is `(nvpar=32, nmu=8, ns=16, nkx=85, nky=32)`.

### 1.3 species model

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

**CFL-adaptive timestep** (`adaptive_dt=True`, default for kinetic electrons):
the timestep is adjusted each step to satisfy two CFL constraints:

1. **Nonlinear ExB CFL**: $\Delta t_{NL} = \sigma \times 2 / \max|\nabla\phi|$,
   computed from the dealiased real-space potential gradient. Safety factor
   $\sigma = 0.95$ by default (`cfl_safety` parameter).

2. **Linear parallel streaming CFL**: $\Delta t_{par} = 0.5 \times \Delta s / \max|v_{\parallel,s}|$
   and $\Delta t_{trap} = 0.5 \times \Delta v_\parallel / \max|v_{trap,s}|$,
   where the characteristic speeds include the per-species $v_{th,s}/v_{th,ref}$
   scaling. For kinetic electrons with $v_{th,e}/v_{th,i} \approx 60$, this is
   the binding constraint (dt ~ 0.002 vs input dt = 0.004).

The effective timestep is $\Delta t = \min(\Delta t_{NL}, \Delta t_{par}, \Delta t_{trap}, \Delta t_{input})$.
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
| `geometry.py` | load geometry from GKW `geom.dat` and `input.dat` |
| `stencils.py` | finite difference coefficient tables |
| `utils.py` | K-dump loading, checkpoint save/load, diagnostics |
| `gksimulate.py` | high-level simulation runner from YAML config |
| `plot_utils.py` | publication-quality visualization |

### 4.2 key interfaces

```python
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
| geometry | `geom.f90` | geometry metric loading |
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
- CFL-adaptive timestep (nonlinear ExB + linear parallel streaming)

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
| CFL vs GKW dtim | all 3 cases | `ratio(dt_est, dtim)` | `0.3 – 3.0` |
| adaptive CFL 20 steps | all 3 cases | finiteness (dt=0.004) | pass |
| adiabatic fallback | 4 iterations | shapes + finiteness | pass |
