# GKW Technical Specification & Reimplementation Guide

## 1. Physical Model: Gyrokinetic Framework

### 1.1 Ordering and Normalization
GKW solves the gyrokinetic equations up to first order in the Larmor radius over the major radius ($\rho_* = \rho_i / R \ll 1$).
- **Length Scale:** Major radius $R$.
- **Velocity Scale:** Thermal velocity $v_{th} = \sqrt{2T/m}$ (Note: GKW manual says $v_{th}$ is used for normalization, but check if factor of 2 is included).
- **Time Scale:** $R / v_{th}$.
- **Potential Scale:** $T/e$.
- **Gradients:**
  - Parallel: $R \nabla_\parallel \approx 1$.
  - Perpendicular: $R \nabla_\perp \approx 1/\rho_*$.
- **Field Ordering:**
  - $\phi \approx (T/e) \rho_*$
  - $A_\parallel \approx (T / e v_{th}) \rho_*$

### 1.2 Lagrangian and Equations of Motion
The gyro-center Lagrangian in a rotating frame is:
$$\Gamma = \left(\frac{e}{c}\mathbf{A} + \frac{e}{c}\langle A_\parallel \rangle \mathbf{b} + m v_\parallel \mathbf{b}\right) \cdot d\mathbf{X} + \mu d\theta - \left(\frac{m}{2}v_\parallel^2 + \mu B + e \langle \phi \rangle\right) dt$$
The resulting equations of motion are:
$$\frac{d\mathbf{X}}{dt} = v_\parallel \mathbf{b} + \mathbf{v}_D + \mathbf{v}_E + \mathbf{v}_{\delta B_\perp}$$
$$m v_\parallel \frac{dv_\parallel}{dt} = \frac{d\mathbf{X}}{dt} \cdot [Z e \mathbf{E} - \mu \nabla B + m \Omega^2 R \nabla R]$$
$$\frac{d\mu}{dt} = 0$$

---

## 2. Complete Equation Set (δf approximation)

The evolution of the perturbed distribution function $g = f + \frac{2 Z}{T} w v_\parallel \langle A_\parallel \rangle F_M$ is given by:
$$\frac{\partial g}{\partial t} = \text{I} + \text{II} + \text{III} + \text{IV} + \text{V} + \text{VI} + \text{VII} + \text{VIII}$$

### 2.1 Operators
- **I: Parallel Streaming**
  $-v_\parallel \mathbf{b} \cdot \nabla f \rightarrow -w v_\parallel \mathcal{F} \frac{\partial f}{\partial s}$
- **II: Magnetic Drift**
  $-\frac{\rho_*}{Z} [T_G E_D \mathcal{D}^\alpha + \dots] \frac{\partial f}{\partial x_\alpha}$
- **III: Nonlinear Term**
  $-\rho_*^2 \frac{\partial \chi}{\partial x_\beta} \mathcal{E}^{\beta \alpha} \frac{\partial g}{\partial x_\alpha}$
- **IV: Mirror Force**
  $+w (\mu B \mathcal{G} + \dots) \frac{\partial f}{\partial v_\parallel}$
- **V: Electric Drive**
  $- \mathbf{v}_\chi \cdot \nabla F_M$
- **VI: Background Drift**
  $- \mathbf{v}_D \cdot \nabla F_M$
- **VII: Parallel Electric Field**
  $-\frac{Z}{T} w v_\parallel \mathcal{F} \frac{\partial \langle \phi \rangle}{\partial s} F_M$
- **VIII: Magnetic Drift of Potential**
  $-\frac{\rho_*}{T} [\dots] \frac{\partial \langle \phi \rangle}{\partial x_\alpha} F_M$

### 2.2 Energy Terms
- $E_D = v_\parallel^2 + \mu B$
- $E_T = \frac{T}{T_G} [v_\parallel^2 + 2\mu B + \mathcal{E}_\Omega] - 1.5$

---

## 3. Numerical Implementation Details

### 3.1 Coordinates and Grids
- **Coordinates:** $(\psi, \zeta, s)$ where $s$ is the parallel coordinate, $\psi$ is radial, and $\zeta$ is binormal.
- **Spectral Representation:** Fourier modes are used for $\zeta$ and sometimes $\psi$ (local limit).
- **Resolution:** $(v_\parallel, \mu, s, x, k_y) = (32, 8, 16, 85, 32)$.

### 3.2 Finite Difference Stencils (Parallel $s$)
- **4th Order Centered:** $v \frac{\partial g}{\partial s} \rightarrow v_i \frac{g_{i-2} - 8g_{i-1} + 8g_{i+1} - g_{i+2}}{12 \Delta s}$
- **Upwind Dissipation:** $-D |v_i| \frac{-g_{i-2} + 4g_{i-1} - 6g_i + 4g_{i+1} - g_{i+2}}{12 \Delta s}$
- **Staggered Order Reduction (at boundaries):**
  - Boundary: 2nd order one-sided.
  - Adjacent: 3rd order.
- **Arakawa Scheme:** Conserves $\int f^2 dV$ and $\int H f dV$ by differencing the Poisson bracket $\{H, f\}$ directly.

### 3.3 Time Stepping: RK4
Standard explicit Runge-Kutta 4th order.
- Each stage includes a field solve (`calculate_fields`).
- Time step $dt = 0.01$.

---

## 4. Field Solve & Adiabatic Response

### 4.1 Quasi-neutrality Solver
1.  **Integrate Ions:** $I = \sum_i Z_i \int \mathcal{J}_0 f_i d^3v$.
2.  **Zonal flow ($k_y = 0$):**
    - Correct for adiabatic electron response using $matz$ and $maty$ matrices.
    - $matz$: maps density to a buffer.
    - $FluxSurfaceAverage$: $\langle \dots \rangle_s = \int \dots ds$.
    - $maty$: maps averaged buffer back to density correction.
3.  **Poisson Diagonal:** $\phi = - I / poisson\_dia$.
    - $poisson\_dia$ includes the polarization term $(\Gamma_0 - 1)$.

---

## 5. Normalization and Geometry Tensors

### 5.1 Unit System
- $t_N = t \cdot (v_{th}/R)$
- $v_N = v / v_{th}$
- $\phi_N = \phi \cdot (e/T \rho_*)$
- $B_N = B / B_{ref}$

### 5.2 Geometry Factors
Loaded from `geom.dat`:
- $D_{eps}, D_{zeta}, D_s$: Curvature and grad-B drift components.
- $g^{\alpha \beta}$: Metric tensor components.
- $bn$: Magnetic field strength $B/B_{ref}$.
- $ints$: Parallel grid spacing $\Delta s$.

---

## 6. JAX Solver Strategy

### 6.1 Modular RHS
```python
def gksolve_rhs(df, geom, fm):
    phi, fluxes = get_integrals(df, geom)
    rhs = L_streaming(df, geom) + L_drift(df, geom) + L_drive(phi, geom, fm)
    return rhs, (phi, fluxes)
```

### 6.2 Key Verification Points
- **Growth Rates:** Compare linear growth $\gamma = \dot{|\phi|} / |\phi|$ with `growth_rates_all_modes`.
- **Fluxes:** Compare $Q = \int v^2 (\mathbf{v}_E \cdot \nabla f) d^3v$ with `fluxes.dat`.
- **Conservation:** Check invariance of $\int f d^3v$ (particle conservation).
