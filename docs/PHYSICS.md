# Physics & Numerical Implementation

Gyaradax implements the gyrokinetic Vlasov-Poisson system in the local flux-tube limit, targeting the electrostatic, adiabatic-electron configuration of GKW.

## Implemented Equations

The solver evolves the perturbed distribution function $f$ in a 5D phase space $(v_\parallel, \mu, s, k_x, k_y)$.

### Active RHS Terms

The following terms from the GKW formulation are currently implemented in `gyaradax/solver.py`:

1.  **Term I (Parallel Advection):** $v_\parallel 
abla_\parallel f$ using fourth-order upwinded finite differences.
2.  **Term II (Drift Advection):** $\mathbf{v}_d \cdot 
abla_\perp f$ representing curvature and $
abla B$ drifts.
3.  **Term III (Nonlinear E x B Advection):** $\mathbf{v}_E \cdot 
abla_\perp f$ evaluated via a pseudospectral method with dealiasing.
4.  **Term IV (Trapping/Mirror):** Parallel velocity space advection due to magnetic field gradients.
5.  **Term V (Equilibrium Drive):** $\mathbf{v}_E \cdot 
abla F_M$ representing background density and temperature gradients.
6.  **Term VII (Parallel Field Drive):** $v_\parallel 
abla_\parallel \phi$ coupling.
7.  **Term VIII (Drift Field Drive):** $\mathbf{v}_d \cdot 
abla \phi$ coupling.

### Dissipation

- **Parallel Dissipation:** Fourth-order damping on the streaming term.
- **Velocity Space Dissipation:** Smoothing in $v_\parallel$ to prevent grid-scale oscillations.
- **Perpendicular Hyper-dissipation:** Fourth-order spectral damping in $(k_x, k_y)$ to absorb energy at the grid cutoff.

## Numerical Schemes

### Time Integration
Gyaradax uses an explicit **Runge-Kutta 4 (RK4)** scheme for the small-step update. The large-step cadence (GKW's `naverage`) is handled via stateful metadata to maintain normalization and growth-rate tracking.

### Spatial Differencing
- **Parallel ($s$):** Fourth-order central and upwinded stencils with complex connectivity across parallel boundaries (ballooning transformation).
- **Parallel Velocity ($v_\parallel$):** Centered fourth-order stencils with zero-padding at the boundaries.
- **Perpendicular ($k_x, k_y$):** Pseudospectral evaluation for the nonlinear bracket, using dealiased FFT grids $(3/2$ rule).

### Connectivity
Parallel boundaries are mapped using `mode_label` metadata to correctly couple $k_x$ chains, reproducing the GKW `mode_box` connectivity.

## Normalization
The solver operates in the standard GKW normalization (quantities scaled by $R_{ref}$, $v_{th,ref}$, etc.). In linear modes, per-toroidal-mode normalization is applied at large-step boundaries to maintain unit potential amplitude.
