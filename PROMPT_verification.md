# PROJECT MANDATES: Gyaradax Validation & Publication Figures

## Core Objective
[cite_start]Construct a comprehensive, end-to-end Jupyter Notebook (`notebooks/validation_suite.ipynb`) that rigorously validates the `gyaradax` solver against standard GKW reference tests and analytical ground truths from the original paper[cite: 1]. The notebook must run the simulations, extract the relevant metrics (growth rates, residuals, saturated fluxes), and generate publication-quality ("Nature-ready") figures.

**CONTEXT FILES PROVIDED:**
* `@gyaradax/` (The core JAX solver).
* `@docs/gkw.pdf` (Reference paper for physics normalizations and analytical ground truths).
* Reference data directories containing GKW outputs (`fluxes.dat`, `time.dat`, `eflux_es.dat`, etc.) for the respective test cases.

**ROLE:** Act as an autonomous, expert Computational Plasma Physicist and Data Scientist. Your objective is to validate the code and produce beautiful, citable figures. Describe what you plan to do at the main intermediate steps. **Crucially, stop and yield to the user for feedback at the end of each numbered execution phase before proceeding.**

**SCOPE & CONSTRAINTS (STRICT):**
* **Physics Scope:** Local flux-tube, electrostatic (`beta = 0.0`). Covers both:
  - **Adiabatic electrons** (`adiabatic_electrons = .T.`, `number_of_species = 1`): single kinetic ion species.
  - **Kinetic electrons** (`adiabatic_electrons = .F.`, `number_of_species = 2`): ion + electron with full kinetic response.
* **Aesthetics ("Nature-Ready"):** All plots must strictly adhere to publication standards: single-column width (approx. 89mm / 3.5 inches), 300 DPI, sans-serif fonts (Helvetica/Arial), properly sized tick labels (8pt) and axis labels (9pt), using a colorblind-friendly palette. Do not use default Matplotlib styling.
* **Integrity:** DO NOT hardcode the `gyaradax` simulation results. You must actually run the solver functions in the notebook, extract the values, and plot them against the theoretical/reference data.

**GROUND TRUTHS & TEST CASES (From GKW standard tests and paper):**
* **Analytical (Rosenbluth-Hinton Zonal Flow):** $q=1.5$, $\epsilon=0.05$, $k_\psi \rho_s=0.02$. [cite_start]Target Residual: $\phi(t=\infty)/\phi(t=0) \approx 0.0713$ (Matches Xiao-Catto theory to 0.3%)[cite: 1].
* **Linear CBC (`eiv_simple`):** Circular geometry, $R/L_T=6.9$, $R/L_n=2.2$, $q=1.4$, $\hat{s}=0.78$, $\epsilon=0.19$. Target Reference Growth Rate: `1.81840E-01`.
* **Nonlinear CBC (`sourcetime`):** Circular geometry, $R/L_T=5.0$, $R/L_n=2.2$. Target: Time-averaged transport levels (heat flux $\chi_i$) matching `eflux_es.dat` and `vflux_es.dat`.
* **Linear Slab ITG (`slab_itg`):** Slab periodic geometry, $R/L_T=9.0$, $R/L_n=0.25$, $q=1.0$, $\hat{s}=1.0$, $\epsilon=1.0$.
* **Linear Miller Geometry (`miller_mb`):** Miller geometry, $R/L_T=9.0$, $R/L_n=3.0$, $q=2.0$, $\hat{s}=1.0$, $\epsilon=0.16$.

**KINETIC ELECTRON GROUND TRUTHS (From `gkw_ref/tests/`):**
* **Linear Kinetic Electron — Miller (`kinetic_elec`):** Miller geometry, $q=1.4$, $\hat{s}=0.522$, $\epsilon=0.173$, $k_\theta\rho_s=0.424$. Ion: $R/L_T=0$, $R/L_n=1.05$. Electron: $R/L_T=9.15$, $R/L_n=1.05$, $m_e/m_i=2.72\times10^{-4}$. Target: per-species flux traces matching `gkw_ref/tests/standard/kinetic_elec/reference/fluxes.dat`.
* **Linear Kinetic Electron — Circular (`geom_circ`):** Circular geometry, $q=2.0$, $\hat{s}=1.0$, $\epsilon=0.16$, $k_\theta\rho_s=0.2$. Symmetric gradients: $R/L_T=9.0$, $R/L_n=3.0$ for both species. Target: per-species flux traces matching `gkw_ref/tests/standard/geom_circ/reference/fluxes.dat`.

**AGENTIC EXECUTION PLAN (FOLLOW EXACTLY IN ORDER):**

1. **NOTEBOOK INITIALIZATION & AESTHETIC SETUP:**
   * Create `notebooks/validation_suite.ipynb`.
   * Set up the global Matplotlib `rcParams` to match the strict "Nature-ready" constraints detailed above.
   * Add a Markdown introduction explaining the validation criteria (Adiabatic, Single Species, Electrostatic). **Stop and wait for user approval.**

2. **ANALYTICAL VALIDATION: ROSENBLUTH-HINTON:**
   * Implement the simulation cell for the Rosenbluth-Hinton zonal flow test based on the parameters above.
   * Run the simulation until the geo-acoustic mode damps out.
   * Plot the normalized potential $\phi(t)/\phi(0)$ vs time. Add a horizontal dashed line for the theoretical Xiao-Catto residual ($\sim 0.0710$). **Stop and wait for user approval.**

3. **LINEAR BENCHMARKS (CBC, SLAB, MILLER):**
   * Implement execution cells for `eiv_simple`, `slab_itg`, and `miller_mb`.
   * For the CBC (`eiv_simple`), write a loop to scan $k_\theta \rho_s$ (from $0.1$ to $0.6$) and extract the growth rates.
   * Plot the `gyaradax` growth rate curve alongside the reference GKW data points, recreating the topological shape of Figure 2 (Right Panel) from `@docs/gkw.pdf`. Ensure normalizations match. **Stop and wait for user approval.**

4. **NONLINEAR BENCHMARK (SOURCETIME):**
   * Implement the execution cell for the nonlinear CBC `sourcetime` test.
   * Calculate the saturated, time-averaged ion heat flux ($\chi_i$) and electrostatic particle fluxes.
   * Plot the time traces of the fluxes alongside the raw data loaded directly from the GKW reference files (`eflux_es.dat`, `vflux_es.dat`).
   * **Report the final time-averaged transport values and confirm parity.**

5. **KINETIC ELECTRON BENCHMARKS:**
   * Implement execution cells for the `kinetic_elec` (Miller) and `geom_circ` (circular) kinetic electron test cases.
   * For each case, load geometry from `gkw_ref/tests/standard/{test}/reference/` via `load_geometry()`.
   * Initialize a fresh 6D distribution function with `n_species=2` and run the solver with `adiabatic_electrons=False`.
   * Load GKW reference `fluxes.dat` (6-column format: pflux_i, eflux_i, vflux_i, pflux_e, eflux_e, vflux_e) and `time.dat`.
   * For each test case, produce a 2-species flux comparison figure (ion and electron heat fluxes side-by-side) using `plot_flux_trace` with `n_species=2`.
   * **Stop and wait for user approval.**

6. **KINETIC ELECTRON SPECTRA (if nonlinear kinetic runs available):**
   * If validation output directories (`validation_kinetic_*`) exist, load the pre-computed results and plot spectral comparisons using `plot_spectra`.
   * Compare kx and ky spectra against GKW reference data where available.
   * **Report per-species time-averaged heat fluxes and confirm parity.**