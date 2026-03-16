# PROJECT MANDATES: JAX Reimplementation of GKW (Kinetic Electrons)

## Documentation & Note-Taking

* **Thoroughness is Mandatory:** You must be extremely thorough in your note-taking. Do not hesitate to produce long, detailed files.
* **Granular Detail:** Document everything required for reimplementation and more: specific subroutine logic, module dependencies, variable mappings, numerical constants, and edge-case handling for the stiff parallel streaming terms.
* **Foundational Reference:** Notes in `@GKW.md` (and similar files) serve as the foundational technical specification for the JAX port. You must build upon the existing adiabatic documentation.

**CONTEXT FILES PROVIDED:**

* `@gyaradax/` (The entire current JAX codebase, fully functional for the adiabatic case).
* `@GKW.md` (Existing notes from the adiabatic implementation).
* `@tests/` (The test suite, specifically `@tests/unit/test_integrals.py` and `@tests/unit/test_nonlinear.py`).
* `gkw_ref` (Source code and LaTeX manuals for the legacy GKW Fortran code).
* `/restricteddata/ukaea/gyrokinetics/raw/kinetic_electrons/` (Reference empirical trajectories for the kinetic case).

**ROLE:** Act as an autonomous, expert Scientific Computing Engineer. Your objective is to extend the existing `gyaradax` JAX solver to support kinetic electrons. Describe what you plan to do at the main intermediate steps. **Crucially, stop and yield to the user for feedback at the end of each numbered execution phase before proceeding to the next.**

**SCOPE & CONSTRAINTS (STRICT):**

* **Physics:** Kinetic electron case (Multi-species: ions + electrons).
* **Fallback:** The kinetic solver must be cleanly isolated. It should be deactivated when `adiabatic_electrons=True` (default), which must perfectly revert the solver back to the one-species adiabatic case.
* **Framework:** JAX. All functions must be pure, differentiable, and `jit`-compatible. Dynamic shapes are strictly forbidden.
* **Precision:** ALWAYS use float64 (fp64), especially for parallel velocity reductions to mitigate non-deterministic error accumulation. JAX must be configured to use 64-bit precision.
* **Integrity:** STRICTLY NO normalization sweeps to force test passes. NO cheating or hacking the outputs to match expected results.
* **Environment:** STRICTLY use GPU:0 for all compilations and tests.

**REQUIRED INTERFACE:**
You MUST maintain the existing purely functional interface for the core solver, ensuring it scales seamlessly to $N$ species:
`next_df, (phi, fluxes) = gksolve(prev_df, ...)`

**EXECUTION PLAN (FOLLOW EXACTLY IN ORDER):**
Do not write the final solver code immediately. You must execute the following steps sequentially. Use your toolset to explore directories, read files, write code, and run tests. Output and save your notes and progress to markdown artifacts for each step before pausing.

1. **INITIAL CONTEXT INGESTION & DATA EXPLORATION:** * Ingest the entire `@gyaradax/` codebase, the `@GKW.md` notes, and the `@tests/` directory.

* Explore the Fortran reference at `gkw_ref` to identify the specific subroutines handling multi-species initialization, parallel velocity derivatives ($v_\parallel$), and the updated field solver (Poisson equation with kinetic electron density).
* Inspect the reference trajectories in `/restricteddata/ukaea/gyrokinetics/raw/kinetic_electrons/` to understand the multi-species data shapes and scales. **Stop and wait for user approval.**

2. **VERIFY & EXTEND JAX INTEGRALS:**

* Review the existing integrals in the JAX codebase.
* Check if the current flux integrals support multiple species via proper broadcasting.
* Extend `@tests/unit/test_integrals.py` to explicitly validate the multi-species kinetic case. **Stop and wait for user approval.**

3. **EXPLORATION, ANALYSIS & PLANNING (NOTE-TAKING):**

* **Source Code & Theory:** Actively analyze the Fortran source code for the kinetic electron formulation. Pay specific attention to the fast parallel streaming terms and how the explicit RK4 integrator handles the stiffness.
* **Synthesize & Plan:** Update `@GKW.md` with detailed notes specifically for the kinetic case. Map out the new multi-species array dimensions, the required changes to the `gksolve` update step, and the mathematical modifications to the field solver. **Stop and wait for user approval.**

4. **ESTABLISH KINETIC EMPIRICAL TESTS (TDD):** * Implement the empirical kinetic electron tests within `@tests/unit/test_nonlinear.py`.

* Hook these tests directly into the reference data found in `/restricteddata/ukaea/gyrokinetics/raw/kinetic_electrons/`.
* Ensure there is a test that verifies the solver falls back correctly to the single-species results when `adiabatic_electrons=True`. **Stop and wait for user approval.**

5. **IMPLEMENT KINETIC `gksolve`:** * Implement the multi-species kinetic dynamics in the main JAX solver based on your notes.

* Ensure the field solver accurately computes $\phi$ using the full kinetic electron density instead of the adiabatic response.
* Iterate on the implementation and testing. You must iterate until the tests in `@tests/unit/test_nonlinear.py` pass with a relative error `< 1e-4`. **Report the final test results and wait for user approval.**
