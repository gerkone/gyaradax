# PROJECT MANDATES: JAX Reimplementation of GKW

## Documentation & Note-Taking
- **Thoroughness is Mandatory:** You must be extremely thorough in your note-taking. Do not hesitate to produce long, detailed files.
- **Granular Detail:** Document everything required for reimplementation and more: specific subroutine logic, module dependencies, variable mappings, numerical constants, and edge-case handling.
- **Foundational Reference:** Notes in `GKW.md` (and similar files) serve as the foundational technical specification for the JAX port.

**CONTEXT FILES PROVIDED:**
* @jax_integrals.py (Contains the newly implemented JAX flux integrals).
* @utils.py (Contains data loading utilities and potential/phi calculation helpers).
* @jax_geometry.py (geometry loading utils)
* @test_jax_integral.py

**ROLE:** Act as an autonomous, expert Scientific Computing Engineer. Your objective is to translate the GKW Fortran code into JAX. Describe what you plan to do at the main intermediate steps. **Crucially, stop and yield to the user for feedback at the end of each numbered execution phase before proceeding to the next.**

**SCOPE & CONSTRAINTS (STRICT):**
* **Physics:** Adiabatic electron case ONLY. Simplify all equations to reflect this.
* **Grid Resolution:** (vpar, mu, s, x, ky) = (32, 8, 16, 85, 32)
* **Time step:** dt = 0.01
* **Framework:** JAX. All functions must be pure, differentiable, and `jit`-compatible.
* **Precision:** ALWAYS use float64 (fp64). JAX must be configured to use 64-bit precision.
* **Integrity:** STRICTLY NO normalization sweeps to force test passes. NO cheating or hacking the outputs to match expected results. NO hardcoded constants—derive everything mathematically from the source code and physical constants.
* **Environment:** STRICTLY use the environment under "/system/apps/userenv/galletti/mhd" for everything.


**ENVIRONMENT:**
Use @utils.py for data loading and potential (phi) calculations. Do NOT reimplement these helpers. 
**CRITICAL:** The flux integrals have already been implemented in JAX within @jax_integrals.py. Your first coding task is to verify that they function correctly before proceeding to the main solver.

**REQUIRED INTERFACE:**
You MUST expose the following purely functional interface for the core solver:
`next_df, (phi, fluxes) = gksolve(prev_df, ...)`

**AGENTIC EXECUTION PLAN (FOLLOW EXACTLY IN ORDER):**
Do not write the final solver code immediately. You must execute the following steps sequentially. Use your toolset to explore directories, read files, write code, and run tests. Output and save your notes and progress to markdown artifacts for each step before pausing.

1. **INITIAL CONTEXT INGESTION:** Before starting any code translation or detailed planning, explore the `gkw_ref/src` (Fortran code) and `gkw_ref/manual` (LaTeX files) directories. Individuate the specific files that are relevant to time integration and the adiabatic electron formulation. Fully load these files into your context. **Stop and wait for the user to approve the identified file list.**
2. **VERIFY JAX FLUX INTEGRALS:** * Review the existing JAX integrals in @jax_integrals.py. 
   * Verify that the geometry loading functions from @jax_geometry.py work seamlessly with this JAX implementation. 
   * Run and adapt @test_jax_integral.py if necessary to confirm the JAX integrals pass all tests. 
   * TAKE NOTES on any specific broadcasting, memory layout, or numerical precision details observed during verification. **Stop and wait for user approval.**
3. **EXPLORATION, ANALYSIS & PLANNING (NOTE-TAKING):**
   * **Source Code & Theory:** Actively analyze the Fortran source code and LaTeX manual files you loaded in Step 1. Focus strictly on time integration and the adiabatic electron formulation.
   * **Reference Data:** Ingest and explore the reference trajectory at `/restricteddata/ukaea/gyrokinetics/raw/iteration_13` to understand data structures, empirical array shapes, and physical scales.
   * **Synthesize & Plan:** Based on the Fortran code, the LaTeX manual, and the reference data, write out a high-level plan for the core JAX architecture. Identify exactly which Fortran modules and subroutines map to the `gksolve` update step, detail the exact mathematical update equations, and map out the variables.
   * **Artifact Generation:** Save all of these findings into a detailed `GKW.md` file. **Stop and wait for user approval.**
4. **ESTABLISH CORE TESTS (TDD):** * Write the test suite for the core simulator before implementing it. These tests must run independently. 
   * **Mandatory:** Write a validation test that checks calculated growth rates against the `growth_rate_all_modes` file.
   * **Mandatory:** Write unit tests verifying array shapes (based on your notes from Step 3) and basic conservation properties. **Stop and wait for user approval.**
5. **IMPLEMENT `gksolve`:** * Write the core `gksolve` function, based on the notes you created. 
   * Proceed to this step ONLY after all prior tests are written and validated against your notes.