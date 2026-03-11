# PROJECT MANDATES: JAX Reimplementation of GKW

## Documentation & Note-Taking
- **Thoroughness is Mandatory:** You must be extremely thorough in your note-taking. Do not hesitate to produce long, detailed files.
- **Granular Detail:** Document everything required for reimplementation and more: specific subroutine logic, module dependencies, variable mappings, numerical constants, and edge-case handling.
- **Foundational Reference:** Notes in `GKW.md` (and similar files) serve as the foundational technical specification for the JAX port.

**CONTEXT FILES PROVIDED:**
* @integrals.py (Contains the existing PyTorch flux integrals that require translation).
* @utils.py (Contains data loading utilities and potential/phi calculation helpers).
* @geometry.py (geometry loading utils)
* @test_integral.py

**ROLE:** Act as an expert Scientific Computing Engineer. Describe what you plan to do at the main intermediate steps. Stop at the end of each phase. Your objective is to translate the GKW Fortran code (https://bitbucket.org/gkw/gkw/src) into JAX. 

**SCOPE & CONSTRAINTS (STRICT):**
* **Physics:** Adiabatic electron case ONLY. Simplify all equations to reflect this.
* **Grid Resolution:** `(vpar, mu, s, x, ky) = (32, 8, 16, 85, 32)`
* **Time step:** `dt = 0.01`
* **Framework:** JAX. All functions must be pure, differentiable, and `jit`-compatible.
* **Precision:** ALWAYS use float64 (fp64). JAX must be configured to use 64-bit precision.

**ENVIRONMENT:**
Use @utils.py for data loading and potential (`phi`) calculations. Do NOT reimplement these helpers. 
**CRITICAL:** The flux integrals currently exist in PyTorch within @integrals.py. You MUST reimplement them into JAX as your very first task.

**REQUIRED INTERFACE:**
You MUST expose the following purely functional interface for the core solver:
`next_df, (phi, fluxes) = gksolve(prev_df, ...)`

**EXECUTION PLAN (FOLLOW EXACTLY IN ORDER):**
Do not write the final solver code immediately. You must execute the following steps sequentially, and write your reasoning / whay you are trying to accomplish between smaller intermediate steps. **Output and write to markdown your notes and progress for each step before moving to the next.**

1. **REIMPLEMENT & VALIDATE FLUX INTEGRALS:** Read the PyTorch integrals in @integrals.py and translate them into JAX. Implement the geometry loading functions so they work with jax from @geometry.py. Verify that your JAX integral pass the @test_integrals.py (after adapting to jax). TAKE NOTES on any differences you encounter in broadcasting, memory layout, or numerical precision.
2. **ANALYZE GKW & TAKE NOTES:** Review the GKW source code focusing strictly on time integration and the adiabatic electron formulation. TAKE DETAILED NOTES on the exact mathematical update equations and how you will map them to the `gksolve` interface. 
3. **EXPLORE REFERENCE DATA & TAKE NOTES:** Ingest and explore the reference trajectory at `/restricteddata/ukaea/gyrokinetics/raw/iteration_13`. TAKE NOTES on the data structures, empirical array shapes, and physical scales you observe.
4. **PLAN:** Based on the previous notes, write out a high-level plan for the core JAX architecture. TAKE NOTES identifying exactly which Fortran modules and subroutines map to the `gksolve` update step.
5. **ESTABLISH CORE TESTS (TDD):** Write the test suite for the core simulator before implementing it. These tests must run independently. 
   * **Mandatory:** Write a validation test that checks calculated growth rates against the `growth_rate_all_modes` file.
   * **Mandatory:** Write unit tests verifying array shapes (based on your notes from Step 4) and basic conservation properties.
6. **IMPLEMENT `gksolve`:** Write the core `gksolve` function using your newly translated JAX flux integrals, the helpers in @utils.py, and the fixed rules. Proceed to this step ONLY after all prior tests are written and validated against your notes.