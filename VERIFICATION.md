# GKW Standard Tests Verification

This document identifies standard tests from the GKW repository that are applicable to the current `gyaradax` solver implementation.

## Selection Criteria
- **Local Flux-Tube:** `local_flux_tube = .T.` or `flux_tube = .T.` (Default in GKW).
- **Adiabatic Electrons:** `adiabatic_electrons = .T.`.
- **Single Species:** `number_of_species = 1` (Targets 1 kinetic species + adiabatic electrons).
- **Electrostatic:** `beta = 0.0`.
- **Benchmark:** Cyclone Base Case (CBC) or similar ITG physics.

## Applicable Test Cases

### 1. `eiv_simple`
- **Type:** Linear Eigenvalue Test (CBC Parameters).
- **Physics:** 1 Kinetic species + adiabatic electrons, electrostatic.
- **Geometry:** Circular (`geom_type = 'circ'`).
- **Parameters:**
  - `SHAT = 0.78`, `Q = 1.4`, `EPS = 0.19`.
  - `rlt = 6.9`, `rln = 2.2`.
- **Utility:** Primary benchmark for linear growth rates and mode frequencies.
- **Reference Growth Rate:** `1.81840E-01` (from `input.dat`).
- **Reference Files:** `fluxes.dat`, `time.dat`.

### 2. `sourcetime`
- **Type:** Nonlinear CBC benchmark.
- **Physics:** 1 Kinetic species + adiabatic electrons, electrostatic.
- **Geometry:** Circular (`geom_type = 'circ'`).
- **Parameters:**
  - `SHAT = 0.78`, `Q = 1.4`, `EPS = 0.19`.
  - `rlt = 5.0`, `rln = 2.2`.
- **Note:** Contains source modulation in `&SOURCE_TIME`.
- **Utility:** Verification of nonlinear transport levels (heat flux).
- **Reference Files:** `eflux_es.dat`, `vflux_es.dat`, `dens_real.dat`.

### 3. `slab_itg`
- **Type:** Linear Slab ITG.
- **Physics:** 1 Kinetic species + adiabatic electrons, electrostatic (`beta = 3e-6`).
- **Geometry:** Slab periodic (`geom_type = 'slab_periodic'`).
- **Parameters:**
  - `SHAT = 1.0`, `Q = 1.0`, `EPS = 1.0`.
  - `rlt = 9.0`, `rln = 0.25`.
- **Utility:** Simplified limit for fundamental ITG verification.
- **Reference Files:** `fluxes.dat`, `time.dat`.

### 4. `miller_mb`
- **Type:** Linear Miller Geometry Test.
- **Physics:** 1 Kinetic species + adiabatic electrons, electrostatic (`beta = 3e-4`).
- **Geometry:** Miller (`geom_type = 'miller'`).
- **Parameters:**
  - `SHAT = 1.0`, `Q = 2.0`, `EPS = 0.16`.
  - `rlt = 9.0`, `rln = 3.0`.
- **Utility:** Verification of the solver's ability to handle complex geometric metrics.
- **Reference Files:** `fluxes.dat`, `time.dat`.

## Future Considerations
- **Multi-species:** Tests like `adiabat_freq` can be used once multi-species support is fully verified.
- **Electromagnetic:** `shat0` and `mode_box_ara` provide electromagnetic benchmarks once $\beta > 0$ is implemented.
- **Collisions:** Standard tests starting with `collisions_` are available for future implementation.
