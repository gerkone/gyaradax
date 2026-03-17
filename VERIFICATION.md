# GKW Standard Tests Verification

This document identifies standard tests from the GKW repository that are applicable to the current `gyaradax` solver implementation.

## Selection Criteria

### Adiabatic Electron Tests
- **Local Flux-Tube:** `local_flux_tube = .T.` or `flux_tube = .T.` (Default in GKW).
- **Adiabatic Electrons:** `adiabatic_electrons = .T.`.
- **Single Species:** `number_of_species = 1` (Targets 1 kinetic species + adiabatic electrons).
- **Electrostatic:** `beta = 0.0`.
- **Benchmark:** Cyclone Base Case (CBC) or similar ITG physics.

### Kinetic Electron Tests
- **Local Flux-Tube:** Default GKW setting.
- **Kinetic Electrons:** `adiabatic_electrons = .F.`, `number_of_species = 2` (ion + electron).
- **Electrostatic:** `beta = 0.0`.
- **Multi-species field solver:** Quasineutrality with full kinetic response for both species.

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

## Kinetic Electron Test Cases

### 5. `kinetic_elec` (standard)
- **Type:** Linear Kinetic Electron Test.
- **Physics:** 2 Kinetic species (ion + electron), electrostatic (`beta = 0.0`).
- **Geometry:** Miller (default `geom_type`).
- **Parameters:**
  - `SHAT = 0.522048`, `Q = 1.4`, `EPS = 0.173240`.
  - `KTHRHO = 0.424264`.
  - Ion: `MASS = 1.0`, `Z = 1.0`, `TEMP = 1.0`, `rlt = 0.0`, `rln = 1.05`.
  - Electron: `MASS = 2.72e-4`, `Z = -1.0`, `TEMP = 1.009574`, `rlt = 9.150731`, `rln = 1.05`.
- **Grid:** `NX = 1`, `N_s_grid = 21`, `N_mu_grid = 4`, `N_vpar_grid = 16`, `nperiod = 2`.
- **Utility:** Primary benchmark for kinetic electron linear physics with electron-driven ITG/ETG.
- **Reference Files:** `fluxes.dat` (6 columns: pflux/eflux/vflux per species), `time.dat`.

### 6. `kinetic_elec` (extra)
- **Type:** Linear Kinetic Electron Test (alternative resolution).
- **Physics:** Same as standard `kinetic_elec`.
- **Geometry:** Miller.
- **Parameters:** Identical species parameters to standard `kinetic_elec`.
- **Utility:** Cross-check with different parallelization / io settings.
- **Reference Files:** `fluxes.dat`, `time.dat`.

### 7. `geom_circ`
- **Type:** Linear Kinetic Electron Test (Circular Geometry).
- **Physics:** 2 Kinetic species (ion + electron), electrostatic (`beta = 0.0`).
- **Geometry:** Circular (`geom_type = 'circ'`).
- **Parameters:**
  - `SHAT = 1.0`, `Q = 2.0`, `EPS = 0.16`.
  - `KTHRHO = 0.2`, `CHIN = 0.2`.
  - Ion: `MASS = 1.0`, `Z = 1.0`, `TEMP = 1.0`, `rlt = 9.0`, `rln = 3.0`.
  - Electron: `MASS = 2.7777e-4`, `Z = -1.0`, `TEMP = 1.0`, `rlt = 9.0`, `rln = 3.0`.
- **Grid:** `NX = 1`, `N_s_grid = 105`, `N_mu_grid = 8`, `N_vpar_grid = 64`, `nperiod = 3`.
- **Utility:** Kinetic electron verification with simple circular geometry and symmetric ion/electron gradients.
- **Reference Files:** `fluxes.dat`, `time.dat`, `geom.dat`.

## Future Considerations
- **Electromagnetic:** Tests like `nonspec_lin_bpar` (`beta = 0.012`, kinetic electrons, circular geometry) and `rota_miller` (`beta_ref = 0.015`, kinetic electrons, Miller geometry with rotation) provide benchmarks once $\beta > 0$ is implemented.
- **Collisions:** `collisions_user_defined` (kinetic electrons, Miller, with Coulomb collisions) is available for future collision operator validation.
- **Implicit Time-Stepping:** `implicit` (kinetic electrons, circular, `beta = 0.004`) tests the implicit method once implemented.
