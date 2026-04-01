---
name: run-gkw
description: Run the GKW Fortran reference code and set up input.dat configurations
disable-model-invocation: true
argument-hint: [input-dir]
---

# Running GKW

The GKW binary path is set via the `GKW_BIN` environment variable. If not set, ask the user to configure it:
```bash
export GKW_BIN=/path/to/gkw.x
```

## Running a simulation

GKW must be run with MPI. The number of MPI ranks must equal `n_procs_s × n_procs_mu × n_procs_vpar` (× `n_procs_sp` for kinetic electrons). GKW reads `input.dat` from the current directory.

```bash
cd /path/to/run/directory
/usr/lib64/openmpi/bin/mpirun -np 64 $GKW_BIN
```

Typical MPI decomposition for the standard grid (32×8×16×85×32):
- Adiabatic: `n_procs_s=4, n_procs_mu=8, n_procs_vpar=2` → 64 ranks
- Kinetic (2 species): add `n_procs_sp=2` → 128 ranks

## input.dat configuration

### Adiabatic electron template

Based on iteration_13 parameters:

```fortran
&control
    silent = .true.
    order_of_the_scheme = 'fourth_order'
    parallel_boundary_conditions = 'open'
    read_file = .false.        ! .true. to restart from dump
    non_linear = .true.
    zonal_adiabatic = .true.
    method = 'EXP'
    meth = 2                   ! RK4
    dtim = 0.01                ! timestep
    ntime = 800                ! output windows
    naverage = 40              ! steps per window
    nlapar = .false.           ! electrostatic
    collisions = .false.
    disp_par = 1.0             ! parallel dissipation
    disp_vp = 0.2              ! velocity dissipation
    disp_x = 0.1               ! radial hyper-dissipation
    disp_y = 0.1               ! binormal hyper-dissipation
    lverbose = .true.
    io_format = 'ascii'
    io_testdata = .true.
    io_legacy = .true.
/

&gridsize
    nx = 85                    ! nkx
    n_s_grid = 16
    n_mu_grid = 8
    n_vpar_grid = 32
    nmod = 32                  ! nky
    nperiod = 1
    number_of_species = 1
    n_procs_s = 4
    n_procs_mu = 8
    n_procs_vpar = 2
/

&mode
    mode_box = .true.
    krhomax = 1.4
    ikxspace = 5
/

&geom
    shat = 3.075              ! magnetic shear
    q = 4.568                 ! safety factor
    eps = 0.19                ! inverse aspect ratio
    geom_type = 'circ'        ! circular geometry
/

&spcgeneral
    beta = 0.0
    adiabatic_electrons = .true.
    finit = 'cosine2'          ! or 'noise', 'sine', 'gnoise'
    amp_init = 0.0001
/

&species
    mass = 1.0
    z = 1.0
    temp = 1.0
    dens = 1.0
    rlt = 10.17               ! R/L_T (temperature gradient)
    rln = 2.61                ! R/L_n (density gradient)
    uprim = 0.0
/
```

### Kinetic electron template

Key differences from adiabatic:
- `number_of_species = 2` and add `n_procs_sp = 2`
- `adiabatic_electrons = .false.`
- `dtim` should be smaller (~0.004) due to electron CFL
- `normalize_per_toroidal_mode = .false.` for nonlinear runs
- Two `&species` blocks (ions + electrons)

```fortran
&control
    ! ... same as adiabatic, plus:
    dtim = 0.004
    naverage = 100
    normalize_per_toroidal_mode = .false.
/

&gridsize
    ! ... same grid, but:
    number_of_species = 2
    n_procs_sp = 2            ! species parallelism
/

&spcgeneral
    adiabatic_electrons = .false.
    amp_init = 0.001
/

&species                       ! ions
    mass = 1.0
    z = 1.0
    temp = 1.0
    dens = 1.0
    rlt = 5.39
    rln = 3.03
    uprim = 0.0
/

&species                       ! electrons
    mass = 0.000272            ! m_e/m_i
    z = -1.0
    dens = 1.0
    temp = 1.0
    rlt = 6.9                  ! electron R/L_T
    rln = 3.03
    uprim = 0.0
/
```

## Important notes

- **Do NOT include** `keep_dumps` or `ndump_ts` — these are gyaradax-specific config keys parsed by `gkw_to_yaml.py`, not GKW namelist variables.
- `read_file = .true.` resumes from existing dump files in the run directory.
- Output files: `fluxes.dat`, `time.dat`, `growth.dat`, `kxspec`, `kyspec`, `geom.dat`.
- This GKW version **DOES NOT** write K-dumps (binary distribution function snapshots).

## Converting GKW runs to gyaradax YAML

```bash
python scripts/gkw_to_yaml.py /path/to/gkw/run/dir output.yaml
```
