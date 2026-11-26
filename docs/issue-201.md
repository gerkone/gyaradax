# Issue 201 electromagnetic instability

GKW and Gyaradax reproduce a rapidly growing, grid-scale mode in the low-`ky`, electromagnetic `3559` case. Stock CGYRO selects a much slower mode in the circular-Miller port at both radial chain lengths tested. The agreement between the two reproducers, the parallel-resolution dependence, and the CGYRO comparison point to a numerical instability in the GKW/Gyaradax discretization. The comparison leaves the converged growth rate of the slow mode open.

This is a local lab investigation. The issue bundle and prior analysis are restricted. Keep this branch local and use the original inputs at the paths below.

## Reproduce the fast branch

The recovered `3559` input is a June 2026 port of the original attachment. It uses circular geometry, kinetic electrons, `kthrho=0.07`, `CHIN=1.8`, `beta=0.0035`, and parallel dissipation `disp_par=2` with `idisp=2`. It evolves the electrostatic potential and parallel vector potential. There is one radial harmonic over three poloidal turns, with 32 points per turn and a 32 × 8 velocity grid. The original rotation and parallel flow gradients were dropped in the surviving port. The bundle also has the separate `base4_01` reproducer for a follow-up regression check. [Bundle README](/system/user/publicwork/volkmann/issue201.I85RLq/issue201/README.md) · [Input provenance](/system/user/publicwork/volkmann/issue201.I85RLq/issue201/SOURCES.md)

### GKW

On a lab host with the GKW build and its MPI runtime available, run the recovered deck in a fresh directory. The wrapper derives eight ranks from the input. Use the double-precision build from `785a215` with conservative parallel dissipation off. The expected late-time rate is γ ≈ 2.60712 and ω ≈ −56.6134 in GKW units.

```bash
issue_bundle=/system/user/publicwork/volkmann/issue201.I85RLq/issue201
issue_runs=$(mktemp -d /system/user/publicwork/volkmann/issue201-repro.XXXXXX)
gkw_bin=/system/user/galletti/git/gkw-issue201/run/gkw_scorpion-DP_785a215-dirty.x
issue_python=/system/apps/userenv/volkmann/jax_env/bin/python

mkdir "$issue_runs/gkw-3559"
cp "$issue_bundle/gkw/3559/input.dat" "$issue_runs/gkw-3559/"
bash "$issue_bundle/scripts/run_gkw.sh" "$issue_runs/gkw-3559" "$gkw_bin"
"$issue_python" "$issue_bundle/scripts/gkw_growth.py" "$issue_runs/gkw-3559"
```

The completed reference reached `t=60`. A later launch on `wasp` encountered Open MPI fabric errors and its output ends at `t=0.10`. Use the completed reference for comparison and check that a fresh run reaches late time before reading its rates. [Reference output](/system/user/galletti/git/gkw-issue201/run/test_3559_off/time.dat) · [Wasp MPI log](/system/user/publicwork/volkmann/issue201-gkw-wasp.Sampe0/3559/gkw.log)

### Gyaradax

Use the bundled `src-3559` snapshot (`95da48b` plus the GKW import patch) to reproduce the recorded GPU run. The patch is needed on that base to retain the finite ballooning angle and match GKW's zero perpendicular dissipation for the single-mode input. The bundled YAML already points to the exported GKW grid and specifies `chin: 1.8`, `disp_x: 0`, `disp_y: 0`, and the electromagnetic fields. The investigation branch starts from current `master`. Port and verify those import settings on this branch before using it for solver comparisons. [GPU config](/system/user/publicwork/volkmann/issue201.I85RLq/issue201/3559.yaml) · [Import patch](/system/user/publicwork/volkmann/issue201.I85RLq/issue201/gyaradax/gkw_import_fix.patch)

Continue in the same shell. Check `nvidia-smi` and set `issue_gpu` to a free GPU index. The recorded run used JAX 0.10.2 in FP64 on a H200 on `wasp`.

```bash
CUDA_VISIBLE_DEVICES="${issue_gpu:?set issue_gpu to a free GPU index}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$issue_bundle/src-3559" \
  "$issue_python" -B -P -m gyaradax.cli run "$issue_bundle/3559.yaml" \
  --backend jax --dp --from-scratch --n-steps 8000 \
  --output-dir "$issue_runs/gyaradax-3559"

"$issue_python" "$issue_bundle/scripts/gyx_growth.py" "$issue_runs/gyaradax-3559"
```

Expect γ ≈ 2.60700 at `t=16` with `dt=0.002`. The recorded ω ≈ −56.61302 came from the potential overlap phase over one additional RK4 step. The growth helper reports γ only. [GPU measurements and frequency method](/system/user/publicwork/volkmann/issue201.I85RLq/issue201/growth-summary.txt)

## CGYRO comparison

On `wasp`, use the lab's GPU-enabled GACODE environment and one MPI rank per run. Copy the two recorded decks into clean run directories and keep the GPU idle before starting. With the GPU-enabled `cgyro` launcher on `PATH`, run the following in the same shell.

```bash
issue_cgyro=/system/user/publicwork/volkmann/issue201-3559-cgyro.k1JmWL

for nr in 4 16; do
  mkdir "$issue_runs/cgyro-nr$nr"
  cp "$issue_cgyro/gpu_nr$nr/input.cgyro" "$issue_runs/cgyro-nr$nr/"
  CUDA_VISIBLE_DEVICES="${issue_gpu:?set issue_gpu to a free GPU index}" \
    cgyro -e "$issue_runs/cgyro-nr$nr" -n 1 > "$issue_runs/cgyro-nr$nr/run.log" 2>&1
done

"$issue_python" \
  /system/user/publicwork/galletti/git/cgyro_study/em/cgyro-compare/scripts/freq_summary.py \
  "$issue_runs/cgyro-nr4" "$issue_runs/cgyro-nr16"
```

Both decks use `N_THETA=32`, `BOX_SIZE=1`, two fields, and `PX0=0.2864789`. Only `N_RADIAL` differs. With `BOX_SIZE=1` it gives the number of linked poloidal turns. Four is the closest supported even count to the three GKW turns. Sixteen tests sensitivity to chain length. The port uses circular Miller geometry, and the `CHIN` to `PX0` mapping still needs a check against the actual \(k_\perp(\theta)\) curves. [Four-mode deck](/system/user/publicwork/volkmann/issue201-3559-cgyro.k1JmWL/gpu_nr4/input.cgyro) · [Sixteen-mode deck](/system/user/publicwork/volkmann/issue201-3559-cgyro.k1JmWL/gpu_nr16/input.cgyro)

| Run | Growth γ | Frequency ω | Measurement |
| --- | ---: | ---: | --- |
| GKW 3559 | 2.60712 | −56.6134 | Late-time fast branch |
| Gyaradax 3559, GPU FP64 | 2.60700 | −56.61302 | Fast branch at t = 16 |
| Stock CGYRO, 4 radial modes | 0.11233 | −0.36256 | Met the time-convergence criterion |
| Stock CGYRO, 16 radial modes | 0.02073 | −0.07071 | Final 30% average, reached the time limit |

Rates are in GKW units, \(v_{\mathrm{th,ref}}/R_{\mathrm{ref}}\). The CGYRO summary multiplies rates by \(1.9732662/\sqrt{2}\) and reverses the frequency sign. The potential snapshots from the 16-mode run gave nearly the same late-time estimate as the frequency log. The factor-of-5.4 change in growth between the two decks leaves the slow-mode reference sensitive to radial chain length. [Four-mode log](/system/user/publicwork/volkmann/issue201-3559-cgyro.k1JmWL/gpu_nr4/run.log) · [Sixteen-mode log](/system/user/publicwork/volkmann/issue201-3559-cgyro.k1JmWL/gpu_nr16/run.log)

## Prior tests and what to verify on this branch

- Prior `3559` scans found that the fast branch disappeared at 48–64 parallel points per turn with `disp_par=2`. It persisted at 64 points per turn with weaker dissipation, and the tested reductions in timestep left it effectively unchanged. Check the spatial operator and mode structure first. [Resolution and timestep scans](/restricteddata/ukaea/gyrokinetics/cgyro/gyaradax_cgyro_em/advisor/ISSUE201.md:38)
- A previous CGYRO build with the upwind moment projection disabled developed a rapidly growing grid mode on the circular-Miller port. Its measured frequency and ballooning-angle dependence differed from the GKW branch, and the frequency had yet to converge. This implicates how parallel dissipation couples to the fields and leaves the identity of the two modes open. [CGYRO projection experiment](/restricteddata/ukaea/gyrokinetics/cgyro/gyaradax_cgyro_em/advisor/ISSUE201.md:58)
- Earlier conservative parallel dissipation removed the original fast branch in GKW but left a residual branch at γ ≈ 0.24647 for `3559`. A later candidate, `disp_par_neutral`, projects the charge created by dissipating the charge-free part of the distribution. Prior linear checks make it a candidate for this branch, with chain-end treatment and velocity-grid sensitivity to verify. [Mitigation measurements](/restricteddata/ukaea/gyrokinetics/cgyro/gyaradax_cgyro_em/advisor/ISSUE201.md:66) · [Candidate operator](/restricteddata/ukaea/gyrokinetics/cgyro/gyaradax_cgyro_em/advisor/ISSUE201.md:293)

Establish the unmodified baseline on this branch with the same parallel grid and finite ballooning angle. Compare the plain and candidate dissipation operators on `3559`, including the field-particle energy transfer and the eigenfunction along the field line. Then check `base4_01` and a resolved physical reference as parallel and velocity resolution and chain length change. A successful fix should remove the grid-scale fast branch without introducing a growing chain-end mode or materially changing the resolved physical mode. Check the JAX and CUDA paths before adopting it for electromagnetic runs.
