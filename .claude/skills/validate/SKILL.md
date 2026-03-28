---
name: validate
description: Compare a gyaradax output directory against its GKW reference
disable-model-invocation: true
argument-hint: <output-dir> <ref-dir>
---

Compare simulation output against GKW reference data.

1. Load `fluxes.npz` from the output dir and `fluxes.dat`/`time.dat` from the ref dir.
2. Time-average the last 80 (sim) and 240 (ref) windows.
3. Report per-flux relative error (pflux, eflux, vflux).
4. If spectra exist (`kyspec.npz`, `kxspec.npz` vs `kyspec`, `kxspec`), compare time-averaged profiles.

```python
import numpy as np
out_dir, ref_dir = "$ARGUMENTS".split()

sim = np.load(f"{out_dir}/fluxes.npz")["fluxes"]
ref = np.loadtxt(f"{ref_dir}/fluxes.dat")

for i, name in enumerate(["pflux", "eflux", "vflux"]):
    sim_avg = np.mean(sim[-80:, i])
    ref_avg = np.mean(ref[-240:, i])
    print(f"  {name}: sim={sim_avg:.4e}  ref={ref_avg:.4e}  rel_err={abs(sim_avg-ref_avg)/max(abs(ref_avg),1e-15):.2e}")
```
