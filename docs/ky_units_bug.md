# The ky-axis convention is producer-dependent, not column-dependent

Found 2026-10-16 calibrating TGLF-SAT2. Supersedes an earlier note of mine
that described this as a difference *between tables*; it is not.

## The rule

The stored binormal wavenumber depends on which code produced the LINEAR
run, so two rows of the same table can be in different units.

```
lin_code == "gkw"        ->  lin_ky = krho              (max = krhomax)
lin_code == "gyaradax"   ->  lin_ky = krho / kthnorm    (max = krhomax/kthnorm)
lin_code is null         ->  unkeyable; 6 kinetic rows
kthnorm = sqrt(g_zz_mid)
        = sqrt( (1/(2*pi*(1+eps)))^2 * (1 + (1-eps^2)*(q/eps)^2) )   [circular]
```

kthnorm ranges 0.71-6.21 and is a strong function of q/eps, so it does not
absorb into any fitted constant.

## Evidence

Each campaign ran at one `krhomax`, so after the correct per-row correction
`max(lin_ky)` must be constant. Scaling only `lin_code == "gyaradax"` rows:

| table | column | raw spread | corrected spread | corrected median |
|---|---|--:|--:|--:|
| adiabatic | lin_ky | 4.69x | **1.00x** | 1.4000 |
| adiabatic | spec_ky | 4.69x | **1.00x** | 1.4000 |
| new_runs_adiabatic | lin_ky | 1.00x | 1.00x | 1.4000 |
| kinetic | lin_ky | 7.84x | **1.00x** | 1.5000 |

Grouped by producer, the corrected values are exact, not approximate:

```
kinetic, lin_code=gyaradax :  ky_max  1.5000..1.5000   (n=382)
                              ky_min  0.0484..0.0484
                              spec_ky 1.5000..1.5000   (n=283)
kinetic, lin_code=null     :  ky_max  0.2753..0.3966   (n=6)  <- unkeyable
```

Independent physics check (same ITG family, so the gamma peak must coincide):

```
adiabatic median ky_gamma_max            = 0.542
kinetic   median ky_gamma_max            = 0.182   disagrees
kinetic   median ky_gamma_max * kthnorm  = 0.484   agrees
```

## Scope: ky only, NOT kx

Applying the same correction to the radial columns makes them worse, so kx is
already consistent across producers and must be left alone:

| column | raw spread | "corrected" spread |
|---|--:|--:|
| kx_max (adiabatic) | 2.22x | 7.34x |
| kx_max (kinetic) | 1.98x | 11.95x |
| spec_kx (kinetic) | 1.82x | 11.20x |

Affected: `lin_ky`, `spec_ky`, `ky_min`, `ky_max`, `ky_gamma_max`.
Unaffected: `kx_max`, `spec_kx`, and all rates/weights — `lin_gamma`,
`lin_omega` and `lin_weight_*` are per-mode ratios and carry no ky scale.

## Observed impact

Two adiabatic rows have `lin_code = "gyaradax"`:
`neurips26/id/extra_iteration_8` and `_9`. Uncorrected, they mispredicted the
SAT2 flux by 89x and 124x and alone took the adiabatic test RMSE from 86 to
3099. SAT2 is especially exposed because `kycut = 0.8*grad_r0` and
`kymin = 0.173*sqrt(2)*grad_r0` are absolute thresholds in k_theta*rho: a
wrong ky axis moves every mode across them.

## Why a dataloader fix is the wrong fix

Keying on `lin_code` at read time is what I am doing to stay unblocked, and
it is not good enough as the permanent answer:

1. it fails outright on the 6 kinetic rows where `lin_code` is null;
2. it silently mis-scales any future producer that is neither name;
3. every consumer must reimplement it, and a consumer that does not even
   gets a plausible-looking wrong answer rather than an error.

## Suggested fix, at write time

1. Normalise in the writer so the stored ky axis has ONE meaning regardless
   of producer. State it in the README next to `lin_ky`.
2. Add a `kthnorm` column unconditionally — physical k_theta*rho is not
   reconstructible from the table without it.
3. Add `ky_convention` (e.g. "krho") so a future producer must declare it
   rather than inherit silently.
4. Put the invariant in `verify_summary.py` as an assertion, e.g.
   `max(lin_ky)` constant within a campaign, and
   `isclose(max(lin_ky), krhomax)`. Any future producer violating the
   convention then fails the build instead of reaching consumers.
5. Resolve the 6 null-`lin_code` kinetic rows, or drop them.
6. Adjacent: `ky_max` is NaN in 306/331 adiabatic and 998/1021 new_runs rows
   (recoverable as `max(lin_ky)`).
