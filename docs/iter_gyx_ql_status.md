# gyx-QL in the TORAX ITER case — paused 2026-10-17

Paused by the user at the cost blocker. Code works; the configuration does
not fit in a transport loop.

## Working code

- gyaradax `feat/sat2-full-linear` @ `51fe47b`
- torax `qlgyx` @ `fde74d33`
- needs `JAX_PLATFORMS=cuda,cpu` — the diagnostics io_callback wants a CPU device
- arms wired in `comparison/iter_baseline/iter_case.py`:
  `gyx_ql` (p 0.648, cn 1.364) and `gyx_ql_p1` (p 1.0, cn 0.197),
  `rho_match` (0.3, 0.45, 0.6, 0.75)

## The blocker

`grid_preset='kinetic'` is the full NONLINEAR grid — nvpar 32, nmu 12,
ns 18, nkx 191, nky 32, ikxspace 16, two species. That was my
recommendation and it was wrong: it is the grid the calibration DATA was
produced on, which is a different requirement from a solve inside a
transport loop. One standalone linear solve at that grid took ~1.3 h on an
A40 in the rerun campaign; the plugin runs four of them vmapped per
transport call and TORAX makes hundreds of calls. kinetic measured one
transport call unfinished after 28 min on a B300. My own grid-timing scan
did not finish the first (full) grid in 10 minutes, which agrees.

## Levers, in the order I would try them

1. **Warm start** from the previous call's eigenmode per radius. The
   profiles barely move between transport steps, so a converged restart
   should need a few hundred steps rather than 4000. Biggest win, but
   gyaradax's linear path always starts from `gk_init` — no resume from a
   stored `df` exists yet, so this is a code change first.
2. **A reduced linear grid.** sat2 does NOT use `<k_perp^2>`, so the kx
   grid only matters through its effect on gamma(ky) and W(ky) — the
   ballooning extent. ikxspace 16 and nkx 191 are far more than a
   converged ITG eigenmode needs. Validate any candidate against the
   full-grid reruns on ~10 ITER-like cases: gamma(ky) and chain-0 W(ky).
3. **Fewer QL updates** — call the model every N steps and freeze chi
   between. Cheapest to implement, but it changes the transport solve
   rather than the model, so it is the least honest of the three.

## The catch that applies to all three

`cn = 1.364` and `p = 0.648` were fitted on weights computed at the FULL
grid. Any reduced grid that shifts gamma(ky) or W(ky) invalidates them, so
the 406 calibration points must be re-run at whatever grid is chosen and
the fit redone. That is cheap once the grid is cheap, but it is not
optional, and it is the step most likely to be skipped.
