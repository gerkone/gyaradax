# ex1: <one line: what the campaign adds to the dataset>

<N> trajectories into `<dataset root>` as ex1_000..ex1_<N-1>, plus <M> reserves. `make_configs.py` writes `configs/`, `design.csv` and `order.txt`.

## Design

- Purpose: <the gap in the existing dataset this fills, or the test set it builds>.
- Parameters and ranges: <table or list>, sampled by <method, seed>; reserves replace banded or diverged runs.
- Training or evaluation: <train: frames kept, head cut after review | evaluation: full history, no frames kept beyond the last>.

## Numerics

- Base config: <kinetic or adiabatic example>, grid <nvpar x nmu x ns x nkx x nky>, frames every 2.2 t to t 500.
- Code: <gyaradax revision>, flags <per GPU family>.

## Log

One line per decision (launch, moves, stops, replacements, finalize), newest last. Run events go to `queue.log`.

- <date>: <decision> (<who asked>)
