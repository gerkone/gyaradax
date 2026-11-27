import os
"""Acceptance summary from gyx/results/ACC.jsonl per deck and variant: largest growth among grid-scale modes (phi centroid above 0.3 of
Nyquist), leading smooth mode (centroid below 0.2), and its shift against the smooth mode of the plain run on the same grid.

usage: acc_table.py [DECK ...]
"""
import json
import sys

rows = {}
for line in open(os.environ.get("ACC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ACC.jsonl"))):
    r = json.loads(line)
    rows[r["tag"][4:]] = r


def split(r):
    fast = [m for m in r["modes"] if m.get("kc_phi", 0) > 0.3 and m["gamma_c"] > 0]
    smooth = [m for m in r["modes"] if m.get("kc_phi", 0) < 0.2]
    return (max(m["gamma_c"] for m in fast) if fast else None), (smooth[0] if smooth else None)


decks = sorted({k.split("_")[0] for k in rows})
for d in decks:
    if len(sys.argv) > 1 and d not in sys.argv[1:]:
        continue
    ref = {}
    for k in sorted(rows):
        if k.split("_")[0] == d and "_pl" in k and "dp" not in k and split(rows[k])[1]:
            ref[(rows[k]["ns"], rows[k]["nkx"])] = split(rows[k])[1]
    for k in sorted(rows):
        if k.split("_")[0] != d:
            continue
        r = rows[k]
        f, s = split(r)
        key = (r["ns"], r["nkx"])
        base = ref.get(key)
        sh = ""
        if s and base and abs(s["omega_c"] - base["omega_c"]) < 0.3 * abs(base["omega_c"]) + 0.02:
            sh = f" shift {100 * (s['gamma_c'] / base['gamma_c'] - 1):+.1f} % / {100 * (s['omega_c'] / base['omega_c'] - 1):+.1f} %"
        print(f"{k:20s} {r['ns']:3d}x{r['nkx']:<2d} dp{r['disp_par']:<4} rho {r['spectral_radius']:6.0f} | fast {'none ' if f is None else f'{f:5.3f}'} | smooth "
              + (f"{s['gamma_c']:.4f}{s['omega_c']:+.3f}i" if s else "   -   ") + sh)
    print()
