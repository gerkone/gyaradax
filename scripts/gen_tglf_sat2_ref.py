"""Dump TGLF SAT2 reference spectra for tests/unit/test_sat2_vs_tglf.py.

Needs a gacode TGLF built through f2py with the per-ky spectrum arrays
exposed (tglf_kyspectrum, tglf_dimensions and the tglf_global spectra), which
the stock torax tglf2py.pyf does not expose. Point TGLFSPEC_DIR at the
directory holding the built tglfspec_lib*.so and run this with that build's
interpreter:

    TGLFSPEC_DIR=<...>/gacode/f2py/tglfspec \\
        <that venv>/bin/python scripts/gen_tglf_sat2_ref.py out.json

The resulting JSON is what TGLF_SAT2_REF must point at.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("TGLFSPEC_DIR", "."))
import tglfspec_lib as T  # noqa: E402

ti = T.tglf_interface

CASES = {
    "base": {},
    "steep_rlt": {"rlt_i": 5.0, "rlt_e": 5.0},
    "low_q": {"q": 1.2, "qp": 2.0},
    "high_shear": {"qp": 12.0},
    "outer": {"rmin": 0.9},
    "peaked_rln": {"rln": 3.0},
}


def run_point(alpha_zf=-1.0, **over):
    ti.tglf_sat_rule_in = 2
    ti.tglf_kygrid_model_in = 4
    ti.tglf_nbasis_max_in, ti.tglf_nbasis_min_in, ti.tglf_nxgrid_in = 4, 2, 16
    ti.tglf_alpha_zf_in, ti.tglf_filter_in = alpha_zf, 2.0
    ti.tglf_use_bper_in = ti.tglf_use_bpar_in = False
    ti.tglf_geometry_flag_in = 1
    ti.tglf_ns_in = 2
    ti.tglf_zs_in[0], ti.tglf_mass_in[0] = -1.0, 2.72e-4
    ti.tglf_as_in[0], ti.tglf_taus_in[0] = 1.0, 1.0
    ti.tglf_zs_in[1], ti.tglf_mass_in[1] = 1.0, 1.0
    ti.tglf_as_in[1], ti.tglf_taus_in[1] = 1.0, 1.0
    ti.tglf_rlns_in[0] = ti.tglf_rlns_in[1] = over.get("rln", 1.0)
    ti.tglf_rlts_in[0] = over.get("rlt_e", 3.0)
    ti.tglf_rlts_in[1] = over.get("rlt_i", 3.0)
    ti.tglf_rmin_loc_in, ti.tglf_rmaj_loc_in = over.get("rmin", 0.5), over.get("rmaj", 3.0)
    ti.tglf_q_loc_in, ti.tglf_q_prime_loc_in = over.get("q", 2.0), over.get("qp", 4.0)
    ti.tglf_p_prime_loc_in = 0.0
    ti.tglf_kappa_loc_in, ti.tglf_delta_loc_in, ti.tglf_drmajdx_loc_in = 1.0, 0.0, 0.0
    ti.tglf_zeff_in, ti.tglf_betae_in, ti.tglf_xnue_in = 1.0, 0.0, 0.0
    ti.tglf_debye_in, ti.tglf_vexb_shear_in = 0.0, 0.0
    ti.tglf_use_transport_model_in = True
    ti.tglf_quiet_flag_in = True
    T.tglf_run()

    nky = int(T.tglf_dimensions.nky)
    g = T.tglf_global
    rln = over.get("rln", 1.0)
    rlts = over.get("rlt_e", 3.0) + over.get("rlt_i", 3.0)
    dlnpdr = min(max(over.get("rmaj", 3.0) * (2.0 * rln + rlts) / 2.0, 4.0), 18.0)
    return dict(
        alpha_zf=alpha_zf,
        nmodes=int(ti.tglf_nmodes_in),
        ky=np.asarray(T.tglf_kyspectrum.ky_spectrum[:nky]).tolist(),
        gamma=np.asarray(g.eigenvalue_spectrum_out)[0, :nky, 0].tolist(),
        phinorm=np.asarray(g.field_spectrum_out)[1, :nky, 0].tolist(),
        kymax=float(g.kymax_out),
        vzf=float(g.vzf_out),
        grad_r0=float(g.grad_r0_out),
        b_geo0=float(g.b_geo0_out),
        bt0=float(g.bt0_out),
        sat_geo0=float(g.sat_geo0_out),
        sat_geo1=float(g.sat_geo1_out),
        sat_geo2=float(g.sat_geo2_out),
        dlnpdr=dlnpdr,
    )


def _emit(azf):
    return {
        f"{name}_azf{int(azf)}": run_point(alpha_zf=azf, **over)
        for name, over in CASES.items()
    }


if __name__ == "__main__":
    # tglf_multiscale_spectrum.f90 latches use_kymin on the first ALPHA_ZF < 0
    # run and never clears it, so each setting needs its own process
    if "--azf" in sys.argv:
        json.dump(_emit(float(sys.argv[sys.argv.index("--azf") + 1])), sys.stdout)
        raise SystemExit

    import subprocess

    out = sys.argv[1] if len(sys.argv) > 1 else "tglf_sat2_ref.json"
    ref = {}
    for azf in (-1.0, 1.0):
        res = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--azf", str(azf)],
            capture_output=True, text=True, check=True,
        )
        ref.update(json.loads(res.stdout))
    with open(out, "w") as fh:
        json.dump(ref, fh)
    print(f"wrote {len(ref)} cases to {out}")
