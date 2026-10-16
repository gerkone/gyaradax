"""Run GACODE TGLF SAT2 at every calibration point.

Gives the original-calibration TGLF-SAT2 prediction to compare against, and
the geometry scalars (grad_r0, B_geo0, SAT_geo*) that SAT2's kx_width and
sat_geo models need. Run with the interpreter that built tglfspec_lib.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("TGLFSPEC_DIR", "."))
import tglfspec_lib as T  # noqa: E402

ti = T.tglf_interface
RMAJ = 3.0          # R/a; rmin_loc = eps*RMAJ keeps eps = r/R
ALPHA_ZF = -1.0     # one sign for the whole sweep: use_kymin latches per process


def run_point(p):
    eps = float(p["eps"])
    rmin = eps * RMAJ
    ti.tglf_sat_rule_in = 2
    ti.tglf_kygrid_model_in = 4
    ti.tglf_nbasis_max_in, ti.tglf_nbasis_min_in, ti.tglf_nxgrid_in = 4, 2, 16
    ti.tglf_alpha_zf_in, ti.tglf_filter_in = ALPHA_ZF, 2.0
    ti.tglf_use_bper_in = ti.tglf_use_bpar_in = False
    ti.tglf_geometry_flag_in = 1
    ti.tglf_adiabatic_elec_in = not bool(p["kinetic"])
    ti.tglf_ns_in = 2
    # species 1 is the electron in TGLF; taus is relative to the electron
    te_ti = float(p["te_ti"]) if p["te_ti"] else 1.0
    ti.tglf_zs_in[0], ti.tglf_mass_in[0] = -1.0, 2.72e-4
    ti.tglf_as_in[0], ti.tglf_taus_in[0] = 1.0, 1.0
    ti.tglf_zs_in[1], ti.tglf_mass_in[1] = 1.0, 1.0
    ti.tglf_as_in[1], ti.tglf_taus_in[1] = 1.0, 1.0 / te_ti
    # our rlt/rln are R/L; TGLF wants a/L, and dlnpdr multiplies by rmaj again
    ti.tglf_rlns_in[0] = ti.tglf_rlns_in[1] = float(p["rln"]) / RMAJ
    ti.tglf_rlts_in[0] = float(p["rlt_e"] or 0.0) / RMAJ
    ti.tglf_rlts_in[1] = float(p["rlt_i"]) / RMAJ
    ti.tglf_rmin_loc_in, ti.tglf_rmaj_loc_in = rmin, RMAJ
    ti.tglf_q_loc_in = abs(float(p["q"]))
    # Q_PRIME_LOC = (q/rmin)^2 * s_hat
    ti.tglf_q_prime_loc_in = (float(p["q"]) / rmin) ** 2 * float(p["shat"])
    ti.tglf_p_prime_loc_in = 0.0
    ti.tglf_kappa_loc_in, ti.tglf_delta_loc_in, ti.tglf_drmajdx_loc_in = 1.0, 0.0, 0.0
    ti.tglf_zeff_in, ti.tglf_betae_in, ti.tglf_xnue_in = 1.0, 0.0, 0.0
    ti.tglf_debye_in, ti.tglf_vexb_shear_in = 0.0, 0.0
    ti.tglf_use_transport_model_in = True
    ti.tglf_quiet_flag_in = True
    T.tglf_run()

    nky = int(T.tglf_dimensions.nky)
    g = T.tglf_global
    ev = np.asarray(g.eigenvalue_spectrum_out)
    return dict(
        name=p["name"], split=p["split"], kinetic=p["kinetic"],
        ky=np.asarray(T.tglf_kyspectrum.ky_spectrum[:nky]).tolist(),
        gamma=ev[0, :nky, 0].tolist(),
        phinorm=np.asarray(g.field_spectrum_out)[1, :nky, 0].tolist(),
        kymax=float(g.kymax_out), vzf=float(g.vzf_out),
        grad_r0=float(g.grad_r0_out), b_geo0=float(g.b_geo0_out), bt0=float(g.bt0_out),
        sat_geo0=float(g.sat_geo0_out), sat_geo1=float(g.sat_geo1_out),
        sat_geo2=float(g.sat_geo2_out),
        nmodes=int(ti.tglf_nmodes_in),
        Qi_tglf=float(ti.tglf_ion_eflux_out[0]), Qe_tglf=float(ti.tglf_elec_eflux_out),
        Gam_tglf=float(ti.tglf_elec_pflux_out),
    )


if __name__ == "__main__":
    pts = json.load(open(sys.argv[1]))
    out = {}
    for ds, rows in pts.items():
        res, bad = [], 0
        for i, p in enumerate(rows):
            try:
                res.append(run_point(p))
            except Exception as e:  # a failed point must not kill the sweep
                bad += 1
                res.append(dict(name=p["name"], split=p["split"],
                                kinetic=p["kinetic"], error=str(e)))
            if (i + 1) % 25 == 0:
                print(f"  {ds} {i+1}/{len(rows)}", flush=True)
        out[ds] = res
        print(f"{ds}: {len(res)} done, {bad} failed", flush=True)
    json.dump(out, open(sys.argv[2], "w"))
    print("wrote", sys.argv[2])
