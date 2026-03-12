import os
import argparse
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gyaradax import load_geometry, load_gkw_k_dump, get_integrals


def validate_spectra(run_dir, dump_name):
    """validate ky and kx spectra from a given dump."""
    print(f"--- validating physics for {run_dir} dump {dump_name} ---")

    geom = load_geometry(run_dir)
    ns, nkx, nky = len(geom["ints"]), len(geom["kxrh"]), len(geom["krho"])
    nvpar, nmu = len(geom["intvp"]), len(geom["intmu"])
    res = (nvpar, nmu, ns, nkx, nky)

    df = load_gkw_k_dump(os.path.join(run_dir, dump_name), res)
    phi, _ = get_integrals(df, geom)

    # average over parallel dimension s
    phi_sq = jnp.abs(phi) ** 2
    phi_avg_s = jnp.mean(phi_sq, axis=0)

    # ky spectrum (summed over kx)
    ky_spec = jnp.sum(phi_avg_s, axis=0)
    # kx spectrum (summed over ky)
    kx_spec = jnp.sum(phi_avg_s, axis=1)

    ky = np.asarray(geom["krho"])
    kx = np.asarray(geom["kxrh"])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.semilogy(ky, ky_spec, "o-")
    plt.xlabel(r"$k_y \rho_{ref}$")
    plt.ylabel(r"$|\phi|^2$")
    plt.title("ky spectrum")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.semilogy(kx, kx_spec, "o-")
    plt.xlabel(r"$k_x \rho_{ref}$")
    plt.ylabel(r"$|\phi|^2$")
    plt.title("kx spectrum")
    plt.grid(True, alpha=0.3)

    out_plot = f"physics_validation_{dump_name}.png"
    plt.tight_layout()
    plt.savefig(out_plot)
    print(f"saved spectra plots to {out_plot}")


def validate_growth_rates(run_dir):
    """compare growth rates vs ky against references."""
    growth_all = np.loadtxt(os.path.join(run_dir, "growth_rates_all_modes"))
    mode_label = np.loadtxt(os.path.join(run_dir, "mode_label"))
    kxrh = np.loadtxt(os.path.join(run_dir, "kxrh"))

    ixzero = np.argmin(np.abs(kxrh))
    cols = mode_label[ixzero].astype(int) - 1

    # take latest growth rates
    gamma_ref = growth_all[-1, cols]
    ky = np.loadtxt(os.path.join(run_dir, "krho"))

    plt.figure(figsize=(8, 5))
    plt.plot(ky, gamma_ref, "s--", label="GKW reference")
    plt.xlabel(r"$k_y \rho_{ref}$")
    plt.ylabel(r"$\gamma [v_{th}/R]$")
    plt.title("growth rate spectrum (kx=0)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_plot = "growth_rate_validation.png"
    plt.savefig(out_plot)
    print(f"saved growth rate plot to {out_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, default="/restricteddata/ukaea/gyrokinetics/raw/iteration_13"
    )
    parser.add_argument("--dump", type=str, default="100")
    args = parser.parse_args()

    if os.path.exists(args.dir):
        validate_spectra(args.dir, args.dump)
        validate_growth_rates(args.dir)
    else:
        print(f"directory {args.dir} not found, skipping validation script execution.")
