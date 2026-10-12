"""Preprocessed dataset output: quantized real-space frames and metadata in the neugk kvikio layout.

A trajectory lives in ``<root>/<name>_ifft_realpotens/``::

    data/timestep_NNNNN.<bits>.bin   df in real space, (re/im, species, vpar, mu, s, x, y);
                                     adiabatic runs have no species axis
    data/poten_NNNNN.<bits>.bin      phi in real space, (x, s, y)
    metadata.pkl                     everything below, including the statistics
    metadata_light.pkl               the same without the per-element df / phi moments
    config.yaml                      the run's effective config
    .gyaradax_run                    the run directory that writes the trajectory

``bits`` is ``zstd16`` (bfloat16 values byte-shuffled and zstd-compressed in 4 MiB chunks behind
a 4 KiB header, ``neugk_jax.dataset.zframe`` v1) or ``bf16`` (the raw values); either way the
metadata records ``quantization: bf16``, the precision of the values. Keys, transforms and
rounding follow ``neugk_jax.dataset.preprocess.preprocess_gyaradax_kinetic``, so the neugk
loaders read the output unchanged. The df / phi statistics are computed from the stored frames
once the run ends (:func:`write_statistics`).
"""

from __future__ import annotations

import glob
import os
import pickle
import re
import shutil
import struct
import warnings
from typing import Any, Callable, Optional

import numpy as np

# geometry entries of the kinetic dataset (neugk_jax.dataset.preprocess.kinetic_geometry)
GEOMETRY_KEYS = ("kxrh", "krho", "intvp", "vpgr", "intmu", "mugr", "ints", "bn", "bt_frac", "rfun", "efun",
                 "little_g", "mas", "tmp", "de", "signz", "vthrat", "d2X", "signB", "parseval")

OWNER_FILE = ".gyaradax_run"

# zstd16 shard layout (neugk_jax.dataset.zframe v1)
MAGIC, VERSION, ALIGN, CHUNK_BYTES = b"NGKZ", 1, 4096, 4 << 20
_FIXED = struct.Struct("<4sIIQQQ")


def trajectory_path(root: str, name: str) -> str:
    return os.path.join(root, f"{name}_ifft_realpotens")


def _claim(traj: str, run_dir: str) -> None:
    """Make ``traj`` the trajectory of ``run_dir``, refusing one another run or tool wrote."""
    owner_path = os.path.join(traj, OWNER_FILE)
    if os.path.islink(traj):
        raise FileExistsError(f"{traj} is a symlink to an existing trajectory; choose another dataset name")
    owner = open(owner_path).read().strip() if os.path.exists(owner_path) else None
    if owner is None and os.path.isdir(traj) and os.listdir(traj):
        raise FileExistsError(f"{traj} exists and was not written by a gyaradax run; choose another dataset name")
    if owner is not None and owner != run_dir:
        raise FileExistsError(f"{traj} belongs to the run in {owner}; choose another dataset name")
    os.makedirs(os.path.join(traj, "data"), exist_ok=True)
    if owner is None:
        _atomic_write(owner_path, (run_dir + "\n").encode())


def frame_path(traj: str, kind: str, index: int, bits: str) -> str:
    return os.path.join(traj, "data", f"{kind}_{int(index):05d}.{bits}.bin")


def _atomic_write(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def zstd16_encode(payload: np.ndarray, level: int = 1) -> bytes:
    """``.zstd16`` bytes of a bf16 payload: per 4 MiB chunk byte-shuffled, one zstd frame each."""
    from compression import zstd

    raw = np.ascontiguousarray(payload).view(np.uint8).ravel()
    chunks = [
        zstd.compress(np.ascontiguousarray(raw[i : i + CHUNK_BYTES].reshape(-1, 2).T).tobytes(), level)
        for i in range(0, raw.size, CHUNK_BYTES)
    ]
    start = -(-(_FIXED.size + 8 * (len(chunks) + 1)) // ALIGN) * ALIGN
    offsets = np.cumsum([start] + [len(c) for c in chunks], dtype=np.uint64)
    head = _FIXED.pack(MAGIC, VERSION, 2, raw.size, CHUNK_BYTES, len(chunks)) + offsets.tobytes()
    return head.ljust(start, b"\0") + b"".join(chunks)


def zstd16_decode(buf: bytes) -> np.ndarray:
    """bf16 bit patterns (uint16) of a ``.zstd16`` shard."""
    from compression import zstd

    magic, version, elem, raw_bytes, chunk, n = _FIXED.unpack_from(buf, 0)
    if magic != MAGIC or version != VERSION or elem != 2:
        raise IOError(f"not a zstd16 v{VERSION} shard (magic {magic!r}, version {version})")
    offsets = np.frombuffer(buf, dtype=np.uint64, count=n + 1, offset=_FIXED.size)
    out = np.empty(raw_bytes, np.uint8)
    for i in range(n):
        dst = out[i * chunk : min((i + 1) * chunk, raw_bytes)]
        dec = np.frombuffer(zstd.decompress(buf[int(offsets[i]) : int(offsets[i + 1])]), np.uint8)
        dst[:] = dec.reshape(2, -1).T.ravel()
    return out.view(np.uint16)


def read_frame(path: str, shape) -> np.ndarray:
    """float32 values of a stored frame."""
    with open(path, "rb") as fh:
        buf = fh.read()
    bits = zstd16_decode(buf) if path.endswith(".zstd16.bin") else np.frombuffer(buf, np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def _write_frame(path: str, bf16_bits: np.ndarray, bits: str) -> None:
    payload = np.ascontiguousarray(bf16_bits)
    _atomic_write(path, zstd16_encode(payload) if bits == "zstd16" else payload.tobytes())


def _to_bf16_bits(x) -> np.ndarray:
    import jax.numpy as jnp

    return np.asarray(jnp.asarray(x).astype(jnp.bfloat16)).view(np.uint16).ravel()


def df_to_real(df):
    """Real-space ``(re/im, ..., x, y)`` float32 of a spectral df (kx centred, ky one-sided)."""
    import jax.numpy as jnp

    phys = jnp.fft.ifftn(jnp.fft.fftshift(df, axes=-2), axes=(-2, -1), norm="forward")
    return jnp.stack([phys.real, phys.imag]).astype(jnp.float32)


def real_to_df(frame):
    """Spectral complex64 df of a real-space ``(re/im, ..., x, y)`` frame (inverse of ``df_to_real``)."""
    import jax
    import jax.numpy as jnp

    x = jnp.asarray(frame, jnp.float32)
    spec = jnp.fft.fftn(jax.lax.complex(x[0], x[1]), axes=(-2, -1), norm="forward")
    return jnp.fft.ifftshift(spec, axes=-2)


def df_frame(df):
    """The stored df frame as a bf16 array: the spectrum rounded to bf16, then ``df_to_real``."""
    import jax
    import jax.numpy as jnp

    df = jnp.asarray(df)
    spec = jax.lax.complex(*(x.astype(jnp.bfloat16).astype(jnp.float32) for x in (df.real, df.imag)))
    return df_to_real(spec).astype(jnp.bfloat16)


def df_frame_bits(df) -> np.ndarray:
    return np.asarray(df_frame(df)).view(np.uint16).ravel()


def _rel_err(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-300))


def quantization_loss(df, frame, phi_real, phi_frame, integrals: Optional[Callable] = None) -> dict:
    """Per-frame bf16 loss: stored phi against phi, and phi and fluxes recomputed from the stored df.

    ``integrals`` maps a spectral df to ``(phi, fluxes)``; it runs on ``df`` and on the df of
    ``frame``, so the errors hold the quantization alone. Without it the df entries are nan.
    """
    import jax.numpy as jnp

    out = {"quant_phi_err": _rel_err(np.asarray(phi_frame, np.float32), phi_real)}
    if integrals is None:
        return {**out, "quant_phi_df_err": np.nan, "quant_flux_err": np.nan}
    phi_ref, flux_ref = integrals(df)
    phi_q, flux_q = integrals(real_to_df(frame).astype(jnp.asarray(df).dtype))
    ref, q = (np.asarray(f, np.float64).reshape(-1) for f in (flux_ref, flux_q))
    scale = np.abs(ref)
    # channels zero to round-off (e.g. the adiabatic particle flux) have no relative error
    with np.errstate(divide="ignore", invalid="ignore"):
        flux_err = np.where(scale > 1e-12 * scale.max(initial=0.0), (q - ref) / scale, np.nan)
    return {**out, "quant_phi_df_err": _rel_err(phi_q, phi_ref), "quant_flux_err": flux_err}


def phi_to_real(phi) -> np.ndarray:
    """Real ``(x, s, y)`` float32 of a spectral phi ``(s, kx, ky)`` (complex64 transform, as neugk)."""
    spec = np.moveaxis(np.asarray(phi), 0, 1).astype(np.complex64)
    nkx, _, nky = spec.shape
    real = np.fft.irfftn(np.fft.ifftshift(spec, axes=0), axes=(0, 2), norm="forward", s=(nkx, nky))
    return real.astype(np.float32)


class RunningStats:
    """Elementwise running mean / var / min / max with the neugk prior (count 1e-4, variance 1)."""

    def __init__(self, prior_count: float = 1e-4):
        self.count = prior_count
        self.mean = self.var = self.min = self.max = None

    def merge(self, mean, var, mn, mx, count: float = 1.0) -> None:
        mean, var = np.asarray(mean, np.float64), np.asarray(var, np.float64)
        if self.mean is None:
            self.mean, self.var = np.zeros(mean.shape), np.ones(mean.shape)
            self.min, self.max = np.full(mean.shape, np.inf), np.full(mean.shape, -np.inf)
        new = self.count + count
        delta = mean - self.mean
        m2 = self.var * self.count + var * count + delta**2 * (self.count * count / new)
        self.mean = self.mean + delta * (count / new)
        self.var = m2 / new
        self.min = np.minimum(self.min, np.asarray(mn, np.float64))
        self.max = np.maximum(self.max, np.asarray(mx, np.float64))
        self.count = new

    def push(self, x) -> None:
        x = np.asarray(x, np.float64)
        self.merge(x, np.zeros_like(x), x, x, 1.0)

    def moments(self, prefix: str) -> dict:
        var = np.asarray(self.var)
        out = {"mean": self.mean, "var": var, "std": np.sqrt(var), "min": self.min, "max": self.max}
        return {f"{prefix}_{k}": np.asarray(v, np.float32) for k, v in out.items()}


def load_metadata(traj: str) -> Optional[dict]:
    for name in ("metadata.pkl", "metadata_light.pkl"):
        path = os.path.join(traj, name)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                return pickle.load(fh)
    return None


def write_metadata(traj: str, metadata: dict) -> None:
    _atomic_write(os.path.join(traj, "metadata.pkl"), pickle.dumps(metadata))
    # metadata_light leaves out the per-element moments, as neugk does
    drop = {f"df_{m}" for m in ("mean", "var", "std", "min", "max")} | {"phi_min", "phi_max", "phi_var"}
    light = {k: v for k, v in metadata.items() if k not in drop}
    _atomic_write(os.path.join(traj, "metadata_light.pkl"), pickle.dumps(light))


def _static_metadata(geometry: dict, params: Any, config_path: str) -> dict:
    kinetic = not params.adiabatic_electrons
    resolution = tuple(len(np.asarray(geometry[k])) for k in ("intvp", "intmu", "ints", "kxrh", "krho"))
    geom = {k: np.array(geometry[k], np.float64) for k in GEOMETRY_KEYS}
    # the neugk flux kernel weights ints twice, so ky > 0 carries 2 * ns
    geom["parseval"][1:] *= len(geom["ints"])
    geom["adiabatic"] = np.array(0.0 if kinetic else 1.0)
    geom["beta"] = np.array(float(getattr(params, "beta", 0.0)))
    geom["nlapar"] = np.array(0.0)
    geom["nlbpar"] = np.array(0.0)
    signz = np.atleast_1d(np.asarray(params.signz, dtype=np.float64))
    tmp = np.atleast_1d(np.asarray(params.tmp, dtype=np.float64))
    rlt = np.atleast_1d(np.asarray(params.rlt, dtype=np.float64))
    rln = np.atleast_1d(np.asarray(params.rln, dtype=np.float64))
    n_species = len(signz) if kinetic else 1
    ion = int(np.argmax(signz > 0))
    elec = int(np.argmax(signz < 0)) if kinetic else -1
    df_shape = (2, n_species, *resolution) if kinetic else (2, *resolution)
    meta = {
        "simulation_code": "gyaradax",
        "source_config": os.path.abspath(config_path),
        "ds": float(geom["ints"][0]),
        "resolution": resolution,
        "df_shape": df_shape,
        "phi_shape": (resolution[3], resolution[2], resolution[4]),
        "axis_order": (
            "df: (re/im, species, vpar, mu, s, x, y); phi: (x, s, y); x, y real space"
            if kinetic
            else "df: (re/im, vpar, mu, s, x, y); phi: (x, s, y); x, y real space"
        ),
        "channel_layout": "reim_species" if kinetic else "reim",
        "quantization": "bf16",
        "real_space_convention": "ifftn(fftshift(spec, axes=kx), axes=(kx, ky), norm='forward')",
        "n_species": n_species,
        "ion_species_index": ion,
        "electron_species_index": elec,
        "ion_temp_grad": np.array([rlt[ion]]),
        "density_grad": np.array([rln[ion]]),
        "rlt": rlt,
        "rln": rln,
        "tmp": tmp,
        "ion_temp": np.array([tmp[ion]]),
        "s_hat": np.array([float(params.shat)]),
        "q": np.array([float(params.q)]),
        "geometry": geom,
        "flux_labels": [f"{f}_s{s}" for s in range(n_species) for f in ("pflux", "eflux", "vflux")],
        # df moments are pooled over (vpar, s, x, y): one value per (re/im, [species,] mu)
        "df_stats_agg_axes": (2, 4, 5, 6) if kinetic else (1, 3, 4, 5),
    }
    if kinetic:
        meta["electron_temp_grad"] = np.array([rlt[elec]])
        meta["electron_temp"] = np.array([tmp[elec]])
        meta["temp_ratio"] = np.array([tmp[elec] / tmp[ion]])
    return meta


# per-frame metadata; the quant_* entries track the bf16 loss (quantization_loss)
SERIES_KEYS = ("timesteps", "solver_steps", "flux", "kyspec", "kxspec", "growth",
               "quant_phi_err", "quant_phi_df_err", "quant_flux_err")


class DatasetWriter:
    """Appends one quantized frame per call to a trajectory of the preprocessed layout.

    ``start_step`` is the solver step the run starts from: frames of a previous leg past that
    step (a resume from an older snapshot) are dropped, earlier ones are kept and the new
    frames continue their numbering. ``finalize`` computes the statistics of the frames from
    time ``stats_from`` on. The trajectory belongs to the run directory holding ``config_path``:
    an existing trajectory of another run, or one not written by a run, is refused.
    """

    def __init__(self, root: str, name: str, geometry: dict, params: Any, config_path: str,
                 bits: str = "zstd16", start_step: int = 0, stats_from: float = 0.0):
        if bits not in ("zstd16", "bf16"):
            raise ValueError(f"dataset bits must be zstd16 or bf16, got {bits!r}")
        self.path = trajectory_path(root, name)
        self.bits = bits
        self.stats_from = float(stats_from)
        _claim(self.path, os.path.dirname(os.path.abspath(config_path)))
        shutil.copyfile(config_path, os.path.join(self.path, "config.yaml"))
        self.static = _static_metadata(geometry, params, os.path.join(self.path, "config.yaml"))
        self.series: dict[str, list] = {k: [] for k in SERIES_KEYS}
        old = load_metadata(self.path)
        if old is not None and len(old.get("timesteps", ())):
            keep = int(np.sum(np.asarray(old["solver_steps"]) <= start_step))
            for k in self.series:
                self.series[k] = list(np.asarray(old[k])[:keep])
            for kind in ("timestep", "poten"):
                for path in glob.glob(os.path.join(self.path, "data", f"{kind}_*.bin")):
                    if int(re.search(rf"{kind}_(\d+)", path).group(1)) >= keep:
                        os.remove(path)

    @property
    def n_frames(self) -> int:
        return len(self.series["timesteps"])

    def add(self, df, phi, fluxes, state, ky_spec, kx_spec, integrals: Optional[Callable] = None) -> None:
        """Write one frame; ``integrals`` (spectral df -> (phi, fluxes)) enables the df quantization loss."""
        import jax.numpy as jnp

        idx = self.n_frames
        frame = df_frame(df)
        _write_frame(frame_path(self.path, "timestep", idx, self.bits), np.asarray(frame).view(np.uint16).ravel(), self.bits)
        phi_real = phi_to_real(phi)
        phi_frame = np.asarray(jnp.asarray(phi_real).astype(jnp.bfloat16))
        _write_frame(frame_path(self.path, "poten", idx, self.bits), phi_frame.view(np.uint16).ravel(), self.bits)
        loss = quantization_loss(df, frame, phi_real, phi_frame, integrals)
        n_flux = np.asarray(fluxes).size
        loss["quant_flux_err"] = np.broadcast_to(np.asarray(loss["quant_flux_err"], np.float64), (n_flux,)).copy()
        for k, v in loss.items():
            self.series[k].append(v)
        self.series["timesteps"].append(float(np.ravel(np.asarray(state.time))[0]))
        self.series["solver_steps"].append(int(np.ravel(np.asarray(state.step))[0]))
        self.series["flux"].append(np.asarray(fluxes, np.float64).reshape(-1))
        self.series["kyspec"].append(np.asarray(ky_spec, np.float64))
        self.series["kxspec"].append(np.asarray(kx_spec, np.float64))
        self.series["growth"].append(np.asarray(state.last_growth_rate, np.float64))
        write_metadata(self.path, self.metadata())

    def metadata(self) -> dict:
        meta = dict(self.static)
        for k, v in self.series.items():
            meta[k] = np.asarray(v)
        return meta

    def finalize(self) -> None:
        if self.n_frames and self.series["timesteps"][-1] >= self.stats_from:
            write_statistics(self.path, self.stats_from)
        if self.n_frames:
            print(f"  dataset {os.path.basename(self.path)}: {quantization_summary(self.metadata())}")


def quantization_summary(meta: dict) -> str:
    """Median bf16 loss over the frames of a trajectory's metadata."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = {k: np.nanmedian(np.abs(np.asarray(meta[k], np.float64)), axis=0)
               for k in ("quant_phi_err", "quant_phi_df_err", "quant_flux_err")}
    eflux = ", ".join(f"{label} {err:.1e}" for label, err in zip(meta["flux_labels"], np.ravel(med["quant_flux_err"]))
                      if label.startswith("eflux"))
    return (f"bf16 loss (median relative error): phi frame {med['quant_phi_err']:.1e}, "
            f"phi from df {med['quant_phi_df_err']:.1e}, {eflux}")


def write_statistics(traj: str, stats_from: float = 0.0) -> dict:
    """df / phi moments of the stored frames from time ``stats_from`` on, written into the metadata."""
    meta = load_metadata(traj)
    if meta is None:
        raise FileNotFoundError(f"no metadata in {traj}")
    times = np.asarray(meta["timesteps"])
    first = int(np.searchsorted(times, stats_from, side="left"))
    if first >= len(times):
        raise ValueError(f"{traj}: no frame at or after t = {stats_from} (last t = {times[-1]:.3f})")
    bits = "zstd16" if os.path.exists(frame_path(traj, "timestep", first, "zstd16")) else "bf16"
    axes = tuple(int(a) for a in meta["df_stats_agg_axes"])
    df_stats, phi_stats = RunningStats(), RunningStats()
    for i in range(first, len(times)):
        df = read_frame(frame_path(traj, "timestep", i, bits), meta["df_shape"])
        count = float(np.prod([df.shape[a] for a in axes]))
        df_stats.merge(
            *(np.asarray(f(df, axis=axes, keepdims=True), np.float64) for f in (np.mean, np.var, np.min, np.max)),
            count,
        )
        phi_stats.push(read_frame(frame_path(traj, "poten", i, bits), meta["phi_shape"]))
    stats = {**df_stats.moments("df"), **phi_stats.moments("phi")}
    meta = {**{k: v for k, v in meta.items() if k not in stats}, **stats}
    write_metadata(traj, meta)
    return meta
