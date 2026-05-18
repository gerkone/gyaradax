"""Geometry module: circular (Lapillonne), s-alpha, Miller.

Public entry points:
- compute_geometry                  build from scalar parameters (thin wrapper)
- compute_geometry_from_input       build from a GKW input.dat
- geometry_from_geom_dat_and_input  build from a GKW geom.dat + input.dat
- build_topology                    discrete topology (numpy, called once)
- compute_continuous_geometry       pure-JAX continuous fields (jit/AD-safe)

Internal helpers `_build_mode_connectivity`, `_build_pos_par_grid_classes`,
`_build_parallel_shift_maps` are re-exported for `utils.load_geometry`.
"""

from gyaradax.geometry.geom import (
    compute_geometry,
    compute_geometry_from_input,
    geometry_from_geom_dat_and_input,
    build_topology,
    compute_continuous_geometry,
    _build_mode_connectivity,
    _build_pos_par_grid_classes,
    _build_parallel_shift_maps,
)

__all__ = [
    "compute_geometry",
    "compute_geometry_from_input",
    "geometry_from_geom_dat_and_input",
    "build_topology",
    "compute_continuous_geometry",
    "_build_mode_connectivity",
    "_build_pos_par_grid_classes",
    "_build_parallel_shift_maps",
]
