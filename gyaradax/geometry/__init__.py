"""Geometry module: circular (Lapillonne), s-alpha, Miller.

Public entry points:
- compute_geometry                 build from scalar parameters
- create_geometry                  build from a GeometrySpec via registry
- compute_geometry_from_input       build from a GKW input.dat
- geometry_from_geom_dat_and_input  build from a GKW geom.dat + input.dat

Internal helpers `_build_mode_connectivity`, `_build_pos_par_grid_classes`,
`_build_parallel_shift_maps` are re-exported for `utils.load_geometry`.
"""

from gyaradax.geometry.geom import (
    compute_geometry,
    create_geometry,
    compute_geometry_from_input,
    geometry_from_geom_dat_and_input,
    _build_mode_connectivity,
    _build_pos_par_grid_classes,
    _build_parallel_shift_maps,
)
from gyaradax.geometry.registry import GeometryModel, get_geometry_model, list_geometry_models
from gyaradax.geometry.spec import GeometrySpec, geometry_spec_from_compute_kwargs

__all__ = [
    "compute_geometry",
    "create_geometry",
    "GeometrySpec",
    "GeometryModel",
    "geometry_spec_from_compute_kwargs",
    "get_geometry_model",
    "list_geometry_models",
    "compute_geometry_from_input",
    "geometry_from_geom_dat_and_input",
    "_build_mode_connectivity",
    "_build_pos_par_grid_classes",
    "_build_parallel_shift_maps",
]
