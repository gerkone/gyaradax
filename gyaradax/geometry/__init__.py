"""Geometry module: circular (Lapillonne), s-alpha, Miller.

Public entry points:
- compute_geometry                 build from scalar parameters
- create_geometry                  build from a GeometrySpec via registry
- compute_geometry_from_input       build from a GKW input.dat
- geometry_from_geom_dat_and_input  build from a GKW geom.dat + input.dat
- load_loaded_geometry              build from a GKW reference/output directory

Internal connectivity helpers are re-exported for compatibility.
"""

from gyaradax.geometry.geom import (
    compute_geometry,
    create_geometry,
    compute_geometry_from_input,
    geometry_spec_from_input_dat,
    geometry_from_geom_dat_and_input,
)
from gyaradax.geometry.loaded import LoadedGKWGeometryModel, load_loaded_geometry
from gyaradax.geometry.registry import GeometryModel, get_geometry_model, list_geometry_models
from gyaradax.geometry.spec import (
    GeometrySpec,
    geometry_spec_from_compute_kwargs,
    geometry_spec_from_config,
)
from gyaradax.geometry.topology import (
    _build_mode_connectivity,
    _build_parallel_shift_maps,
    _build_pos_par_grid_classes,
)

__all__ = [
    "compute_geometry",
    "create_geometry",
    "GeometrySpec",
    "GeometryModel",
    "geometry_spec_from_compute_kwargs",
    "geometry_spec_from_config",
    "geometry_spec_from_input_dat",
    "get_geometry_model",
    "list_geometry_models",
    "compute_geometry_from_input",
    "geometry_from_geom_dat_and_input",
    "LoadedGKWGeometryModel",
    "load_loaded_geometry",
    "_build_mode_connectivity",
    "_build_pos_par_grid_classes",
    "_build_parallel_shift_maps",
]
