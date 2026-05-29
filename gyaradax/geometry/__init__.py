"""Public geometry construction API.

Analytic construction entry points:
- ``compute_geometry`` builds directly from scalar parameters.
- ``compute_geometry_from_config`` builds from a YAML/OmegaConf config.
- ``compute_geometry_from_input`` builds from a GKW ``input.dat``.
- ``create_geometry`` builds from a normalized ``GeometrySpec`` via registry.

Loaded/reference construction entry points:
- ``load_loaded_geometry`` reads a GKW reference/output directory.
- ``geometry_from_geom_dat_and_input`` builds from ``reference/geom.dat`` plus
  grids/config in a sibling GKW ``input.dat``.

The geometry spec, registry protocol types, and model lookup helpers are also
exported here. Private connectivity helpers remain re-exported only for
backward compatibility with older internal imports.
"""

from gyaradax.geometry.geom import (
    compute_geometry,
    create_geometry,
    compute_geometry_from_config,
    compute_geometry_from_input,
    geometry_spec_from_input_dat,
    geometry_from_geom_dat_and_input,
)
from gyaradax.geometry.loaded import LoadedGKWGeometryModel, load_loaded_geometry
from gyaradax.geometry.registry import (
    ContinuousGeometryModel,
    GeometryModel,
    get_geometry_model,
    list_geometry_models,
)
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
    "compute_geometry_from_config",
    "GeometrySpec",
    "GeometryModel",
    "ContinuousGeometryModel",
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
