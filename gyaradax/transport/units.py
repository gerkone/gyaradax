"""GyroBohm unit conventions and the TORAX quasilinear inputs of the gyaradax models.

gyaradax returns fluxes in GKW gyroBohm units; every conversion to TORAX happens here.
A convention fixes the thermal velocity (sqrt(2T/m) or sqrt(T/m)), the reference length
of the flux and the temperature inside the chi factor. See docs/TORAX.md.
"""

import dataclasses
import math
from typing import Literal

import jax
from torax._src.transport_model.quasilinear_utils import NormalizedLogarithmicGradients
from torax._src.transport_model.quasilinear_utils import calculate_chiGB


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QuasilinearInputs:
    """Face-grid chiGB and logarithmic gradients, in the TORAX_QL convention."""

    chiGB: jax.Array
    Rmin: jax.Array
    Rmaj: jax.Array
    lref_over_lti: jax.Array
    lref_over_lte: jax.Array
    lref_over_lne: jax.Array
    lref_over_lni0: jax.Array
    lref_over_lni1: jax.Array


# (sqrt 2)^3 from v_th = sqrt(2T/m) against sqrt(T/m)
VTH_SQRT2_FLUX_FACTOR = 2.0 * math.sqrt(2.0)


@dataclasses.dataclass(frozen=True)
class GyroBohmConvention:
    name: str
    flux_reference_length: Literal["R_major", "a_minor"]
    vth_includes_sqrt2: bool
    chi_reference_temperature: Literal["T_i", "T_e"]


GYARADAX = GyroBohmConvention("gyaradax-gkw", "R_major", True, "T_i")
TORAX_QL = GyroBohmConvention("torax-quasilinear", "R_major", False, "T_i")
QLKNN_HYPER = GyroBohmConvention("qlknn-hyper", "a_minor", False, "T_e")


def flux_reference_length(convention, geo):
    return geo.R_major if convention.flux_reference_length == "R_major" else geo.a_minor


def gradient_reference_length(geo):
    return geo.R_major


def chi_reference_temperature(convention, core_profiles):
    if convention.chi_reference_temperature == "T_i":
        return core_profiles.T_i.face_value()
    return core_profiles.T_e.face_value()


def chi_gb(convention, core_profiles, geo):
    return calculate_chiGB(
        reference_temperature=chi_reference_temperature(convention, core_profiles),
        reference_magnetic_field=geo.B_0,
        reference_mass=core_profiles.A_i,
        reference_length=flux_reference_length(convention, geo),
    )


def gb_flux_conversion_factor(from_convention, to_convention, geo, core_profiles=None):
    """Factor Q_ref(from) / Q_ref(to) converting a gyroBohm flux between conventions.

    A scalar, or a face-grid profile when the two chi reference temperatures differ
    (core_profiles is then required).
    """
    s_from = 1.0 if from_convention.vth_includes_sqrt2 else 0.0
    s_to = 1.0 if to_convention.vth_includes_sqrt2 else 0.0
    factor = VTH_SQRT2_FLUX_FACTOR ** (s_from - s_to)
    l_from = flux_reference_length(from_convention, geo)
    l_to = flux_reference_length(to_convention, geo)
    factor = factor * (l_to / l_from) ** 2
    if from_convention.chi_reference_temperature != to_convention.chi_reference_temperature:
        if core_profiles is None:
            raise ValueError(
                f"Converting {from_convention.name} -> {to_convention.name} changes the "
                "chi reference temperature; core_profiles is required."
            )
        t_from = chi_reference_temperature(from_convention, core_profiles)
        t_to = chi_reference_temperature(to_convention, core_profiles)
        factor = factor * (t_from / t_to) ** 1.5
    return factor


def convert_gb_flux(value, from_convention, to_convention, geo, core_profiles=None):
    return value * gb_flux_conversion_factor(from_convention, to_convention, geo, core_profiles)


def build_quasilinear_inputs(core_profiles, geo) -> QuasilinearInputs:
    log_grads = NormalizedLogarithmicGradients.from_profiles(
        core_profiles=core_profiles,
        radial_coordinate=geo.r_mid,
        radial_face_coordinate=geo.r_mid_face,
        reference_length=gradient_reference_length(geo),
    )
    return QuasilinearInputs(
        chiGB=chi_gb(TORAX_QL, core_profiles, geo),
        Rmaj=geo.R_major,
        Rmin=geo.a_minor,
        lref_over_lti=log_grads.lref_over_lti,
        lref_over_lte=log_grads.lref_over_lte,
        lref_over_lne=log_grads.lref_over_lne,
        lref_over_lni0=log_grads.lref_over_lni0,
        lref_over_lni1=log_grads.lref_over_lni1,
    )
