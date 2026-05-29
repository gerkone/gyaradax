"""Characterization tests for GKW input.dat parsing and species defaults.

These tests freeze current parser quirks before loader/geometry consolidation.
They intentionally cover public APIs rather than introducing new schema rules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gyaradax.geometry import compute_geometry
from gyaradax.params import gkparams_from_input_and_geometry
from gyaradax.utils import parse_input_dat


def test_parse_input_dat_preserves_repeated_section_suffix_order(tmp_path: Path) -> None:
    """Repeated namelist sections keep today's species/species0/species00 keys."""
    input_dat = tmp_path / "input.dat"
    input_dat.write_text(
        """
 &SPECIES
 MASS = 1.0
 Z = 1.0
 /
 &SPECIES
 MASS = 2.0
 Z = -1.0
 /
 &SPECIES
 MASS = 3.0
 Z = 2.0
 /
""",
        encoding="utf-8",
    )

    parsed = parse_input_dat(str(input_dat))

    assert [key for key in parsed if key.startswith("species")] == [
        "species",
        "species0",
        "species00",
    ]
    assert parsed["species"]["mass"] == 1.0
    assert parsed["species0"]["mass"] == 2.0
    assert parsed["species00"]["z"] == 2.0


def test_parse_input_dat_keeps_comments_and_commas_inside_quotes(tmp_path: Path) -> None:
    """Inline comments and top-level comma splitting respect quoted strings."""
    input_dat = tmp_path / "input.dat"
    input_dat.write_text(
        """
 &CONTROL
 METHOD = 'EX,P ! not comment', NON_LINEAR = .true. ! real comment
 DTIM = 1.0d-2
 /
 &SPCGENERAL
 finit = "cos,! still string"
 /
""",
        encoding="utf-8",
    )

    parsed = parse_input_dat(str(input_dat))

    assert parsed["control"]["method"] == "EX,P ! not comment"
    assert parsed["control"]["non_linear"] is True
    assert parsed["control"]["dtim"] == 1.0e-2
    assert parsed["spcgeneral"]["finit"] == "cos,! still string"


def test_parse_input_dat_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    assert parse_input_dat(str(tmp_path / "missing.input.dat")) == {}


def _minimal_species_input(adiabatic_electrons: bool) -> str:
    ae = ".true." if adiabatic_electrons else ".false."
    return f"""
 &CONTROL
 DTIM = 0.02
 NAVERAGE = 7
 /
 &GRIDSIZE
 number_of_species = 2
 adiabatic_electrons = {ae}
 /
 &SPCGENERAL
 adiabatic_electrons = {ae}
 beta = 0.0
 /
 &SPECIES
 MASS = 1.0
 Z = 1.0
 TEMP = 2.0
 DENS = 3.0
 RLT = 4.0
 RLN = 5.0
 /
 &SPECIES
 MASS = 0.25
 Z = -1.0
 TEMP = 0.5
 DENS = 0.75
 RLT = 6.0
 RLN = 7.0
 /
"""


def test_gkparams_from_input_and_geometry_filters_adiabatic_electrons_but_keeps_collision_backgrounds(
    tmp_path: Path,
) -> None:
    """Adiabatic path evolves ion species only, but collision backgrounds keep all species."""
    input_dat = tmp_path / "input.dat"
    input_dat.write_text(_minimal_species_input(adiabatic_electrons=True), encoding="utf-8")
    geometry = compute_geometry(q=1.4, shat=0.8, eps=0.2, ns=8, nkx=3, nky=2, nvpar=6, nmu=4)

    params = gkparams_from_input_and_geometry(str(input_dat), geometry)

    assert params.adiabatic_electrons is True
    assert float(np.asarray(params.mas)) == 1.0
    assert float(np.asarray(params.signz)) == 1.0
    assert float(np.asarray(params.tmp)) == 2.0
    np.testing.assert_allclose(np.asarray(params.coll_bg_mas), [1.0, 0.25])
    np.testing.assert_allclose(np.asarray(params.coll_bg_signz), [1.0, -1.0])


def test_gkparams_from_input_and_geometry_keeps_all_species_for_kinetic_electrons(
    tmp_path: Path,
) -> None:
    """Kinetic-electron path preserves species order from repeated GKW sections."""
    input_dat = tmp_path / "input.dat"
    input_dat.write_text(_minimal_species_input(adiabatic_electrons=False), encoding="utf-8")
    geometry = compute_geometry(q=1.4, shat=0.8, eps=0.2, ns=8, nkx=3, nky=2, nvpar=6, nmu=4)

    params = gkparams_from_input_and_geometry(str(input_dat), geometry)

    assert params.adiabatic_electrons is False
    np.testing.assert_allclose(np.asarray(params.mas), [1.0, 0.25])
    np.testing.assert_allclose(np.asarray(params.signz), [1.0, -1.0])
    np.testing.assert_allclose(np.asarray(params.rlt), [4.0, 6.0])
    np.testing.assert_allclose(np.asarray(params.rln), [5.0, 7.0])
