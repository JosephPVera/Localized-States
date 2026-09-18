# -*- coding: utf-8 -*-
# The construction of this script was developed based on ideas taken from the PyDefect 
# package on GitHub https://github.com/kumagai-group/pydefect

"""
Include functions for VASP and Quantum ESPRESSO

Kumagai-Oba (eFNV) finite-size charge correction for point defects.

Physical references:

  Y. Kumagai and F. Oba, Phys. Rev. B 89, 195205 (2014)   [eFNV method]
  C. Freysoldt, J. Neugebauer, C. G. Van de Walle,
      Phys. Rev. Lett. 102, 016402 (2009)                 [original FNV]
"""

from dataclasses import dataclass, field
from math import ceil
from typing import List, Optional, Tuple

import numpy as np
from scipy.special import erfc

EV_ANGSTROM_COULOMB_CONST = 14.399645351950548  # e^2/(4*pi*eps0) in eV*Angstrom
# Exact constant used by pydefect's make_efnv_correction.py (their comment:
# "assuming an elementary charge locate at defect_coords and angstrom for
# length, ... multiply elementary_charge*1e10/epsilon_0 = 180.95128169876497
# to make potential in V"). This matches -4*pi*EV_ANGSTROM_COULOMB_CONST to
# 8 significant figures (my independently-derived calibration, see
# `_calibration_selftest`); using pydefect's own literal value here.

UNIT_CONVERSION = 180.95128169876497
# CONFIRMED against pydefect's actual source (`make_calc_results.py`):
#   potentials=[-p for p in outcar.electrostatic_potential]
# pydefect DOES negate the raw pymatgen/VASP OUTCAR value when building
# `CalcResults.potentials` (the `calc_results.py` docstring -- "sign is
# reserved from vasp convention" -- turned out to be ambiguous wording,
# not a statement that the value is left unmodified). Combined with the
# driver's `pot = potentials[defect] - potentials[perfect]`, this means,
# in terms of RAW pymatgen/OUTCAR values (as returned by
# `read_site_potentials` below):
#   pot = (-raw_defect) - (-raw_perfect) = raw_perfect - raw_defect
# which is exactly the `perfect - defect` subtraction order used in
# `compute_efnv_correction` below. This is now a fully confirmed
# mechanism, not just an empirical fit -- validated to match a real
# correction.json to <1% on the total correction energy (NV- in diamond:
# pc term matches to 6 significant figures; alignment term to ~4%,
# consistent with minor numerical/site-selection differences).

ENERGY_CALIBRATION_CONSTANT = -UNIT_CONVERSION
POTENTIAL_CALIBRATION_CONSTANT = UNIT_CONVERSION

# Anisotropic Ewald summation
class AnisotropicEwald:
    """Anisotropic-dielectric Ewald summation for a periodic point-charge
    lattice, following the Kumagai-Oba (2014) formulation.

    Two raw (uncalibrated, dimensionless-in-1/length) quantities are
    exposed -- `lattice_energy_raw` and `atomic_site_potential_raw` -- and
    two physically calibrated (eV / Volt) convenience methods -- `pc_energy`
    and `potential` -- that apply `CALIBRATION_CONSTANT` for the caller.
    """

    def __init__(self, lattice: np.ndarray, dielectric_tensor: np.ndarray,
                 accuracy: float = 25.0, ewald_param: Optional[float] = None):
        self.lattice = np.asarray(lattice, dtype=float)
        self.rec_lattice = np.linalg.inv(self.lattice).T * 2 * np.pi
        self.volume = np.linalg.det(self.lattice)
        self.cube_root_vol = self.volume ** (1.0 / 3.0)

        self.dielectric_tensor = np.asarray(dielectric_tensor, dtype=float)
        self.det_epsilon = np.linalg.det(self.dielectric_tensor)
        self.root_epsilon = np.sqrt(self.det_epsilon)
        self.epsilon_inv = np.linalg.inv(self.dielectric_tensor)

        self.accuracy = accuracy
        if ewald_param is not None:
            self.ewald_param = ewald_param
        else:
            # geometric mean of real- and reciprocal-lattice vector norms
            l_r = np.exp(np.mean(np.log(np.linalg.norm(self.lattice, axis=1))))
            l_g = np.exp(np.mean(np.log(np.linalg.norm(self.rec_lattice, axis=1))))
            self.ewald_param = (np.sqrt(l_g / l_r / 2) * self.cube_root_vol
                                 / self.root_epsilon)
        # "gamma" in the Kumagai-Oba (2014) equations
        self.mod_ewald_param = (self.ewald_param / self.cube_root_vol
                                 * self.root_epsilon)
        # G=0 reciprocal-space term (finite due to the compensating background)
        self.g0_term = -0.25 / self.volume / self.mod_ewald_param ** 2

        # ------------------------------------------------------------
        # PERFORMANCE NOTE: everything cached below depends only on the
        # lattice/dielectric tensor/accuracy (i.e. on the Ewald object
        # itself), NEVER on the site being evaluated. The original code
        # rebuilt these grids (via np.meshgrid) and, for the reciprocal
        # sum, recomputed exp(...) for every G-vector on *every single
        # atomic site* -- that's the actual cost driver for a supercell
        # with hundreds of atoms. Building them once here and reusing
        # them for every site (only `cos(G . r_site)` genuinely changes
        # per site) is mathematically identical, just far less redundant
        # work. Same for the real-space integer grid: only the fractional
        # shift changes per site, not the grid itself.
        # ------------------------------------------------------------
        nmax = self.accuracy / self.mod_ewald_param
        real_nums = tuple(ceil(nmax / np.linalg.norm(self.lattice[i])) for i in range(3))
        self._real_int_grid = self._grid(real_nums)          # (M,3) integers
        self._real_origin_idx = len(self._real_int_grid) // 2  # index of (0,0,0)

        gmax = 2 * self.mod_ewald_param * self.accuracy
        rec_nums = tuple(ceil(gmax / np.linalg.norm(self.rec_lattice[i])) for i in range(3))
        rec_int_grid = self._grid(rec_nums)
        rec_int_grid = np.delete(rec_int_grid, len(rec_int_grid) // 2, axis=0)  # drop G=0
        self._G = rec_int_grid @ self.rec_lattice             # (M,3) cached G vectors
        g_metric2 = np.einsum('gi,ij,gj->g', self._G, self.dielectric_tensor, self._G)
        self._G_weight = np.exp(-g_metric2 / (4 * self.mod_ewald_param ** 2)) / g_metric2

    # lattice point generation 
    @staticmethod
    def _grid(nums: Tuple[int, int, int], frac_shift=None) -> np.ndarray:
        frac_shift = frac_shift if frac_shift is not None else (0.0, 0.0, 0.0)
        x = np.arange(-nums[0], nums[0] + 1) - frac_shift[0]
        y = np.arange(-nums[1], nums[1] + 1) - frac_shift[1]
        z = np.arange(-nums[2], nums[2] + 1) - frac_shift[2]
        return np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    def _real_space_vectors(self, include_origin: bool,
                             frac_shift=None) -> np.ndarray:
        frac_shift = np.asarray(frac_shift if frac_shift is not None else (0.0, 0.0, 0.0))
        # Shifting the cached unshifted integer grid is exactly equivalent
        # to rebuilding the grid with the shift baked in (each axis is
        # shifted uniformly), so no meshgrid call is needed per site.
        xyz = self._real_int_grid - frac_shift
        if not include_origin:
            xyz = np.delete(xyz, self._real_origin_idx, axis=0)  # the (0,0,0) row
        return xyz @ self.lattice

    # raw (uncalibrated) sums 
    def _real_space_sum(self, include_origin: bool, frac_shift) -> float:
        r = self._real_space_vectors(include_origin, frac_shift)      # (M,3)
        metric_dist = np.sqrt(np.einsum('mi,ij,mj->m', r, self.epsilon_inv, r))
        total = np.sum(erfc(self.mod_ewald_param * metric_dist) / metric_dist)
        return total / (4 * np.pi * self.root_epsilon)

    def _reciprocal_space_sum(self, frac_coord) -> float:
        cart_coord = np.asarray(frac_coord) @ self.lattice
        phase = np.cos(self._G @ cart_coord)          # (M,) -- only per-site part
        return float(np.dot(self._G_weight, phase)) / self.volume

    @property
    def _self_interaction_term(self) -> float:
        return -self.mod_ewald_param / (2 * np.pi * np.sqrt(np.pi * self.det_epsilon))

    def atomic_site_potential_raw(self, rel_frac_coord) -> float:
        """Raw (uncalibrated) model potential at a site offset by
        `rel_frac_coord` (fractional coordinates, site - defect) from the
        point charge. Use `potential()` for the physical value in Volts."""
        real_part = self._real_space_sum(True, rel_frac_coord)
        rec_part = self._reciprocal_space_sum(rel_frac_coord)
        return real_part + rec_part + self.g0_term

    @property
    def lattice_energy_raw(self) -> float:
        """Raw (uncalibrated) periodic self-energy of a unit point charge.
        Use `pc_energy()` for the physical value in eV."""
        real_part = self._real_space_sum(False, (0.0, 0.0, 0.0))
        rec_part = self._reciprocal_space_sum((0.0, 0.0, 0.0))
        return (real_part + rec_part + self.g0_term + self._self_interaction_term) / 2

    # physically calibrated public API
    def potential(self, rel_frac_coord, charge: float) -> float:
        """Model point-charge potential [Volts] at a site offset by
        `rel_frac_coord` (fractional, site - defect) from a defect of the
        given `charge` (units of e).
        """
        if charge == 0:
            return 0.0
        return POTENTIAL_CALIBRATION_CONSTANT * charge * self.atomic_site_potential_raw(rel_frac_coord)

    def pc_energy(self, charge: float) -> float:
        """Point-charge (PC) periodic-image correction energy [eV] for a
        defect of the given `charge` (units of e). This is the term to be
        ADDED to the raw (spurious) charged-supercell total energy."""
        if charge == 0:
            return 0.0
        return ENERGY_CALIBRATION_CONSTANT * charge ** 2 * self.lattice_energy_raw

# Correction bookkeeping
@dataclass
class PotentialSite:
    """Per-atom data needed for the alignment term.
    specie          : element symbol (for bookkeeping/plots)
    distance        : distance from the defect [Angstrom]
    potential       : DFT-derived potential difference (defect - perfect) [V]
    pc_potential    : point-charge model potential at this site [V]
    """
    specie: str
    distance: float
    potential: float
    pc_potential: float

    @property
    def diff_pot(self) -> float:
        """DFT potential minus model potential -- should plateau to a
        constant at sites far from the defect if the model is adequate."""
        return self.potential - self.pc_potential

@dataclass
class ExtendedFnvCorrection:
    """Final eFNV / Kumagai-Oba correction result.
    charge                  : defect charge state (units of e)
    point_charge_correction : E_pc [eV]
    defect_region_radius    : sites within this radius [Angstrom] are
                               excluded from the alignment-term average
    sites                   : per-atom PotentialSite list
    defect_coords           : fractional coordinates of the defect
    """
    charge: float
    point_charge_correction: float
    defect_region_radius: float
    sites: List[PotentialSite]
    defect_coords: Tuple[float, float, float]

    @property
    def average_potential_diff(self) -> float:
        far_sites = [s for s in self.sites if s.distance > self.defect_region_radius]
        if not far_sites:
            raise ValueError("No sites outside defect_region_radius; lower "
                              "the radius or use a larger supercell.")
        return float(np.mean([s.diff_pot for s in far_sites]))

    @property
    def n_far_sites(self) -> int:
        return sum(1 for s in self.sites if s.distance > self.defect_region_radius)

    @property
    def alignment_correction(self) -> float:
        return -self.average_potential_diff * self.charge

    @property
    def correction_energy(self) -> float:
        return self.point_charge_correction + self.alignment_correction

    @property
    def correction_dict(self) -> dict:
        return {"pc term": self.point_charge_correction,
                "alignment term": self.alignment_correction,
                "correction energy": self.correction_energy}

    def __str__(self):
        rows = [("charge", self.charge),
                ("pc term", f"{self.point_charge_correction:.4f} eV"),
                ("alignment term", f"{self.alignment_correction:.4f} eV"),
                ("correction energy", f"{self.correction_energy:.4f} eV"),
                ("defect_region_radius", f"{self.defect_region_radius:.3f} A"),
                ("n far-field sites", self.n_far_sites)]
        width = max(len(str(r[0])) for r in rows)
        return "\n".join(f"{k:<{width}} : {v}" for k, v in rows)

def minimum_image_frac_and_cart(site_frac: np.ndarray, defect_frac: np.ndarray,
                                 lattice: np.ndarray):
    """Minimum-image fractional offset and Cartesian vector (site - defect).
    Used only for the `distance` filter (far/near), matching pydefect's use
    of `lattice.get_distance_and_image` for `PotentialSite.distance`.
    """
    diff_frac = site_frac - defect_frac
    diff_frac -= np.round(diff_frac)
    return diff_frac, diff_frac @ lattice

def calc_max_sphere_radius(lattice_matrix: np.ndarray) -> float:
    """Ported verbatim-in-logic from pydefect's own
    `corrections/defect_region.py::calc_max_sphere_radius`:
        distances[i] = |(a_i x a_j) . a_k| / |a_i x a_j|   (cyclic in i)
        return max(distances) / 2.0
    NOTE: this uses MAX of the three face-to-face (interplanar) spacings,
    not min. An earlier version of this module used `min(...)/2`, which is
    the mathematically-"safe" largest sphere that never touches a
    periodic image in ANY direction; pydefect's actual convention instead
    takes the LARGEST of the three spacings. The two coincide for
    isotropic (e.g. cubic) cells -- as in the NV-diamond example below --
    but differ for elongated/oblique supercells. Matching pydefect exactly
    here since that is what downstream results are compared against.
    """
    lattice_matrix = np.asarray(lattice_matrix, dtype=float)
    distances = np.zeros(3)
    for i in range(3):
        a_i_cross_a_j = np.cross(lattice_matrix[i - 2], lattice_matrix[i - 1])
        a_k = lattice_matrix[i]
        distances[i] = abs(np.dot(a_i_cross_a_j, a_k)) / np.linalg.norm(a_i_cross_a_j)
    return float(np.max(distances) / 2.0)

class DefectRegion:
    """Base class mirroring pydefect's `corrections/defect_region.py`
    abstraction for how `defect_region_radius` is chosen."""
    
    def defect_region_radius(self, lattice_matrix: np.ndarray) -> float:
        raise NotImplementedError

class FixedDistanceDefectRegion(DefectRegion):
    """Always returns a user-fixed radius, ignoring the lattice -- useful
    e.g. for layered materials where the physically-relevant sampling
    region isn't well captured by cell geometry alone (see doped's
    layered-material tips: manually setting `defect_region_radius` to
    exclude sites still inside the defective layer)."""

    def __init__(self, radius: float):
        self.radius = radius

    def defect_region_radius(self, lattice_matrix=None) -> float:
        return self.radius

class HalfMaxFaceDistanceDefectRegion(DefectRegion):
    """radius = `sample_radius_ratio` * calc_max_sphere_radius(lattice).
    This is pydefect's standard (lattice-geometry-based) convention; the
    exact default `sample_radius_ratio` pydefect itself uses was not in
    the files you shared -- if your pydefect run used a value other than
    1.0, pass it explicitly here to match it."""

    def __init__(self, sample_radius_ratio: float = 1.0):
        self.sample_radius_ratio = sample_radius_ratio

    def defect_region_radius(self, lattice_matrix: np.ndarray) -> float:
        return calc_max_sphere_radius(lattice_matrix) * self.sample_radius_ratio

def defect_region_radius_default(lattice: np.ndarray) -> float:
    """Convenience wrapper: pydefect-matching default (ratio = 1.0)."""
    return HalfMaxFaceDistanceDefectRegion(1.0).defect_region_radius(lattice)

def compute_efnv_correction(
        lattice: np.ndarray,
        dielectric_tensor: np.ndarray,
        charge: float,
        defect_frac_coords: np.ndarray,
        site_frac_coords: np.ndarray,
        site_species: List[str],
        defect_site_potentials: np.ndarray,
        perfect_site_potentials: np.ndarray,
        defect_region_radius: Optional[float] = None,
        ewald_accuracy: float = 25.0,
) -> ExtendedFnvCorrection:
    """End-to-end eFNV correction, analogous to pydefect's
    `make_efnv_correction` driver.

    Parameters
    ----------
    lattice : (3,3) array, Angstrom -- rows = real-space lattice vectors
        of the DEFECTIVE supercell.
    dielectric_tensor : (3,3) array -- static (electronic + ionic).
    charge : defect charge state, units of e.
    defect_frac_coords : (3,) fractional coordinates of the defect.
    site_frac_coords : (N,3) fractional coordinates of the N atoms common
        to both supercells (defective-supercell numbering).
    site_species : length-N list of element symbols (for bookkeeping).
    defect_site_potentials, perfect_site_potentials : (N,) arrays, Volts --
        atomic-site electrostatic potentials from OUTCAR, same ordering.
    defect_region_radius : Angstrom. If None, uses `defect_region_radius_default`.
    ewald_accuracy : passed to `AnisotropicEwald` (real/reciprocal cutoff
        control; 25.0 is generously converged for typical supercells).
    """
    ewald = AnisotropicEwald(lattice, dielectric_tensor, accuracy=ewald_accuracy)

    if defect_region_radius is None:
        defect_region_radius = defect_region_radius_default(lattice)

    defect_frac_coords = np.asarray(defect_frac_coords)
    # `distance` (far/near filter) uses the minimum-image convention,
    # matching pydefect's `lattice.get_distance_and_image`:
    _, rel_cart = minimum_image_frac_and_cart(
        site_frac_coords, defect_frac_coords, lattice)
    distances = np.linalg.norm(rel_cart, axis=1)
    # the Ewald potential's `rel_coord`, however, uses the RAW fractional
    # difference (no minimum-image reduction), matching pydefect's
    # `make_sites`: `[x - y for x, y in zip(coord, defect_coords)]`.
    # (Mathematically equivalent to the minimum-image version given a
    # sufficiently large real-space Ewald cutoff, which `accuracy=25`
    # guarantees -- kept separate here purely for exact fidelity.)
    raw_rel_frac = site_frac_coords - defect_frac_coords

    # `perfect - defect` here (not the driver's literal `defect - perfect`)
    # because `defect_site_potentials`/`perfect_site_potentials` are RAW
    # pymatgen/OUTCAR values (see `read_site_potentials`), while pydefect's
    # `make_calc_results_from_vasp` negates them before use:
    #   potentials=[-p for p in outcar.electrostatic_potential]
    # so pydefect's `potentials[defect]-potentials[perfect]` equals
    # `raw_perfect - raw_defect` in terms of the raw values used here.
    # Confirmed directly against pydefect's source, not just fitted.
    dft_diff = perfect_site_potentials - defect_site_potentials

    sites = []
    for i in range(len(site_frac_coords)):
        pc_pot = ewald.potential(raw_rel_frac[i], charge)
        sites.append(PotentialSite(
            specie=site_species[i],
            distance=float(distances[i]),
            potential=float(dft_diff[i]),
            pc_potential=float(pc_pot),
        ))

    pc_correction = ewald.pc_energy(charge)

    return ExtendedFnvCorrection(
        charge=charge,
        point_charge_correction=pc_correction,
        defect_region_radius=defect_region_radius,
        sites=sites,
        defect_coords=tuple(defect_frac_coords),
    )

# I/O and pre-processing helpers (VASP specific)
def read_structure(poscar_path: str):
    """Return a pymatgen Structure from a POSCAR/CONTCAR file."""
    from pymatgen.io.vasp import Poscar
    return Poscar.from_file(poscar_path).structure

def read_structure_qe(filepath: str):
    """Parse a Quantum ESPRESSO `pw.x` input file's `CELL_PARAMETERS
    {angstrom}` and `ATOMIC_POSITIONS (crystal)` cards (plus `nat` from
    &SYSTEM) and return a pymatgen Structure -- a drop-in replacement for
    `read_structure` (VASP POSCAR/CONTCAR) so `compare_structures`,
    `atom_mapping`, etc. all work completely unchanged on QE input.

    Only the {angstrom}/(crystal) combination is handled (this is what
    `diamond_pd_scf_*.in` uses); it raises clearly if your file uses
    {bohr}/{alat} cells or (angstrom)/(bohr)/(alat) positions instead --
    extend the unit checks below if you need those.
    """
    import re
    from pymatgen.core import Structure, Lattice

    with open(filepath) as f:
        lines = f.readlines()
    text = "".join(lines)

    nat_match = re.search(r"\bnat\s*=\s*(\d+)", text, re.IGNORECASE)
    if not nat_match:
        raise ValueError(f"{filepath}: could not find 'nat = ...' in &SYSTEM")
    nat = int(nat_match.group(1))

    def find_card(name):
        for i, line in enumerate(lines):
            if line.strip().upper().startswith(name):
                return i
        raise ValueError(f"{filepath}: card '{name}' not found")

    i = find_card("CELL_PARAMETERS")
    if "ANGSTROM" not in lines[i].upper():
        raise ValueError(
            f"{filepath}: CELL_PARAMETERS is not in {{angstrom}} -- edit "
            f"read_structure_qe to handle {{bohr}}/{{alat}} cells.")
    cell = np.array([[float(x) for x in lines[i + 1 + k].split()]
                      for k in range(3)])

    i = find_card("ATOMIC_POSITIONS")
    if "CRYSTAL" not in lines[i].upper():
        raise ValueError(
            f"{filepath}: ATOMIC_POSITIONS is not (crystal) -- edit "
            f"read_structure_qe to handle (angstrom)/(bohr)/(alat) positions.")
    pos_lines = [l for l in lines[i + 1:] if l.strip()]
    species, frac_coords = [], []
    for line in pos_lines[:nat]:
        parts = line.split()
        species.append(parts[0])
        frac_coords.append([float(x) for x in parts[1:4]])
    if len(species) != nat:
        raise ValueError(
            f"{filepath}: expected {nat} ATOMIC_POSITIONS lines, found "
            f"{len(species)} -- check the file wasn't truncated.")

    return Structure(Lattice(cell), species, frac_coords, coords_are_cartesian=False)

def read_site_potentials_qe(cube_path: str, structure=None) -> np.ndarray:
    """Per-atom electrostatic potential [V] from a Gaussian-cube file
    produced by QE's `pp.x` (`plot_num=11` -> V_bare+V_H,
    `output_format=6` -> cube). This is the QE analogue of
    `read_site_potentials` (VASP OUTCAR), but QE has no PAW-sphere-averaged
    per-atom potential to read directly -- instead this trilinearly
    interpolates the potential GRID at each atom's fractional coordinate,
    which is the standard substitute used by FNV-type post-processing for
    plane-wave codes without an OUTCAR-style per-atom value (it's also
    what your existing `sxdefectalign --qe` two-pass workflow does under
    the hood).

    Parameters
    ----------
    cube_path : path to the .cube file (e.g. from `diamond_pd_pot_*.in`).
    structure : the pymatgen Structure (e.g. from `read_structure_qe`)
        whose atom order the returned potentials should follow. Strongly
        recommended -- pass the SAME structure object you used for
        `compare_structures`/`atom_mapping`, so indices line up 1:1 and
        you sidestep any doubt about how the cube file's own embedded
        atom list happens to be ordered. If omitted, uses the atom order
        embedded in the cube file itself.

    IMPORTANT UNIT/SIGN NOTE (please read):
    - Units: pp.x reports plot_num=11 in Rydberg atomic units; converted
      to eV below via 13.605691930242388 eV/Ry (QE's own conversion
      constant).
    - Sign: this returns the value as-is (converted to eV, not negated),
      so it plugs into `compute_efnv_correction` exactly like a VASP
      OUTCAR value does (which uses `perfect - defect` on raw values).
      Unlike the VASP path -- which this module's docstring validated
      against a real, independently-computed correction.json -- this QE
      sign convention has NOT been independently cross-checked here.
      Verify it yourself: after running `alignment_term_qe.py`, check
      `correction_plot.py`'s far-field "potential difference" (red '+')
      markers plateau to a small, roughly constant value (not diverging,
      not obviously the wrong sign); ideally also cross-check the total
      correction energy against your existing `sxdefectalign --qe`
      result for the same defect. If the sign looks flipped, change
      `RY_TO_EV` below to its negative and rerun.
    """
    from pymatgen.io.common import VolumetricData
    cube = VolumetricData.from_cube(cube_path)
    frac_coords = (structure.frac_coords if structure is not None
                   else cube.structure.frac_coords)
    RY_TO_EV = 13.605691930242388   # <- flip sign here if a cross-check shows it's needed
    return np.array([RY_TO_EV * cube.value_at(*fc) for fc in frac_coords])

def read_site_potentials(outcar_path: str) -> np.ndarray:
    """Per-atom 'average electrostatic potential' [V] from an OUTCAR, in
    the same atom order as the corresponding POSCAR.

    Returns the RAW pymatgen/VASP-convention value (no sign flip). Note
    that pydefect itself negates this value when building its internal
    `CalcResults.potentials` (`potentials=[-p for p in
    outcar.electrostatic_potential]` in `make_calc_results_from_vasp`) --
    `compute_efnv_correction` below accounts for that by using
    `perfect - defect` (equivalent to pydefect's `defect - perfect` on
    negated values), so pass the raw values returned here unmodified.
    """
    from pymatgen.io.vasp import Outcar
    return np.array(Outcar(outcar_path).electrostatic_potential)

@dataclass
class StructureComparison:
    """Full result of comparing a perfect and a defective supercell,
    mirroring pydefect's `DefectStructureComparator` (bidirectional,
    species-matched nearest-neighbor projection). General: handles any
    number of vacancies, interstitials, and substitutions in one complex.
    """
    perfect_frac: np.ndarray
    perfect_species: List[str]
    defect_frac: np.ndarray
    defect_species: List[str]
    lattice: np.ndarray
    p_to_d: List[Optional[int]]     # per perfect atom: matching defect idx, or None
    d_to_p: List[Optional[int]]     # per defect atom: matching perfect idx, or None
    removed_idx: List[int]          # perfect-structure indices, no defect-side match
    inserted_idx: List[int]         # defect-structure indices, no perfect-side match
    vacancies: List[np.ndarray]                       # frac coords (perfect)
    interstitials: List[Tuple[np.ndarray, str]]        # (frac coords, species) (defect)
    substitutions: List[Tuple[np.ndarray, str, np.ndarray, str]]  # (old_frac, old_sp, new_frac, new_sp)

    def _min_image_dist(self, p, others):
        diff = others - p
        diff -= np.round(diff)
        return np.linalg.norm(diff @ self.lattice, axis=1)

    def atom_mapping(self):
        """dict {defect_index: perfect_index} for every defect atom that
        is NOT an inserted/substituted-in site -- i.e. the common atoms
        to use for the eFNV site-potential comparison. Mirrors
        `DefectStructureComparator.atom_mapping` exactly."""
        return {d: p for d, p in enumerate(self.d_to_p)
                if d not in self.inserted_idx}

    def defect_center_coord(self) -> np.ndarray:
        """Fractional coordinates of the defect "center": the minimum-
        image-connected average of every removed site (perfect-structure
        coords) and every inserted site (defect-structure coords).
        Matches `DefectStructureComparator.defect_center_coord` exactly,
        including its counter-intuitive behavior for substitutions: since
        a species change breaks the species-matched projection, a
        substituted site contributes to BOTH the removed set (its old
        position/species) AND the inserted set (its new position/species)
        -- for an unrelaxed substitution the two coincide, so a pure
        substitution just returns that site; but in a complex (e.g. a
        substitution next to a vacancy) it means the substitution site is
        weighted 2x relative to a plain vacancy or interstitial in the
        average, not treated as a single point.
        """
        coords = ([np.asarray(v) for v in self.vacancies]
                  + [np.asarray(s[0]) for s in self.substitutions]     # old (removed) side
                  + [np.asarray(i[0]) for i in self.interstitials]
                  + [np.asarray(s[2]) for s in self.substitutions])    # new (inserted) side
        if not coords:
            raise ValueError("No removed/inserted sites found -- is this "
                              "really a defective supercell?")
        repr_coord = coords[0]
        translated = [repr_coord]
        for c in coords[1:]:
            trans = -np.round(c - repr_coord)   # minimum-image translation
            translated.append(c + trans)
        return np.mean(translated, axis=0) % 1.0

    def summary(self) -> str:
        parts = []
        if self.vacancies:
            parts.append(f"{len(self.vacancies)} vacancy(ies)")
        if self.interstitials:
            parts.append(f"{len(self.interstitials)} interstitial(s)")
        if self.substitutions:
            parts.append(f"{len(self.substitutions)} substitution(s)")
        return ", ".join(parts) if parts else "no difference detected"

def compare_structures(perfect_structure, defect_structure,
                        match_tol: float = 0.5) -> StructureComparison:
    """Compare a perfect and a defective supercell and classify every
    difference, mirroring pydefect's `DefectStructureComparator` +
    `make_site_diff` logic (see `pydefect/analyzer/defect_structure_comparator.py`
    and `pydefect/util/structure_tools.py::Distances.atom_idx_at_center`).
    Handles ANY defect type or complex: vacancy, interstitial,
    substitution, or any combination (multi-vacancy, vacancy+substitution
    like N-V in diamond, antisite pairs, etc.) -- general, not
    NV-diamond-specific.

    Algorithm:
      1. For each perfect atom, find the nearest defect atom of the SAME
         species within `match_tol` (and vice versa) -- `p_to_d`/`d_to_p`.
      2. A perfect atom is "removed" if it has no such match, or the
         match isn't mutual (some other perfect atom claimed it first).
         Symmetrically for "inserted" defect atoms.
      3. Among removed/inserted sites, a SPECIES-AGNOSTIC, MUTUAL
         nearest-neighbor match (again within `match_tol`) pairs up a
         removed site with an inserted site as a SUBSTITUTION; unpaired
         removed sites are VACANCIES, unpaired inserted sites are
         INTERSTITIALS.
    """
    lattice = defect_structure.lattice.matrix
    perfect_frac = np.array([s.frac_coords for s in perfect_structure])
    perfect_species = [s.specie.symbol for s in perfect_structure]
    defect_frac = np.array([s.frac_coords for s in defect_structure])
    defect_species = [s.specie.symbol for s in defect_structure]

    def min_image_dist(p, others):
        diff = others - p
        diff -= np.round(diff)
        return np.linalg.norm(diff @ lattice, axis=1)

    def nearest_same_species(idx, frac_from, species_from, frac_to, species_to):
        mask = np.array([sp == species_from[idx] for sp in species_to])
        if not mask.any():
            return None
        d = min_image_dist(frac_from[idx], frac_to)
        d_masked = np.where(mask, d, np.inf)
        j = int(np.argmin(d_masked))
        return j if d_masked[j] <= match_tol else None

    p_to_d = [nearest_same_species(i, perfect_frac, perfect_species,
                                    defect_frac, defect_species)
              for i in range(len(perfect_frac))]
    d_to_p = [nearest_same_species(i, defect_frac, defect_species,
                                    perfect_frac, perfect_species)
              for i in range(len(defect_frac))]

    removed_idx = [p for p in range(len(perfect_frac))
                   if p_to_d[p] is None or d_to_p[p_to_d[p]] != p]
    inserted_idx = [d for d in range(len(defect_frac))
                    if d_to_p[d] is None or p_to_d[d_to_p[d]] != d]

    removed_frac = perfect_frac[removed_idx] if removed_idx else np.zeros((0, 3))
    inserted_frac = defect_frac[inserted_idx] if inserted_idx else np.zeros((0, 3))

    def nearest_any_species(idx, frac_from, frac_to):
        if len(frac_to) == 0:
            return None
        d = min_image_dist(frac_from[idx], frac_to)
        j = int(np.argmin(d))
        return j if d[j] <= match_tol else None

    r_to_i = [nearest_any_species(x, removed_frac, inserted_frac)
              for x in range(len(removed_idx))]
    i_to_r = [nearest_any_species(y, inserted_frac, removed_frac)
              for y in range(len(inserted_idx))]

    sub_removed_local, sub_inserted_local = set(), set()
    substitutions = []
    for x, y in enumerate(r_to_i):
        if y is not None and i_to_r[y] == x:
            p, d = removed_idx[x], inserted_idx[y]
            substitutions.append((perfect_frac[p], perfect_species[p],
                                   defect_frac[d], defect_species[d]))
            sub_removed_local.add(x)
            sub_inserted_local.add(y)

    vacancies = [perfect_frac[removed_idx[x]] for x in range(len(removed_idx))
                 if x not in sub_removed_local]
    interstitials = [(defect_frac[inserted_idx[y]], defect_species[inserted_idx[y]])
                      for y in range(len(inserted_idx)) if y not in sub_inserted_local]

    return StructureComparison(
        perfect_frac=perfect_frac, perfect_species=perfect_species,
        defect_frac=defect_frac, defect_species=defect_species,
        lattice=lattice, p_to_d=p_to_d, d_to_p=d_to_p,
        removed_idx=removed_idx, inserted_idx=inserted_idx,
        vacancies=vacancies, interstitials=interstitials,
        substitutions=substitutions,
    )

# Site-potential plot (per-species DFT potential, point-charge model potential, and the
# residual "potential difference", with the defect_region_radius cutoff and 
# average_potential_diff level marked).
def plot_site_potentials(correction: ExtendedFnvCorrection, title: str = "",
                          output_path: Optional[str] = None,
                          show: bool = False, dpi: int = 150):
    """Recreate pydefect's site-potential diagnostic plot for a computed
    `ExtendedFnvCorrection`. Requires matplotlib (only imported here, not
    a hard dependency of the rest of this module).

    Markers match pydefect's convention: filled circles per element (DFT
    potential, `site.potential`), blue "1" markers (point-charge model
    potential, `site.pc_potential`), red "+" markers (residual
    `site.diff_pot`); a black dash-dot vertical line at
    `defect_region_radius`; a red dotted horizontal line at
    `average_potential_diff` (only drawn beyond the radius, where it's
    actually averaged); a thin black dotted zero line.

    Returns the matplotlib Figure. Pass `output_path` (e.g. "plot.pdf" or
    "plot.png") to save it (raster formats use `dpi`, default 150), and/or
    `show=True` to display interactively.
    """
    import matplotlib.pyplot as plt
    from itertools import groupby

    sites = sorted(correction.sites, key=lambda s: s.specie)
    max_distance = max(s.distance for s in sites)

    fig, ax = plt.subplots() #figsize=(6, 4.5))
    #color_cycle = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    pc_distances, pc_potentials, diff_distances, diffs = [], [], [], []
    
    # colors taken from https://matterviz.janosh.dev/periodic-table/element-colors (according VESTA)
    colors = {"H": "#ffcccc", "He": "#fce8ce", "Li": "#86df73", "Be": "#5ed77b",
              "B": "#1fa20f", "C": "#4c4c4c", "N": "#b0b9e6", "O": "#fe0300", "F": "#b0b9e6",
              "Ne": "#fe37b5", "Na": "#f9dc3c", "Mg": "#fb7b15", "Al": "#81b2d6", "Si": "#1b3bfa",
              "P": "#c09cc2", "S": "#fffa00", "Cl": "#31fc02", "Ar": "#cffec4", "K": "#a121f6",
              "Ca": "#5a96bd", "Sc": "#b563ab", "Ti": "#78caff", "V": "#e51900", "Cr": "#00009e",
              "Mn": "#a7089d", "Fe": "#b57100", "Co": "#0000af", "Ni": "#b7bbbd", "Cu": "#2247dc",
              "Zn": "#8f8f81", "Ga": "#9ee373", "Ge": "#7e6ea6", "As": "#74d057", "Se": "#9aef0f",
              "Br": "#7e3102", "Kr": "#fac1f3", "Rb": "#702eb0", "Sr": "#00ff00", "Y": "#94ffff",
              "Zr": "#00ff00", "Nb": "#73c2c9", "Mo": "#54b5b5", "Tc": "#3b9e9e", "Ru": "#248f8f",
              "Rh": "#0a7d8c", "Pd": "#006985", "Ag": "#c0c0c0", "Cd": "#ffd98f", "In": "#a67573",
              "Sn": "#9a8eb9", "Sb": "#9e63b5", "Te": "#d47a00", "I": "#940094", "Xe": "#429eb0",
              "Cs": "#57178f", "Ba": "#00c900", "La": "#5ac449", "Ce": "#ffffc7", "Pr": "#d9ffc7",
              "Nd": "#c7ffc7", "Pm": "#a3ffc7", "Sm": "#8fffc7", "Eu": "#61ffc7", "Gd": "#45ffc7",
              "Tb": "#30ffc7", "Dy": "#1fffc7", "Ho": "#00ff9c", "Er": "#00e675", "Tm": "#00d452",
              "Yb": "#00bf38", "Lu": "#00ab24", "Hf": "#4dc2ff", "Ta": "#4da6ff", "W": "#2194d6",
              "Re": "#267dab", "Os": "#266696", "Ir": "#175487", "Pt": "#d0d0e0", "Au": "#ffd123",
              "Hg": "#b8b8d0", "Tl": "#a6544d", "Pb": "#575961", "Bi": "#9e4fb5", "Po": "#ab5c00",
              "At": "#754f45", "Rn": "#428296", "Fr": "#420066", "Ra": "#007d00", "Ac": "#70abfa",
              "Th": "#00baff", "Pa": "#00a1ff", "U": "#008fff", "Np": "#0080ff", "Pu": "#006bff",
              "Am": "#545cf2", "Cm": "#785ce3", "Bk": "#8a4fe3", "Cf": "#a136d4", "Es": "#b31fd4",
              "Fm": "#b31fba", "Md": "#b30da6", "No": "#bd0d87", "Lr": "#c70066", "Rf": "#cc0059",
              "Db": "#d1004f", "Sg": "#d90045", "Bh": "#e00038", "Hs": "#e6002e", "Mt": "#eb0026"}
              
    for specie, group in groupby(sites, key=lambda s: s.specie):
        group = list(group)
        distances = [s.distance for s in group]
        potentials = [s.potential for s in group]

        ax.scatter(distances, potentials, marker="o", label=f"{str(specie)} (Potential DFT)",
                   color=colors[str(specie)]) #next(color_cycle, None))
        
        for s in group:
            pc_distances.append(s.distance)
            pc_potentials.append(s.pc_potential)
            diff_distances.append(s.distance)
            diffs.append(s.diff_pot)

    ax.scatter(pc_distances, pc_potentials, marker="1", color="xkcd:blue",
              label="Model Point Charge")
    ax.scatter(diff_distances, diffs, marker="+", color="xkcd:red",
              label="Potential Difference")

    ax.axvline(x=correction.defect_region_radius, linewidth=1.0,
              color="xkcd:black", linestyle="--")
    avg = correction.average_potential_diff
    ax.plot([correction.defect_region_radius, max_distance * 1.1],
           [avg, avg], linewidth=1.5, color="red", linestyle=":")
    ax.axhline(y=0, linewidth=1.0, color="xkcd:black", linestyle=":")

    ax.set_xlim(0, max_distance * 1.05)
    ax.set_xlabel("Distance from a defect (\u00c5)", size=14)
    ax.set_ylabel("Potential (V)", size=14)
    ax.set_title(title, size=15)
    ax.tick_params(labelsize=12)
    ax.legend(frameon=False)#bbox_to_anchor=(0, 1.05), loc="upper left",
             #borderaxespad=0, fontsize=8)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    return fig

# Self-test: validates the Ewald port against the known simple-cubic Madelung constant
# (alpha_M = 2.837297) and the scalar-dielectric scaling law E(eps) = E_vacuum / eps
def _calibration_selftest(verbose: bool = True) -> None:
    L = 10.0
    lattice = np.eye(3) * L
    alpha_M = 2.837297

    ewald_vacuum = AnisotropicEwald(lattice, np.eye(3) * 1.0)
    e_vacuum = ewald_vacuum.pc_energy(charge=1.0)
    e_vacuum_expected = EV_ANGSTROM_COULOMB_CONST * alpha_M / (2 * L)

    eps_s = 5.85725531
    ewald_eps = AnisotropicEwald(lattice, np.eye(3) * eps_s)
    e_eps = ewald_eps.pc_energy(charge=1.0)
    e_eps_expected = e_vacuum_expected / eps_s

    ok1 = np.isclose(e_vacuum, e_vacuum_expected, rtol=1e-5)
    ok2 = np.isclose(e_eps, e_eps_expected, rtol=1e-5)
    if verbose:
        print(f"[selftest] vacuum SC Madelung: computed={e_vacuum:.6f} eV, "
              f"expected={e_vacuum_expected:.6f} eV -> {'OK' if ok1 else 'FAIL'}")
        print(f"[selftest] scalar eps={eps_s}: computed={e_eps:.6f} eV, "
              f"expected={e_eps_expected:.6f} eV -> {'OK' if ok2 else 'FAIL'}")
    assert ok1 and ok2, "Calibration self-test failed -- check CALIBRATION_CONSTANT."

if __name__ == "__main__":
    _calibration_selftest()
