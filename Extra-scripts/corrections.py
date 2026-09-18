#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2026-09

"""
Usage:
      python3 corrections.py [--qe]      

eFNV is a model point charge.

Computes the eFNV energy correction (pc term + alignment term) for a
defect calculation. Runs the VASP workflow by default; pass --qe to run
the Quantum ESPRESSO workflow instead.

This script computes:
    - pc term:        Depends of lattice parameters (supercell), dielectric
                      tensor and charge state.
                      VASP: CONTCAR and primitive/primitive.json.
                      QE:   .in and primitive/primitive.json.
    - alignment term: Depends of electrostatic potentials, defect position,
                      and defect region radius.
                      VASP: OUTCAR and CONTCAR's.
                      QE:   potential/.cube and .in's (the scf/.in file is
                            used because it contains the relaxed system).
"""

import argparse
import glob
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
from efnv_corrections import (
    read_structure, read_structure_qe, read_site_potentials,
    read_site_potentials_qe, compare_structures,
    HalfMaxFaceDistanceDefectRegion, FixedDistanceDefectRegion,
    compute_efnv_correction, AnisotropicEwald,
)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

parser = argparse.ArgumentParser()
parser.add_argument("--qe", action="store_true",
                     help="Use Quantum ESPRESSO instead of VASP (default).")
args = parser.parse_args()
QE = args.qe

# Direct correspondence between Hermann-Mauguin (crystallographic) and
# Schoenflies notation for the 32 crystallographic point groups.
HM_TO_SCHOENFLIES = {
    "1": "C_1", "-1": "C_i",
    "2": "C_2", "m": "C_s", "2/m": "C_2h",  # C_s = C_1h
    "222": "D_2", "mm2": "C_2v", "mmm": "D_2h",
    "4": "C_4", "-4": "S_4", "4/m": "C_4h",
    "422": "D_4", "4mm": "C_4v",
    "-42m": "D_2d", "-4m2": "D_2d",
    "4/mmm": "D_4h",
    "3": "C_3", "-3": "C_3i",
    "32": "D_3", "3m": "C_3v", "-3m": "D_3d",
    "6": "C_6", "-6": "C_3h", "6/m": "C_6h",
    "622": "D_6", "6mm": "C_6v",
    "-6m2": "D_3h", "-62m": "D_3h",
    "6/mmm": "D_6h",
    "23": "T", "m-3": "T_h",
    "432": "O", "-43m": "T_d", "m-3m": "O_h",
}

def get_point_group(structure, symprec=0.1):
    """Compute the crystallographic (Hermann-Mauguin) point group of a
    structure and derive its corresponding Schoenflies notation via the
    standard one-to-one correspondence between the two conventions for
    the 32 crystallographic point groups (same correspondence used by
    webqc.org/symmetry.php).

    - structure: pymatgen Structure (defect supercell).
    - symprec:   tolerance for SpacegroupAnalyzer.
    """
    crystallographic = None
    schoenflies = None
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        crystallographic = sga.get_point_group_symbol()
        schoenflies = HM_TO_SCHOENFLIES.get(crystallographic)
        if schoenflies is None:
            print(f"  WARNING: no Schoenflies correspondence found for "
                  f"Hermann-Mauguin symbol '{crystallographic}'.")
    except Exception as e:
        print(f"  WARNING: point group determination failed: {e}")

    return {"crystallographic": crystallographic, "schoenflies": schoenflies}

def load_primitive_json(primitive_json_path):
    """Load primitive.json, expected to contain (at least):
        {"Dielectric": {"Total": [[...],[...],[...]], ...}, ...}
    """
    if not primitive_json_path.exists():
        raise FileNotFoundError(
            f"{primitive_json_path} not found. This file must contain "
            f"the host's dielectric tensor.")
    with open(primitive_json_path) as f:
        return json.load(f)

def find_single_file(folder, extension):
    """Return the only file with the given extension inside folder.
    Only needed for the QE workflow (VASP file names are fixed).

    - Raises FileNotFoundError if no file with that extension is found.
    - If more than one is found, prints a warning and uses the first one.
    """
    pattern = os.path.join(folder, f"*.{extension}")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No .{extension} file found in '{folder}'.")

    if len(files) > 1:
        print(f"Multiple .{extension} files found in '{folder}': {files}")
        print(f"Using the first one: {files[0]}")

    return files[0]

# Valence electrons from OUTCAR (VASP only)
ZVAL_PATTERN = re.compile(r"ZVAL\s*=\s*([\d.]+)")

# Total number of electrons from OUTCAR (VASP only)
NELECT_PATTERN = re.compile(r"NELECT\s*=\s*([\d.]+)")

# tot_charge from a QE scf .in file (QE only)
TOT_CHARGE_RE = re.compile(r"tot_charge\s*=\s*([+-]?\d+\.?\d*)")

def get_species_counts(structure):
    """Atom counts per species, in the order they appear in the
    structure (i.e. the same order as the CONTCAR species line)."""
    counts = OrderedDict()
    for site in structure:
        symbol = site.specie.symbol
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts

def get_zval_per_species(outcar_path, species_order):
    """Read ZVAL (valence electrons) per species from the OUTCAR's
    POTCAR block. Takes the first len(species_order) ZVAL matches,
    which appear in the same order as the species in the CONTCAR. This
    function serves to identify the defect with neutral charge state
    (NELECT_reference)"""
    zvals = []
    with open(outcar_path) as f:
        for line in f:
            m = ZVAL_PATTERN.search(line)
            if m:
                zvals.append(float(m.group(1)))
            if len(zvals) >= len(species_order):
                break
    if len(zvals) < len(species_order):
        raise ValueError(
            f"Only found {len(zvals)} ZVAL entries in {outcar_path} but "
            f"there are {len(species_order)} species: {species_order}")
    return dict(zip(species_order, zvals))

def get_nelect(outcar_path):
    """Grep the actual NELECT used in the calculation from an OUTCAR."""
    with open(outcar_path) as f:
        for line in f:
            m = NELECT_PATTERN.search(line)
            if m:
                return float(m.group(1))
    raise ValueError(f"NELECT not found in {outcar_path}")

def detect_charge_vasp(structure, outcar_path):
    """charge = NELECT_reference (from CONTCAR composition x ZVAL)
    minus NELECT_defect (actual value used in the calculation)."""
    counts = get_species_counts(structure)
    zvals = get_zval_per_species(outcar_path, list(counts.keys()))
    nelect_reference = sum(counts[sp] * zvals[sp] for sp in counts)
    nelect_defect = get_nelect(outcar_path)
    charge_raw = nelect_reference - nelect_defect
    charge = round(charge_raw)
    if not np.isclose(charge_raw, charge, atol=1e-3):
        print(f"  WARNING: non-integer charge detected ({charge_raw:.6f}). "
              f"Check ZVAL/species counts for {outcar_path}")
    details = {
        "species_counts": dict(counts),
        "zval_per_species": zvals,
        "nelect_reference": nelect_reference,
        "nelect_defect": nelect_defect,
    }
    return charge, details

def get_tot_charge_qe(in_path):
    """Read 'tot_charge' directly from a QE scf .in file. Defaults to 0
    (neutral) if the keyword isn't present."""
    with open(in_path, "r", errors="ignore") as f:
        for line in f:
            m = TOT_CHARGE_RE.search(line)
            if m:
                return round(float(m.group(1)))
    return 0

if QE:
    PERFECT_STRUCT_FILE = find_single_file("../perfect/scf", "in")
    DEFECT_STRUCT_FILE = find_single_file("scf", "in")
    PERFECT_POT_FILE = find_single_file("../perfect/potential", "cube")
    DEFECT_POT_FILE = find_single_file("potential", "cube")
    print(f"Reading: {PERFECT_STRUCT_FILE}")
    print(f"Reading: ./{DEFECT_STRUCT_FILE}")
    print(f"Reading: {PERFECT_POT_FILE}")
    print(f"Reading: ./{DEFECT_POT_FILE}")
else:
    PERFECT_STRUCT_FILE = "../perfect/CONTCAR"
    DEFECT_STRUCT_FILE = "CONTCAR"
    PERFECT_POT_FILE = "../perfect/OUTCAR"
    DEFECT_POT_FILE = "OUTCAR"

# Static dielectric tensor
PRIMITIVE_JSON = Path("../../primitive/primitive.json")
_primitive_data = load_primitive_json(PRIMITIVE_JSON)
DIELECTRIC_TENSOR = np.array(_primitive_data["Dielectric"]["Total"])

# defect_region_radius: sites closer than this to the defect are excluded
# from the alignment-term average. AUTO_RADIUS=True refines it
# automatically by checking where the far-field potential is flat.
AUTO_RADIUS = True
RADIUS_ANGSTROM = False   # only used if AUTO_RADIUS = False

# Optional: override the automatically-detected defect position (None ->
# use compare_structures(...).defect_center_coord(), recommended).
DEFECT_FRAC_COORDS_OVERRIDE = None

OUTPUT_JSON = "correction.json"

# STEP 1: pc term (lattice parameters + dielectric tensor + charge state)
if QE:
    defect_struct = read_structure_qe(DEFECT_STRUCT_FILE)
    perfect_struct = read_structure_qe(PERFECT_STRUCT_FILE)
    CHARGE = get_tot_charge_qe(DEFECT_STRUCT_FILE)
    charge_details = None
else:
    defect_struct = read_structure(DEFECT_STRUCT_FILE)
    perfect_struct = read_structure(PERFECT_STRUCT_FILE)
    # Charge detection (self-contained -- only needs DEFECT_STRUCT_FILE +
    # DEFECT_POT_FILE):
    #   1. Count atoms per species in the defect CONTCAR.
    #   2. Read each species' valence (ZVAL) from the defect OUTCAR.
    #   3. NELECT_reference = sum(count_species * ZVAL_species)
    #   4. NELECT_defect = actual NELECT reported in the defect OUTCAR.
    #   5. charge = NELECT_reference - NELECT_defect
    CHARGE, charge_details = detect_charge_vasp(defect_struct, DEFECT_POT_FILE)

lattice = defect_struct.lattice.matrix

ewald = AnisotropicEwald(lattice, DIELECTRIC_TENSOR)
pc_term = ewald.pc_energy(CHARGE)

#print(f"pc term : {pc_term:.16f} eV")

# STEP 2: alignment term
if QE:
    perfect_pot = read_site_potentials_qe(PERFECT_POT_FILE, structure=perfect_struct)
    defect_pot = read_site_potentials_qe(DEFECT_POT_FILE, structure=defect_struct)
else:
    perfect_pot = read_site_potentials(PERFECT_POT_FILE)
    defect_pot = read_site_potentials(DEFECT_POT_FILE)

# sanity check: structure and potential must be the same calculation
if len(perfect_struct) != len(perfect_pot):
    raise ValueError(
        f"{PERFECT_STRUCT_FILE} has {len(perfect_struct)} atoms but "
        f"{PERFECT_POT_FILE} reports {len(perfect_pot)} site potentials. "
        f"These two files don't correspond to the same calculation")
if len(defect_struct) != len(defect_pot):
    raise ValueError(
        f"{DEFECT_STRUCT_FILE} has {len(defect_struct)} atoms but "
        f"{DEFECT_POT_FILE} reports {len(defect_pot)} site potentials. "
        f"These two files don't correspond to the same calculation.")

# Auto-identify the defect
comparison = compare_structures(perfect_struct, defect_struct)
#print(f"\nDetected: {comparison.summary()}")

if DEFECT_FRAC_COORDS_OVERRIDE is not None:
    defect_frac_coords = np.array(DEFECT_FRAC_COORDS_OVERRIDE)
else:
    defect_frac_coords = comparison.defect_center_coord()
#print(f"defect_center_coord: {defect_frac_coords}")

# Match atoms common to both supercells
mapping = comparison.atom_mapping()
defect_idx = np.array(sorted(mapping.keys()))
perfect_idx = np.array([mapping[d] for d in defect_idx])

site_frac_coords = comparison.defect_frac[defect_idx]
site_species = [comparison.defect_species[d] for d in defect_idx]
defect_site_potentials = defect_pot[defect_idx]
perfect_site_potentials = perfect_pot[perfect_idx]

# Defect_region_radius
correction = compute_efnv_correction(
    lattice=lattice, dielectric_tensor=DIELECTRIC_TENSOR, charge=CHARGE,
    defect_frac_coords=defect_frac_coords,
    site_frac_coords=site_frac_coords, site_species=site_species,
    defect_site_potentials=defect_site_potentials,
    perfect_site_potentials=perfect_site_potentials,
    defect_region_radius=HalfMaxFaceDistanceDefectRegion(sample_radius_ratio=1.0)
        .defect_region_radius(lattice),
)

if AUTO_RADIUS:
    radius = HalfMaxFaceDistanceDefectRegion(sample_radius_ratio=1.0) \
        .defect_region_radius(lattice)
    #print(f"defect_region_radius (pydefect default, max inscribed sphere): "
    #      f"{radius:.3f} A")
else:
    radius = FixedDistanceDefectRegion(RADIUS_ANGSTROM).defect_region_radius(lattice)
    #print(f"defect_region_radius (fixed): {radius:.3f} A")

# alignment term
correction.defect_region_radius = radius
alignment_term = correction.alignment_correction

# Sanity check: compute_efnv_correction recomputes its own pc term
# internally. it should match the value computed in step 1 above to numerical precision.
if not np.isclose(correction.point_charge_correction, pc_term, rtol=1e-10):
    print(f"\nWARNING: pc term recomputed in step 2 "
          f"({correction.point_charge_correction:.16f}) differs from the "
          f"pc term computed in step 1 above ({pc_term:.16f}). Check that "
          f"CHARGE/DIELECTRIC_TENSOR/lattice are consistent.")

correction_energy = pc_term + alignment_term

# STEP 3: point group symmetry (perfect + defect): Hermann-Mauguin
# (crystallographic) + corresponding Schoenflies notation
point_group_perfect = get_point_group(perfect_struct)
point_group_defect = get_point_group(defect_struct)

#print("\n---point group ---")
#print(f"perfect : Hermann-Mauguin = {point_group_perfect['crystallographic']}, "
#      f"Schoenflies = {point_group_perfect['schoenflies']}")
#print(f"defect  : Hermann-Mauguin = {point_group_defect['crystallographic']}, "
#      f"Schoenflies = {point_group_defect['schoenflies']}")

#print("\n--- eFNV correction ---")
#print(f"pc term        : {pc_term:.16f} eV")
#print(f"alignment term : {alignment_term:.16f} eV")
#print(f"correction energy : {correction_energy:.16f} eV")

# save EVERYTHING into a json file
summary = {
    "@module": "Quantum ESPRESSO" if QE else "VASP",
    "@class": "Energy Correction Summary",
    "charge": CHARGE,
    "lattice": lattice.tolist(),
    "dielectric_tensor": np.asarray(DIELECTRIC_TENSOR).tolist(),
    #"point_charge_correction": pc_term,
    "defect_region_radius": radius,
    "defect_frac_coords": list(defect_frac_coords),
    "point group": {
        "perfect": {
            "Hermann-Mauguin": point_group_perfect["crystallographic"],
            "Schoenflies": point_group_perfect["schoenflies"],
        },
        "defect": {
            "Hermann-Mauguin": point_group_defect["crystallographic"],
            "Schoenflies": point_group_defect["schoenflies"],
        },
    },
    "energy_corrections": {
        "pc term": pc_term,
        "alignment term": alignment_term,
        "correction energy": correction_energy,
    },
    "sites": [
        {"specie": s.specie, "distance": s.distance,
         "potential": s.potential, "pc_potential": s.pc_potential}
        for s in correction.sites
    ],
}
if charge_details is not None:
    summary["charge_details"] = charge_details

with open(OUTPUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)
    
#print(f"\nSaved file: {OUTPUT_JSON}")
