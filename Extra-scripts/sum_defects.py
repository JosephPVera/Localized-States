#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2026-09

"""
Usage:
      python3 defects.py [--qe]     

Collects the VBM, CBM, and gap from ../primitive/primitive.json. The
energy corrections and charge are also collected from each defect folder
(defect/correction.json). In addition, it also extracts the total
energies for each defect and perfect folder. Finally, the chemical
potentials are collected from ../cpd/chem_pot.json. This information is
saved in a summary_defects.json file.

Defect Folder Tree (VASP, default):

    project/
    |-- perfect/
    |   |-- CONTCAR
    |   |-- OUTCAR
    |   `-- vasprun.xml
    |-- N_C-V_C_0/
    |   |-- CONTCAR
    |   |-- vasprun.xml
    |   `-- correction.json
    |-- N_C-V_C_-1/
    |   `-- ...
    |-- C_Si_0/
    |   `-- ...
    |-- V_C_0/
    |   `-- ...
    ...

Defect Folder Tree (--qe, Quantum ESPRESSO):

    project/
    |-- perfect/
    |   `-- scf/
    |       `-- perfect_scf.in, perfect_scf.out
    |-- N_C-V_C_0/
    |   |-- scf/
    |   |   `-- <name>.in, <name>.out
    |   `-- correction.json
    |-- N_C-V_C_-1/
    |   `-- ...
    |-- C_Si_0/
    |   `-- ...
    ...

Chemical potentials (atoms added/removed per species, relative to the
perfect supercell): n_species = count_in_defect - count_in_perfect.
Negative -> atoms removed (vacancy), positive -> atoms added
(substitution/interstitial).
"""

import argparse
import copy
import glob
import json
import re
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from efnv_corrections import read_structure, read_structure_qe

from math import gcd
from functools import reduce

parser = argparse.ArgumentParser()
parser.add_argument("--qe", action="store_true",
                     help="Use Quantum ESPRESSO instead of VASP (default).")
args = parser.parse_args()
QE = args.qe

PROJECT_ROOT = Path(".")
PERFECT_DIR = PROJECT_ROOT / "perfect"

# Dielectric tensor, VBM, CBM and gap
PRIMITIVE_JSON = Path("../primitive/primitive.json")

OUTPUT_JSON = "summary_defects.json"

# Energy correction (VASP: filename inside each defect folder; QE: same
# filename, also inside each defect folder)
CORRECTION_JSON_FILENAME = "correction.json"

# Elemental and limits chemical potentials
CHEM_POT_JSON = Path("../cpd/chem_pot.json")

# VASP-only: override the host reference used for the formation enthalpy
# (None -> use PERFECT_DIR/CONTCAR and PERFECT_DIR/vasprun.xml).
HOST_PRIMITIVE_CONTCAR = None   # e.g. Path("references/BN_primitive/CONTCAR")
HOST_PRIMITIVE_VASPRUN = None   # e.g. Path("references/BN_primitive/vasprun.xml")

RY_TO_EV = 13.605693122994

ENERGY_RE = re.compile(r"total energy\s*=\s*(-?\d+\.\d+)\s*(\w+)")

FOLDER_NAME_PATTERN = re.compile(r"^(?P<defect_type>.+)_(?P<charge_suffix>-?\d+)$")

def load_primitive_json(primitive_json_path):
    if not primitive_json_path.exists():
        raise FileNotFoundError(
            f"{primitive_json_path} not found. This file must contain "
            f"the host's dielectric tensor and band edges (VBM, CBM, gap).")
    with open(primitive_json_path) as f:
        return json.load(f)

_primitive_data = load_primitive_json(PRIMITIVE_JSON)
BAND_EDGES = _primitive_data["Band_edges"]

def find_file(folder, ext):
    """Return the first non-slurm file matching folder/*.<ext>, or None.
    Only needed for the QE workflow (VASP file names are fixed)."""
    pattern = str(Path(folder) / f"*.{ext}")
    files = glob.glob(pattern)
    if ext == "out":
        files = [f for f in files if not re.match(r"^slurm-\d+\.out$", Path(f).name)]

    if not files:
        return None

    files = sorted(files)
    if len(files) > 1:
        print(f"Multiple .{ext} files found in '{folder}': {files}")
        print(f"Using the first one: {files[0]}")

    return Path(files[0])

def get_total_energy_vasp(vasprun_path):
    """Total energy (e_wo_entrp, last occurrence) from a vasprun.xml file."""
    root = ET.parse(vasprun_path).getroot()

    last_value = None
    for energy_tag in root.findall(".//i[@name='e_wo_entrp']"):
        if energy_tag.text:
            last_value = float(energy_tag.text)

    if last_value is None:
        raise ValueError(f"total energy (e_wo_entrp) not found in {vasprun_path}")
    return last_value

def get_total_energy_qe(out_path):
    """Return the last total energy (converted from Ry to eV) found in a
    QE .out file. Prefers the final '!!' converged-energy line; falls
    back to '!' lines."""
    last_energy_single = None
    last_energy_double = None

    with open(out_path, "r", errors="ignore") as f:
        for line in f:
            if re.match(r"^\s*!!\s*total energy\s*=", line):
                m = ENERGY_RE.search(line)
                if m:
                    last_energy_double = float(m.group(1))
            elif re.match(r"^\s*!\s*total energy\s*=", line):
                m = ENERGY_RE.search(line)
                if m:
                    last_energy_single = float(m.group(1))

    last_energy_ry = last_energy_double if last_energy_double is not None else last_energy_single
    if last_energy_ry is None:
        raise ValueError(f"total energy not found in {out_path}")
    return last_energy_ry * RY_TO_EV

def read_structure_any(path):
    """Read a structure with the reader that matches the active code."""
    return read_structure_qe(str(path)) if QE else read_structure(path)

def get_total_energy_any(path):
    """Total energy (eV) with the reader that matches the active code."""
    return get_total_energy_qe(path) if QE else get_total_energy_vasp(path)

def locate_calc_files(dirpath):
    """Return (structure_file, energy_file) for a perfect/defect folder.

    VASP: dirpath/CONTCAR, dirpath/vasprun.xml.
    QE:   the .in and .out found inside dirpath/scf/.
    """
    if QE:
        scf_dir = dirpath / "scf"
        struct_file = find_file(scf_dir, "in")
        energy_file = find_file(scf_dir, "out")
        if struct_file is None or energy_file is None:
            raise FileNotFoundError(f"{dirpath}: missing .in/.out in scf/")
        return struct_file, energy_file
    return dirpath / "CONTCAR", dirpath / "vasprun.xml"

def get_species_counts(structure):
    """Atom counts per species, in the order they appear in the
    structure (i.e. the same order as the CONTCAR species line)."""
    counts = OrderedDict()
    for site in structure:
        symbol = site.specie.symbol
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts

def load_correction_json(correction_json_path):
    if not correction_json_path.exists():
        raise FileNotFoundError(
            f"{correction_json_path} not found. Each defect folder must "
            f"contain its own {correction_json_path.name} with charge "
            f"and energy_corrections.")
    with open(correction_json_path) as f:
        data = json.load(f)
    return {
        "charge": data["charge"],
        "energy_corrections": data["energy_corrections"],
    }

def compute_chemical_potentials(perfect_struct, defect_struct):
    """Number of atoms added/removed per species, relative to the perfect
    supercell: n_species = count_in_defect - count_in_perfect.
    Negative -> atoms removed (e.g. a vacancy), positive -> atoms added
    (e.g. a substitution or interstitial). Every species that makes up
    the perfect material and every species involved in the defect are
    always reported, even when its count did not change (diff = 0).
    """
    perfect_counts = get_species_counts(perfect_struct)
    defect_counts = get_species_counts(defect_struct)
    all_species = sorted(set(perfect_counts) | set(defect_counts))

    chemical_potentials = {}
    for sp in all_species:
        n_perfect = perfect_counts.get(sp, 0)
        n_defect = defect_counts.get(sp, 0)
        chemical_potentials[sp] = n_defect - n_perfect
    return chemical_potentials

def get_formula_unit_counts(structure):
    """Reduce a structure's atom counts to per-formula-unit counts, e.g.
    8 B + 8 N -> ({'B': 1, 'N': 1}, 8 formula units)."""

    counts = get_species_counts(structure)
    g = reduce(gcd, counts.values())
    stoichiometry = {sp: n // g for sp, n in counts.items()}
    n_formula_units = g
    return stoichiometry, n_formula_units

def load_elemental_chemical_potentials(chem_pot_json_path):
    """Elemental mu_i^0 (energy_per_atom_eV), keyed by the element symbol
    stored in each entry's "compound" field (not the top-level folder-name
    key, e.g. "C_mp_14" or "mol_N2")."""
    with open(chem_pot_json_path) as f:
        data = json.load(f)
    return {
        entry["compound"]: entry["energy_per_atom_eV"]
        for entry in data.values()
        if isinstance(entry, dict)
        and "compound" in entry
        and "energy_per_atom_eV" in entry
    }

def compute_formation_enthalpy(host_struct_file, host_energy_file, mu0):
    """Formation enthalpy of the host material per formula unit:
        dH_f = E_host/f.u. - sum_i (stoichiometry_i * mu_i^0)
    Also returns the host's stoichiometry (e.g. {'B': 1, 'N': 1})."""
    host_struct = read_structure_any(host_struct_file)
    stoichiometry, n_formula_units = get_formula_unit_counts(host_struct)

    missing = set(stoichiometry) - set(mu0)
    if missing:
        raise ValueError(
            f"Missing elemental reference(s) for {sorted(missing)} in "
            f"{CHEM_POT_JSON}.")

    e_host_total = get_total_energy_any(host_energy_file)
    e_host_per_fu = e_host_total / n_formula_units
    delta_Hf = e_host_per_fu - sum(
        stoichiometry[sp] * mu0[sp] for sp in stoichiometry)
    return delta_Hf, stoichiometry, e_host_per_fu

def compute_limiting_chemical_potentials(stoichiometry, delta_Hf):
    """Limiting (Delta_mu_i) values at each 'rich' vertex, from:
        sum_i (stoichiometry_i * Delta_mu_i) = delta_Hf   (host stays stable)
        Delta_mu_i <= 0 for every species                 (no elemental precipitation)

    - Host with a single species (e.g. pure C): there is nothing to be
      "rich/poor" in relative to -- returns None.
    - Host with 2 species (e.g. BN): returns the standard two vertices
      A-rich / B-rich.
    - Host with 3+ species: the two conditions above alone don't pin a
      unique vertex per species (real cases need extra secondary
      competing-phase constraints). As a simplifying default, each
      "X-rich" vertex sets Delta_mu_X = 0 and splits delta_Hf equally
      (weighted by stoichiometry) among the remaining species. Treat
      these as approximate reference points, not rigorous phase-diagram
      vertices, unless secondary phases are added.
    """
    species = list(stoichiometry.keys())
    if len(species) < 2:
        return None

    limits = {}
    for rich_sp in species:
        other_sp = [sp for sp in species if sp != rich_sp]
        denom = sum(stoichiometry[sp] for sp in other_sp)
        shared_delta_mu = delta_Hf / denom
        delta_mu = {rich_sp: 0.0}
        delta_mu.update({sp: shared_delta_mu for sp in other_sp})
        limits[f"{rich_sp}-rich"] = delta_mu
    return limits

def is_defect_case(d):
    """A defect case folder has its structure/energy files (see
    locate_calc_files) and its own correction.json.

    VASP: dirpath/CONTCAR, dirpath/vasprun.xml, dirpath/correction.json.
    QE:   dirpath/scf/*.in, dirpath/scf/*.out, dirpath/correction.json.
    """
    has_correction = (d / CORRECTION_JSON_FILENAME).is_file()
    if QE:
        scf_dir = d / "scf"
        if not scf_dir.is_dir():
            return False
        has_in = bool(glob.glob(str(scf_dir / "*.in")))
        has_out = bool(glob.glob(str(scf_dir / "*.out")))
        return has_in and has_out and has_correction
    return (d / "CONTCAR").exists() and (d / "vasprun.xml").exists() and has_correction

def find_defect_folders(project_root, perfect_dir):
    """Find every directory directly under project_root (excluding
    perfect_dir) that looks like a defect case (see is_defect_case).
    Non-matching entries are silently skipped."""
    perfect_resolved = perfect_dir.resolve()
    folders = []
    for d in sorted(project_root.iterdir()):
        if not d.is_dir():
            continue
        if d.resolve() == perfect_resolved:
            continue
        if is_defect_case(d):
            folders.append(d)
    return folders

def defect_type_and_case_label(case_dir, project_root):
    """Split a case folder's own name (e.g. "N_C-V_C_-2", "C_Si_0") into
    (defect_type, case_label).

    defect_type is everything before the trailing "_<integer>" suffix
    (e.g. "N_C-V_C", "C_Si"). This is how different defect types are
    told apart and how results get grouped in the output JSON.
    """
    name = case_dir.name
    m = FOLDER_NAME_PATTERN.match(name)
    if not m:
        raise ValueError(
            f"Folder name '{name}' does not match the expected "
            f"'<defect_type>_<charge>' pattern (e.g. 'N_C-V_C_-2', "
            f"'C_Si_0'). Rename it or adjust FOLDER_NAME_PATTERN.")
    defect_type = m.group("defect_type")
    case_label = name
    return defect_type, case_label

def with_missing_species_zeroed(host_chem_pot_extra, species_involved):
    """Deep copy of host_chem_pot_extra, adapted to the species that
    actually matter for one specific defect (species_involved = every
    species of the perfect material plus every species involved in the
    defect, i.e. the keys of that defect's chemical_potentials/defect):

    - "limits": any species in species_involved that isn't already a
      key gets added with value 0.0. This covers species a defect
      involves (e.g. a dopant atom) that don't belong to the perfect
      host material, so they don't naturally appear in the host-only
      "limits" section.
    - "elementals": filtered down to only the species in
      species_involved, dropping any other element present in
      CHEM_POT_JSON that isn't part of the perfect material or this
      defect.
    """
    result = copy.deepcopy(host_chem_pot_extra)
    species_involved = set(species_involved)

    limits = result.get("limits", {})
    elementals = result.get("elementals")

    if limits is elementals:
        limits = copy.deepcopy(limits)
        result["limits"] = limits

    if limits:
        first_value = next(iter(limits.values()))
        if isinstance(first_value, dict):
            # 2+ species host: {"<sp>-rich": {sp: val, ...}, ...}
            for vertex_dict in limits.values():
                for sp in species_involved:
                    vertex_dict.setdefault(sp, 0.0)
        else:
            # Single-species host: {sp: val, ...}
            for sp in species_involved:
                limits.setdefault(sp, 0.0)

    elementals = result.get("elementals")
    if elementals:
        result["elementals"] = {
            sp: val for sp, val in elementals.items() if sp in species_involved
        }

    return result

def process_defect(defect_dir, perfect_struct, host_chem_pot_extra):
    struct_file, energy_file = locate_calc_files(defect_dir)
    defect_struct = read_structure_any(struct_file)

    # Charge and energy corrections: always taken from the defect's own
    # correction.json (e.g. N_C-V_C_-2/correction.json)
    correction_data = load_correction_json(defect_dir / CORRECTION_JSON_FILENAME)
    charge = correction_data["charge"]
    energy_corrections = correction_data["energy_corrections"]

    defect_chem_pot = compute_chemical_potentials(perfect_struct, defect_struct)
    chemical_potentials = {
        "defect": defect_chem_pot,
    }
    if host_chem_pot_extra:
        chemical_potentials.update(
            with_missing_species_zeroed(host_chem_pot_extra, defect_chem_pot.keys())
        )

    total_energy = get_total_energy_any(energy_file)

    return {
        "charge": charge,
        "chemical_potentials": chemical_potentials,
        "energy_corrections": energy_corrections,
        "total_energy": {
            "TOTEN": total_energy,
        },
    }

def compute_host_chemical_potentials(host_struct_file, host_energy_file):
    """Chemical-potential info that depends only on the host material
    (not on any individual defect). computed once in main() and then
    merged into every defect's own chemical_potentials section:
        - "elementals": all elemental mu_i^0 (energy_per_atom_eV) read
          directly from CHEM_POT_JSON, always included regardless of
          how many species the host has.
        - "limits":
            - host with a single species -> {that species: 0.0} (no
              rich/poor vertex makes sense with only one species, so
              its Delta_mu is just fixed at 0; any extra species from a
              given defect, e.g. a dopant, is later added at 0 too)
            - host with 2+ species       -> {<sp>-rich: {...}, ...}
    Returns None (and prints why) if chem_pot.json doesn't exist.
    """
    if not CHEM_POT_JSON.exists():
        print(f"\nSkipping elemental/limiting chemical potentials: "
              f"{CHEM_POT_JSON} not found. Fill in chem_pot.json at the "
              f"top of the script to enable this.")
        return None

    mu0 = load_elemental_chemical_potentials(CHEM_POT_JSON)
    delta_Hf, stoichiometry, e_host_per_fu = compute_formation_enthalpy(
        host_struct_file, host_energy_file, mu0)

    #print(f"\n--- Chemical potentials (host) ---")
    #print(f"  host stoichiometry (per f.u.) : {stoichiometry}")
    #print(f"  host energy per f.u.          : {e_host_per_fu:.6f} eV")
    #print(f"  elemental mu_i^0              : {mu0}")
    #print(f"  formation enthalpy dH_f(host) : {delta_Hf:.6f} eV")

    if len(stoichiometry) < 2:
        limits = {sp: 0.0 for sp in stoichiometry}
        #print(f"  limits (host species fixed at 0) : {limits}")
        return {"limits": limits, "elementals": mu0}

    limits = compute_limiting_chemical_potentials(stoichiometry, delta_Hf)
    #print(f"  limiting chemical potentials  : {limits}")
    return {"limits": limits, "elementals": mu0}

def main():
    struct_file, energy_file = locate_calc_files(PERFECT_DIR)
    perfect_struct = read_structure_any(struct_file)
    perfect_total_energy = get_total_energy_any(energy_file)

    # Host reference used for the formation enthalpy. VASP allows an
    # explicit override (HOST_PRIMITIVE_CONTCAR/HOST_PRIMITIVE_VASPRUN);
    # QE always uses the perfect/scf files found above.
    if QE:
        host_struct_file, host_energy_file = struct_file, energy_file
    else:
        host_struct_file = HOST_PRIMITIVE_CONTCAR or struct_file
        host_energy_file = HOST_PRIMITIVE_VASPRUN or energy_file

    # Computed once for the whole project (depends only on the host
    # material): {"limits": {...}} -- rich/poor vertices for a 2+
    # species host, or all elemental mu_i^0 for a single-species host.
    # Merged into every defect's own chemical_potentials section below.
    host_chem_pot_extra = compute_host_chemical_potentials(
        host_struct_file, host_energy_file)

    defect_folders = find_defect_folders(PROJECT_ROOT, PERFECT_DIR)
    if not defect_folders:
        raise RuntimeError(
            f"No defect folders found under {PROJECT_ROOT.resolve()} "
            f"(excluding {PERFECT_DIR}); see is_defect_case for what "
            f"each folder must contain).")

    # Group results by defect type
    grouped_results = defaultdict(dict)
    for defect_dir in defect_folders:
        defect_type, case_label = defect_type_and_case_label(defect_dir, PROJECT_ROOT)
        rel_path = defect_dir.relative_to(PROJECT_ROOT)
        try:
            grouped_results[defect_type][case_label] = process_defect(
                defect_dir, perfect_struct, host_chem_pot_extra)
        except Exception as exc:
            print(f"\n  SKIPPED [{rel_path}]: {exc}")

    # Within each defect type, sort case entries by (declared) charge
    defects_summary = {}
    for defect_type in sorted(grouped_results.keys()):
        cases = grouped_results[defect_type]
        defects_summary[defect_type] = dict(
            sorted(cases.items(), key=lambda kv: kv[1]["charge"])
        )

    summary = {
        "@module": "Quantum ESPRESSO" if QE else "VASP",
        "@class": "Defects Summary",
        "band_edges": BAND_EDGES,
        "perfect": {
            "total_energy": {
                "TOTEN": perfect_total_energy,
            },
        },
        "defects": defects_summary,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    #print(f"Saved file: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
