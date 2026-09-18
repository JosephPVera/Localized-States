#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2026-09

"""
Usage:
      python3 chem_pot.py [--qe]        

Extracts the total energy, the number of atoms, and the chemical formula
from each calculation folder, computes the energy per atom, and saves
everything to chem_pot.json. 

Each top-level species folder may contain the DFT outputs directly, or
inside a "scf" subfolder alongside other unrelated subfolders (relax,
dos, bands, etc., which must be ignored):

    N/vasprun.xml   (or N/*.out for --qe)
    Si/scf/vasprun.xml   (or Si/scf/*.out for --qe)
    Si/relax/...          <- ignored
    Si/dos/...            <- ignored

For each species folder, only an output found directly in it or inside
its "scf" subfolder is used. Any other subfolder is skipped.
"""

import argparse
import glob
import json
import os
import re
import xml.etree.ElementTree as ET

RY_TO_EV = 13.605693122994

ENERGY_RE = re.compile(r"total energy\s*=\s*(-?\d+\.\d+)\s*(\w+)")
NATOMS_RE = re.compile(r"number of atoms/cell\s*=\s*(\d+)")
SPECIES_HEADER_RE = re.compile(r"atomic species\s+valence\s+mass\s+pseudopotential")
SPECIES_LINE_RE = re.compile(r"\s*(\S+)\s+[\d.]+\s+[\d.]+\s+\S")

# ---------------------------------------------------------------------
# VASP
# ---------------------------------------------------------------------
def vasp_find_calc_dir(species_dir):
    """Locates vasprun.xml directly in species_dir or in species_dir/scf."""
    flat_vasprun = os.path.join(species_dir, "vasprun.xml")
    if os.path.isfile(flat_vasprun):
        return species_dir

    scf_dir = os.path.join(species_dir, "scf")
    scf_vasprun = os.path.join(scf_dir, "vasprun.xml")
    if os.path.isfile(scf_vasprun):
        return scf_dir

    return None

def vasp_get_compound_formula(vasprun_path):
    """
    Builds the chemical formula straight from vasprun.xml's <atominfo>
    section, i.e. from the actual elements in the calculation, not the
    folder name. Appends only each element's symbol (no atom counts).
    Returns None if <atominfo>/"atomtypes" can't be found or parsed.
    """
    root = ET.parse(vasprun_path).getroot()

    atominfo = root.find("atominfo")
    if atominfo is None:
        return None

    atomtypes = atominfo.find("array[@name='atomtypes']")
    if atomtypes is None:
        return None

    rows = atomtypes.find("set")
    if rows is None:
        return None

    formula = ""
    for rc in rows.findall("rc"):
        cells = rc.findall("c")
        if len(cells) < 2:
            continue
        element = cells[1].text.strip()
        formula += element

    return formula or None

def vasp_get_total_energy(vasprun_path):
    """Returns the last e_wo_entrp value found in vasprun.xml."""
    root = ET.parse(vasprun_path).getroot()

    last_value = None
    for energy_tag in root.findall(".//i[@name='e_wo_entrp']"):
        if energy_tag.text:
            last_value = float(energy_tag.text)

    return last_value

def vasp_get_num_atoms(vasprun_path):
    """Returns the number of atoms read from vasprun.xml's <atominfo><atoms>."""
    root = ET.parse(vasprun_path).getroot()

    atominfo = root.find("atominfo")
    if atominfo is None:
        return None

    atoms_tag = atominfo.find("atoms")
    if atoms_tag is None or atoms_tag.text is None:
        return None

    return int(atoms_tag.text.strip())

def vasp_collect_results(base_directory):
    results = {"@module": "VASP", "@class": "Chemical Potentials Summary"}

    for entry in sorted(os.listdir(base_directory)):
        species_dir = os.path.join(base_directory, entry)
        if not os.path.isdir(species_dir):
            continue

        calc_dir = vasp_find_calc_dir(species_dir)
        if calc_dir is None:
            continue  # no vasprun.xml directly nor inside scf/

        vasprun_path = os.path.join(calc_dir, "vasprun.xml")

        total_energy = vasp_get_total_energy(vasprun_path)
        num_atoms = vasp_get_num_atoms(vasprun_path)
        compound = vasp_get_compound_formula(vasprun_path)

        results[entry] = build_entry(compound, total_energy, num_atoms)

    return results


# ---------------------------------------------------------------------
# Quantum ESPRESSO
# ---------------------------------------------------------------------

def qe_find_out_file(folder):
    """Return the first non-slurm .out file found inside folder, or None."""
    pattern = os.path.join(folder, "*.out")
    all_out_files = glob.glob(pattern)
    out_files = [f for f in all_out_files if not re.match(r"^slurm-\d+\.out$", os.path.basename(f))]

    if not out_files:
        return None

    out_files = sorted(out_files)
    if len(out_files) > 1:
        print(f"Multiple .out files found in '{folder}': {out_files}")
        print(f"Using the first one: {out_files[0]}")

    return out_files[0]

def qe_find_target_folders():
    """
    Decides which directory actually holds the QE outputs for a given
    species folder: directly inside it, or inside its "scf" subfolder.
    Any other subfolder (relax, dos, bands, etc.) is ignored.
    """
    targets = {}

    for entry in sorted(os.listdir(".")):
        if not os.path.isdir(entry):
            continue

        scf_path = os.path.join(entry, "scf")
        if os.path.isdir(scf_path):
            targets[entry] = scf_path
            continue

        if qe_find_out_file(entry) is not None:
            targets[entry] = entry

    return targets

def qe_get_total_energy_ev(filepath):
    """Return the last total energy (converted from Ry to eV) found in a QE .out file.
    Prefers the final '!!' converged-energy line; falls back to '!' lines."""
    last_energy_single = None
    last_energy_double = None

    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            if re.match(r"^\s*!!\s*total energy\s*=", line):
                match = ENERGY_RE.search(line)
                if match:
                    last_energy_double = float(match.group(1))
            elif re.match(r"^\s*!\s*total energy\s*=", line):
                match = ENERGY_RE.search(line)
                if match:
                    last_energy_single = float(match.group(1))

    last_energy_ry = last_energy_double if last_energy_double is not None else last_energy_single
    return last_energy_ry * RY_TO_EV if last_energy_ry is not None else None

def qe_get_num_atoms(filepath):
    """Return the number of atoms/cell found in a QE .out file, or None."""
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            match = NATOMS_RE.search(line)
            if match:
                return int(match.group(1))
    return None

def qe_get_compound_formula(filepath):
    """
    Builds the chemical formula straight from the .out file's
    "atomic species   valence    mass     pseudopotential" table, i.e.
    from the actual species used in the calculation, not the folder
    name. Returns None if the table can't be found.
    """
    species = []
    in_block = False

    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            if SPECIES_HEADER_RE.search(line):
                in_block = True
                continue
            if in_block:
                if not line.strip():
                    break
                match = SPECIES_LINE_RE.match(line)
                if match:
                    species.append(match.group(1))

    formula = "".join(species)
    return formula or None

def qe_collect_results():
    results = {"@module": "Quantum ESPRESSO", "@class": "Chemical Potentials Summary"}

    for element_folder, scan_folder in qe_find_target_folders().items():
        filename = qe_find_out_file(scan_folder)
        if filename is None:
            print(f"No .out file found in '{scan_folder}'.")
            continue

        print(f"Reading: {filename}")

        total_energy_ev = qe_get_total_energy_ev(filename)
        num_atoms = qe_get_num_atoms(filename)
        compound = qe_get_compound_formula(filename)

        results[element_folder] = build_entry(compound, total_energy_ev, num_atoms)

    return results

# ---------------------------------------------------------------------
# VASP and QE
# ---------------------------------------------------------------------

def build_entry(compound, total_energy, num_atoms):
    """Builds the per-folder result dict, computing the energy per atom.
    Shared by both the VASP and QE code paths."""
    energy_per_atom = total_energy / num_atoms if total_energy is not None and num_atoms else None

    return {
        "compound": compound,
        "total_energy_eV": total_energy,
        "num_atoms": num_atoms,
        "energy_per_atom_eV": energy_per_atom,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qe", action="store_true", help="Use Quantum ESPRESSO instead of VASP (default)")
    args = parser.parse_args()

    results = qe_collect_results() if args.qe else vasp_collect_results(os.getcwd())

    with open("chem_pot.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
