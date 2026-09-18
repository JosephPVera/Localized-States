#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2026-09

"""
Usage:
      python3 primitive.py [--qe]        

Extracts the VBM, CBM and gap values, and the electronic, ionic, and total
dielectric tensors, and saves everything to primitive.json.

VASP reads from:
      dos/vasprun.xml
      dielectric/vasprun.xml

QE reads from:
      nscf/.out
      dielectric/dynmat/.out
"""

import argparse
import glob
import json
import os
import re
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------
# VASP
# ---------------------------------------------------------------------
def vasp_detect_spins_and_kpoints(root):
    """
    Detect which spin and kpoint numbers exist inside the
    eigenvalues block of the vasprun.xml file.
    """
    spin_numbers = []
    for spin_set in root.findall(".//set[@comment]"):
        match = re.match(r"spin (\d+)", spin_set.get("comment", ""))
        if match:
            spin_numbers.append(int(match.group(1)))

    spin_numbers = sorted(set(spin_numbers))

    kpoint_numbers = []
    if spin_numbers:
        first_spin = root.find(f".//set[@comment='spin {spin_numbers[0]}']")
        for kp_set in first_spin.findall("set"):
            match = re.match(r"kpoint (\d+)", kp_set.get("comment", ""))
            if match:
                kpoint_numbers.append(int(match.group(1)))

    kpoint_numbers = sorted(set(kpoint_numbers))
    return spin_numbers, kpoint_numbers

def vasp_find_band_edges(root, spin_numbers, kpoint_numbers):
    """
    Finds the VBM (max energy with occupancy 1.000) and the CBM
    (min energy with occupancy 0.000) across the given spins/kpoints.
    """
    max_energy_1000 = float('-inf')  # VBM
    min_energy_0000 = float('inf')   # CBM

    for spin_number in spin_numbers:
        spin_set = root.find(f".//set[@comment='spin {spin_number}']")

        if spin_set is None:
            print(f"Superblock 'spin {spin_number}' not found.")
            continue

        for kpoint_number in kpoint_numbers:
            kpoint_block = spin_set.find(f".//set[@comment='kpoint {kpoint_number}']")

            if kpoint_block is None:
                print(f"Block 'kpoint {kpoint_number}' not found in 'spin {spin_number}'.")
                continue

            last_1_000_energy = None   # last energy with occupancy 1.000
            first_0_000_energy = None  # first energy with occupancy 0.000

            for child in kpoint_block:
                if not child.text:
                    continue
                columns = child.text.split()
                if len(columns) < 2:
                    continue

                energy = float(columns[0])
                occupancy = float(columns[1])

                if occupancy == 1.000:
                    last_1_000_energy = energy

                if occupancy == 0.000 and first_0_000_energy is None:
                    first_0_000_energy = energy

            if last_1_000_energy is not None:
                max_energy_1000 = max(max_energy_1000, last_1_000_energy)

            if first_0_000_energy is not None:
                min_energy_0000 = min(min_energy_0000, first_0_000_energy)

    return max_energy_1000, min_energy_0000

def vasp_get_gap(folder):
    """Compute VBM, CBM and gap from the vasprun.xml inside "dos/vasprun.xml"."""
    xml_path = f"{folder}/vasprun.xml"
    root = ET.parse(xml_path).getroot()

    spin_numbers, kpoint_numbers = vasp_detect_spins_and_kpoints(root)

    vbm, cbm = vasp_find_band_edges(root, spin_numbers, kpoint_numbers)

    gap = cbm - vbm if (vbm != float('-inf') and cbm != float('inf')) else None

    return {
        "VBM": vbm,
        "CBM": cbm,
        "gap": gap,
    }

def vasp_get_dielectric_tensor(folder):
    """Extract the ionic, electronic, and total dielectric tensors from the vasprun.xml inside "dielectric/vasprun.xml"."""
    xml_path = f"{folder}/vasprun.xml"
    root = ET.parse(xml_path).getroot()

    def extract_tensor(varray_name):
        varray = root.find(f".//varray[@name='{varray_name}']")
        if varray is None:
            return None
        return [
            [float(x) for x in v.text.split()]
            for v in varray.findall("v")
        ]

    epsilon_ionic = extract_tensor("epsilon_ion")
    epsilon_electronic = extract_tensor("epsilon")

    return {
        "Ionic": epsilon_ionic,
        "Electronic": epsilon_electronic,
        "Total": sum_matrices(epsilon_ionic, epsilon_electronic),
    }

def vasp_collect_results():
    results = {"@module": "VASP", "@class": "Primitive Summary"}
    results["Band_edges"] = vasp_get_gap("dos")
    results["Dielectric"] = vasp_get_dielectric_tensor("dielectric")
    return results

# ---------------------------------------------------------------------
# Quantum ESPRESSO
# ---------------------------------------------------------------------
HEADER_DIELECTRIC_RE = re.compile(r"Dielectric constant in cartesian axis")
HEADER_ELECTRONIC_RE = re.compile(
    re.escape("Electronic dielectric permittivity tensor (relative, adimensional)")
)
HEADER_TOTAL_RE = re.compile(re.escape("... with zone-center polar mode contributions"))

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

def qe_has_dielectric_info(filename):
    """Return True if the file contains at least one of the dielectric tensor headers."""
    with open(filename, "r", errors="ignore") as f:
        content = f.read()
    return bool(
        HEADER_DIELECTRIC_RE.search(content)
        or HEADER_ELECTRONIC_RE.search(content)
        or HEADER_TOTAL_RE.search(content)
    )

def qe_find_dielectric_out_file(folder):
    """Return the .out file inside folder to use for dielectric tensor extraction."""
    pattern = os.path.join(folder, "*.out")
    all_out_files = glob.glob(pattern)
    out_files = [f for f in all_out_files if not re.match(r"^slurm-\d+\.out$", os.path.basename(f))]

    if not out_files:
        return None

    out_files = sorted(out_files)

    if len(out_files) == 1:
        return out_files[0]

    with_info = [f for f in out_files if qe_has_dielectric_info(f)]

    if with_info:
        if len(with_info) > 1:
            print(f"Multiple .out files with dielectric tensor info found in '{folder}': {with_info}")
            print(f"Using the first one: {with_info[0]}")
        return with_info[0]

    print(f"Multiple .out files found in '{folder}', but none contain dielectric tensor info: {out_files}")
    print(f"Using the first one: {out_files[0]}")
    return out_files[0]

def qe_get_last_block(lines, header_re):
    """Return the last block matching header_re as a string, or None if not found.
    A block is: the header line, any blank line(s) right after it, and the
    matrix rows until the next blank line."""
    indices = [i for i, line in enumerate(lines) if header_re.search(line)]
    if not indices:
        return None

    start = indices[-1]
    block = [lines[start]]
    i = start + 1

    while i < len(lines) and lines[i].strip() == "":
        block.append(lines[i])
        i += 1

    while i < len(lines) and lines[i].strip() != "":
        block.append(lines[i])
        i += 1

    return "".join(block).rstrip("\n")

def qe_parse_matrix(block):
    """Parse a block's text (header line + matrix rows) into a 3x3 list of floats.
    Returns None if the block doesn't contain a well-formed 3x3 matrix."""
    if block is None:
        return None

    rows = []
    for line in block.splitlines()[1:]:  # skip the header line
        if not line.strip():
            continue
        values = [float(x) for x in line.split()]
        if values:
            rows.append(values)

    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        return None
    return rows

def qe_get_gap(folder="nscf"):
    """Compute VBM, CBM and gap (or Fermi energy) from the .out file inside folder."""
    filename = qe_find_out_file(folder)
    if filename is None:
        print(f"No .out file found in '{folder}'.")
        return {"VBM": None, "CBM": None, "gap": None}

    print(f"Reading: {filename}")

    with open(filename, "r", errors="ignore") as f:
        lines = f.readlines()

    vbm = None
    cbm = None
    fermi = None

    for line in lines:
        match = re.search(
            r"highest occupied, lowest unoccupied level\s*\(ev\):\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)",
            line
        )
        if match:
            vbm = float(match.group(1))
            cbm = float(match.group(2))

        fermi_match = re.search(
            r"the Fermi energy is\s*(-?\d+\.\d+)\s*ev",
            line,
            re.IGNORECASE
        )
        if fermi_match:
            fermi = float(fermi_match.group(1))  # keep overwriting -> ends up as the LAST match

    if vbm is not None and cbm is not None:
        return {"VBM": vbm, "CBM": cbm, "gap": cbm - vbm}
    elif fermi is not None:
        return {"Fermi_energy": fermi}
    else:
        print("Neither 'highest occupied, lowest unoccupied level' nor 'the Fermi energy is' line found.")
        return {"VBM": None, "CBM": None, "gap": None}

def qe_get_dielectric_tensor(folder="dielectric/dynmat"):
    """Extract the ionic, electronic, and total dielectric tensors from the .out file inside folder."""
    filename = qe_find_dielectric_out_file(folder)
    if filename is None:
        print(f"No .out file found in '{folder}'.")
        return {"Ionic": None, "Electronic": None, "Total": None}

    print(f"Reading: {filename}")

    with open(filename, "r", errors="ignore") as f:
        lines = f.readlines()

    dielectric_block = qe_get_last_block(lines, HEADER_DIELECTRIC_RE)
    electronic_block = qe_get_last_block(lines, HEADER_ELECTRONIC_RE)
    total_block = qe_get_last_block(lines, HEADER_TOTAL_RE)

    dielectric_matrix = qe_parse_matrix(dielectric_block)
    electronic_matrix = qe_parse_matrix(electronic_block) or dielectric_matrix
    total_matrix = qe_parse_matrix(total_block)

    ionic_matrix = None
    if electronic_matrix is not None and total_matrix is not None:
        ionic_matrix = subtract_matrices(total_matrix, electronic_matrix)

    return {
        "Ionic": ionic_matrix,
        "Electronic": electronic_matrix,
        "Total": total_matrix,
    }

def qe_collect_results():
    results = {"@module": "Quantum ESPRESSO", "@class": "Primitive Summary"}
    results["Band_edges"] = qe_get_gap("nscf")
    results["Dielectric"] = qe_get_dielectric_tensor("dielectric/dynmat")
    return results

# ---------------------------------------------------------------------
# VASP and QE
# ---------------------------------------------------------------------
def sum_matrices(matrix_a, matrix_b):
    """Elementwise a + b for two same-shape matrices. Used by VASP to build
    the total dielectric tensor from the ionic and electronic ones."""
    if matrix_a is None or matrix_b is None:
        return None
    return [
        [a + b for a, b in zip(row_a, row_b)]
        for row_a, row_b in zip(matrix_a, matrix_b)
    ]

def subtract_matrices(matrix_a, matrix_b):
    """Elementwise a - b for two same-shape matrices. Used by QE to back out
    the ionic dielectric tensor from the total and electronic ones."""
    if matrix_a is None or matrix_b is None:
        return None
    return [
        [a - b for a, b in zip(row_a, row_b)]
        for row_a, row_b in zip(matrix_a, matrix_b)
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qe", action="store_true", help="Use Quantum ESPRESSO instead of VASP (default)")
    args = parser.parse_args()

    results = qe_collect_results() if args.qe else vasp_collect_results()

    with open("primitive.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
