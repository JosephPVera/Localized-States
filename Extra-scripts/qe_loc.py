#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2026-08

"""
Usage
     python3 qe_loc.py [--no-orbitals] [--no-band] [--nsp] [--ctb]

Use --nsp to treat the calculation as NON spin-polarized instead: 
          all k-point blocks are then kept as a single set, with no 
          SPIN UP / SPIN DOWN split.

Use --no-orbitals to skip the (potentially very large) per-band
                  atomic-orbital projection tables and only write the 
                  two-column band/energy tables.

Use --no-band to skip the two-column band/energy tables and only write
              the per-band atomic-orbital projection tables (each of 
              which already shows its own band number and energy in its 
              header line).

Use --ctb (contribution) to only print, in each per-band orbital table,
          the TOP_N_CONTRIB (6) atom rows with the highest 'tot' value. 
          These are then printed in ascending atom-number order. All other 
          atoms are omitted from the printed table, but the column sums 
          and the final 'tot' summary row still account for ALL atoms.

Extracts, from a Quantum ESPRESSO projwfc.x .out file:
  1) Band number and band energy (eV) for every k-point, split into
     SPIN UP / SPIN DOWN sections.
  2) (optional, on by default) For every band, a table with the squared
     projection coefficients (|c|^2) onto atomic s / p_x / p_y / p_z
     orbitals, summed per atom.

How k-point blocks are assigned to spin channels
--------------------------------------------------
In a spin-polarized projwfc.x run, "nkstot" (found near the top of the file,
in the "Problem Sizes" block) is the TOTAL number of k-point blocks printed,
already counting both spin channels. If there are N distinct k-points in the
calculation, then nkstot = 2*N:
    - The first  N blocks (in the order they appear) = SPIN UP
    - The last   N blocks (in the order they appear) = SPIN DOWN

A new k-point block begins every time a line of the form:
    k =   kx  ky  kz
is found. Within a block, band/energy pairs are lines of the form:
    ==== e(   1) =    -8.29100 eV ====
followed by one or more lines of projection terms:
    psi = 0.011*[# 857]+0.008*[#  53]+ ...
             +0.007*[# 689]+ ...
    |psi|^2 = 0.994

Atomic-orbital projection table
--------------------------------
The "Atomic states used for projection" section maps each state index
(the number inside [# ...]) to an atom and a set of quantum numbers
(l, m), e.g.:
    state #   1: atom   1 (C  ), wfc  1 (l=0 m= 1)
    state #   2: atom   1 (C  ), wfc  2 (l=1 m= 1)

Quantum-number -> orbital-type mapping used here:
    l=0, m=1  ->  s
    l=1, m=1  ->  p_x
    l=1, m=2  ->  p_y
    l=1, m=3  ->  p_z

For each band, every term coeff*[# idx] in its psi expansion contributes
coeff**2 to the (atom, orbital_type) cell it maps to. States with other
(l, m) combinations (e.g. d orbitals) are not currently mapped to a
column and are skipped (a warning with the count is printed).
"""

import glob
import os
import re
import sys
from pathlib import Path

KPOINT_RE = re.compile(r"^\s*k\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")
BAND_RE = re.compile(r"====\s*e\(\s*(\d+)\s*\)\s*=\s*([-\d.]+)\s*eV\s*====")
NKSTOT_RE = re.compile(r"nkstot\s*=\s*(\d+)")
NBND_RE = re.compile(r"nbnd\s*=\s*(\d+)")
NATOMWFC_RE = re.compile(r"natomwfc\s*=\s*(\d+)")
STATE_RE = re.compile(
    r"state\s*#\s*(\d+):\s*atom\s*(\d+)\s*\(\s*(\S+?)\s*\),\s*wfc\s*(\d+)\s*"
    r"\(l=\s*(-?\d+)\s*m=\s*(-?\d+)\)")
TERM_RE = re.compile(r"([\d.]+)\*\[#\s*(\d+)\]")
PSI2_RE = re.compile(r"\|psi\|\^2\s*=\s*([\d.]+)")

MASTER_ORBITAL_COLUMNS = [
    "s",
    "p_x", "p_y", "p_z",
    "d_z2", "d_xz", "d_yz", "d_x2-y2", "d_xy",
    "f_z3", "f_xz2", "f_yz2", "f_z(x2-y2)", "f_xyz", "f_x(x2-3y2)", "f_y(3x2-y2)",]

def orbital_label(l, m):
    """Map (l, m) quantum numbers to one of the supported orbital columns,
    or None if unsupported (e.g. g orbitals and beyond)."""
    if l == 0 and m == 1:
        return "s"
    if l == 1 and m == 1:
        return "p_x"
    if l == 1 and m == 2:
        return "p_y"
    if l == 1 and m == 3:
        return "p_z"
    if l == 2 and m == 1:
        return "d_z2"
    if l == 2 and m == 2:
        return "d_xz"
    if l == 2 and m == 3:
        return "d_yz"
    if l == 2 and m == 4:
        return "d_x2-y2"
    if l == 2 and m == 5:
        return "d_xy"
    if l == 3 and m == 1:
        return "f_z3"
    if l == 3 and m == 2:
        return "f_xz2"
    if l == 3 and m == 3:
        return "f_yz2"
    if l == 3 and m == 4:
        return "f_z(x2-y2)"
    if l == 3 and m == 5:
        return "f_xyz"
    if l == 3 and m == 6:
        return "f_x(x2-3y2)"
    if l == 3 and m == 7:
        return "f_y(3x2-y2)"
    return None

def active_orbital_columns(state_map):
    """Return the subset of MASTER_ORBITAL_COLUMNS that actually occur in
    state_map (i.e. the orbital types present in this .out file), keeping
    the canonical s -> p -> d -> f order."""
    used = {label for (_atom, label) in state_map.values() if label is not None}
    return [col for col in MASTER_ORBITAL_COLUMNS if col in used]

def find_out_file(folder="."):
    all_out_files = glob.glob(os.path.join(folder, "*.out"))
    # Exclude files like slurm-0001.out, slurm-5768.out, etc. (checked by filename only)
    out_files = [f for f in all_out_files if not re.match(r"^slurm-\d+\.out$", os.path.basename(f))]
    if not out_files:
        raise FileNotFoundError(
            "No valid .out file was found in the folder "
        )
    if len(out_files) > 1:
        print(f"Warning: multiple .out files were found {sorted(out_files)}, using: {out_files[0]}")
    return out_files[0]

def parse_file(path):
    """Single pass parse of the .out file.
    Returns:
        nkstot, nbnd, natomwfc : ints or None
        state_map : dict[int -> (atom_number:int, orbital_label:str|None)]
        n_atoms   : int, highest atom number seen in the state list
        n_skipped_states : int, states with unsupported (l, m) (e.g. d orbitals)
        blocks    : list of {'kpoint': (kx,ky,kz),
                              'bands': [{'band':int,'energy':float,'terms':[(coeff,idx),...]}, ...]}
    """
    nkstot = nbnd = natomwfc = None
    state_map = {}
    n_atoms = 0
    n_skipped_states = 0

    blocks = []
    current_block = None
    current_band = None
    collecting_psi = False

    with open(path, "r", errors="ignore") as f:
        for line in f:
            if nkstot is None:
                m = NKSTOT_RE.search(line)
                if m:
                    nkstot = int(m.group(1))
            if nbnd is None:
                m = NBND_RE.search(line)
                if m:
                    nbnd = int(m.group(1))
            if natomwfc is None:
                m = NATOMWFC_RE.search(line)
                if m:
                    natomwfc = int(m.group(1))

            ms = STATE_RE.search(line)
            if ms:
                idx = int(ms.group(1))
                atom = int(ms.group(2))
                l = int(ms.group(5))
                m_q = int(ms.group(6))
                label = orbital_label(l, m_q)
                if label is None:
                    n_skipped_states += 1
                state_map[idx] = (atom, label)
                if atom > n_atoms:
                    n_atoms = atom
                continue

            mk = KPOINT_RE.match(line)
            if mk:
                current_block = {
                    "kpoint": (mk.group(1), mk.group(2), mk.group(3)),
                    "bands": [],
                }
                blocks.append(current_block)
                current_band = None
                collecting_psi = False
                continue

            mb = BAND_RE.search(line)
            if mb:
                band = int(mb.group(1))
                energy = float(mb.group(2))
                current_band = {"band": band, "energy": energy, "terms": [], "psi2": None}
                if current_block is not None:
                    current_block["bands"].append(current_band)
                collecting_psi = True
                for c, i in TERM_RE.findall(line):
                    current_band["terms"].append((float(c), int(i)))
                continue

            if collecting_psi:
                if "|psi|^2" in line:
                    m2 = PSI2_RE.search(line)
                    if m2 and current_band is not None:
                        current_band["psi2"] = float(m2.group(1))
                    collecting_psi = False
                    continue
                for c, i in TERM_RE.findall(line):
                    current_band["terms"].append((float(c), int(i)))
                continue

    return nkstot, nbnd, natomwfc, state_map, n_atoms, n_skipped_states, blocks

ATOM_COL_WIDTH = 8
ORB_COL_WIDTH = 15
TOP_N_CONTRIB = 6

def write_orbital_table(f, band_entry, state_map, n_atoms, orbital_columns, only_significant=False):
    """Write the per-atom orbital squared-coefficient table for one band,
    using only the orbital columns actually present in the file.

    If only_significant is True (--ctb), only the TOP_N_CONTRIB atom rows
    with the highest 'tot' value are printed, sorted from highest to
    lowest. The column sums and the final 'tot' row still account for ALL
    atoms, regardless of this filter."""
    n_cols = len(orbital_columns)
    table = [[0.0] * n_cols for _ in range(n_atoms)]

    for coeff, idx in band_entry["terms"]:
        entry = state_map.get(idx)
        if entry is None:
            continue
        atom, label = entry
        if label is None or label not in orbital_columns:
            continue
        col = orbital_columns.index(label)
        table[atom - 1][col] += coeff ** 2

    f.write(f"Band {band_entry['band']}   energy = {band_entry['energy']:.5f} eV\n")
    header = f"{'atom':<{ATOM_COL_WIDTH}}"
    header += "".join(f"{col:<{ORB_COL_WIDTH}}" for col in orbital_columns)
    header += f"{'tot':<{ORB_COL_WIDTH}}"
    f.write(header + "\n")

    col_sums = [0.0] * n_cols
    atom_rows = []
    for atom_i in range(n_atoms):
        row = table[atom_i]
        tot = sum(row)
        col_sums = [col_sums[c] + row[c] for c in range(n_cols)]
        atom_rows.append((atom_i + 1, row, tot))

    if only_significant:
        rows_to_print = sorted(atom_rows, key=lambda r: r[2], reverse=True)[:TOP_N_CONTRIB]
        rows_to_print = sorted(rows_to_print, key=lambda r: r[0])
    else:
        rows_to_print = atom_rows

    for atom_num, row, tot in rows_to_print:
        line = f"{atom_num:<{ATOM_COL_WIDTH}}"
        line += "".join(f"{v:<{ORB_COL_WIDTH}.6f}" for v in row)
        line += f"{tot:<{ORB_COL_WIDTH}.6f}"
        f.write(line + "\n")

    psi2 = band_entry.get("psi2")
    psi2_str = f"{psi2:.6f}" if psi2 is not None else "NA"
    tot_line = f"{'tot':<{ATOM_COL_WIDTH}}"
    tot_line += "".join(f"{v:<{ORB_COL_WIDTH}.6f}" for v in col_sums)
    tot_line += f"{psi2_str:<{ORB_COL_WIDTH}}"
    f.write(tot_line.rstrip() + "\n")
    f.write("\n")

def write_output(out_path, nkstot, nbnd, natomwfc, state_map, n_atoms, blocks, include_orbitals=True, polarized=True, include_band_table=True, only_significant=False):
    n_total_blocks = len(blocks)
    orbital_columns = active_orbital_columns(state_map) if include_orbitals else []

    if polarized:
        if nkstot is not None and nkstot > 0 and nkstot % 2 == 0:
            n_kpts_per_spin = nkstot // 2
        else:
            # Fallback: split the detected blocks evenly in half
            n_kpts_per_spin = n_total_blocks // 2

        spin_sections = [
            ("SPIN UP", blocks[:n_kpts_per_spin]),
            ("SPIN DOWN", blocks[n_kpts_per_spin:2 * n_kpts_per_spin]),
        ]
    else:
        # Non spin-polarized: keep all k-point blocks together, no spin split
        spin_sections = [(None, blocks)]

    with open(out_path, "w") as f:
        in_path = Path(find_out_file("."))
        f.write(f"# Extracted band/energy data from {in_path}\n")
        if nkstot is not None:
            f.write(f"# nkstot   = {nkstot}\n")
        if nbnd is not None:
            f.write(f"# nbnd     = {nbnd}\n")
        if natomwfc is not None:
            f.write(f"# natomwfc = {natomwfc}\n")
        f.write(f"# n_atoms  = {n_atoms}\n")
        f.write("\n")
        f.write(f"# Total k-point blocks found in file = {n_total_blocks}\n")
        f.write(f"# Analysis mode = {'SPIN POLARIZED' if polarized else 'NON SPIN-POLARIZED'}\n")
        f.write("\n")

        for spin_label, spin_blocks in spin_sections:
            if spin_label is not None:
                f.write("=" * 60 + "\n")
                f.write(f"{spin_label}\n")
                f.write("=" * 60 + "\n\n")

            for idx, block in enumerate(spin_blocks, start=1):
                kx, ky, kz = block["kpoint"]
                f.write(f"--- k-point {idx}  (k = {kx} {ky} {kz}) ---\n\n")

                if include_band_table:
                    f.write(f"{'Band':>6}\t{'Energy(eV)':>12}\n")
                    for band_entry in block["bands"]:
                        f.write(f"{band_entry['band']:>6}\t{band_entry['energy']:>12.5f}\n")
                    f.write("\n")

                if include_orbitals and n_atoms > 0:
                    for band_entry in block["bands"]:
                        write_orbital_table(f, band_entry, state_map, n_atoms, orbital_columns, only_significant=only_significant)

    if polarized:
        n_up = len(spin_sections[0][1])
        n_down = len(spin_sections[1][1])
        return n_up, n_down
    else:
        n_blocks = len(spin_sections[0][1])
        return n_blocks, None

def main():
    args = sys.argv[1:]

    include_orbitals = True
    if "--no-orbitals" in args:
        include_orbitals = False
        args.remove("--no-orbitals")

    include_band_table = True
    if "--no-band" in args:
        include_band_table = False
        args.remove("--no-band")

    if not include_orbitals and not include_band_table:
        print("Error: --no-orbitals and --no-band cannot be used together "
              "(there would be no information left to write).")
        sys.exit(1)

    polarized = True
    if "--nsp" in args:
        polarized = False
        args.remove("--nsp")

    only_significant = False
    if "--ctb" in args:
        only_significant = True
        args.remove("--ctb")

    in_path = Path(find_out_file("."))
    out_path = in_path.with_name("localization.dat")

    print(f"Detected .out file: {in_path}")
    #print(f"Analysis mode: {'SPIN POLARIZED' if polarized else 'NO POLARIZED (--nsp)'}")

    nkstot, nbnd, natomwfc, state_map, n_atoms, n_skipped_states, blocks = parse_file(in_path)

    if not blocks:
        print("No k-point blocks found. Check that the file contains "
              "'k =  ...' lines followed by '==== e( n ) = ... eV ====' lines.")
        sys.exit(1)

    result = write_output(
        out_path, nkstot, nbnd, natomwfc, state_map, n_atoms, blocks,
        include_orbitals=include_orbitals, polarized=polarized,
        include_band_table=include_band_table, only_significant=only_significant,)

    #print(f"Parsed nkstot = {nkstot}, nbnd = {nbnd}, natomwfc = {natomwfc}")
    #print(f"Atoms detected: {n_atoms}")
    if n_skipped_states:
        print(f"Warning: {n_skipped_states} states with unsupported (l, m) values "
              f"(e.g., g orbitals or higher) were ignored in the orbital tables.")
    #print(f"Total k-point blocks detected in file: {len(blocks)}")

    if polarized:
        n_up, n_down = result
        #print(f"SPIN UP blocks written: {n_up}")
        #print(f"SPIN DOWN blocks written: {n_down}")
    else:
        n_blocks, _ = result
        print(f"K-point blocks written (no spin split): {n_blocks}")

    #print(f"Output written to: {out_path}")

if __name__ == "__main__":
    main()
