#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2025-06

"""
Usage:
    python ccd.py [--qe] [--n-images] 
    
Generate a set of inputs along a linear configuration coordinate (CCD)
between a relaxed ground-state structure and a relaxed excited-state
structure, for both electronic configurations.

Default mode: VASP.
    Reads POSCAR_ground / POSCAR_excited, linearly interpolates positions
    and cell between them, and writes one POSCAR per lambda value into
    ground_state/<lambda>/POSCAR and excited_state/<lambda>/POSCAR. The
    two trees get an identical POSCAR per lambda -- what differs between
    a "ground" and "excited" VASP run is the INCAR (occupations/
    NUPDOWN/etc), not the POSCAR.

--qe mode: Quantum ESPRESSO.
    Reads two QE 'scf' input templates, ground_state.in and
    excited_state.in, that already contain everything needed to run a
    calculation (CONTROL, SYSTEM, ELECTRONS, ATOMIC_SPECIES, K_POINTS,
    CELL_PARAMETERS, ATOMIC_POSITIONS, and for the excited template
    only an OCCUPATIONS card). Builds the same lambda-interpolated
    geometries and writes one QE input per lambda into
    ground_configs/<lambda>/scf.in and excited_configs/<lambda>/scf.in,
    each using its own template's settings (SYSTEM/ELECTRONS/OCCUPATIONS)
    but the shared geometry.

    ATOMIC_POSITIONS may be 'crystal', 'angstrom' or 'bohr' (not
    'alat', see note below), and CELL_PARAMETERS may be 'angstrom' or
    'bohr'. Internally everything is converted to a common
    representation (cell in Angstrom, positions as fractional/crystal
    coordinates), the atomic displacement is corrected with the
    minimum-image convention and the result is converted back to whatever 
    units the templates used.
    Both templates must use the same units as each other; 'alat' is
    rejected because it depends on celldm(1)/A and becomes ambiguous
    once the cell itself is interpolated.
"""

import argparse
import os
import re
import numpy as np

def get_lambdas(n_images):
    """Generate lambda values from 0 to 1.
    Example:
        n_images = 9
        -> 0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000
    """
    if n_images < 2:
        raise ValueError("n_images must be >= 2")
    return np.linspace(0.0, 1.0, n_images)

# VASP utilities
def minimum_distance(atoms):
    """Calculate the minimum periodic interatomic distance using ASE.
    Returns:
        minimum distance in Angstrom.
    """
    distances = atoms.get_all_distances(mic=True)

    # Remove diagonal (atom-to-itself) distances
    distances = distances + np.eye(len(atoms)) * 1.0e6

    return np.min(distances)

def check_vasp_structures(R_g, R_e):
    """Check compatibility between ground and excited structures."""

    # Number of atoms
    if len(R_g) != len(R_e):
        raise ValueError("Ground and excited structures have different atom counts.")

    # Chemical species / atom ordering
    symbols_g = list(R_g.get_chemical_symbols())
    symbols_e = list(R_e.get_chemical_symbols())

    if symbols_g != symbols_e:
        raise ValueError("Atom ordering/species mismatch between ground and excited structures.\n"
                         "The atom order must be identical in both POSCAR files.")

    # Check cells
    if not np.allclose(R_g.cell.array, R_e.cell.array, atol=1.0e-8):
        raise ValueError("\nGround and excited cells are different.\n"
                         "This script assumes the same simulation cell for the two relaxed structures.\n"
                         "For a configuration-coordinate calculation, this is normally the desired setup.")

    print("\nVASP structure check")
    print("--------------------")
    print(f"Number of atoms       : {len(R_g)}")
    print(f"Chemical composition  : {R_g.get_chemical_formula()}")
    print("Atom ordering         : OK")
    #print("Simulation cell       : identical")
    print()

# innecesary function, it can be delete
def calculate_displacements(R_g, R_e):
    """Calculate the minimum-image displacement between ground and excited
    structures. 
    Returns:
        dfrac : Nx3 fractional-coordinate displacement
        dcart : Nx3 Cartesian displacement
    """
    frac_g = R_g.get_scaled_positions(wrap=False)
    frac_e = R_e.get_scaled_positions(wrap=False)

    # Fractional displacement
    dfrac = frac_e - frac_g

    # Apply minimum-image convention
    dfrac -= np.round(dfrac)

    # Convert to Cartesian
    dcart = np.dot(dfrac, R_g.cell.array)

    return dfrac, dcart

def run_vasp(n_images):
    """Generate interpolated VASP structures. The same geometry is used for the ground-state 
    and excited-state single-point calculations.
    """

    from ase.io import read, write
    from ase import Atoms

    ground_outdir = "ground_state"
    excited_outdir = "excited_state"
    
    # Read relaxed structures
    print("Reading relaxed structures...")

    R_g = read("POSCAR_ground", format="vasp")
    R_e = read("POSCAR_excited", format="vasp")

    # Check structures
    check_vasp_structures(R_g, R_e)

    # Get fractional coordinates
    frac_g = R_g.get_scaled_positions(wrap=False)
    frac_e = R_e.get_scaled_positions(wrap=False)

    # Minimum-image displacement
    dfrac = frac_e - frac_g
    dfrac -= np.round(dfrac)

    # Cartesian displacement
    dcart = np.dot(dfrac, R_g.cell.array)

    # Report total structural displacement
    displacement_squared = np.sum(np.sum(dcart ** 2, axis=1))

    total_displacement = np.sqrt(displacement_squared)

    max_displacement = np.max(np.sqrt(np.sum(dcart ** 2, axis=1)))

    atom_max = np.argmax(np.sqrt(np.sum(dcart ** 2, axis=1))) + 1

    #print("Structural displacement")
    #print("-----------------------")
    #print(f"Mass-independent RMS displacement  : {np.sqrt(np.mean(np.sum(dcart**2, axis=1))):.6f} Å")
    #print(f"Maximum atomic displacement        : {max_displacement:.6f} Å")
    #print(f"Atom with maximum displacement     : {atom_max}")
    #print(f"Total geometric displacement       : {total_displacement:.6f} Å")
    #print()

    # Create output directories
    os.makedirs(ground_outdir, exist_ok=True)
    os.makedirs(excited_outdir, exist_ok=True)

    # Lambda values
    lambdas = get_lambdas(n_images)

    print("Generating interpolated structures")
    print("-" * 48)
    print(f"{'lambda':>10} {'min distance (Å)':>20} {'status':>12}")
    print("-" * 48)

    problematic = []
    
    # Interpolation
    for lam in lambdas:

        # Minimum-image fractional interpolation
        frac_i = frac_g + lam * dfrac

        # Create interpolated structure
        img = R_g.copy()

        # Keep original cell
        img.set_cell(R_g.cell)

        # Set interpolated fractional coordinates
        img.set_scaled_positions(frac_i)

        # Wrap coordinates into unit cell
        img.wrap()

        # Minimum periodic distance
        min_dist = minimum_distance(img)

        if min_dist < 1.0:
            status = "WARNING"
            problematic.append((lam, min_dist))
        elif min_dist < 1.2:
            status = "CHECK"
            problematic.append((lam, min_dist))
        else:
            status = "OK"

        print(f"{lam:10.3f} {min_dist:20.6f} {status:>12}")

        # Directory names
        lam_str = f"{lam:.3f}"

        gdir = os.path.join(ground_outdir, lam_str)

        edir = os.path.join(excited_outdir, lam_str)

        os.makedirs(gdir, exist_ok=True)
        os.makedirs(edir, exist_ok=True)

        # POSCAR paths
        gpath = os.path.join(gdir, "POSCAR")
        epath = os.path.join(edir, "POSCAR")

        # Write POSCAR
        write(gpath, img, format="vasp", direct=True, sort=False)
        write(epath, img, format="vasp", direct=True, sort=False)

    # Final report
    print()
    print("-" * 48)
    print("Interpolation completed")
    print("-" * 48)

    print(f"Number of images : {n_images}")
    print(f"Ground directory : {ground_outdir}/")
    print(f"Excited directory: {excited_outdir}/")

    if problematic:
        print()
        print("WARNING: Some images have short interatomic distances:")
        print()

        for lam, dist in problematic:
            print(f"lambda = {lam:.3f}, minimum distance = {dist:.6f} Å")
        print()
        print("Inspect these POSCAR files before running VASP.")
    else:
        print()
        print("No unusually short interatomic distances were detected.")
    print()

# QE utilities
def get_nat(text):
    """Extract nat from a QE input"""
    m = re.search(r"nat\s*=\s*(\d+)", text, re.IGNORECASE)

    if not m:
        raise ValueError("Could not find 'nat' in &SYSTEM")
        
    return int(m.group(1))

def parse_cell(text):
    """Return:
        cell matrix
        exact CELL_PARAMETERS header
    """
    lines = text.splitlines()

    for i, line in enumerate(lines):
        if line.strip().upper().startswith("CELL_PARAMETERS"):
            header = line.strip()
            vecs = []
            for j in range(i + 1, i + 4):
                vecs.append([float(x) for x in lines[j].split()])

            return np.array(vecs), header

    raise ValueError("CELL_PARAMETERS block not found")

def parse_positions(text, nat):
    """Return:
        species : list of nat chemical symbols
        coords  : Nx3 array of RAW coordinates, exactly as written in
                  the template (units depend on the header -- see
                  parse_positions_units)
        extras  : list of nat lists with any extra trailing tokens per
                  atom (e.g. if_pos constraint flags such as "0 0 1").
                  Preserved verbatim so constraints are not silently
                  dropped when the file is rewritten.
        header  : exact ATOMIC_POSITIONS header line
        start   : index of the header line
        end     : index of the line after the last atom
    """
    lines = text.splitlines()
    
    idx = next(i for i, l in enumerate(lines) if l.strip().upper().startswith("ATOMIC_POSITIONS"))

    header = lines[idx].strip()

    species = []
    coords = []
    extras = []

    for j in range(idx + 1, idx + 1 + nat):
        parts = lines[j].split()
        species.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
        extras.append(parts[4:])

    return (species, np.array(coords), extras, header, idx, idx + 1 + nat)

# ----------------------------------------------------------------------
# Units handling
#
# QE lets ATOMIC_POSITIONS be 'crystal' (fractional), 'angstrom',
# 'bohr' or 'alat', and CELL_PARAMETERS be 'angstrom', 'bohr' or
# 'alat'. To interpolate correctly and apply the minimum-image
# convention regardless of which convention the templates use, both
# structures are converted internally to a single common
# representation: cell in Angstrom, positions as fractional
# coordinates. 'alat' is rejected: it depends on celldm(1)/A from
# &SYSTEM, and its meaning becomes ambiguous once the cell itself is
# being interpolated, so it's safer to ask for an explicit template.
# ----------------------------------------------------------------------

BOHR_TO_ANG = 0.52917721067

def parse_positions_units(header):
    """Extract the coordinate unit from an ATOMIC_POSITIONS header.
    Supported: 'crystal', 'angstrom', 'bohr'."""
    m = re.search(r"\(?\s*(crystal|angstrom|bohr|alat)\s*\)?", header, re.IGNORECASE)

    if not m:
        raise ValueError(
            f"Could not find explicit units in '{header}'.\n"
            "This script requires ATOMIC_POSITIONS to explicitly state "
            "'crystal', 'angstrom' or 'bohr'.")

    unit = m.group(1).lower()

    if unit == "alat":
        raise ValueError(
            "ATOMIC_POSITIONS alat is not supported: alat depends on "
            "celldm(1)/A, which becomes ambiguous once the cell is "
            "interpolated. Please rewrite your templates using "
            "'ATOMIC_POSITIONS crystal' or 'ATOMIC_POSITIONS angstrom'.")

    return unit

def parse_cell_units(header):
    """Extract the length unit from a CELL_PARAMETERS header.
    Supported: 'angstrom', 'bohr'."""
    m = re.search(r"\(?\s*(angstrom|bohr|alat)\s*\)?", header, re.IGNORECASE)

    if not m:
        raise ValueError(
            f"Could not find explicit units in '{header}'.\n"
            "This script requires CELL_PARAMETERS to explicitly state "
            "'angstrom' or 'bohr'.")

    unit = m.group(1).lower()

    if unit == "alat":
        raise ValueError(
            "CELL_PARAMETERS alat is not supported here. Please rewrite "
            "your templates using 'CELL_PARAMETERS angstrom' or "
            "'CELL_PARAMETERS bohr'.")

    return unit

def cell_to_angstrom(cell, unit):
    """Convert a raw CELL_PARAMETERS matrix to Angstrom."""
    if unit == "angstrom":
        return cell.copy()
    elif unit == "bohr":
        return cell * BOHR_TO_ANG

    raise ValueError(f"Unsupported cell unit: {unit}")

def cell_from_angstrom(cell_ang, unit):
    """Convert a cell matrix in Angstrom back to the template's unit."""
    if unit == "angstrom":
        return cell_ang.copy()
    elif unit == "bohr":
        return cell_ang / BOHR_TO_ANG

    raise ValueError(f"Unsupported cell unit: {unit}")

def positions_to_fractional(coords, unit, cell_ang):
    """
    Convert RAW atomic coordinates (as written in the template) to
    fractional/crystal coordinates, using the cell in Angstrom.
    Works whether the template uses cartesian (angstrom/bohr) or
    already-fractional (crystal) positions.
    """
    if unit == "crystal":
        return coords.copy()

    if unit == "angstrom":
        cart_ang = coords
    elif unit == "bohr":
        cart_ang = coords * BOHR_TO_ANG
    else:
        raise ValueError(f"Unsupported positions unit: {unit}")

    # cartesian (Angstrom) -> fractional
    return np.dot(cart_ang, np.linalg.inv(cell_ang))

def fractional_to_positions(frac, unit, cell_ang):
    """
    Convert fractional coordinates back to whatever unit/format the
    template originally used (inverse of positions_to_fractional).
    """
    if unit == "crystal":
        return frac.copy()

    cart_ang = np.dot(frac, cell_ang)

    if unit == "angstrom":
        return cart_ang
    elif unit == "bohr":
        return cart_ang / BOHR_TO_ANG

    raise ValueError(f"Unsupported positions unit: {unit}")

def minimum_distance_fractional(frac, cell_ang):
    """
    Minimum periodic interatomic distance (Angstrom), computed directly
    from fractional coordinates + cell with the minimum-image
    convention. Pure numpy -- no ASE dependency, so the --qe branch
    works even without ASE installed.
    """
    diff = frac[:, None, :] - frac[None, :, :]
    diff -= np.round(diff)

    cart = np.tensordot(diff, cell_ang, axes=([2], [0]))
    dist = np.sqrt(np.sum(cart ** 2, axis=-1))

    np.fill_diagonal(dist, np.inf)

    return np.min(dist)

def check_qe_structures(sp_g, sp_e, pos_unit_g, pos_unit_e, cell_unit_g, cell_unit_e):
    """
    Check compatibility between ground and excited QE templates:
    atom ordering/species, and that both use the same units --
    otherwise the interpolation would silently mix incompatible
    coordinate systems.
    """
    if sp_g != sp_e:
        raise ValueError(
            "Atom ordering/species mismatch between ground and excited "
            "QE templates.\n"
            "The atom order must be identical in both templates.")

    if pos_unit_g != pos_unit_e:
        raise ValueError(
            "ATOMIC_POSITIONS units differ between templates "
            f"('{pos_unit_g}' vs '{pos_unit_e}').\n"
            "Use the same units in both ground_state.in and "
            "excited_state.in.")

    if cell_unit_g != cell_unit_e:
        raise ValueError(
            "CELL_PARAMETERS units differ between templates "
            f"('{cell_unit_g}' vs '{cell_unit_e}').\n"
            "Use the same units in both templates.")

    print("\nQE structure check")
    print("-------------------")
    print(f"Number of atoms       : {len(sp_g)}")
    print("Atom ordering         : OK")
    print(f"Positions units       : {pos_unit_g}")
    print(f"Cell units            : {cell_unit_g}")
    print()

def set_param(text, key, value):
    """Replace a scalar namelist parameter"""
    pattern = re.compile(r"(^\s*" + re.escape(key) + r"\s*=\s*).*?(,?\s*)$", re.MULTILINE | re.IGNORECASE)

    if isinstance(value, str):
        repl = r"\g<1>'" + value + r"'\g<2>"
    else:
        repl = r"\g<1>" + str(value) + r"\g<2>"

    new_text, n = pattern.subn(repl, text, count=1)

    if n == 0:
        raise ValueError(f"Parameter '{key}' not found to replace")

    return new_text

def build_image(template_text, nat, species, new_coords, new_cell, extras=None):
    """
    Construct a QE input using interpolated coordinates and cell.

    new_coords must already be expressed in the same unit/format as
    the template's ATOMIC_POSITIONS card (crystal, angstrom or bohr --
    see positions_to_fractional / fractional_to_positions), and
    new_cell in the same unit as the template's CELL_PARAMETERS card.

    extras (optional): list of nat lists with trailing tokens per atom
    (e.g. if_pos flags) to preserve; if omitted, nothing is appended.
    """

    if extras is None:
        extras = [[] for _ in range(nat)]

    lines = template_text.splitlines()

    # Replace ATOMIC_POSITIONS
    _, _, _, pos_header, start, end = parse_positions(template_text, nat)

    new_pos_lines = []

    for sp, c, ex in zip(species, new_coords, extras):

        line = (f"{sp:<3s} {c[0]: .10f} {c[1]: .10f} {c[2]: .10f}")

        if ex:
            line += " " + " ".join(ex)

        new_pos_lines.append(line)
    
    lines = lines[:start] + [pos_header] + new_pos_lines + lines[end:]

    text = "\n".join(lines) + "\n"

    # Replace CELL_PARAMETERS
    _, cell_header = parse_cell(text)

    clines = text.splitlines()
    
    ci = next(i for i, l in enumerate(clines) if l.strip().upper().startswith("CELL_PARAMETERS"))

    new_cell_lines = []

    for row in new_cell:
        new_cell_lines.append(f"  {row[0]: .10f}  {row[1]: .10f}  {row[2]: .10f}")
    clines = clines[:ci] + [cell_header] + new_cell_lines + clines[ci + 4:]    

    text = "\n".join(clines) + "\n"

    m = re.search( r"prefix\s*=\s*'([^']+)'", text, re.IGNORECASE)
    base_prefix = m.group(1) if m else "calc"

    text = set_param(text, "prefix", base_prefix)

    # Preserve outdir
    m = re.search(r"outdir\s*=\s*'([^']+)'", text, re.IGNORECASE)
    base_outdir = m.group(1).rstrip("/") if m else "../tmp"
    text = set_param(text, "outdir", base_outdir)

    return text

def run_qe(n_images):
    """
    Generate interpolated QE inputs.

    Both templates are converted internally to a common representation
    (cell in Angstrom, atomic positions as fractional coordinates), so
    the interpolation works regardless of whether ATOMIC_POSITIONS is
    'crystal', 'angstrom' or 'bohr', and CELL_PARAMETERS is 'angstrom'
    or 'bohr'. The minimum-image convention is applied to the
    fractional displacement -- same idea as the VASP branch -- so an
    atom near a cell boundary interpolates along the short physical
    path instead of wrapping the long way around the cell.
    """

    ground_outdir = "ground_state"
    excited_outdir = "excited_state"

    # Read QE templates
    print("Reading QE templates...")

    with open("ground_state.in") as f:
        ground_text = f.read()

    with open("excited_state.in") as f:
        excited_text = f.read()

    # Number of atoms
    nat_g = get_nat(ground_text)
    nat_e = get_nat(excited_text)

    if nat_g != nat_e:
        raise ValueError("Ground and excited templates have different nat.")

    nat = nat_g

    # Atomic positions (raw, exactly as written in the templates)
    sp_g, R_g_raw, extras_g, pos_header_g, _, _ = parse_positions(ground_text, nat)
    sp_e, R_e_raw, extras_e, pos_header_e, _, _ = parse_positions(excited_text, nat)

    pos_unit_g = parse_positions_units(pos_header_g)
    pos_unit_e = parse_positions_units(pos_header_e)

    # Cells (raw, exactly as written in the templates)
    cell_g_raw, cell_header_g = parse_cell(ground_text)
    cell_e_raw, cell_header_e = parse_cell(excited_text)

    cell_unit_g = parse_cell_units(cell_header_g)
    cell_unit_e = parse_cell_units(cell_header_e)

    # Check structures / units compatibility
    check_qe_structures(sp_g, sp_e, pos_unit_g, pos_unit_e, cell_unit_g, cell_unit_e)

    pos_unit = pos_unit_g
    cell_unit = cell_unit_g

    # Convert to a common internal representation:
    # cell -> Angstrom, positions -> fractional (crystal)
    cell_g_ang = cell_to_angstrom(cell_g_raw, cell_unit)
    cell_e_ang = cell_to_angstrom(cell_e_raw, cell_unit)

    frac_g = positions_to_fractional(R_g_raw, pos_unit, cell_g_ang)
    frac_e = positions_to_fractional(R_e_raw, pos_unit, cell_e_ang)

    # Minimum-image displacement (same idea as the VASP branch)
    dfrac = frac_e - frac_g
    dfrac -= np.round(dfrac)

    dcart = np.dot(dfrac, cell_g_ang)

    per_atom = np.sqrt(np.sum(dcart ** 2, axis=1))
    total_displacement = np.sqrt(np.sum(dcart ** 2))
    max_displacement = np.max(per_atom)
    atom_max = np.argmax(per_atom) + 1

    #print("Structural displacement")
    #print("-----------------------")
    #print(f"Mass-independent RMS displacement  : {np.sqrt(np.mean(per_atom ** 2)):.6f} Å")
    #print(f"Maximum atomic displacement        : {max_displacement:.6f} Å")
    #print(f"Atom with maximum displacement     : {atom_max}")
    #print(f"Total geometric displacement       : {total_displacement:.6f} Å")
    #print()

    # Create output directories
    os.makedirs(ground_outdir, exist_ok=True)
    os.makedirs(excited_outdir, exist_ok=True)

    # Lambda values
    lambdas = get_lambdas(n_images)

    print("Generating interpolated structures")
    print("-" * 48)
    print(f"{'lambda':>10} {'min distance (Å)':>20} {'status':>12}")
    print("-" * 48)

    problematic = []

    # Interpolation
    for lam in lambdas:
        # Minimum-image fractional interpolation + cell interpolation
        frac_i = frac_g + lam * dfrac
        cell_i_ang = (1 - lam) * cell_g_ang + lam * cell_e_ang

        # Minimum periodic distance (pure numpy, no ASE needed here)
        min_dist = minimum_distance_fractional(frac_i, cell_i_ang)

        if min_dist < 1.0:
            status = "WARNING"
            problematic.append((lam, min_dist))
        elif min_dist < 1.2:
            status = "CHECK"
            problematic.append((lam, min_dist))
        else:
            status = "OK"

        print(f"{lam:10.3f} {min_dist:20.6f} {status:>12}")

        # Convert back to the units/format used in the templates
        coords_i = fractional_to_positions(frac_i, pos_unit, cell_i_ang)
        cell_i_raw = cell_from_angstrom(cell_i_ang, cell_unit)

        ground_out = build_image(ground_text, nat, sp_g, coords_i, cell_i_raw, extras_g)
        excited_out = build_image(excited_text, nat, sp_e, coords_i, cell_i_raw, extras_e)

        # Directory names
        lam_str = f"{lam:.3f}"

        gdir = os.path.join(ground_outdir, lam_str)
        edir = os.path.join(excited_outdir, lam_str)

        os.makedirs(gdir, exist_ok=True)
        os.makedirs(edir, exist_ok=True)

        gpath = os.path.join(gdir, "scf.in")
        epath = os.path.join(edir, "scf.in")

        with open(gpath, "w") as f:
            f.write(ground_out)

        with open(epath, "w") as f:
            f.write(excited_out)

    # Final report
    print()
    print("-" * 48)
    print("Interpolation completed")
    print("-" * 48)
    print(f"Number of images : {n_images}")
    print(f"Ground directory : {ground_outdir}/")
    print(f"Excited directory: {excited_outdir}/")

    if problematic:
        print()
        print("WARNING: Some images have short interatomic distances:")
        print()

        for lam, dist in problematic:
            print(f"lambda = {lam:.3f}, minimum distance = {dist:.6f} Å")
        print()
        print("Inspect these scf.in files before running Quantum ESPRESSO.")
    else:
        print()
        print("No unusually short interatomic distances were detected.")
    print()

def parse_args():
    parser = argparse.ArgumentParser(
        description=("Generate configuration-coordinate interpolated structures between "
                     "ground- and excited-state geometries."))
    parser.add_argument("--qe", action="store_true", help=("Use Quantum ESPRESSO mode instead "
                        "of VASP mode."))
    parser.add_argument("--n-images", type=int, default=9, help=("Number of images between lambda=0 "
                        "and lambda=1. Default: 9."))
    return parser.parse_args()

def main():
    args = parse_args()
    if args.qe:
        run_qe(args.n_images)
    else:
        run_vasp(args.n_images)

if __name__ == "__main__":
    main()
