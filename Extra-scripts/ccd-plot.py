#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2025-06

"""
Usage:
      python3 ccd-plot.py [--qe]            
      
Configuration Coordinate Diagram (CCD) builder and Huang-Rhys analysis.

Note on Stokes / anti-Stokes naming (see e.g. Huang, Ke & Lei,
J. Appl. Phys. 137, 134303 (2025), Fig. 5):
    Stokes shift (Delta S)       = relaxation energy on the EXCITED
                                    branch (dE_excited)
    anti-Stokes shift (Delta AS) = relaxation energy on the GROUND
                                    branch (dE_ground)                                    
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

parser = argparse.ArgumentParser(
    description="Build a configuration coordinate diagram (CCD) and "
                 "Huang-Rhys analysis from VASP or Quantum ESPRESSO energies."
)
parser.add_argument(
    "--qe", action="store_true",
    help="Use the Quantum ESPRESSO inputs/energies instead of VASP (default).",
)
args = parser.parse_args()

# Relaxed structures. Filenames/format switch automatically with --qe.
if args.qe:
    # QE input files (ATOMIC_POSITIONS + CELL_PARAMETERS cards),
    # read with ASE's 'espresso-in' format.
    GROUND_STRUCT = "ground_state.in"
    EXCITED_STRUCT = "excited_state.in"
    READ_FORMAT = "espresso-in"
else:
    # VASP POSCAR files, read with ASE's 'vasp' format.
    GROUND_STRUCT = "POSCAR_ground"
    EXCITED_STRUCT = "POSCAR_excited"
    READ_FORMAT = "vasp"

RY_TO_EV = 13.605703976

GROUND_DIR = "ground_state"
EXCITED_DIR = "excited_state"

# VASP extraction
VASP_ENERGY_PATTERN = re.compile(r'<i name="e_wo_entrp">\s*([-+]?\d*\.\d+)\s*</i>')

def extract_vasp_energy(folder: str) -> float:
    """Return the last e_wo_entrp value found in <folder>/vasprun.xml, in eV."""
    vasprun_path = os.path.join(folder, "vasprun.xml")
    if not os.path.isfile(vasprun_path):
        raise FileNotFoundError(f"vasprun.xml not found in {folder}")
    last_value = None
    with open(vasprun_path, "r", encoding="utf-8") as f:
        for line in f:
            match = VASP_ENERGY_PATTERN.search(line)
            if match:
                last_value = match.group(1)
    if last_value is None:
        raise ValueError(f"Total energy not found in {vasprun_path}")
    return float(last_value)

# QE extraction
QE_SLURM_PATTERN = re.compile(r"^slurm-\d+\.out$")
QE_DOUBLE_BANG_PATTERN = re.compile(r"^\s*!!\s*total energy\s*=")
QE_SINGLE_BANG_PATTERN = re.compile(r"^\s*!\s*total energy\s*=")
QE_ENERGY_VALUE_PATTERN = re.compile(r"total energy\s*=\s*(-?\d+\.\d+)\s*(\w+)")

def extract_qe_energy(folder: str) -> float:
    """Return the last total energy (preferring the final '!!' line over
    an SCF-step '!' line) found in the QE .out file inside <folder>,
    in Rydberg."""
    out_files = sorted(
        f for f in glob.glob(os.path.join(folder, "*.out"))
        if not QE_SLURM_PATTERN.match(os.path.basename(f))
    )
    if not out_files:
        raise FileNotFoundError(f"No QE .out file found in {folder}")

    last_energy_single = None
    last_energy_double = None
    with open(out_files[0], "r", encoding="utf-8") as f:
        for line in f:
            if QE_DOUBLE_BANG_PATTERN.match(line):
                match = QE_ENERGY_VALUE_PATTERN.search(line)
                if match:
                    last_energy_double = match.group(1)
            elif QE_SINGLE_BANG_PATTERN.match(line):
                match = QE_ENERGY_VALUE_PATTERN.search(line)
                if match:
                    last_energy_single = match.group(1)

    energy = last_energy_double if last_energy_double is not None else last_energy_single
    if energy is None:
        raise ValueError(f"Total energy not found in {out_files[0]}")
    return float(energy)

def get_lambda_folders(base_dir: str) -> list[str]:
    """Return the lambda-value subfolder names of base_dir (e.g. '0.000',
    '0.125', ...), sorted numerically from smallest to largest."""
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    lam_folders = []
    for entry in os.listdir(base_dir):
        if not os.path.isdir(os.path.join(base_dir, entry)):
            continue
        try:
            float(entry)
        except ValueError:
            continue
        lam_folders.append(entry)
    if not lam_folders:
        raise FileNotFoundError(f"No lambda-value subfolders found in {base_dir}")
    lam_folders.sort(key=float)
    return lam_folders

def get_branch_energies(base_dir: str, lam_folders: list[str], use_qe: bool) -> list[float]:
    """Extract the total energy for each lambda subfolder of base_dir."""
    extractor = extract_qe_energy if use_qe else extract_vasp_energy
    return [extractor(os.path.join(base_dir, lam)) for lam in lam_folders]

lam_folder_names = get_lambda_folders(GROUND_DIR)
lam_branch = [float(lam) for lam in lam_folder_names]

E_ground_branch = get_branch_energies(GROUND_DIR, lam_folder_names, args.qe)
E_excited_branch = get_branch_energies(EXCITED_DIR, lam_folder_names, args.qe)

if args.qe:
    E_ground_branch = [E * RY_TO_EV for E in E_ground_branch]
    E_excited_branch = [E * RY_TO_EV for E in E_excited_branch]

E_g_Qg, E_g_Qe = E_ground_branch[0], E_ground_branch[-1]
E_e_Qg, E_e_Qe = E_excited_branch[0], E_excited_branch[-1]

# ---------------------------------------------------------------- #
# 1. Mass-weighted displacement, dQ
# ---------------------------------------------------------------- #

atoms_g = read(GROUND_STRUCT, format=READ_FORMAT)
atoms_e = read(EXCITED_STRUCT, format=READ_FORMAT)

if len(atoms_g) != len(atoms_e):
    raise ValueError("Ground and excited structures have different atom counts.")

if list(atoms_g.get_chemical_symbols()) != list(atoms_e.get_chemical_symbols()):
    raise ValueError("Atom ordering/species mismatch between ground and excited structures.\n"
                      "The atom order must be identical in both structure files.")

if not np.allclose(atoms_g.cell.array, atoms_e.cell.array, atol=1.0e-6):
    raise ValueError("Ground and excited cells differ. This script assumes the same "
                      "simulation cell for both relaxed structures.")

frac_g = atoms_g.get_scaled_positions(wrap=False)
frac_e = atoms_e.get_scaled_positions(wrap=False)

dfrac = frac_e - frac_g
dfrac -= np.round(dfrac)                     # minimum-image convention

disp = np.dot(dfrac, atoms_g.cell.array)     # Cartesian displacement, Angstrom
masses = atoms_g.get_masses()                # amu

per_atom_disp = np.sqrt(np.sum(disp**2, axis=1))
max_disp = np.max(per_atom_disp)
atom_max = np.argmax(per_atom_disp) + 1

dQ2 = np.sum(masses[:, None] * disp**2)
dQ = np.sqrt(dQ2)                                           # amu^1/2 . Angstrom

print(f"ΔQ = {dQ:.4f} amu^(1/2)*Angstrom")
#print(f"Maximum atomic displacement    = {max_disp:.6f} Å (atom {atom_max})")
if max_disp > 1.0:
    print("WARNING: a large single-atom displacement was found, double check "
          "this atom didn't just get wrapped across a periodic boundary.")
          
# ---------------------------------------------------------------- #
# 2. Relaxation energies (two-point method)
# ---------------------------------------------------------------- #

dE_ground = E_g_Qe - E_g_Qg   # ground state relaxes going Qe -> Qg  == anti-Stokes shift
dE_excited = E_e_Qg - E_e_Qe  # excited state relaxes going Qg -> Qe == Stokes shift

stokes_shift = dE_excited
anti_stokes_shift = dE_ground

print("")
print("---------------------------------------")
print("Relaxation energy")
print("---------------------------------------")
print(f"Anti-Stokes shift (ground)  = {dE_ground:.4f} eV")
print(f"Stokes shift (excited) = {dE_excited:.4f} eV")

# ---------------------------------------------------------------- #
# 3. Effective phonon frequencies, Huang-Rhys factors, and
#    Debye-Waller factors
#
#    hbar*omega_i = hbar * sqrt(2*dE_i) / dQ
#    S_i          = dE_i / (hbar*omega_i)
#    DW_i         = exp(-S_i)
#
# eV . amu^-1/2 . Angstrom^-1 -> meV conversion factor below is the
# standard prefactor used in the defect-CCD literature (e.g.
# Alkauskas et al., New J. Phys. 16, 073026 (2014)); double check
# against a published example before trusting the numeric value.
# ---------------------------------------------------------------- #

MEV_PER_UNIT = 64.654148  # meV per sqrt(eV) / (amu^1/2 . Angstrom)

def effective_frequency_meV(dE, dQ):
    """hbar*omega in meV from a relaxation energy (eV) and dQ."""
    return MEV_PER_UNIT * np.sqrt(2.0 * dE) / dQ

def huang_rhys(dE_eV, homega_meV):
    """Dimensionless Huang-Rhys factor. Same energy units, no conversion needed."""
    return (dE_eV * 1000.0) / homega_meV

def debye_waller_factor(S):
    """
    Debye-Waller factor W = exp(-S), the fraction of the total
    transition intensity carried by the zero-phonon line (T = 0 K,
    single effective mode approximation). S is the Huang-Rhys factor.
    """
    return np.exp(-S)

# two-point estimate, used unless a parabola fit overrides it
homega_g = effective_frequency_meV(dE_ground, dQ)
homega_e = effective_frequency_meV(dE_excited, dQ)

def fit_parabola_frequency(lam_list, E_list, lam_min, dQ):
    """
    Fit E = E0 + 0.5*k*(lam - lam_min)^2 * dQ^2 to interpolated points
    and return hbar*omega in meV. Falls back to None if fewer than 3
    points are provided.
    """
    if len(lam_list) < 3:
        return None
    lam = np.array(lam_list)
    E = np.array(E_list)
    Q = lam * dQ  # approximate: assumes linear interpolation in Q
    coeffs = np.polyfit(Q, E, 2)          # E = a*Q^2 + b*Q + c
    a = coeffs[0]                          # a = 0.5 * k  (eV / (amu.Angstrom^2))
    if a <= 0:
        return None
    k_eff = 2 * a
    # hbar*omega = hbar*sqrt(k_eff) in the same mass-weighted units
    return MEV_PER_UNIT * np.sqrt(k_eff)

fit_g = fit_parabola_frequency(lam_branch, E_ground_branch, 0.0, dQ)
fit_e = fit_parabola_frequency(lam_branch, E_excited_branch, 1.0, dQ)

if fit_g is not None:
    homega_g = fit_g
    #print("Using parabola fit for ground branch frequency.")
if fit_e is not None:
    homega_e = fit_e
    #print("Using parabola fit for excited branch frequency.")

S_g = huang_rhys(dE_ground, homega_g)
S_e = huang_rhys(dE_excited, homega_e)

DW_g = debye_waller_factor(S_g)
DW_e = debye_waller_factor(S_e)

print("")
print("---------------------------------------")
print("Effective phonon modes (frecuencies)")
print("---------------------------------------")
print(f"ℏω (ground) = {homega_g:.2f} meV")
print(f"ℏω (excited) = {homega_e:.2f} meV")
print("")
print("---------------------------------------")
print("Huang-Rhys factor")
print("---------------------------------------")
print(f"S (ground) = {S_g:.3f}")
print(f"S (excited) = {S_e:.3f}")
print("")
print("---------------------------------------")
print("Debye-Waller factor")
print("---------------------------------------")
print(f"D (ground) = {DW_g:.4f}")
print(f"D (excited) = {DW_e:.4f}")

zpl = E_e_Qe - E_g_Qg
print("")
print("---------------------------------------")
print("Zero Phonon Line (ZPL)")
print("---------------------------------------")
print(f"ZPL = {zpl:.4f} eV")
print("")
print("---------------------------------------")
print("Absorption and emission energy")
print("---------------------------------------")
print(f"E (absorption) = {E_e_Qg - E_g_Qg:.4f} eV")
print(f"E (emission) = {E_e_Qe - E_g_Qe:.4f} eV")

# ---------------------------------------------------------------- #
# 4. Plot the CCD
# ---------------------------------------------------------------- #

Q_g_min = 0.0
Q_e_min = dQ

k_g = (homega_g / MEV_PER_UNIT) ** 2   
k_e = (homega_e / MEV_PER_UNIT) ** 2

Q = np.linspace(-2.6 * dQ, 4.6 * dQ, 400)
E_ground_curve = E_g_Qg + 0.5 * k_g * (Q - Q_g_min) ** 2
E_excited_curve = E_e_Qe + 0.5 * k_e * (Q - Q_e_min) ** 2

fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(Q, E_ground_curve, color="xkcd:blue", label="Ground state")
ax.plot(Q, E_excited_curve, color="xkcd:orange", label="Excited state")

if len(lam_branch) > 0:
    Q_ground_pts = np.array(lam_branch) * dQ
    ax.plot(Q_ground_pts, E_ground_branch, "o", color="xkcd:blue",
            markerfacecolor="white", markersize=5)#, label="Ground state (data)")

    Q_excited_pts = np.array(lam_branch) * dQ
    ax.plot(Q_excited_pts, E_excited_branch, "o", color="xkcd:orange",
            markerfacecolor="white", markersize=5)#, label="Excited state (data)")

ax.plot(Q_g_min, E_g_Qg, "o", color="xkcd:blue")
ax.plot(Q_e_min, E_e_Qe, "o", color="xkcd:orange")

# Absorption: vertical line at Q_g_min from ground min to excited curve
E_abs_top = E_e_Qe + 0.5 * k_e * (Q_g_min - Q_e_min) ** 2
ax.annotate(
    "", xy=(Q_g_min, E_abs_top), xytext=(Q_g_min, E_g_Qg),
    arrowprops=dict(arrowstyle="->", color="xkcd:red", lw=1.5),
)
ax.text(Q_g_min, (E_abs_top + E_g_Qg) / 2, rf"E$_{{abs}}$ = {E_e_Qg - E_g_Qg:.2f} eV", rotation=90, va="center", ha="right", color="xkcd:black",)

# Emission: vertical line at Q_e_min from excited min to ground curve
E_em_bottom = E_g_Qg + 0.5 * k_g * (Q_e_min - Q_g_min) ** 2
ax.annotate(
    "", xy=(Q_e_min, E_em_bottom), xytext=(Q_e_min, E_e_Qe),
    arrowprops=dict(arrowstyle="->", color="xkcd:green", lw=1.5),
)
ax.text(Q_e_min, (E_em_bottom + E_e_Qe) / 2, rf"E$_{{em}}$ = {E_e_Qe - E_g_Qe:.2f} eV", rotation=-90, va="center", ha="left", color="xkcd:black",)

# ZPL
ax.plot([-0.5*dQ, 4*dQ], [E_e_Qe, E_e_Qe], color="xkcd:black", lw=0.8, linestyle="--")
ax.plot([-0.5*dQ, 4*dQ], [E_g_Qg, E_g_Qg], color="xkcd:black", lw=0.8, linestyle="--")
ax.annotate(
    "", xy=(3.8*dQ, E_g_Qg), xytext=(3.8*dQ, E_e_Qe),
    arrowprops=dict(arrowstyle="<->", color="xkcd:purple", lw=1.5),
)
ax.text(3.8*dQ, (E_g_Qg + E_e_Qe) / 2, rf"E$_{{ZPL}}$ = {zpl:.2f} eV", rotation=90, va="center", ha="right", color="xkcd:black",)

ax.set_xlabel(r"Configuration coordinate $Q$ (amu$^{1/2}$ $\AA$)", fontsize=14)
ax.set_ylabel("Total energy (eV)", fontsize=14)
#ax.set_title("Configuration coordinate diagram")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("ccd.png", dpi=150)
#print("Saved plot to ccd.png")

# ---------------------------------------------------------------- #
# 5. Save results to ccd.dat
# ---------------------------------------------------------------- #

with open("ccd.dat", "w") as f:
    f.write("Configuration Coordinate Diagram\n")
    f.write("\n")
    f.write(f"ΔQ = {dQ:.6f} amu^(1/2)*Angstrom\n")
    f.write("\n")
    f.write("-----------------------------------------\n")
    f.write(f"Zero Phonon Line (ZPL)\n")
    f.write("-----------------------------------------\n")
    f.write(f"ZPL = {zpl:.6f} eV\n")
    f.write("\n")
    f.write("-----------------------------------------\n")
    f.write(f"Absorption and emission energy\n")
    f.write("-----------------------------------------\n")
    f.write(f"E (absorption) = {E_e_Qg - E_g_Qg:.6f} eV\n")
    f.write(f"E (emission) = {E_e_Qe - E_g_Qe:.6f} eV\n")
    f.write("\n")
    f.write("-----------------------------------------\n")
    f.write(f"Relaxation energy\n")
    f.write("-----------------------------------------\n")
    #f.write(f"Relaxation_energy_ground_eV  {dE_ground:.6f}\n")
    #f.write(f"Relaxation_energy_excited_eV {dE_excited:.6f}\n")
    f.write(f"Anti-Stokes shift (ground) = {anti_stokes_shift:.6f} eV\n")
    f.write(f"Stokes shift (excited) = {stokes_shift:.6f} eV\n")
    f.write("\n")
    f.write("-----------------------------------------\n")
    f.write(f"Effective phonon modes (frecuencies)\n") # ℏω is the energy quantum of a mode, while ω or Ω is the angular frequency of the effective mode
    f.write("-----------------------------------------\n")
    f.write(f"ℏω (ground) = {homega_g:.4f} meV\n")
    f.write(f"ℏω (excited) = {homega_e:.4f} meV\n")
    f.write("\n")
    f.write("-----------------------------------------\n")
    f.write(f"Huang-Rhys factor\n")
    f.write("-----------------------------------------\n")   
    f.write(f"S (ground) = {S_g:.4f}\n")
    f.write(f"S (excited) = {S_e:.4f}\n")
    f.write("\n")
    f.write("-----------------------------------------\n")
    f.write(f"Debye-Waller factor\n") # e^(-S)
    f.write("-----------------------------------------\n")   
    f.write(f"D (ground) = {DW_g:.6f}\n")
    f.write(f"D (excited) = {DW_e:.6f}\n")
#print("Saved results to ccd.dat")
