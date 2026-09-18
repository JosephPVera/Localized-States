#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2026-08

"""
Usage:
     python3 qe_eig.py [--band]

    Use --band to print the band index (starting from 1) next to each plotted
    point (if several are overlapping or separated by <= 0.1 eV, they are grouped
    into a list, e.g. "12, 13, 14, 15").
    
Reads a Quantum ESPRESSO .out file (spin-polarized calculation),
extracts the SPIN UP / SPIN DOWN blocks, and for each k-point extracts
the band energies (eV) and their associated occupation numbers.

Plotting rule (scatter, x = k-point index, y = energy eV):
    occ > 0.9             -> occupied (blue) 
    occ < 0.1             -> unoccupied (red)
    0.1 <= occ <= 0.9     -> partially occupied (green)
"""

import json

import os
import re
import glob
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INPUT_FILE = None   # path to your .dat file, or None to automatically search in the current folder (ignoring slurm-*.out)
OUTPUT_FILE = "eigenplot_qe.png"  

# Path to the primitive.json file containing VBM/CBM (Band_edges).
PRIMITIVE_JSON_PATH = "../../../primitive/primitive.json"

# Read VBM and CBM from primitive.json 
with open(PRIMITIVE_JSON_PATH, "r") as f:
    primitive_data = json.load(f)

band_edges = primitive_data["Band_edges"]
VBM = band_edges["VBM"]
CBM = band_edges["CBM"]
 
RES = VBM   # rescale energies with respect to VBM (set RES = VBM); 0.0 = no rescaling

FLOAT_RE = re.compile(r"[-+]?\d+\.\d+")
KPOINT_HEADER_RE = re.compile(r"k\s*=\s*([-+0-9.\s]+?)\(\s*(\d+)\s*PWs\)\s*bands\s*\(ev\):", re.IGNORECASE)

def extract_floats(line):
    """Returns the list of floating-point numbers found in a line."""
    return [float(x) for x in FLOAT_RE.findall(line)]

def find_out_file(folder="."):
    all_out_files = glob.glob(os.path.join(folder, "*.out"))
    out_files = [f for f in all_out_files if not re.match(r"^slurm-\d+\.out$", os.path.basename(f))]

    if not out_files:
        raise FileNotFoundError(
            "No valid .out file was found in the folder "
        )
    if len(out_files) > 1:
        print(f"Warning: multiple .out files were found {sorted(out_files)}, using: {out_files[0]}")

    return out_files[0]

def split_spin_sections(text):
    up_marker = re.search(r"-+\s*SPIN\s+UP\s*-+", text, re.IGNORECASE)
    down_marker = re.search(r"-+\s*SPIN\s+DOWN\s*-+", text, re.IGNORECASE)

    if up_marker and down_marker:
        up_text = text[up_marker.end():down_marker.start()]
        down_text = text[down_marker.end():]
        next_up = re.search(r"-+\s*SPIN\s+UP\s*-+", down_text, re.IGNORECASE)
        if next_up:
            down_text = down_text[: next_up.start()]
        return {"UP": up_text, "DOWN": down_text}
    else:
        return {"UP": text, "DOWN": ""}

def parse_section(section_text):
    """
    Parses a section (UP or DOWN) and returns a list of k-points, each one
    containing its band energies and occupations:
    [ {"k": (kx,ky,kz), "energies": [...], "occupations": [...]}, ... ]
    """
    kpoints = []

    # locate all headers "k = ... bands (ev):"
    headers = list(KPOINT_HEADER_RE.finditer(section_text))
    if not headers:
        return kpoints

    for i, h in enumerate(headers):
        kvec_str = h.group(1).split()
        try:
            kvec = tuple(float(v) for v in kvec_str[:3])
        except ValueError:
            kvec = (None, None, None)

        block_start = h.end()
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(section_text)
        block_text = section_text[block_start:block_end]

        # separate energies (before "occupation numbers") from occupations (after)
        occ_split = re.split(r"occupation numbers", block_text, flags=re.IGNORECASE)
        energy_text = occ_split[0]
        occ_text = occ_split[1] if len(occ_split) > 1 else ""

        energies = []
        for line in energy_text.splitlines():
            if line.strip() == "" or set(line.strip()) <= {"."}:
                continue
            energies.extend(extract_floats(line))

        occupations = []
        for line in occ_text.splitlines():
            if line.strip() == "" or set(line.strip()) <= {"."}:
                continue
            occupations.extend(extract_floats(line))

        if energies:
            if not occupations:
                occupations = [float("nan")] * len(energies)
            n = min(len(energies), len(occupations))
            kpoints.append(
                {
                    "k": kvec,
                    "energies": energies[:n],
                    "occupations": occupations[:n],
                }
            )

    return kpoints

def color_for_occupation(occ):
    if occ != occ: 
        return "gray"
    if occ > 0.9:
        return "xkcd:blue" # occupied
    elif occ < 0.1: 
        return "xkcd:red"  # unoccupied
    else: 
        return "xkcd:green" # partially occupied

def plot_spin_channel(ax, kpoints, title, vbm, cbm, res=0, show_index=False, show_ylabel=True):
    if not kpoints:
        ax.set_title(f"{title} (no data)")
        return []

    shift = res
    n_k = len(kpoints)

    for idx, kp in enumerate(kpoints, start=1):
        xs = [idx] * len(kp["energies"])
        ys = [e - shift for e in kp["energies"]]
        colors = [color_for_occupation(o) for o in kp["occupations"]]
        ax.scatter(xs, ys, c=colors, s=50, edgecolors="none", zorder=3)

        if show_index:
            # group band indices when they are overlapping or separated by
            # <= 0.1 eV (chain grouping: energies are sorted and consecutive
            # points are merged when their difference is <= 0.1 eV)
            order = sorted(range(1, len(ys) + 1), key=lambda b: ys[b - 1])
            groups = []
            current_group = [order[0]]
            current_y = ys[order[0] - 1]
            for band_idx in order[1:]:
                y_val = ys[band_idx - 1]
                if abs(y_val - current_y) <= 0.1:
                    current_group.append(band_idx)
                else:
                    groups.append(current_group)
                    current_group = [band_idx]
                current_y = y_val
            groups.append(current_group)

            for group in groups:
                group_sorted = sorted(group)
                y_mean = sum(ys[b - 1] for b in group_sorted) / len(group_sorted)
                # if there are more than 8 values, split them into rows of 8 (one below another)
                chunks = [group_sorted[i:i + 7] for i in range(0, len(group_sorted), 7)]
                label = "\n".join(", ".join(str(b) for b in chunk) for chunk in chunks)
                ax.annotate(
                    label,
                    xy=(idx, y_mean),
                    xytext=(4, 0),
                    textcoords="offset points",
                    fontsize=8,
                    va="center",
                    ha="left",
                    zorder=4,
                )

    ax.set_xlabel("K-point coordinates", fontsize=14)
    if show_ylabel:
        ax.set_ylabel("Energy (eV)" if shift == 0 else f"Energy (eV)", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(1, n_k + 1))
    ax.set_xticklabels([r'$\Gamma$'] + [str(i) for i in range(2, n_k + 1)], fontsize=8, size=10)
    #ax.grid(alpha=0.3, zorder=0)

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="xkcd:blue", markersize=8, label="Occupied"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="xkcd:green", markersize=8, label="Partially Occupied"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="xkcd:red", markersize=8, label="Unoccupied"),
    ]

    # shading of the valence band (blue) / conduction band (red)
    vb_line = vbm - shift
    cb_line = cbm - shift

    ax.set_xlim(min(range(1, n_k + 1)) - 0.5, max(range(1, n_k + 1)) + 0.5)
    ax.set_ylim(vbm - 1.7945 - res, cbm + 1.7551 - res)

    ymin, ymax = ax.get_ylim()
    ax.axhspan(ymin, vb_line, color="blue", alpha=0.15, zorder=1)
    ax.axhspan(cb_line, ymax, color="red", alpha=0.15, zorder=1)
    #ax.axhline(vb_line, color="blue", linewidth=0.8, linestyle="--", alpha=0.6)
    #ax.axhline(cb_line, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
    legend_elems.append(Line2D([0], [0], color="blue", lw=6, alpha=0.15, label=f"Valence Band"))
    legend_elems.append(Line2D([0], [0], color="red", lw=6, alpha=0.15, label=f"Conduction Band"))

    return legend_elems

def main():
    parser = argparse.ArgumentParser(description="Plot QE spin-polarized bands colored by occupation.")
    parser.add_argument("outfile", nargs="?", default=None, help=".out file of Quantum ESPRESSO")
    parser.add_argument("--band", action="store_true", help="Print the band index (starting from 1)")
    args = parser.parse_args()

    vbm = VBM
    cbm = CBM
    res = RES

    if args.outfile is not None:
        infile = args.outfile
    elif INPUT_FILE is not None:
        infile = INPUT_FILE
    else:
        infile = find_out_file(".")
        print(f"Detected .dat file: {infile}")

    with open(infile, "r", errors="ignore") as f:
        text = f.read()

    sections = split_spin_sections(text)
    up_kpoints = parse_section(sections["UP"])
    down_kpoints = parse_section(sections["DOWN"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True, constrained_layout=True)
    plot_spin_channel(axes[0], up_kpoints, "Spin Up", vbm=vbm, cbm=cbm, res=res,
                       show_index=args.band, show_ylabel=True)
    legend_elems = plot_spin_channel(axes[1], down_kpoints, "Spin Down", vbm=vbm, cbm=cbm, res=res,
                                      show_index=args.band, show_ylabel=False)

    if legend_elems:
        fig.legend(
            handles=legend_elems,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            bbox_transform=axes[1].transAxes,
            fontsize=10,)

    title = "Kohn-Sham Level Diagram"
    if res != 0:
        title += f"\nrescale: E - {res} eV"
    #fig.suptitle(title)
    #fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    #print(f"Saved figure in: {OUTPUT_FILE}")
    #print(f"SPIN UP: {len(up_kpoints)} k-points found")
    #print(f"SPIN DOWN: {len(down_kpoints)} k-points found")

if __name__ == "__main__":
    main()
