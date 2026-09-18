#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2026-08

"""
Usage:
     python3 qe_locplot.py [--band] [--tot]

Use --band to print the band index (starting from 1) next to each plotted
           point (if several are overlapping or separated by <= 0.1 eV, they 
           are grouped into a list, e.g. "12, 13, 14, 15").

Use --tot so that the degree of localization of each band is the value from the
          summary tot row already provided in the file (last column), instead of 
          the manual sum of the individual atom rows.
    
Reads the localization.dat file generated for qe_loc.py
"""

import json

import os
import re
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

INPUT_FILE = None   # path to your .dat file, or None to automatically search in the current folder (ignoring slurm-*.out)
OUTPUT_FILE = "eigenplot_localization.png"

# Path to the primitive.json file containing VBM/CBM (Band_edges).
PRIMITIVE_JSON_PATH = "../../../primitive/primitive.json"

# Read VBM and CBM from primitive.json 
with open(PRIMITIVE_JSON_PATH, "r") as f:
    primitive_data = json.load(f)

band_edges = primitive_data["Band_edges"]
VBM = band_edges["VBM"]
CBM = band_edges["CBM"]
  
RES = VBM   # rescale energies with respect to VBM (set RES = VBM); 0.0 = no rescaling

CMAP = "viridis"    

FLOAT_RE = re.compile(r"[-+]?\d+\.\d+")

SPIN_UP_RE = re.compile(r"=+\s*\n\s*SPIN\s+UP\s*\n\s*=+", re.IGNORECASE)
SPIN_DOWN_RE = re.compile(r"=+\s*\n\s*SPIN\s+DOWN\s*\n\s*=+", re.IGNORECASE)

KPOINT_HEADER_RE = re.compile(r"---\s*k-point\s+(\d+)\s*\(\s*k\s*=\s*([-+0-9.\s]+?)\)\s*---", re.IGNORECASE,)

BAND_HEADER_RE = re.compile(r"Band\s+(\d+)\s+energy\s*=\s*([-+]?\d+\.\d+)\s*eV", re.IGNORECASE,)

def find_dat_file(folder="."):
    all_dat_files = glob.glob(os.path.join(folder, "*.dat"))
    dat_files = [f for f in all_dat_files if not re.match(r"^slurm-\d+\.out$", os.path.basename(f))]

    if not dat_files:
        raise FileNotFoundError(
            "No valid .out file was found in the folder "
        )
    if len(dat_files) > 1:
        print(f"Warning: multiple .dat files were found {sorted(dat_files)}, using: {dat_files[0]}")

    return dat_files[0]

def split_spin_sections(text):
    up_marker = SPIN_UP_RE.search(text)
    down_marker = SPIN_DOWN_RE.search(text)

    if up_marker and down_marker:
        up_text = text[up_marker.end():down_marker.start()]
        down_text = text[down_marker.end():]
        return {"UP": up_text, "DOWN": down_text}
    else:
        return {"UP": text, "DOWN": ""}

def parse_atom_table(block_text, use_summary_row=False):
    """
    Given the text of a band (atom / s / p_x / p_y / p_z / tot table):
    - use_summary_row=False (default): sums the 'tot' column from ALL atom rows
      present in the block (the set of listed atoms varies from band to band /
      k-point to k-point, so it is not filtered using a fixed list). Ignores the
      header row ("atom  s  p_x ...") and the summary row ("tot   ..."), which does
      not represent an individual atom.

    - use_summary_row=True: instead of summing the atom rows, uses directly the
      value of the 'tot' column from the summary row
      ("tot   ...   0.997000"), which is already calculated in the file.
    """
    total = 0.0
    summary_value = 0.0
    for line in block_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("atom"):
            continue
        parts = stripped.split()
        if parts[0].lower() == "tot":
            # summary row: "tot   s_sum  px_sum  py_sum  pz_sum  tot_sum"
            floats = extract_floats_from_parts(parts[1:])
            if floats:
                summary_value = floats[-1]
            continue
        try:
            int(parts[0])  # check that the row corresponds to an atom
        except ValueError:
            continue
        floats = extract_floats_from_parts(parts[1:])
        if floats:
            total += floats[-1]  # last column = "tot" of that row

    return summary_value if use_summary_row else total

def extract_floats_from_parts(parts):
    vals = []
    for p in parts:
        m = FLOAT_RE.fullmatch(p)
        if m:
            vals.append(float(p))
    return vals

def parse_section(section_text, use_summary_row=False):
    """
    Parses a section (UP or DOWN) and returns a list of k-points, each one
    containing its energies and localization degrees:
    [ {"k": (kx,ky,kz), "energies": [...], "localization": [...]}, ... ]
    """
    kpoints = []

    kp_headers = list(KPOINT_HEADER_RE.finditer(section_text))
    if not kp_headers:
        return kpoints

    for i, kh in enumerate(kp_headers):
        kvec_str = kh.group(2).split()
        try:
            kvec = tuple(float(v) for v in kvec_str[:3])
        except ValueError:
            kvec = (None, None, None)

        kp_start = kh.end()
        kp_end = kp_headers[i + 1].start() if i + 1 < len(kp_headers) else len(section_text)
        kp_text = section_text[kp_start:kp_end]

        band_headers = list(BAND_HEADER_RE.finditer(kp_text))
        energies = []
        localization = []

        for j, bh in enumerate(band_headers):
            energy = float(bh.group(2))
            b_start = bh.end()
            b_end = band_headers[j + 1].start() if j + 1 < len(band_headers) else len(kp_text)
            band_block = kp_text[b_start:b_end]

            loc = parse_atom_table(band_block, use_summary_row)

            energies.append(energy)
            localization.append(loc)

        if energies:
            kpoints.append({"k": kvec, "energies": energies, "localization": localization})

    return kpoints

def plot_spin_channel(ax, kpoints, title, vbm, cbm, norm, cmap, res=0,
                       show_index=False, show_ylabel=True):
    if not kpoints:
        ax.set_title(f"{title} (no data)")
        return None

    shift = res
    n_k = len(kpoints)
    scatter_obj = None

    for idx, kp in enumerate(kpoints, start=1):
        xs = [idx] * len(kp["energies"])
        ys = [e - shift for e in kp["energies"]]
        cs = kp["localization"]
        scatter_obj = ax.scatter(xs, ys, c=cs, cmap=cmap, norm=norm, s=50,
                                  edgecolors="none", zorder=3)

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
                    zorder=4,)

    ax.set_xlabel("K-point coordinates", fontsize=14)
    if show_ylabel:
        ax.set_ylabel("Energy (eV)" if shift == 0 else f"Energy (eV)", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(1, n_k + 1))
    ax.set_xticklabels([r'$\Gamma$'] + [str(i) for i in range(2, n_k + 1)], fontsize=8, size=10)

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

    return scatter_obj

def main():
    parser = argparse.ArgumentParser(
        description="Plot bands colored by atomic localization degree (from localization.dat)."
    )
    parser.add_argument("datfile", nargs="?", default=None, help="localization.dat file")
    parser.add_argument("--band", action="store_true", help="Print the band index (starting from 1)")
    parser.add_argument("--tot", action="store_true", help="Use the value from the summary row 'tot' of each band instead of summing the individual atom rows.")
    args = parser.parse_args()

    vbm = VBM
    cbm = CBM
    res = RES

    if args.datfile is not None:
        infile = args.datfile
    elif INPUT_FILE is not None:
        infile = INPUT_FILE
    else:
        infile = find_dat_file(".")
        print(f"Detected .dat file: {infile}")

    with open(infile, "r", errors="ignore") as f:
        text = f.read()

    sections = split_spin_sections(text)
    up_kpoints = parse_section(sections["UP"], use_summary_row=args.tot)
    down_kpoints = parse_section(sections["DOWN"], use_summary_row=args.tot)

    all_vals = []
    for kp in up_kpoints + down_kpoints:
        all_vals.extend(kp["localization"])
    vmin = min(all_vals) if all_vals else 0.0
    vmax = max(all_vals) if all_vals else 1.0
    if vmin == vmax:
        vmax = vmin + 1e-9
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(CMAP)

    fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True, constrained_layout=True)
    plot_spin_channel(axes[0], up_kpoints, "Spin Up", vbm=vbm, cbm=cbm, norm=norm, cmap=cmap,
                       res=res, show_index=args.band, show_ylabel=True)
    scatter_obj = plot_spin_channel(axes[1], down_kpoints, "Spin Down", vbm=vbm, cbm=cbm, norm=norm, cmap=cmap,
                                     res=res, show_index=args.band, show_ylabel=False)

    if scatter_obj is None:
        # fallback: if SPIN DOWN has no data, use the SPIN UP scatter for the colorbar
        for coll in axes[0].collections:
            scatter_obj = coll
            break

    if scatter_obj is not None:
        cbar = fig.colorbar(scatter_obj, ax=axes, location="right", pad=0.02, fraction=0.05)
        cbar.set_label('Localization Factor - Projected Density of States (PDOS)', fontsize=14)
        cbar.ax.yaxis.set_label_position('left')
        #if args.tot:
        #    cbar.set_label("Localization degree (the 'tot' value from the summary row)")
        #else:
        #    cbar.set_label("Localization degree (sum of 'tot' from the listed atoms)")

    title = "Localization Factor"
    if res != 0:
        title += f"\nrescale: E - {res} eV"
    fig.savefig(OUTPUT_FILE, dpi=150)
    #print(f"Saved figure in: {OUTPUT_FILE}")
    #print(f"SPIN UP: {len(up_kpoints)} k-points found")
    #print(f"SPIN DOWN: {len(down_kpoints)} k-points found")
    #print(f"Localization range: [{vmin:.6f}, {vmax:.6f}]")

if __name__ == "__main__":
    main()
