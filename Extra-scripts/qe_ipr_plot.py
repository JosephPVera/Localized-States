#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2026-09

"""
Usage:
    python3 qe_ipr_plot.py [--band]

Use --band to print the band index (starting from 1) next to each
           plotted point (if several are overlapping or separated by <= 0.1 eV,
           they are grouped into a list, e.g. "12, 13, 14, 15").

Data source:
    - ipr.dat: single file with columns
          1) band index
          2) ipr_up
          3) ipr_norm_up   <-- SPIN UP intensity value
          4) ipr_dw
          5) ipr_norm_dw   <-- SPIN DOWN intensity value
          6) Energy_up (eV)
          7) Energy_dw (eV)
      The first line is a comment with the k-point, e.g.:
          # k-point: 0.000000  0.000000  0.000000
      If that k-point is (0, 0, 0), the plot uses the symbol \\Gamma;
      otherwise it shows the raw k-point coordinates.
"""

import json
import re
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

IPR_FILE = "ipr.dat"
OUTPUT_FILE = "eigenplot_ipr.png"

# Path to the primitive.json file containing VBM/CBM (Band_edges).
PRIMITIVE_JSON_PATH = "../../../primitive/primitive.json"

with open(PRIMITIVE_JSON_PATH, "r") as f:
    primitive_data = json.load(f)

band_edges = primitive_data["Band_edges"]
VBM = band_edges["VBM"]
CBM = band_edges["CBM"]

RES = VBM   # rescale energies with respect to VBM (set RES = VBM); 0.0 = no rescaling

CMAP = "viridis"

KPOINT_RE = re.compile(
    r"k-point:\s*([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)",
    re.IGNORECASE,
)

def parse_kpoint_label(first_line):
    """
    Reads the k-point from the file's first comment line and returns the
    label to use on the x-axis: \\Gamma if the k-point is (0,0,0),
    otherwise the raw coordinates.
    """
    m = KPOINT_RE.search(first_line)
    if not m:
        return "k"
    kx, ky, kz = (float(v) for v in m.groups())
    if abs(kx) < 1e-6 and abs(ky) < 1e-6 and abs(kz) < 1e-6:
        return r"$\Gamma$"
    return f"({kx:.3f}, {ky:.3f}, {kz:.3f})"

def parse_ipr_dat(path):
    """
    Reads ipr.dat and returns (kpoint_label, up_kpoint, down_kpoint).

    up_kpoint uses columns 1 (band), 3 (ipr_norm_up), 6 (Energy_up).
    down_kpoint uses columns 1 (band), 5 (ipr_norm_dw), 7 (Energy_dw).
    """
    up = {"band_indices": [], "energies": [], "localization": []}
    down = {"band_indices": [], "energies": [], "localization": []}
    kpoint_label = "k"

    with open(path, "r", errors="ignore") as f:
        first_line = f.readline()
        kpoint_label = parse_kpoint_label(first_line)

        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            try:
                band_idx = int(parts[0])
                ipr_norm_up = float(parts[2])
                ipr_norm_dw = float(parts[4])
                e_up = float(parts[5])
                e_dw = float(parts[6])
            except (ValueError, IndexError):
                continue

            up["band_indices"].append(band_idx)
            up["energies"].append(e_up)
            up["localization"].append(ipr_norm_up)

            down["band_indices"].append(band_idx)
            down["energies"].append(e_dw)
            down["localization"].append(ipr_norm_dw)

    return kpoint_label, up, down

def plot_spin_channel(ax, kpoint, title, vbm, cbm, norm, cmap, kpoint_label, res=0,
                       show_index=False, show_ylabel=True):
    if kpoint is None or not kpoint["energies"]:
        ax.set_title(f"{title} (no data)")
        return None

    shift = res
    idx = 1  # single k-point
    scatter_obj = None

    xs = [idx] * len(kpoint["energies"])
    ys = [e - shift for e in kpoint["energies"]]
    cs = kpoint["localization"]
    scatter_obj = ax.scatter(xs, ys, c=cs, cmap=cmap, norm=norm, s=50,
                              edgecolors="none", zorder=3)

    if show_index:
        # group band indices when they are overlapping or separated by
        # <= 0.1 eV (chain grouping: energies are sorted and consecutive
        # points are merged when their difference is <= 0.1 eV)
        band_indices = kpoint["band_indices"]
        order = sorted(range(len(ys)), key=lambda b: ys[b])
        groups = []
        current_group = [order[0]]
        current_y = ys[order[0]]
        for pos in order[1:]:
            y_val = ys[pos]
            if abs(y_val - current_y) <= 0.1:
                current_group.append(pos)
            else:
                groups.append(current_group)
                current_group = [pos]
            current_y = y_val
        groups.append(current_group)

        for group in groups:
            group_sorted = sorted(group, key=lambda p: band_indices[p])
            y_mean = sum(ys[p] for p in group_sorted) / len(group_sorted)
            labels = [str(band_indices[p]) for p in group_sorted]
            chunks = [labels[i:i + 7] for i in range(0, len(labels), 7)]
            label = "\n".join(", ".join(chunk) for chunk in chunks)
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
        ax.set_ylabel("Energy (eV)", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.set_xticks([1])
    ax.set_xticklabels([kpoint_label], fontsize=8, size=10)

    # shading of the valence band (blue) / conduction band (red)
    vb_line = vbm - shift
    cb_line = cbm - shift

    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(vbm - 1.7945 - res, cbm + 1.7551 - res)

    ymin, ymax = ax.get_ylim()
    ax.axhspan(ymin, vb_line, color="blue", alpha=0.15, zorder=1)
    ax.axhspan(cb_line, ymax, color="red", alpha=0.15, zorder=1)

    return scatter_obj

def main():
    parser = argparse.ArgumentParser(
        description="Plot bands colored by IPR (from ipr.dat)."
    )
    parser.add_argument("--band", action="store_true", help="Print the band index (starting from 1)")
    args = parser.parse_args()

    vbm = VBM
    cbm = CBM
    res = RES

    kpoint_label, up_kpoint, down_kpoint = parse_ipr_dat(IPR_FILE)

    all_vals = up_kpoint["localization"] + down_kpoint["localization"]
    vmin = min(all_vals) if all_vals else 0.0
    vmax = max(all_vals) if all_vals else 1.0
    if vmin == vmax:
        vmax = vmin + 1e-9
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(CMAP)

    fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True, constrained_layout=True)
    plot_spin_channel(axes[0], up_kpoint, "Spin Up", vbm=vbm, cbm=cbm, norm=norm, cmap=cmap,
                       kpoint_label=kpoint_label, res=res, show_index=args.band, show_ylabel=True)
    scatter_obj = plot_spin_channel(axes[1], down_kpoint, "Spin Down", vbm=vbm, cbm=cbm, norm=norm, cmap=cmap,
                                     kpoint_label=kpoint_label, res=res, show_index=args.band, show_ylabel=False)

    if scatter_obj is None:
        # fallback: if SPIN DOWN has no data, use the SPIN UP scatter for the colorbar
        for coll in axes[0].collections:
            scatter_obj = coll
            break

    if scatter_obj is not None:
        cbar = fig.colorbar(scatter_obj, ax=axes, location="right", pad=0.02, fraction=0.05)
        cbar.set_label('Localization Factor - Inverse Participation Ratio (IPR)', fontsize=14)
        cbar.ax.yaxis.set_label_position('left')

    title = "IPR"
    if res != 0:
        title += f"\nrescale: E - {res} eV"
    fig.savefig(OUTPUT_FILE, dpi=150)

if __name__ == "__main__":
    main()
