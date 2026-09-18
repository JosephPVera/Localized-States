#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2026-09

"""
Usage:
      python3 formation_energy.py [--json] [--all] [--nc] [--A] [--B] [--charges] [--y] [--x]
      
Defect formation energy diagram, built directly from a summary_defects.json
file.

    E_form^q(E_F) = E_def^q - E_perf - sum_i n_i * mu_i + q * E_VBM + E_correction + q * E_F
"""

import argparse
import json
import re
import numpy as np
import matplotlib.pyplot as plt

def to_latex_name(name):
    """N_C-V_C -> N_{C}-V_{C} (wrap the subscript after each underscore in braces)."""
    return re.sub(r"_([A-Za-z0-9]+)", r"_{\1}", name)

parser = argparse.ArgumentParser(description="Defect formation energy plot from a summary_corrections.json file")
parser.add_argument("--json", type=str, default="summary_defects.json",
                     help="Path to the summary_defects.json file (default: summary_defects.json)")
parser.add_argument("--all", action="store_true",
                     help="Also plot every individual charge-state line for each defect, in addition to "
                          "the minimum curve. By default, only the minimum curve per defect is plotted.")
parser.add_argument("--nc", action="store_true",
                     help="Do not apply finite-size corrections (E_correction). "
                          "By default, corrections are applied.")
parser.add_argument("--A", action="store_true",
                     help="When a defect's chemical potential 'limits' defines multiple "
                          "growth conditions (e.g. 'B-rich' / 'N-rich'), use the FIRST "
                          "condition (in the order it appears in the JSON).")
parser.add_argument("--B", action="store_true",
                     help="When a defect's chemical potential 'limits' defines multiple "
                          "growth conditions (e.g. 'B-rich' / 'N-rich'), use the SECOND "
                          "condition (in the order it appears in the JSON).")
parser.add_argument("--charges", action="store_true",
                     help="Label each straight-line segment of the minimum-envelope "
                          "curve with its charge state q.")
parser.add_argument("--y", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"),
                     help="Set the y-axis (formation energy) range, e.g. --y 0 15. "
                          "By default, the range is chosen automatically.")
parser.add_argument("--x", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"),
                     help="Set the x-axis (Fermi level) range, e.g. --x -2 5. "
                          "By default, the range is [0, Gap]. When set, E_F is sampled "
                          "over [XMIN, XMAX] instead of [0, Gap], and the regions "
                          "XMIN-0 and Gap-XMAX are shaded blue/red respectively.")
args = parser.parse_args()

if args.A and args.B:
    parser.error("Use only one of --A or --B, not both.")

with open(args.json, "r") as f:
    data = json.load(f)

def resolve_mu_limits(raw_limits, defect_name, use_A, use_B):
    """Return (flat_limits, condition_label).

    Handles two 'limits' formats found under chemical_potentials:
      1. Plain:   {"C": 4.2, "N": 2.1}                       -> species: mu value
      2. Multi-condition: {"B-rich": {"B": 0, "N": 2.3},
                            "N-rich": {"B": 2.3, "N": 0}}    -> condition: {species: mu value}

    For format 2, --A selects the first condition listed in the JSON and
    --B selects the second (order preserved, since JSON keys keep insertion order).
    """
    is_multi_condition = bool(raw_limits) and all(
        isinstance(v, dict) for v in raw_limits.values()
    )

    if not is_multi_condition:
        return raw_limits, None

    condition_names = list(raw_limits.keys())
    if not use_A and not use_B:
        raise SystemExit(
            f"Error: defect '{defect_name}' has multiple chemical-potential conditions "
            f"({', '.join(condition_names)}). Use --A to select '{condition_names[0]}' "
            f"or --B to select '{condition_names[1] if len(condition_names) > 1 else condition_names[0]}'."
        )

    if use_A:
        chosen = condition_names[0]
    else:
        if len(condition_names) < 2:
            raise SystemExit(
                f"Error: defect '{defect_name}' only has one condition "
                f"('{condition_names[0]}'); --B has nothing to select. Use --A instead."
            )
        chosen = condition_names[1]

    return raw_limits[chosen], chosen

# Band edges and perfect-supercell energy
E_VBM = data["band_edges"]["VBM"]
E_CBM = data["band_edges"]["CBM"]
Gap = E_CBM - E_VBM

E_perfect = data["perfect"]["total_energy"]["TOTEN"]

# Fermi energy range
if args.x is not None:
    E_F = np.linspace(args.x[0], args.x[1], 500)
else:
    E_F = np.linspace(0, Gap, 500)

defects = data["defects"]
colors = ["xkcd:green", "xkcd:red", "xkcd:blue", "xkcd:orange", "xkcd:purple",
          "xkcd:brown", "xkcd:magenta", "xkcd:gold", "xkcd:crimson",
          "xkcd:darkblue", "xkcd:navy", "xkcd:olive", "xkcd:black", "xkcd:indigo"]

plt.figure()

results = {}  # defect_name -> (unique_intersections, corr_status)

for idx, (defect_name, charge_states) in enumerate(defects.items()):

    latex_name = to_latex_name(defect_name)

    q_list, E_def_list, E_corr_list, n_i_list = [], [], [], []
    mu_limits = None
    mu_elementals = None

    for state in charge_states.values():
        q_list.append(state["charge"])
        E_def_list.append(state["total_energy"]["TOTEN"])
        E_corr_list.append(state["energy_corrections"]["correction energy"])
        n_i_list.append(state["chemical_potentials"]["defect"])
        # assumed constant across charge states of the same defect
        mu_limits = state["chemical_potentials"]["limits"]
        mu_elementals = state["chemical_potentials"].get("elementals")

    order = np.argsort(q_list)
    q = np.array(q_list)[order]
    E_def = np.array(E_def_list)[order]
    E_corr = np.array(E_corr_list)[order]
    n_i = n_i_list[order[0]]  # {species: n_i}, same for every charge state

    mu_limits_flat, condition_used = resolve_mu_limits(mu_limits, defect_name, args.A, args.B)

    # Absolute chemical potential: mu_i = mu_i^elemental + delta_mu_i
    # delta_mu_i comes from the (chosen) "limits" entry; mu_i^elemental comes
    # from "elementals" if the JSON provides it.
    if mu_elementals is None:
        print(f"Warning: {defect_name} - no 'elementals' section found in the JSON; "
              f"using delta_mu directly as mu_i (results may be offset/incorrect).")
        mu_elementals = {}

    missing_species = [sp for sp in n_i if sp not in mu_limits_flat]
    if missing_species:
        cond_note = f" (condition '{condition_used}')" if condition_used else ""
        print(f"Note: {defect_name}{cond_note} - no limit given for "
              f"{', '.join(missing_species)}; using delta_mu = 0 for it.")

    missing_elemental = [sp for sp in n_i if sp not in mu_elementals]
    if missing_elemental:
        print(f"Note: {defect_name} - no elemental reference given for "
              f"{', '.join(missing_elemental)}; using mu_elemental = 0 for it.")

    mu_i = {
        species: mu_elementals.get(species, 0.0) + mu_limits_flat.get(species, 0.0)
        for species in n_i
    }
    mu_sum = sum(n_i[species] * mu_i[species] for species in n_i)

    E_corr_used = np.zeros_like(E_corr) if args.nc else E_corr

    # Formation energy is linear in E_F for every charge state: E_form_i(E_F) = a_i * E_F + b_i
    a = q.astype(float)
    b = E_def - E_perfect - mu_sum + q * E_VBM + E_corr_used

    E_form_all = np.array([a[i] * E_F + b[i] for i in range(len(q))])

    # Minimum curve
    E_min = np.min(E_form_all, axis=0)
    q_min_idx = np.argmin(E_form_all, axis=0)

    # Find the exact intersection points where the dominant charge state switches
    intersections = []
    for k in range(len(q_min_idx) - 1):
        i, j = q_min_idx[k], q_min_idx[k + 1]
        if i != j:
            if a[i] == a[j]:
                continue
            x_cross = (b[j] - b[i]) / (a[i] - a[j])
            if E_F[0] <= x_cross <= E_F[-1]:
                y_cross = a[i] * x_cross + b[i]
                intersections.append((x_cross, y_cross, q[i], q[j]))

    # Remove near-duplicate points (in case a crossing lands exactly on a grid node)
    unique_intersections = []
    for pt in intersections:
        if not any(np.isclose(pt[0], up[0], atol=1e-6) for up in unique_intersections):
            unique_intersections.append(pt)

    corr_status = "without corrections" if args.nc else "with corrections"
    results[defect_name] = (unique_intersections, corr_status, condition_used)

    if args.all:
        for i in range(len(q)):
            plt.plot(E_F, E_form_all[i], linewidth=1, alpha=0.4,
                     color=colors[i % len(colors)])
        min_color = "xkcd:black"
    else:
        min_color = colors[idx % len(colors)]

    plt.plot(E_F, E_min, color=min_color, linewidth=0.8, label=f"${latex_name}$")

    if args.charges:
        # Label each contiguous segment of the minimum-envelope curve (i.e. each
        # run where the same charge state q is dominant) with that charge value.
        seg_start = 0
        n_points = len(q_min_idx)
        for k in range(1, n_points + 1):
            if k == n_points or q_min_idx[k] != q_min_idx[seg_start]:
                seg_end = k - 1
                mid = (seg_start + seg_end) // 2
                q_val = q[q_min_idx[seg_start]]
                plt.annotate(f"{q_val:+d}", xy=(E_F[mid], E_min[mid]),
                             xytext=(0, 6), textcoords="offset points",
                             ha="center", va="bottom", fontsize=9, color=min_color)
                seg_start = k

    for x_cross, y_cross, qi, qj in unique_intersections:
        plt.plot(x_cross, y_cross, "o", color=min_color, markersize=4, zorder=5)

#plt.axvline(Gap, linestyle="-", color="xkcd:black", linewidth=0.8)
#plt.axvline(0, color="xkcd:black", linewidth=0.8)

if args.x is not None:
    plt.xlim(args.x[0], args.x[1])
    if args.x[0] < 0:
        plt.axvspan(args.x[0], 0, color="xkcd:blue", alpha=0.15, zorder=1)
    if args.x[1] > Gap:
        plt.axvspan(Gap, args.x[1], color="xkcd:red", alpha=0.15, zorder=1)
else:
    plt.xlim(0, Gap)

if args.y is not None:
    plt.ylim(args.y[0], args.y[1])

plt.xlabel("Fermi level (eV)", fontsize=14)
plt.ylabel("Formation energy (eV)", fontsize=14)

plt.legend(frameon=False)

plt.tight_layout()

suffix = "_nc" if args.nc else ""
if args.A:
    suffix += "_A"
elif args.B:
    suffix += "_B"
plt.savefig(f"formation_energy{suffix}.png", dpi=150)

# Print the (x, y) values of the intersection points for every defect,
# and collect them into a JSON file (one entry per curve/defect)
transitions_out = {}
for defect_name, (unique_intersections, corr_status, condition_used) in results.items():
    latex_name = to_latex_name(defect_name)
    condition_note = f", chemical-potential condition: {condition_used}" if condition_used else ""
    #print(f"\n{latex_name} - charge-transition levels ({corr_status}{condition_note}):")
    if unique_intersections:
        for x_cross, y_cross, qi, qj in unique_intersections:
            pass
            #print(f"  q={qi:+d}/{qj:+d}  ->  E_F = {x_cross:.4f} eV,  E_form = {y_cross:.4f} eV")
    else:
        print("No transitions found within the plotted Fermi-level range.")

    transitions_out[defect_name] = {
        "latex_name": latex_name,
        "correction_status": corr_status,
        #"chemical_potential_condition": condition_used,
        "transitions": [
            {
                "q1/q2": f"{int(qi)}/{int(qj)}",
                #"q_to": int(qj),
                "E_F_eV": float(x_cross),
                "E_form_eV": float(y_cross),
            }
            for x_cross, y_cross, qi, qj in unique_intersections
        ],
    }

if args.A:
    with open(f"transitions_A.json", "w") as f:
        json.dump(transitions_out, f, indent=2)
    
elif args.B:
    with open(f"transitions_B.json", "w") as f:
        json.dump(transitions_out, f, indent=2)
else:
    with open(f"transitions.json", "w") as f:
        json.dump(transitions_out, f, indent=2)
                    
#with open("transitions.json", "w") as f:
#    json.dump(transitions_out, f, indent=2)
