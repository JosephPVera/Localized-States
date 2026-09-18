#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2026-09

"""
Usage:
      python3 corrections_plot.py
   
Reads the correction.json file produced by corrections.py script.
"""

import json
from efnv_corrections import (
    ExtendedFnvCorrection, PotentialSite, plot_site_potentials,
)

SUMMARY_JSON = "correction.json"
TITLE = None
OUTPUT_PATH = "correction_plot.png"
DPI = 150
SHOW = False                          

with open(SUMMARY_JSON) as f:
    data = json.load(f)

if "sites" not in data:
    raise ValueError(f"{SUMMARY_JSON} has no 'sites' entry.")

sites = [PotentialSite(specie=s["specie"], distance=s["distance"],
                       potential=s["potential"], pc_potential=s["pc_potential"])
          for s in data["sites"]]

correction = ExtendedFnvCorrection(
    charge=data["charge"],
    point_charge_correction=data["energy_corrections"]["pc term"],
    defect_region_radius=data["defect_region_radius"],
    sites=sites,
    defect_coords=tuple(data["defect_frac_coords"]),
)

#print(f"Loaded {len(sites)} sites from {SUMMARY_JSON}")
#print(correction)

fig = plot_site_potentials(correction, title=TITLE,
                           output_path=OUTPUT_PATH, show=SHOW, dpi=DPI)
                           
#print(f"\nSaved file: {OUTPUT_PATH}")
