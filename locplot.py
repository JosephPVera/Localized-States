#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2025-02

import json

from LSPD.reader.reader import VasprunReader
from LSPD.analyzer.main_variables import VariablesExtractor
from LSPD.analyzer.get_results import ResultsExtractor
from LSPD.plotter.loc_plotter import LocalizedPlotter
from LSPD.arg.commands import CommandLineArgs

"""Plot the localization states in each kpoint
Usage:
     python3 locplot.py [--band] [--tot]
"""

# Use --tot command for plot: Energy versus tot column (PROCAR). By default plot: Energy versus sum (the 5 heaviest values from tot (each band)).
args = CommandLineArgs()

# Path to the primitive.json file containing VBM/CBM (Band_edges).
PRIMITIVE_JSON_PATH = "../../primitive/primitive.json"

# Read VBM and CBM from primitive.json instead of hardcoding them.
with open(PRIMITIVE_JSON_PATH, "r") as f:
    primitive_data = json.load(f)

band_edges = primitive_data["Band_edges"]
vbm = band_edges["VBM"]
cbm = band_edges["CBM"]

# res is optional to rescale the energy
res = vbm

# Read the file
#xml_reader = VasprunReader("vasprun.xml")
xml_reader = VasprunReader()

# Prepare the vasprun.xml file to parse
vasp_data = VariablesExtractor(xml_reader)

# Find the main variables in vasprun.xml file: spin, kpoints and bands.
vasp_data.find_spin_numbers()
vasp_data.find_kpoint_numbers()
vasp_data.find_band_numbers()

# Prepare the extraction results with the main variables
results_extractor = ResultsExtractor(vasp_data.spin_numbers, vasp_data.kpoint_numbers, vasp_data.band_numbers)

# Extract results (spin, kpoint, band, tot, sum) and energy_occupancy (energy, occupancy) values in columns.
results_extractor.extract_results()
results_extractor.extract_energy_occupancy()

# Merge the results and energy_occupancy in one list to plot.
total_results = results_extractor.create_total_results()

# Extract k-point coordinates and labels for x-axis as xticks to plot .
vasp_data.extract_kpoint_coordinates()
vasp_data.generate_x_labels()

# Prepare the plotter by declaring its variables
plotter = LocalizedPlotter(vasp_data.spin_numbers, vasp_data.kpoint_numbers, vbm, cbm, args.tot_mode, vasp_data.generate_x_labels, args.band_mode, res)

# Use the total_results list to plot
plotter.store_final_results(total_results)

# Plot the localized states
plotter.plot_localized()
