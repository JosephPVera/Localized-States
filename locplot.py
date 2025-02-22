#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2025-02

from LSPD.reader.reader import VasprunReader
from LSPD.analyzer.main_variables import VariablesExtractor
from LSPD.analyzer.get_results import ResultsExtractor
from LSPD.plotter.loc_plotter import LocalizedPlotter
from LSPD.arg.commands import CommandLineArgs

"Plot the localization states in each kpoint"

# Use --tot command for plot: Energy versus tot column (PROCAR). By default plot: Energy versus sum (the 5 heaviest values from tot (each band)).
args = CommandLineArgs()

# Variables following the valence band maximum (VBM) and conduction band minimum (CBM).
vbm = 6.7056 # AEXX = 0.25 (AEXX = 0.33 --> 6.4979)  
cbm = 12.5198 # AEXX = 0.25 (AEXX = 0.33 --> 12.7609)

# Read the file
xml_reader = VasprunReader("vasprun.xml")

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

# Prepare the plotter by declaring its variables
plotter = LocalizedPlotter(vasp_data.spin_numbers, vasp_data.kpoint_numbers, vbm, cbm, args.tot_mode, args.band_mode)

# Use the total_results list to plot
plotter.store_final_results(total_results)

# Plot the localized states
plotter.plot_localized()
