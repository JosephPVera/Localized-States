#!/usr/bin/env python
# Written by Joseph P.Vera
# 2024-11

from LSPD.reader.reader import VasprunReader
from LSPD.analyzer.main_variables import VariablesExtractor
from LSPD.analyzer.get_results import ResultsExtractor
from LSPD.plotter.ipr_plotter import IPRPlotter

# Variables following the valence band maximum (VBM) and conduction band minimum (CBM).
vbm = 7.2945  
cbm = 11.7449

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

# Extract results (spin, kpoint, band, IPR) and energy_occupancy (energy, occupancy) values in columns.
results_extractor.IPR()
results_extractor.extract_energy_occupancy()

# Merge the results and energy_occupancy in one list to plot.
total_results = results_extractor.create_total_results()

# Prepare the plotter by declaring its variables
plotter = IPRPlotter(vasp_data.spin_numbers, vasp_data.kpoint_numbers, vbm, cbm)#, args.tot_mode)

# Use the total_results list to plot
plotter.store_final_results(total_results)

# Plot the localized states
plotter.plot_ipr()

