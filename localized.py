#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2024-11

import os
from LSPD.reader.reader import VasprunReader
from LSPD.analyzer.main_variables import VariablesExtractor
from LSPD.analyzer.localized_results import VasprunParser

"Get specific information about the localized states"

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

# It can be a list like ["Occupied", "Partially Occupied"] or "Occupied" or "Partially Occupied" or "Unoccupied". By default is None, it will search to all.
filter_occupancy = None  

# Prepare 
parser = VasprunParser(vbm, cbm, vasp_data.spin_numbers, vasp_data.kpoint_numbers, filter_occupancy)

# Get information same to the EIGENVAL and PROCAR files, but in vasprun.xml file.
parser.parse_eigenval()
parser.parse_procar()

# Save the information
folder_name = os.path.basename(os.getcwd())
localized_folder = f'localized-defects/{folder_name}/Data'
if not os.path.exists(localized_folder):
    os.makedirs(localized_folder)

output_file = os.path.join(localized_folder, f'localized_{folder_name}.dat')

with open(output_file, 'w') as f:
    f.write(f"Defect: {folder_name}\n")
    f.write(f"\nVBM = {vbm} eV\n")  # Replace vbm with actual value
    f.write(f"CBM = {cbm} eV\n\n\n")
    f.write("###########################################################\n")
    f.write("           vasprun.xml file (EIGENVAL information)                            \n")
    f.write("###########################################################\n")
    f.write("\n".join(parser.eigen_val) + "\n")
    f.write("\n\n\n\n########################################################################\n")
    f.write("                  vasprun.xml file (PROCAR information)\n")
    f.write("########################################################################\n")
    f.write("\n".join(parser.vasprun_val) + "\n")

print(f"Data saved to {output_file}")
