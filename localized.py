#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2024-11

import json

import os
from LSPD.reader.reader import VasprunReader
from LSPD.analyzer.main_variables import VariablesExtractor
from LSPD.analyzer.localized_results import VasprunParser
#from LSPD.arg.commands import CommandLineArgs

"Get specific information about the localized states"

# Path to the primitive.json file containing VBM/CBM (Band_edges).
PRIMITIVE_JSON_PATH = "../../primitive/primitive.json"

# Read VBM and CBM from primitive.json instead of hardcoding them.
with open(PRIMITIVE_JSON_PATH, "r") as f:
    primitive_data = json.load(f)

band_edges = primitive_data["Band_edges"]
vbm = band_edges["VBM"]
cbm = band_edges["CBM"]

# res is optional to rescale the Kohn-Sham (eigenvalues) plot with respect to VBM, it may also be off.
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

# It can be a list like ["Occupied", "Partially Occupied"] or "Occupied" or "Partially Occupied" or "Unoccupied". By default is None, it will search to all.
filter_occupancy = None  

# Prepare 
parser = VasprunParser(vbm, cbm, vasp_data.spin_numbers, vasp_data.kpoint_numbers, res, filter_occupancy)

# Get information same to the EIGENVAL and PROCAR files, but in vasprun.xml file.
parser.parse_eigenval()
parser.parse_procar()

# Save the information
folder_name = os.path.basename(os.getcwd())
#localized_folder = f'localized-defects/{folder_name}/Data'
#if not os.path.exists(localized_folder):
#    os.makedirs(localized_folder)

#output_file = os.path.join(localized_folder, f'localized_{folder_name}.dat')
output_file = f'localized_{folder_name}.dat'

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
