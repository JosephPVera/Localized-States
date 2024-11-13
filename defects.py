#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2024-11

from LSPD.analyzer.get_defects import DefectAnalysis  

"Find defects by comparing POSCAR_perfect and POSCAR_defect"
# Introduce the path for the defect/POSCAR and perfect/POSCAR. By default the code use POSCAR_perfect and POSCAR_defect in the current folder.
defect_file = "POSCAR"
perfect_file = "../perfect/POSCAR"

# Introducing the path
defect_analysis = DefectAnalysis(defect_file,perfect_file)

# without introduce the path
# defect_analysis = DefectAnalysis()

# Print vacancy defects
defect_analysis.print_closest_to_vacancy()

# Print substitutional defects
defect_analysis.print_closest_to_substitutional()

# Print the interstitial defects
defect_analysis.print_closest_to_interstitial()

# Save the information found. It can save the information directly without printing the defects.
defect_analysis.save_defect_data()
