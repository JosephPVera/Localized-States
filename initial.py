#!/usr/bin/env python3

from LSPD.reader.reader import VASPXMLReader
from LSPD.analyzer.main_variables import VariablesExtractor

# Read the file
xml_reader = VASPXMLReader("vasprun.xml")

# Prepare the vasprun.xml file to parse
extractor = VariablesExtractor(xml_reader)

# Find the main variables in vasprun.xml file: spin, kpoints and bands.
extractor.find_spin_numbers()
extractor.find_kpoint_numbers()
extractor.find_band_numbers()

print("Spin numbers:", extractor.spin_numbers)
print("Kpoint numbers:", extractor.kpoint_numbers)
print("Band numbers:", extractor.band_numbers)
