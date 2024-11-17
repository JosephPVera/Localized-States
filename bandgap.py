#!/usr/bin/env python
# Written by Joseph P.Vera
# 2024-10

from LSPD.reader.reader import VasprunReader
from LSPD.analyzer.main_variables import VariablesExtractor
from LSPD.analyzer.get_gap import GapAnalyzer

# Read the file
xml_reader = VasprunReader("vasprun.xml")
extractor = VariablesExtractor(xml_reader)

extractor.find_spin_numbers()
extractor.find_kpoint_numbers()
extractor.find_band_numbers()

analyzer = GapAnalyzer(extractor.spin_numbers, extractor.kpoint_numbers)
analyzer.analyze()
vbm, cbm = analyzer.get_results()
print(f"VBM: {vbm}, CBM: {cbm}")
print(f"Bandgap = {cbm - vbm}")
