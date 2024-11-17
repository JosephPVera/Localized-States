#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2024-11

from LSPD.reader.reader import VasprunReader
from LSPD.analyzer.get_dielectric import DielectricAnalyzer

xml_reader = VasprunReader("vasprun.xml")

parser = DielectricAnalyzer(xml_reader)

parser.parse_dielectric_tensor()
