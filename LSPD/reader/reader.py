# Written by Joseph P.Vera
# 2024-11

import xml.etree.ElementTree as ET

class VasprunReader:
    "Class for reading and parsing the vasprun.xml file."
    def __init__(self, xml_file):
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()

    def get_root(self):
        "Returns the root of the XML tree."
        return self.root
