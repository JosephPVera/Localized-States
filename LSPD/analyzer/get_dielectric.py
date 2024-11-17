from LSPD.reader.reader import VasprunReader

class DielectricAnalyzer:
    def __init__(self, xml_reader):
        self.root = xml_reader.get_root()

    def parse_dielectric_tensor(self):
        # Parsing the ionic dielectric tensor
        epsilon_ion = self.root.find(".//varray[@name='epsilon_ion']")
        if epsilon_ion is not None:
            print("Ionic dielectric tensor:")
            for vector in epsilon_ion.findall("v"):
                print(vector.text)
        else:
            print("The ionic dielectric tensor was not found in the vasprun.xml file.")

        # Parsing the electronic dielectric tensor
        epsilon = self.root.find(".//varray[@name='epsilon']")
        if epsilon is not None:
            print("\nElectronic dielectric tensor:")
            for vector in epsilon.findall("v"):
                print(vector.text)
        else:
            print("The electronic dielectric tensor was not found in the vasprun.xml file.")
