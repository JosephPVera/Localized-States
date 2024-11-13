# Written by Joseph P.Vera
# 2024-11

from LSPD.reader.reader import VasprunReader
from fractions import Fraction

class VariablesExtractor:
    def __init__(self, xml_reader):
        self.root = xml_reader.get_root()
        self.spin_numbers = []
        self.kpoint_numbers = []
        self.band_numbers = []
        self.kpoint_coordinates = []
        self.result = []

    def find_spin_numbers(self):
        """Finds unique spin numbers in the XML data."""
        for spin_set in self.root.findall(".//set"):
            comment = spin_set.get('comment')
            if comment and comment.startswith('spin'):
                spin_number = int(comment.replace('spin', ''))
                if spin_number not in self.spin_numbers:
                    self.spin_numbers.append(spin_number)

    def find_kpoint_numbers(self):
        """Finds unique kpoint numbers for each spin."""
        for spin_number in self.spin_numbers:
            spin_set = self.root.find(f".//set[@comment='spin{spin_number}']")
            if spin_set is not None:
                for kpoint_set in spin_set.findall(".//set"):
                    comment = kpoint_set.get('comment')
                    if comment and comment.startswith('kpoint'):
                        kpoint_number = int(comment.replace('kpoint ', ''))
                        if kpoint_number not in self.kpoint_numbers:
                            self.kpoint_numbers.append(kpoint_number)

    def find_band_numbers(self):
        """Finds unique band numbers for each kpoint."""
        for spin_number in self.spin_numbers:
            spin_set = self.root.find(f".//set[@comment='spin{spin_number}']")
            if spin_set is not None:
                for kpoint_number in self.kpoint_numbers:
                    kpoint_block = spin_set.find(f".//set[@comment='kpoint {kpoint_number}']")
                    if kpoint_block is not None:
                        for band_set in kpoint_block.findall(".//set"):
                            comment = band_set.get('comment')
                            if comment and comment.startswith('band'):
                                band_number = int(comment.replace('band ', ''))
                                if band_number not in self.band_numbers:
                                    self.band_numbers.append(band_number)

 
    def extract_kpoint_coordinates(self):
        "Extract k-point coordinates from the XML tree."
        
        kpointlist = self.root.find(".//varray[@name='kpointlist']")
        
        if kpointlist is not None:
            for v in kpointlist.findall("v"):
                coordinates = [float(coord) for coord in v.text.strip().split()]
                self.kpoint_coordinates.append(coordinates)
        else:
            print("The <varray name='kpointlist'> tag was not found in the file.")
        
        return self.kpoint_coordinates

    def generate_x_labels(self, line_break="\n"):
        "Generate x-axis labels from k-point coordinates."
        
        for k in self.kpoint_coordinates:
            x_label = []
            for i in k:
                frac = Fraction(i).limit_denominator(10)
                if frac.numerator == 0:
                    x_label.append("0")
                else:
                    x_label.append(f"{frac.numerator}/{frac.denominator}")
            if x_label == ["0", "0", "0"]:
                self.result.append("Γ")
            else:
                self.result.append(line_break.join(x_label))
        return self.result
