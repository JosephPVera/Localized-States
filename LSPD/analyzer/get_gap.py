from LSPD.reader.reader import VasprunReader

class GapAnalyzer:
    def __init__(self, spin_numbers, kpoint_numbers, xml_reader="vasprun.xml"):
        self.xml_reader = VasprunReader(xml_reader)
        self.root = self.xml_reader.get_root()
        self.spin_numbers = spin_numbers
        self.kpoint_numbers = kpoint_numbers
        self.max_energy_1000 = float('-inf')  # store the max energy (VBM)
        self.min_energy_0000 = float('inf')   # store the min energy (CBM)

    def analyze(self):
        for spin_number in self.spin_numbers:
            for kpoint_number in self.kpoint_numbers:
                # spin superblock
                spin_set = self.root.find(f".//set[@comment='spin {spin_number}']")
                
                if spin_set is not None:
                    # kpoint block
                    kpoint_block = spin_set.find(f".//set[@comment='kpoint {kpoint_number}']")
                    
                    if kpoint_block is not None:
                        last_1_000_energy = None  # last energy with occupancy 1.000
                        first_0_000_energy = None  # first energy with occupancy 0.000
                        
                        for child in kpoint_block:
                            if child.text:
                                columns = child.text.split()
                                if len(columns) >= 2:
                                    energy = float(columns[0])
                                    occupancy = float(columns[1])
                                    
                                    # Identify the last line with occupancy 1.000 in each kpoint
                                    if occupancy == 1.000:
                                        last_1_000_energy = energy
                                    
                                    # Identify the first line with occupancy 0.000 in each kpoint
                                    if occupancy == 0.000 and first_0_000_energy is None:
                                        first_0_000_energy = energy
                        
                        # Update maximum energy with occupancy 1.000
                        if last_1_000_energy is not None:
                            self.max_energy_1000 = max(self.max_energy_1000, last_1_000_energy)
                        
                        # Update the minimum energy wuth occupancy 0.000
                        if first_0_000_energy is not None:
                            self.min_energy_0000 = min(self.min_energy_0000, first_0_000_energy)
                    else:
                        print(f"Block 'kpoint {kpoint_number}' not found in 'spin {spin_number}'.")
                else:
                    print(f"Superblock 'spin {spin_number}' not found.")

    def get_results(self):
        return self.max_energy_1000, self.min_energy_0000
