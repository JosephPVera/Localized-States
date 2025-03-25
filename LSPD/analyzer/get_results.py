# Written by Joseph P.Vera
# 2024-11

from LSPD.reader.reader import VasprunReader
from vaspwfc import vaspwfc
import numpy as np

class ResultsExtractor:
    def __init__(self, spin_numbers, kpoint_numbers, band_numbers, gamma=False, xml_reader="vasprun.xml", wav_file="WAVECAR"):
        self.xml_reader = VasprunReader(xml_reader)
        self.root = self.xml_reader.get_root()
        self.wav_file = wav_file
        self.spin_numbers = spin_numbers
        self.kpoint_numbers = kpoint_numbers
        self.band_numbers = band_numbers
        self.gamma = gamma
        self.results = []
        self.energy_values = []
        self.occupancy_list = []

    def extract_results(self):
        "Extracts results, including total sum and closest sums for bands.Information same to the PROCAR file"
        self.results.append(f"{'Spin':<6} {'k-point':<10} {'Band':<10} {'tot':<10} {'sum':<10}")

        for spin_number in self.spin_numbers:
            spin_set = self.root.find(f".//set[@comment='spin{spin_number}']")
            if spin_set is not None:
                for kpoint_number in self.kpoint_numbers:
                    kpoint_block = spin_set.find(f".//set[@comment='kpoint {kpoint_number}']")
                    if kpoint_block is not None:
                        for band_number in self.band_numbers:
                            band_subblock = kpoint_block.find(f".//set[@comment='band {band_number}']")
                            if band_subblock is not None:
                                total_sum = 0.0
                                tot_values = []

                                for child in band_subblock:
                                    columns = child.text.split()
                                    total = float(columns[0]) + float(columns[1]) + float(columns[2])
                                    total_sum += total
                                    tot_values.append(total)

                                closest_to_one = sorted(tot_values, key=lambda x: abs(x - 1))[:5]
                                closest_sum = sum(closest_to_one)

                                self.results.append(f"{spin_number:<6} {kpoint_number:<10} {band_number:<10} {total_sum:<10.3f} {closest_sum:<10.3f}")

    def extract_energy_occupancy(self):
        "Extract energy and occupancy values. Information same to the EIGENVAL file"
        for spin_number in self.spin_numbers:
            for kpoint_number in self.kpoint_numbers:
                spin_set = self.root.find(f".//set[@comment='spin {spin_number}']")
                if spin_set is not None:
                    kpoint_block = spin_set.find(f".//set[@comment='kpoint {kpoint_number}']")
                    if kpoint_block is not None:
                        block_values = []
                        block_occu = []
                        for child in kpoint_block:
                            if child.text:
                                columns = child.text.split()
                                if len(columns) >= 2:
                                    block_values.append(float(columns[0]))
                                    block_occu.append(float(columns[1]))

                        if block_values:
                            self.energy_values.append(block_values)
                        if block_occu:
                            self.occupancy_list.append(block_occu)
    
    def IPR(self):
        # Initialize the VASP wavefunction object
        if self.gamma:
            wfc = vaspwfc(self.wav_file, lgamma=True)
        else:
            wfc = vaspwfc(self.wav_file)
    
        if wfc._nspin == 2:
            spins = [1, 2]  # Spin-polarized calculation
        else:
            spins = [1]  # Non-spin-polarized calculation

        self.results.append(f"{'Spin':<6} {'k-point':<10} {'Band':<10} {'IPR':<10}")

        # Loop through all spins, k-points, and bands to calculate IPR
        for spin in spins:
            for k in range(wfc._nkpts):  # k-points
                for b in range(wfc._nbands):  # number of bands
                    # Extract the real-space wavefunction for the current spin, k-point, and band
                    psi_r = wfc.wfc_r(ispin=spin, ikpt=k+1, iband=b+1)
                
                    # Calculate |psi(r)|^2
                    psi_squared = np.abs(psi_r)**2
                
                    # Calculate the numerator and denominator for IPR
                    ipr_numerator = np.sum(psi_squared**2)  # Sum of |\Psi(r)|^4
                    ipr_denominator = np.sum(psi_squared)**2  # (Sum of |\Psi(r)|^2)^2
                
                    ipr_value = ipr_numerator / ipr_denominator
                
                    self.results.append(f"{spin:<6} {k+1:<10} {b+1:<10} {ipr_value:<10.6f}")

    def create_total_results(self):
        "Create the total results with energy and occupancy values."
        total_results = []
        for i, result in enumerate(self.results):
            if i == 0:
                total_results.append(f"{result:<10} {'Energy':<10} {'Occ':<10}")
            else:
                block_index = (i - 1) // len(self.band_numbers)
                row_index = (i - 1) % len(self.band_numbers)
                energy_value = self.energy_values[block_index][row_index] if block_index < len(self.energy_values) and row_index < len(self.energy_values[block_index]) else ''
                occupancy_value = self.occupancy_list[block_index][row_index] if block_index < len(self.occupancy_list) and row_index < len(self.occupancy_list[block_index]) else ''
                total_results.append(f"{result:<10} {energy_value:<10.3f} {occupancy_value:<10.3f}")

                if row_index == len(self.band_numbers) - 1 and block_index < len(self.energy_values) - 1:
                    total_results.append("")

        return total_results


class ResultsPrinter:
    def __init__(self, results_extractor):
        self.results_extractor = results_extractor

    def print_total_results(self):
        "Print the final total results."
        total_results = self.results_extractor.create_total_results()
        for line in total_results:
            print(line)
