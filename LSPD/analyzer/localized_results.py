# Written by Joseph P.Vera
# 2024-11

from LSPD.reader.reader import VasprunReader

class VasprunParser:
    def __init__(self, vbm, cbm, spin_numbers, kpoint_numbers, filter_occupancy=None, xml_reader = 'vasprun.xml'):
        self.xml_reader = VasprunReader(xml_reader)
        self.root = self.xml_reader.get_root()
        self.vbm = vbm
        self.cbm = cbm
        self.spin_numbers = spin_numbers
        self.kpoint_numbers = kpoint_numbers
        self.filter_occupancy = filter_occupancy

        # Initialize lists to store information
        self.energy_values = []
        self.occupancy_list = []
        self.occupation_status = {}
        self.band_index_list_up = []
        self.kpoint_list_up = []
        self.spin_list_up = []
        self.band_index_list_down = []
        self.kpoint_list_down = []
        self.spin_list_down = []
        self.eigen_val = []
        self.vasprun_val = []

    def parse_eigenval(self):
        for spin_number in self.spin_numbers:
            self.occupation_status[spin_number] = {}  

            self.eigen_val.append("###########################################################")
            self.eigen_val.append(f"                           {'SPIN UP' if spin_number == 1 else 'SPIN DOWN'}                     ")
            self.eigen_val.append("###########################################################")
            
            for kpoint_number in self.kpoint_numbers:
                spin_set = self.root.find(f".//set[@comment='spin {spin_number}']")
                if spin_set is not None:
                    kpoint_block = spin_set.find(f".//set[@comment='kpoint {kpoint_number}']")
                    if kpoint_block is not None:
                        block_values, block_occu, block_status, band_indices = [], [], [], []

                        for band_index, child in enumerate(kpoint_block, 1):
                            if child.text:
                                columns = child.text.split()
                                if len(columns) >= 2:
                                    energy = float(columns[0])
                                    occupancy = float(columns[1])
                                    # Determine occupancy status
                                    if occupancy == 1.0:
                                        status = "Occupied"
                                    elif occupancy > 0.9:
                                        status = "Occupied"
                                    elif occupancy < 0.1:
                                        status = "Unoccupied"
                                    else:
                                        status = "Partially Occupied"

                                    # Store values
                                    block_values.append(energy)
                                    block_occu.append(occupancy)
                                    block_status.append((energy, status))
                                    band_indices.append(band_index)

                        if block_values: self.energy_values.append(block_values)
                        if block_occu: self.occupancy_list.append(block_occu)
                        if block_status:
                            self.occupation_status[spin_number][kpoint_number] = block_status

                        for idx in range(1, len(block_occu)):
                            if block_occu[idx - 1] == 1.0 and block_occu[idx] < 1.0:
                                start_idx = max(idx - 500, 0)
                                end_idx = min(idx + 500, len(block_occu))
                                energies_in_range = block_values[start_idx:end_idx]
                                occupancies_in_range = block_occu[start_idx:end_idx]
                                indices_in_range = band_indices[start_idx:end_idx]

                                self.eigen_val.append("###########################################################")
                                self.eigen_val.append(f"                          kpoint {kpoint_number}                   ")
                                self.eigen_val.append("###########################################################")
                                self.eigen_val.append(f"{'Band':<10} {'Energy':<14} {'Occ':<10} {'Occupancy'}")

                                for band_index, (energy, occupancy) in zip(indices_in_range, zip(energies_in_range, occupancies_in_range)):
                                    if self.vbm <= energy <= self.cbm:
                                        label = "Occupied" if occupancy > 0.9 else "Unoccupied" if occupancy < 0.1 else "Partially Occupied"
                                        
                                        # Apply filter if set
                                        if not self.filter_occupancy or label in self.filter_occupancy:
                                            if spin_number == 1:
                                                self.band_index_list_up.append(band_index)
                                                self.kpoint_list_up.append(kpoint_number)
                                                self.spin_list_up.append(spin_number)
                                            elif spin_number == 2:
                                                self.band_index_list_down.append(band_index)
                                                self.kpoint_list_down.append(kpoint_number)
                                                self.spin_list_down.append(spin_number)
                                            
                                            self.eigen_val.append(f"{band_index:<10} {energy:<14.6f} {occupancy:<10.6f} {label}")
                    else:
                        print(f"Block 'kpoint {kpoint_number}' not found in 'spin {spin_number}'.")
                else:
                    print(f"Superblock 'spin {spin_number}' not found.")

            if spin_number == 1:
                self.eigen_val.append("\n")

    def parse_procar(self):
        # Spin Up Analysis
        if self.band_index_list_up and self.kpoint_list_up and self.spin_list_up:
            self.vasprun_val.append("########################################################################")
            self.vasprun_val.append("                               SPIN UP                                  ")
            self.vasprun_val.append("########################################################################")

            # Group information by k-point
            current_kpoint = None
            band_info = []

            for i in range(len(self.band_index_list_up)):
                spin_number = self.spin_list_up[i]
                kpoint_number = self.kpoint_list_up[i]
                band_number = self.band_index_list_up[i]

                # Find the spin up (1) superblock
                spin_set = self.root.find(f".//set[@comment='spin{spin_number}']")

                if spin_set is not None:
                    # Find the block corresponding to the kpoint
                    kpoint_block = spin_set.find(f".//set[@comment='kpoint {kpoint_number}']")

                    if kpoint_block is not None:
                        # If we are in a different k-point from the previous one, write the accumulated information
                        if current_kpoint != kpoint_number:
                            if band_info:
                                # Write accumulated information of the previous k-point
                                self.vasprun_val.append("\n########################################################################")
                                self.vasprun_val.append(f"                               KPOINT {current_kpoint}                             ")
                                self.vasprun_val.append("########################################################################")
                                self.vasprun_val.extend(band_info)

                                band_info = []  # Reset for the next k-point

                            current_kpoint = kpoint_number  # Update the current k-point

                        # Process the bands for this k-point
                        band_subblock = kpoint_block.find(f".//set[@comment='band {band_number}']")

                        if band_subblock is not None:
                            band_info.append(f"\nInformation of {band_subblock.attrib['comment']}:")
                            band_info.append(f"{'index':<6} {'s':<10} {'p':<10} {'d':<10} {'tot':<10}")

                            # Add information of the band
                            for j, child in enumerate(band_subblock):
                                columns = child.text.split()
                                total_sum = sum(float(value) for value in columns)

                                # Set the decimals
                                formatted_values = [f"{float(value):.3f}" for value in columns]
                                formatted_sum = f"{total_sum:.3f}"

                                # Only print if the total (s+p+d) is greater than 0.1
                                if float(formatted_sum) > 0.1:
                                    band_info.append(f"{j + 1:<6} {formatted_values[0]:<10} {formatted_values[1]:<10} {formatted_values[2]:<10} {formatted_sum:<10}")
                        else:
                            band_info.append(f"Subblock 'band {band_number}' not found in 'kpoint {kpoint_number}'.\n")

            # Write the information of the last k-point if it exists
            if band_info:
                self.vasprun_val.append("\n########################################################################")
                self.vasprun_val.append(f"                               KPOINT {current_kpoint}                             ")
                self.vasprun_val.append("########################################################################")
                self.vasprun_val.extend(band_info)

        # Spin Down Analysis
        if self.band_index_list_down and self.kpoint_list_down and self.spin_list_down:
            self.vasprun_val.append("\n\n\n")
            self.vasprun_val.append("########################################################################")
            self.vasprun_val.append("                               SPIN DOWN                                ")
            self.vasprun_val.append("########################################################################")

            # Group information by k-point
            current_kpoint = None
            band_info = []

            for i in range(len(self.band_index_list_down)):
                spin_number = self.spin_list_down[i]
                kpoint_number = self.kpoint_list_down[i]
                band_number = self.band_index_list_down[i]

                # Find the spin down (2) superblock
                spin_set = self.root.find(f".//set[@comment='spin{spin_number}']")

                if spin_set is not None:
                    # Find the block corresponding to the kpoint
                    kpoint_block = spin_set.find(f".//set[@comment='kpoint {kpoint_number}']")

                    if kpoint_block is not None:
                        # If we are in a different k-point from the previous one, write the accumulated information
                        if current_kpoint != kpoint_number:
                            if band_info:
                                # Write accumulated information of the previous k-point
                                self.vasprun_val.append("\n########################################################################")
                                self.vasprun_val.append(f"                               KPOINT {current_kpoint}                             ")
                                self.vasprun_val.append("########################################################################")
                                self.vasprun_val.extend(band_info)

                                band_info = []  # Reset for the next k-point

                            current_kpoint = kpoint_number  # Update the current k-point

                        # Process the bands for this k-point
                        band_subblock = kpoint_block.find(f".//set[@comment='band {band_number}']")

                        if band_subblock is not None:
                            band_info.append(f"\nInformation of {band_subblock.attrib['comment']}:")
                            band_info.append(f"{'index':<6} {'s':<10} {'p':<10} {'d':<10} {'tot':<10}")

                            # Add information of the band
                            for j, child in enumerate(band_subblock):
                                columns = child.text.split()
                                total_sum = sum(float(value) for value in columns)

                                # Set the decimals
                                formatted_values = [f"{float(value):.3f}" for value in columns]
                                formatted_sum = f"{total_sum:.3f}"

                                # Only print if the total (s+p+d) is greater than 0.1
                                if float(formatted_sum) > 0.1:
                                    band_info.append(f"{j + 1:<6} {formatted_values[0]:<10} {formatted_values[1]:<10} {formatted_values[2]:<10} {formatted_sum:<10}")
                        else:
                            band_info.append(f"Subblock 'band {band_number}' not found in 'kpoint {kpoint_number}'.\n")

            # Write the information of the last k-point if it exists
            if band_info:
                self.vasprun_val.append("\n########################################################################")
                self.vasprun_val.append(f"                               KPOINT {current_kpoint}                             ")
                self.vasprun_val.append("########################################################################")
                self.vasprun_val.extend(band_info)
