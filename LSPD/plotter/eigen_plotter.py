# Written by Joseph P.Vera
# 2025-03

import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

class EigenvaluesPlotter:
    def __init__(self, vbm, cbm, kpoint_coordinates, generate_x_labels, res=0.0, band_mode=False):
        self.final_result = []
        self.vbm = vbm 
        self.cbm = cbm 
        self.res = res
        self.kpoint_coordinates = kpoint_coordinates 
        self.generate_x_labels = generate_x_labels
        self.band_mode = band_mode  # Store the band_mode value

    def store_final_results(self, total_results):
        "Store total results into final results."
        self.final_result = total_results.copy()
        
    def plot_eigenvalues(self):
        "Plot eigenvalues based on k-point coordinates and formatted labels."
        content = '\n'.join(self.final_result[1:])  # Skip the first line
        blocks = content.strip().split('\n\n')

        num_blocks = len(blocks)
        blocks_up = blocks[:num_blocks // 2]
        blocks_down = blocks[num_blocks // 2:]

        kpoint_vals_up, energy_vals_up, colors_up = [], [], []
        kpoint_vals_down, energy_vals_down, colors_down = [], [], []

        band_numbers_up, band_numbers_down = [], []  # Para almacenar los números de banda
        printed_bands_up, printed_bands_down = set(), set()

        for i, block in enumerate(blocks):
            data = pd.read_csv(StringIO(block), sep=r'\s+', header=None)

            """Column 0 ----> spin number, \
             Column 1 ----> kpoint number, \
             Column 2 ----> band number, \
             Column 3 ----> tot, \
             Column 4 ----> sum, \
             Column 5 ----> Energy, \
             Column 6 ----> occupancies"""
            if data.shape[1] >= 7:
                subset = data.iloc[:, [1, 5, 6, 2]]  # Include band number (column 2)
                subset.columns = ['kpoint', 'Energy', 'occ', 'band']

                occupancy = subset['occ'].to_list()
                rupture_point = next((j for j, val in enumerate(occupancy) if val < 1.0), None)
            
                if rupture_point is not None and rupture_point >= 10 and rupture_point < len(occupancy) - 10:
                    kpoint_vals = subset['kpoint'][rupture_point - 14:rupture_point + 11].to_list()
                    energy_vals = subset['Energy'][rupture_point - 14:rupture_point + 11].to_list()
                    occupancy_group = occupancy[rupture_point - 14:rupture_point + 11]
                    band_numbers = subset['band'][rupture_point - 14:rupture_point + 11].to_list()

                    if i < num_blocks // 2:
                        kpoint_vals_up.extend(kpoint_vals)
                        energy_vals_up.extend(energy_vals)
                        colors_up.extend(['blue' if val > 0.9 else 'red' if val < 0.1 else 'green' for val in occupancy_group])
                        band_numbers_up.extend(band_numbers)
                    else:
                        kpoint_vals_down.extend(kpoint_vals)
                        energy_vals_down.extend(energy_vals)
                        colors_down.extend(['blue' if val > 0.9 else 'red' if val < 0.1 else 'green' for val in occupancy_group])
                        band_numbers_down.extend(band_numbers)

        rescale_up = [valor - self.res for valor in energy_vals_up]
        rescale_down = [valor - self.res for valor in energy_vals_down]
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 8))
        kpoint_labels = self.generate_x_labels()

        unique_kpoints = sorted(set(kpoint_vals_up + kpoint_vals_down))
        x_tick_labels = [kpoint_labels[unique_kpoints.index(kpt)] if kpt in unique_kpoints else '' for kpt in unique_kpoints]

        if self.band_mode:
            printed_bands_per_kpoint_up = {}
            printed_bands_per_kpoint_down = {}
            
            for kpt, energy, band in zip(kpoint_vals_up, rescale_up, band_numbers_up):
                if self.vbm - self.res <= energy <= self.cbm - self.res:
                    if kpt not in printed_bands_per_kpoint_up:
                        printed_bands_per_kpoint_up[kpt] = set()
                    
                    similar_bands = [band]
                    for kpt2, energy2, band2 in zip(kpoint_vals_up, rescale_up, band_numbers_up):
                        if kpt == kpt2 and abs(energy - energy2) <= 0.1 and band != band2:
                            if self.vbm - self.res <= energy2 <= self.cbm - self.res:
                                similar_bands.append(band2)
                    
                    similar_bands_sorted = tuple(sorted(set(similar_bands)))
                    
                    if similar_bands_sorted not in printed_bands_per_kpoint_up[kpt]:
                        printed_bands_per_kpoint_up[kpt].add(similar_bands_sorted)
                        axs[0].text(kpt + 0.05, energy, ', '.join(map(str, similar_bands_sorted)), fontsize=10, color='black')

            for kpt, energy, band in zip(kpoint_vals_down, rescale_down, band_numbers_down):
                if self.vbm - self.res <= energy <= self.cbm - self.res:
                    if kpt not in printed_bands_per_kpoint_down:
                        printed_bands_per_kpoint_down[kpt] = set()
                    
                    similar_bands = [band]
                    for kpt2, energy2, band2 in zip(kpoint_vals_down, rescale_down, band_numbers_down):
                        if kpt == kpt2 and abs(energy - energy2) <= 0.1 and band != band2:
                            if self.vbm - self.res <= energy2 <= self.cbm - self.res:
                                similar_bands.append(band2)
                    
                    similar_bands_sorted = tuple(sorted(set(similar_bands)))
                    
                    if similar_bands_sorted not in printed_bands_per_kpoint_down[kpt]:
                        printed_bands_per_kpoint_down[kpt].add(similar_bands_sorted)
                        axs[1].text(kpt + 0.05, energy, ', '.join(map(str, similar_bands_sorted)), fontsize=10, color='black')
                        
        # Subplot Spin up
        axs[0].scatter(kpoint_vals_up, rescale_up, color=colors_up, label='Spin Up', s=30)
        axs[0].set_xlabel('K-point coordinates', fontsize=14)
        axs[0].set_title('Spin up', fontsize=14)
        axs[0].set_ylabel('Energy (eV)', fontsize=14)
        axs[0].set_xlim(min(kpoint_vals_up) - 0.5, max(kpoint_vals_up) + 0.5)
        axs[0].set_ylim(self.vbm - 1.7945 - self.res, self.cbm + 1.7551 - self.res)
        axs[0].axhspan(self.vbm - self.res, self.vbm - 1.7945 - self.res, color='lightblue', alpha=0.4)
        axs[0].axhspan(self.cbm - self.res, self.cbm + 1.7551 - self.res, color='thistle', alpha=0.4)
        axs[0].set_xticks(unique_kpoints)
        axs[0].set_xticklabels(x_tick_labels, rotation=0, fontsize=8, size=10)

        # Subplot Spin down
        axs[1].scatter(kpoint_vals_down, rescale_down, color=colors_down, label='Spin Down', s=30)
        axs[1].set_xlabel('K-point coordinates', fontsize=14)
        axs[1].set_title('Spin down', fontsize=14)
        axs[1].tick_params(axis='y', which='both', left=True, right=False, labelleft=False)
        axs[1].set_xlim(min(kpoint_vals_down) - 0.5, max(kpoint_vals_down) + 0.5)
        axs[1].set_ylim(self.vbm - 1.7945 - self.res, self.cbm + 1.7551 - self.res)
        axs[1].axhspan(self.vbm - self.res, self.vbm - 1.7945 - self.res, color='lightblue', alpha=0.4)
        axs[1].axhspan(self.cbm - self.res, self.cbm + 1.7551 - self.res, color='thistle', alpha=0.4)
        axs[1].set_xticks(unique_kpoints)
        axs[1].set_xticklabels(x_tick_labels, rotation=0, fontsize=8, size=10)

        occupied_patch = plt.Line2D([0], [0], marker='o', color='w', label='Occupied', markerfacecolor='blue', markersize=10)
        unoccupied_patch = plt.Line2D([0], [0], marker='o', color='w', label='Unoccupied', markerfacecolor='red', markersize=10)
        partially_occupied_patch = plt.Line2D([0], [0], marker='o', color='w', label='Partially occupied', markerfacecolor='green', markersize=10)
        vbm_patch = plt.Line2D([0], [0], color='lightblue', label='VBM')
        cbm_patch = plt.Line2D([0], [0], color='thistle', label='CBM')
        plt.legend(handles=[occupied_patch, unoccupied_patch, partially_occupied_patch, vbm_patch, cbm_patch], bbox_to_anchor=(1.56, 0.7))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.03)
        plt.tight_layout()
        plt.savefig('kohn-sham-states.png', dpi=150)
