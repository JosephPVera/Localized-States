# Written by Joseph P.Vera
# 2024-11

import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

class EigenvaluesPlotter:
    def __init__(self, vbm, cbm, kpoint_coordinates, generate_x_labels, res=0.0):
        self.final_result = []
        self.vbm = vbm 
        self.cbm = cbm 
        self.res = res
        self.kpoint_coordinates = kpoint_coordinates 
        self.generate_x_labels = generate_x_labels

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

        for i, block in enumerate(blocks):
            data = pd.read_csv(StringIO(block), sep=r'\s+', header=None)

            if data.shape[1] >= 7:
                subset = data.iloc[:, [1, 5, 6]]
                subset.columns = ['kpoint', 'Energy', 'occ']

                occupancy = subset['occ'].to_list()
                rupture_point = next((j for j, val in enumerate(occupancy) if val < 1.0), None)
            
                if rupture_point is not None and rupture_point >= 10 and rupture_point < len(occupancy) - 10:
                    kpoint_vals = subset['kpoint'][rupture_point - 14:rupture_point + 11].to_list()
                    energy_vals = subset['Energy'][rupture_point - 14:rupture_point + 11].to_list()
                    occupancy_group = occupancy[rupture_point - 14:rupture_point + 11]

                    if i < num_blocks // 2:
                        kpoint_vals_up.extend(kpoint_vals)
                        energy_vals_up.extend(energy_vals)
                        colors_up.extend(['blue' if val > 0.9 else 'red' if val < 0.1 else 'green' for val in occupancy_group])
                    else:
                        kpoint_vals_down.extend(kpoint_vals)
                        energy_vals_down.extend(energy_vals)
                        colors_down.extend(['blue' if val > 0.9 else 'red' if val < 0.1 else 'green' for val in occupancy_group])

        rescale_up = [valor - self.res for valor in energy_vals_up]
        rescale_down = [valor - self.res for valor in energy_vals_down]
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 8))
        kpoint_labels = self.generate_x_labels()

        unique_kpoints = sorted(set(kpoint_vals_up + kpoint_vals_down))
        x_tick_labels = [kpoint_labels[unique_kpoints.index(kpt)] if kpt in unique_kpoints else '' for kpt in unique_kpoints]
        
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
        axs[1].set_xlabel('K-point coordinates', fontsize=12)
        axs[1].set_title('Spin down', fontsize=12)
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
