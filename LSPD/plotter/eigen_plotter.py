# Written by Joseph P.Vera
# 2025-02

import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

class EigenvaluesPlotter:
    def __init__(self, vbm, cbm, kpoint_coordinates, generate_x_labels, res=0.0, band_mode=False, split_mode=False):
        self.final_result = []
        self.vbm = vbm 
        self.cbm = cbm 
        self.res = res
        self.kpoint_coordinates = kpoint_coordinates 
        self.generate_x_labels = generate_x_labels
        self.band_mode = band_mode  
        self.split_mode = split_mode
        
    def store_final_results(self, total_results):
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

        band_numbers_up, band_numbers_down = [], []  
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
                subset = data.iloc[:, [1, 5, 6, 2]]  
                subset.columns = ['kpoint', 'Energy', 'occ', 'band']

                # Filer states inside the band gap
                if self.split_mode:
                    bandgap_states = subset[(subset['Energy'] >= self.vbm - 1.6) & (subset['Energy'] <= self.cbm + 2.6)] # 1.6
                else:
                    bandgap_states = subset[(subset['Energy'] >= self.vbm - 1.6) & (subset['Energy'] <= self.cbm + 1.6)] # 0.4              1.6

                if not bandgap_states.empty:
                    kpoint_vals = bandgap_states['kpoint'].to_list()
                    energy_vals = bandgap_states['Energy'].to_list()
                    occupancy_group = bandgap_states['occ'].to_list()
                    band_numbers = bandgap_states['band'].to_list()

                    if i < num_blocks // 2: 
                        kpoint_vals_up.extend(kpoint_vals)
                        energy_vals_up.extend(energy_vals)
                        colors_up.extend([
                            'xkcd:blue' if val > 0.9 else 'xkcd:red' if val < 0.1 else 'xkcd:green'
                            for val in occupancy_group])
                        band_numbers_up.extend(band_numbers)

                    else:  
                        kpoint_vals_down.extend(kpoint_vals)
                        energy_vals_down.extend(energy_vals)
                        colors_down.extend([
                            'xkcd:blue' if val > 0.9 else 'xkcd:red' if val < 0.1 else 'xkcd:green'
                            for val in occupancy_group])
                        band_numbers_down.extend(band_numbers)

        rescale_up = [valor - self.res for valor in energy_vals_up]
        rescale_down = [valor - self.res for valor in energy_vals_down]
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 8), constrained_layout=True)
        kpoint_labels = self.generate_x_labels()

        unique_kpoints = sorted(set(kpoint_vals_up + kpoint_vals_down))
        x_tick_labels = [kpoint_labels[unique_kpoints.index(kpt)] if kpt in unique_kpoints else '' for kpt in unique_kpoints]

        if self.band_mode:
            def label_bands(ax, kpoint_vals, energy_vals, band_numbers):
                unique_kpts = sorted(set(kpoint_vals))

                for kpt in unique_kpts:
                    # Points from this k-point that fall inside the gap
                    idxs = [i for i, k in enumerate(kpoint_vals)
                            if k == kpt and self.vbm - self.res <= energy_vals[i] <= self.cbm - self.res]
                    if not idxs:
                        continue

                    # Chain grouping: sort by energy and merge consecutive
                    # points that are <= 0.1 eV apart (same idea as qe_locplot.py)
                    idxs_sorted = sorted(idxs, key=lambda i: energy_vals[i])

                    groups = []
                    current_group = [idxs_sorted[0]]
                    current_y = energy_vals[idxs_sorted[0]]
                    for i in idxs_sorted[1:]:
                        y_val = energy_vals[i]
                        if abs(y_val - current_y) <= 0.1:
                            current_group.append(i)
                        else:
                            groups.append(current_group)
                            current_group = [i]
                        current_y = y_val
                    groups.append(current_group)

                    for group in groups:
                        group_bands = sorted(set(band_numbers[i] for i in group))
                        y_mean = sum(energy_vals[i] for i in group) / len(group)
                        chunks = [group_bands[c:c + 5] for c in range(0, len(group_bands), 5)]
                        label = "\n".join(", ".join(str(b) for b in chunk) for chunk in chunks)
                        ax.annotate(
                            label,
                            xy=(kpt, y_mean),
                            xytext=(4, 0),
                            textcoords="offset points",
                            fontsize=10,
                            va="center",
                            ha="left",
                            zorder=4,
                        )

            label_bands(axs[0], kpoint_vals_up, rescale_up, band_numbers_up)
            
            label_bands(axs[1], kpoint_vals_down, rescale_down, band_numbers_down)

        if self.split_mode:
            def find_degenerate_states(energy_list, threshold=0.006): # 0.006
                degenerate_states = []
                used_indices = set()

                for i, energy in enumerate(energy_list):
                    if i not in used_indices:
                        degenerate_group = [energy]
                        for j, energy2 in enumerate(energy_list):
                            if i != j and abs(energy - energy2) <= threshold:
                                degenerate_group.append(energy2)
                                used_indices.add(j)
                        degenerate_states.append(degenerate_group)

                return degenerate_states

            degenerate_groups_up = find_degenerate_states(energy_vals_up)
            
            size_arrow_spin = 0.21 
            for degenerate_group in degenerate_groups_up:
                energy = degenerate_group[0]
                idx = energy_vals_up.index(energy)
                color = colors_up[idx]

                if len(degenerate_group) == 1:
                    axs[0].plot([0.95, 1.05], [energy - self.res, energy - self.res], color=color, lw=1)
                    if color != 'xkcd:red':  # Avoid plotting if the color is red
                        axs[0].scatter([1], [energy - self.res], color=color, marker='o')
                        axs[0].annotate(
                            '',
                            xy=(1, energy - self.res + size_arrow_spin), 
                            xytext=(1, energy - self.res - size_arrow_spin),
                            arrowprops=dict(arrowstyle='->', color=color, lw=0.5),
                            zorder=4)

                elif len(degenerate_group) == 2:
                    axs[0].plot([0.875, 0.975], [energy - self.res, energy - self.res], color=color, lw=1)
                    axs[0].plot([1.025, 1.125], [energy - self.res, energy - self.res], color=color, lw=1)

                    if color != 'xkcd:red':
                        x_centers = [(0.875 + 0.975) / 2, (1.025 + 1.125) / 2]
                        for x_center in x_centers:
                            axs[0].plot(x_center, energy - self.res, marker='o', color=color)
                            axs[0].annotate(
                                '',
                                xy=(x_center, energy - self.res + size_arrow_spin),
                                xytext=(x_center, energy - self.res - size_arrow_spin),
                                arrowprops=dict(arrowstyle='->', color=color, lw=0.5),
                                zorder=4)

                elif len(degenerate_group) == 3:
                    axs[0].plot([0.8, 0.9], [energy - self.res, energy - self.res], color=color, lw=1)
                    axs[0].plot([0.95, 1.05], [energy - self.res, energy - self.res], color=color, lw=1)
                    axs[0].plot([1.1, 1.2], [energy - self.res, energy - self.res], color=color, lw=1)
                    if color != 'xkcd:red':
                        x_centers = [(0.8 + 0.9) / 2, (0.95 + 1.05) / 2, (1.1 + 1.2) / 2]
                        for x_center in x_centers:
                            axs[0].plot(x_center, energy - self.res, marker='o', color=color)
                            axs[0].annotate(
                                '',
                                xy=(x_center, energy - self.res + size_arrow_spin),
                                xytext=(x_center, energy - self.res - size_arrow_spin),
                                arrowprops=dict(arrowstyle='->', color=color, lw=0.5),
                                zorder=4)
                        axs[0].scatter([1], [energy - self.res], color=color, marker='o')

            degenerate_groups_down = find_degenerate_states(energy_vals_down)

            for degenerate_group in degenerate_groups_down:
                energy = degenerate_group[0]
                idx = energy_vals_down.index(energy)
                color = colors_down[idx]

                if len(degenerate_group) == 1:
                    axs[1].plot([0.95, 1.05], [energy - self.res, energy - self.res], color=color, lw=1)

                    if color != 'xkcd:red':
                        axs[1].scatter([1], [energy - self.res], color=color, marker='o')
                        x_center = 1.0
                        y_center = energy - self.res
                        axs[1].plot(x_center, y_center, marker='o', color=color)
                        axs[1].annotate(
                            '',
                            xy=(x_center, y_center + size_arrow_spin),
                            xytext=(x_center, y_center - size_arrow_spin),
                            arrowprops=dict(arrowstyle='<-', color=color, lw=0.5),
                            zorder=4)

                elif len(degenerate_group) == 2:
                    axs[1].plot([0.875, 0.975], [energy - self.res, energy - self.res], color=color, lw=1)
                    axs[1].plot([1.025, 1.125], [energy - self.res, energy - self.res], color=color, lw=1)

                    if color != 'xkcd:red':
                        x_centers = [(0.875 + 0.975) / 2, (1.025 + 1.125) / 2]
                        for x_center in x_centers:
                            axs[1].plot(x_center, energy - self.res, marker='o', color=color)
                            axs[1].annotate(
                                '',
                                xy=(x_center, energy - self.res + size_arrow_spin),
                                xytext=(x_center, energy - self.res - size_arrow_spin),
                                arrowprops=dict(arrowstyle='<-', color=color, lw=0.5),
                                zorder=4)

                elif len(degenerate_group) == 3:
                    axs[1].plot([0.8, 0.9], [energy - self.res, energy - self.res], color=color, lw=1)
                    axs[1].plot([0.95, 1.05], [energy - self.res, energy - self.res], color=color, lw=1)
                    axs[1].plot([1.1, 1.2], [energy - self.res, energy - self.res], color=color, lw=1)

                    if color != 'xkcd:red':
                        x_centers = [(0.8 + 0.9) / 2, (0.95 + 1.05) / 2, (1.1 + 1.2) / 2]
                        for x_center in x_centers:
                            axs[1].plot(x_center, energy - self.res, marker='o', color=color)
                            axs[1].annotate(
                                '',
                                xy=(x_center, energy - self.res + size_arrow_spin),
                                xytext=(x_center, energy - self.res - size_arrow_spin),
                                arrowprops=dict(arrowstyle='<-', color=color, lw=0.5),
                                zorder=4)
                        axs[1].scatter([1], [energy - self.res], color=color, marker='o')                                                  
        else:
             axs[0].scatter(kpoint_vals_up, rescale_up, color=colors_up, label='Spin Up', s=30)   
             axs[1].scatter(kpoint_vals_down, rescale_down, color=colors_down, label='Spin Down', s=30)
                     
        # Subplot Spin up
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
        plt.legend(handles=[occupied_patch, unoccupied_patch, partially_occupied_patch, vbm_patch, cbm_patch], bbox_to_anchor=(1.5, 0.6))
        plt.savefig('kohn-sham-states.png', dpi=150)
