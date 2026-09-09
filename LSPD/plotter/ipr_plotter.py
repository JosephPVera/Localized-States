# Written by Joseph P.Vera
# 2024-11

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

class IPRPlotter:
    def __init__(self, spin_numbers, kpoint_numbers, vbm, cbm, generate_x_labels, band_mode=False, res=0.0):
        self.spin_numbers = spin_numbers
        self.kpoint_numbers = kpoint_numbers
        self.vbm = vbm  
        self.cbm = cbm  
        self.generate_x_labels = generate_x_labels
        self.band_mode = band_mode  
        self.res = res
        self.final_result = []

    def store_final_results(self, total_results):
        self.final_result = total_results.copy()

    def plot_ipr(self):
        #folder_name = os.path.basename(os.getcwd())
        #localized_folder = f'localized-defects/{folder_name}/Figures'
        #os.makedirs(localized_folder, exist_ok=True)

        content = '\n'.join(self.final_result[1:])
        blocks = content.strip().split('\n\n')

        if not self.spin_numbers or not self.kpoint_numbers:
            print("Error: Spin numbers or kpoint numbers are empty.")
            return

        total_combinations = len(self.spin_numbers) * len(self.kpoint_numbers)

        if len(blocks) > total_combinations:
            print(f"Warning: More blocks ({len(blocks)}) than combinations ({total_combinations}).")

        kpoint_vals_up, energy_vals_up, ipr_vals_up, band_numbers_up = [], [], [], []
        kpoint_vals_down, energy_vals_down, ipr_vals_down, band_numbers_down = [], [], [], []

        for i, block in enumerate(blocks):
            if i >= total_combinations:
                break

            data = pd.read_csv(StringIO(block), sep=r'\s+', header=None)

            """Column 0 ----> spin number, \
             Column 1 ----> kpoint number, \
             Column 2 ----> band number, \
             Column 3 ----> IPR, \
             Column 4 ----> Energy, \
             Column 5 ----> occupancies"""
            if data.shape[1] < 6:
                print(f"Warning: Block {i + 1} does not have enough columns.")
                continue

            spin_index = i // len(self.kpoint_numbers)
            kpoint_index = i % len(self.kpoint_numbers)

            spin = self.spin_numbers[spin_index]
            kpoint = self.kpoint_numbers[kpoint_index]

            rescaled_energy = np.array([valor - self.res for valor in data[4]])
            ipr_values = data[3]
            finite_mask = np.isfinite(ipr_values).to_numpy()

            if spin == 1:
                target_kpoint, target_energy, target_ipr, target_band = (
                    kpoint_vals_up, energy_vals_up, ipr_vals_up, band_numbers_up)
            else:
                target_kpoint, target_energy, target_ipr, target_band = (
                    kpoint_vals_down, energy_vals_down, ipr_vals_down, band_numbers_down)

            target_kpoint.extend([kpoint] * int(finite_mask.sum()))
            target_energy.extend(rescaled_energy[finite_mask])
            target_ipr.extend(ipr_values.to_numpy()[finite_mask])
            target_band.extend(data[2].to_numpy()[finite_mask].tolist())

        if not (energy_vals_up or energy_vals_down):
            print("Error: No valid data points to plot.")
            return

        fig, axs = plt.subplots(1, 2, figsize=(10, 8), constrained_layout=True)

        cmap = 'viridis'
        all_ipr_vals = ipr_vals_up + ipr_vals_down
        vmin, vmax = min(all_ipr_vals), max(all_ipr_vals)

        sc = None
        if kpoint_vals_up:
            sc = axs[0].scatter(kpoint_vals_up, energy_vals_up, c=ipr_vals_up, cmap=cmap,
                                 vmin=vmin, vmax=vmax, s=30)
        if kpoint_vals_down:
            sc_down = axs[1].scatter(kpoint_vals_down, energy_vals_down, c=ipr_vals_down, cmap=cmap,
                                      vmin=vmin, vmax=vmax, s=30)
            sc = sc if sc is not None else sc_down

        if self.band_mode:
            def label_bands(ax, kpoint_vals, energy_vals, band_numbers):
                unique_kpts = sorted(set(kpoint_vals))

                for kpt in unique_kpts:
                    idxs = [i for i, k in enumerate(kpoint_vals)
                            if k == kpt and self.vbm - self.res <= energy_vals[i] <= self.cbm - self.res]
                    if not idxs:
                        continue

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
                        chunks = [group_bands[c:c + 7] for c in range(0, len(group_bands), 7)]
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

            label_bands(axs[0], kpoint_vals_up, energy_vals_up, band_numbers_up)
            label_bands(axs[1], kpoint_vals_down, energy_vals_down, band_numbers_down)

        kpoint_labels = self.generate_x_labels()

        unique_kpoints = sorted(set(kpoint_vals_up + kpoint_vals_down))
        x_tick_labels = [kpoint_labels[unique_kpoints.index(kpt)] if kpt in unique_kpoints else ''
                          for kpt in unique_kpoints]

        all_energies = energy_vals_up + energy_vals_down
        y_min = min(all_energies) - 0.9
        y_max = max(all_energies) + 0.9

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

        cbar = fig.colorbar(sc, ax=axs, orientation='vertical', fraction=0.046, pad=0.04)

        output_file = 'eigenplot_localization-IPR.png'
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Saved figure: {output_file}")
