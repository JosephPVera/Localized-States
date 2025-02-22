# Written by Joseph P.Vera
# 2025-02

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

class LocalizedPlotter:
    def __init__(self, spin_numbers, kpoint_numbers, vbm, cbm, tot_mode, band_mode=False, res=0.0):
        self.spin_numbers = spin_numbers
        self.kpoint_numbers = kpoint_numbers
        self.vbm = vbm  
        self.cbm = cbm  
        self.tot_mode = tot_mode 
        self.band_mode = band_mode 
        self.res = res
        self.final_result = []

    def store_final_results(self, total_results):
        "Store total results into final results."
        self.final_result = total_results.copy()

    def plot_localized(self):
        "Generate plots based on final results, spin numbers, and kpoint numbers."
        folder_name = os.path.basename(os.getcwd())
        localized_folder = f'localized-defects/{folder_name}/Figures'
        os.makedirs(localized_folder, exist_ok=True)

        content = '\n'.join(self.final_result[1:])
        blocks = content.strip().split('\n\n')

        if not self.spin_numbers or not self.kpoint_numbers:
            print("Error: Spin numbers or kpoint numbers are empty.")
            return

        total_combinations = len(self.spin_numbers) * len(self.kpoint_numbers)

        if len(blocks) > total_combinations:
            print(f"Warning: More blocks ({len(blocks)}) than combinations ({total_combinations}).")

        for i, block in enumerate(blocks):
            if i >= total_combinations:
                break

            data = pd.read_csv(StringIO(block), sep=r'\s+', header=None)

            # Validate number of columns
            if data.shape[1] < 7:
                print(f"Warning: Block {i + 1} does not have enough columns.")
                continue

            subset = data.iloc[:, [5, 3, 6]] if self.tot_mode else data.iloc[:, [5, 4, 6]]
            subset.columns = ['Energy', 'sum', 'occ']

            spin_index = i // len(self.kpoint_numbers)
            kpoint_index = i % len(self.kpoint_numbers)

            spin = self.spin_numbers[spin_index]
            kpoint = self.kpoint_numbers[kpoint_index]

            # Rescale the energy values
            rescaled_energy = [valor - self.res for valor in data[5]]

            gap_data = data[(np.array(rescaled_energy) >= self.vbm - self.res) & (np.array(rescaled_energy) <= self.cbm - self.res)]  
            band_numbers = gap_data[2].values  
            energies = np.array(rescaled_energy)[(np.array(rescaled_energy) >= self.vbm - self.res) & (np.array(rescaled_energy) <= self.cbm - self.res)]

            printed_bands = set()  # Track printed bands to avoid duplication

            # Plotting
            plt.figure(figsize=(10, 6))

            # Scatter points with band numbers next to them
            for j, (energy, sum_val, occ, band) in enumerate(zip(rescaled_energy, data[3] if self.tot_mode else data[4], data[6], data[2])):
                if np.isfinite(sum_val):
                    color = 'blue' if occ > 0.9 else 'red' if occ < 0.1 else 'green'
                    plt.scatter(energy, sum_val, marker='o', color=color)
                    
                    if self.band_mode:
                    # Check if the point is within the gap
                        if self.vbm - self.res <= energy <= self.cbm - self.res:
                        # Group similar band numbers
                            similar_bands = [band]
                            for energy2, band2 in zip(energies, band_numbers):
                                if abs(energy - energy2) <= 0.1 and band != band2 and self.vbm - self.res <= energy2 <= self.cbm - self.res:
                                    similar_bands.append(band2)
                        
                        # Sort and remove duplicates
                            similar_bands_sorted = sorted(set(similar_bands))
                        
                        # Only print once per unique set of band numbers
                            if tuple(similar_bands_sorted) not in printed_bands:
                                printed_bands.add(tuple(similar_bands_sorted))
                            # Label next to scatter point
                                plt.text(energy + 0.6, sum_val, ', '.join(map(str, similar_bands_sorted)), 
                                         fontsize=10, color='black')

            # Legend
            occupied_patch = plt.Line2D([0], [0], marker='o', color='w', label='Occupied', markerfacecolor='blue', markersize=10)
            unoccupied_patch = plt.Line2D([0], [0], marker='o', color='w', label='Unoccupied', markerfacecolor='red', markersize=10)
            partially_occupied_patch = plt.Line2D([0], [0], marker='o', color='w', label='Partially occupied', markerfacecolor='green', markersize=10)
            vbm_patch = plt.Line2D([0], [0], color='lightblue', label='VBM')
            cbm_patch = plt.Line2D([0], [0], color='thistle', label='CBM')
            plt.legend(handles=[occupied_patch, unoccupied_patch, partially_occupied_patch, vbm_patch, cbm_patch])

            # VBM and CBM shading
            plt.axvspan(subset['Energy'].min() - 0.9 - self.res, self.vbm - self.res, color='lightblue', alpha=0.4)
            plt.axvspan(self.cbm - self.res, subset['Energy'].max() + 0.9  + self.res, color='thistle', alpha=0.4)

            plt.xlabel('Energy (eV)', fontsize=14)
            plt.ylabel('Localization', fontsize=14)
            plt.xlim(subset['Energy'].min() - 0.9 - self.res, subset['Energy'].max() + 0.9 - self.res)

            # Title and Save Plot
            if spin == 1:
                plt.title(f'Spin up - kpoint {kpoint}', fontsize=14)
                plot_filename = f'Spin_up-kpoint_{kpoint}.png'
                output_file = os.path.join(localized_folder, plot_filename)
            else:
                plt.title(f'Spin down - kpoint {kpoint}')
                plot_filename = f'Spin_down-kpoint_{kpoint}.png'
                output_file = os.path.join(localized_folder, plot_filename)      
            plt.savefig(output_file, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"Saved figure: {output_file}")
