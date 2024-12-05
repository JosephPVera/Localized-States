# Written by Joseph P.Vera
# 2024-11

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

class IPRPlotter:
    def __init__(self, spin_numbers, kpoint_numbers, vbm, cbm):#, tot_mode):
        self.spin_numbers = spin_numbers
        self.kpoint_numbers = kpoint_numbers
        self.vbm = vbm  
        self.cbm = cbm  
#        self.tot_mode = tot_mode  
        self.final_result = []

    def store_final_results(self, total_results):
        "Store total results into final results."
        self.final_result = total_results.copy()

    def plot_ipr(self):
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

            """Column 0 ----> spin number, \
             Column 1 ----> kpoint number, \
             Column 2 ----> band number, \
             Column 3 ----> IPR, \
             Column 4 ----> Energy, \
             Column 5 ----> occupancies"""
            if data.shape[1] < 6:
                print(f"Warning: Block {i + 1} does not have enough columns.")
                continue

            subset = data.iloc[:, [4, 3, 5]] #if self.tot_mode else data.iloc[:, [4, 3, 5]]
            subset.columns = ['Energy', 'ipr', 'occ']

            spin_index = i // len(self.kpoint_numbers)
            kpoint_index = i % len(self.kpoint_numbers)

            spin = self.spin_numbers[spin_index]
            kpoint = self.kpoint_numbers[kpoint_index]

            # Plotting
            plt.figure(figsize=(10, 6))

            for j in range(len(subset['ipr'])):
                if np.isfinite(subset['ipr'].iloc[j]):
                    color = 'blue' if subset['occ'].iloc[j] > 0.9 else 'red' if subset['occ'].iloc[j] < 0.1 else 'green'
                    plt.scatter(subset['Energy'].iloc[j], subset['ipr'].iloc[j], marker='o', color=color)

            # Legend
            occupied_patch = plt.Line2D([0], [0], marker='o', color='w', label='Occupied', markerfacecolor='blue', markersize=10)
            unoccupied_patch = plt.Line2D([0], [0], marker='o', color='w', label='Unoccupied', markerfacecolor='red', markersize=10)
            partially_occupied_patch = plt.Line2D([0], [0], marker='o', color='w', label='Partially occupied', markerfacecolor='green', markersize=10)
            vbm_patch = plt.Line2D([0], [0], color='lightblue', label='VBM')
            cbm_patch = plt.Line2D([0], [0], color='thistle', label='CBM')
            plt.legend(handles=[occupied_patch, unoccupied_patch, partially_occupied_patch, vbm_patch, cbm_patch])

            plt.axvspan(subset['Energy'].min() - 0.9, self.vbm, color='lightblue', alpha=0.4)
            plt.axvspan(self.cbm, subset['Energy'].max() + 0.9, color='thistle', alpha=0.4)

            plt.xlabel('Energy (eV)', fontsize=14)
            plt.ylabel('Inverse Participation Ratio (IPR)', fontsize=14)
            plt.xlim(subset['Energy'].min() - 0.9, subset['Energy'].max() + 0.9)

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
#            print("Saving figures ... ")
            print(f"Saved figure: {output_file}")