# Written by Joseph P.Vera
# 2024-11

from ase.io import read
import numpy as np
import os

class DefectAnalysis:
    def __init__(self, defect_file= "POSCAR_defect", perfect_file= "POSCAR_perfect"):
        self.defect_file = defect_file
        self.perfect_file = perfect_file

        self.poscar_defect = read(defect_file)
        self.poscar_perfect = read(perfect_file)
        self.lattice_matrix = self.poscar_defect.get_cell()
        self.frac_positions_defect = self.poscar_defect.get_scaled_positions()
        self.frac_positions_perfect = self.poscar_perfect.get_scaled_positions()
        self.symbols_defect = self.poscar_defect.get_chemical_symbols()
        self.symbols_perfect = self.poscar_perfect.get_chemical_symbols()
        self.tolerance = 0.001

    def cartesian_distance(self, frac_pos_a, frac_pos_b):
        cart_pos_a = np.dot(frac_pos_a, self.lattice_matrix)
        cart_pos_b = np.dot(frac_pos_b, self.lattice_matrix)
        return np.linalg.norm(cart_pos_a - cart_pos_b)

    def find_vacancy(self):
        vacancies = []
        for i, pos_a in enumerate(self.frac_positions_perfect):
            found = False
            for pos_b in self.frac_positions_defect:
                if self.cartesian_distance(pos_a, pos_b) < self.tolerance:
                    found = True
                    break
            if not found:
                vacancies.append((self.symbols_perfect[i], pos_a, i + 1))
        return vacancies

    def find_susbstitutional(self):
        susbstitutional = []
        for i, pos_a in enumerate(self.frac_positions_defect):
            for j, pos_b in enumerate(self.frac_positions_perfect):
                if self.cartesian_distance(pos_a, pos_b) < self.tolerance:
                    if self.symbols_defect[i] != self.symbols_perfect[j]:
                        susbstitutional.append((self.symbols_defect[i], self.symbols_perfect[j], pos_a, i + 1, j + 1))
                    break
        return susbstitutional

    def find_interstitial(self):
        interstitial = []
        for i, pos_a in enumerate(self.frac_positions_defect):
            found = False
            for pos_b in self.frac_positions_perfect:
                if self.cartesian_distance(pos_a, pos_b) < self.tolerance:
                    found = True
                    break
            if not found:
                interstitial.append((self.symbols_defect[i], pos_a, i + 1))
        return interstitial

    def find_closest_atoms(self, target_frac_position):
        distances = []
        for i, frac_pos in enumerate(self.frac_positions_defect):
            if not np.array_equal(target_frac_position, frac_pos):
                distance = self.cartesian_distance(target_frac_position, frac_pos)
                distances.append((distance, self.symbols_defect[i], frac_pos, i + 1))

        distances.sort(key=lambda x: x[0])

        if distances:
            closest_distance = distances[0][0]
            same_distance_atoms = [atom for atom in distances if np.isclose(atom[0], closest_distance, atol=0.001)]
            return same_distance_atoms
        return []

    def print_closest_to_vacancy(self):
        vacancies = self.find_vacancy()
        folder_name = os.path.basename(os.getcwd())
        localized_folder = f'localized-defects/{folder_name}/Data'
        if not os.path.exists(localized_folder):
            os.makedirs(localized_folder)

        if vacancies:
            for symbol, frac_position, index in vacancies:
                print(f"\nVacancy: V_{symbol}")
                print(f"Index in /perfect/POSCAR: {index}")
                print(f"Position: {frac_position}")

                closest_atoms = self.find_closest_atoms(frac_position)
                print(f"\nClosest neighbors to the V_{symbol} defect in {folder_name}/POSCAR:")
                print(f"{'Index':<10} {'Atom':<10} {'Position':<30} {'Distance (A)':<10}")

                for distance, neighbor_symbol, neighbor_frac_position, neighbor_index in closest_atoms:
                    frac_pos_str = ' '.join(f"{coord:.6f}" for coord in neighbor_frac_position)
                    print(f"{neighbor_index:<10} {neighbor_symbol:<10} {frac_pos_str:<30} {distance:<10.4f}")

    def print_closest_to_substitutional(self):
        vacancies = self.find_vacancy()
        susbstitutional = self.find_susbstitutional()
        folder_name = os.path.basename(os.getcwd())
        localized_folder = f'localized-defects/{folder_name}/Data'
        if not os.path.exists(localized_folder):
            os.makedirs(localized_folder)

        if susbstitutional:
            for new_symbol, old_symbol, frac_position, old_index, new_index in susbstitutional:
                print("\n##################################################################")
                print(f"\nSubstitutional: {new_symbol}_{old_symbol}")
                print(f"Index in /perfect/POSCAR: {new_index}")
                print(f"Index in {folder_name}/POSCAR: {old_index}")
                print(f"Position: {frac_position}")

                closest_atoms = self.find_closest_atoms(frac_position)
                for missed_atom in vacancies:
                    missed_symbol, missed_position, missed_index = missed_atom
                    if self.cartesian_distance(frac_position, missed_position) < self.tolerance:
                        closest_atoms.append((self.cartesian_distance(frac_position, missed_position), missed_symbol, missed_position, missed_index))

                if closest_atoms:
                    closest_distance = closest_atoms[0][0]
                    same_distance_atoms = [atom for atom in closest_atoms if np.isclose(atom[0], closest_distance)]
                    same_distance_atoms.sort(key=lambda x: x[0])

                    print(f"\nClosest neighbors to the {new_symbol}_{old_symbol} defect in {folder_name}/POSCAR:")
                    print(f"{'Index':<10} {'Atom':<10} {'Position':<30} {'Distance (A)':<10}")
                    for distance, neighbor_symbol, neighbor_frac_position, neighbor_index in same_distance_atoms:
                        frac_pos_str = ' '.join(f"{coord:.6f}" for coord in neighbor_frac_position)
                        print(f"{neighbor_index:<10} {neighbor_symbol:<10} {frac_pos_str:<30} {distance:<10.4f}")

    def print_closest_to_interstitial(self):
        interstitial = self.find_interstitial()
        folder_name = os.path.basename(os.getcwd())
        localized_folder = f'localized-defects/{folder_name}/Data'
        if not os.path.exists(localized_folder):
            os.makedirs(localized_folder)

        if interstitial:
            for symbol, frac_position, index in interstitial:
                print("\n##################################################################")
                print(f"Interstitial: {symbol}_i")
                print(f"Index in {folder_name}/POSCAR: {index}")
                print(f"Position: {frac_position}")

                closest_atoms = self.find_closest_atoms(frac_position)
                print(f"\nClosest neighbors to the {symbol}_i defect in {folder_name}/POSCAR:")
                print(f"{'Index':<10} {'Atom':<10} {'Position':<30} {'Distance (A)':<10}")
                for distance, neighbor_symbol, neighbor_frac_position, neighbor_index in closest_atoms:
                    frac_pos_str = ' '.join(f"{coord:.6f}" for coord in neighbor_frac_position)
                    print(f"{neighbor_index:<10} {neighbor_symbol:<10} {frac_pos_str:<30} {distance:<10.4f}")
            
    def save_defect_data(self):
        vacancies = self.find_vacancy()
        susbstitutional = self.find_susbstitutional()
        interstitial = self.find_interstitial()
        
        folder_name = os.path.basename(os.getcwd())
        localized_folder = f'localized-defects/{folder_name}/Data'
        if not os.path.exists(localized_folder):
            os.makedirs(localized_folder)

        output_file = os.path.join(localized_folder, f'neighbor_atoms.dat')

        with open(output_file, "w") as file:
            if vacancies:
                for symbol, frac_position, index in vacancies:
                    file.write(f"\nVacancy: V_{symbol}\n")
                    file.write(f"Index in ../perfect/POSCAR: {index}\n")
                    file.write(f"Position: {frac_position}\n")

                    closest_atoms = self.find_closest_atoms(frac_position)
                    file.write(f"\nClosest neighbors to the V_{symbol} defect in {folder_name}/POSCAR:\n")
                    file.write(f"{'Index':<10} {'Atom':<10} {'Position':<30} {'Distance (A)':<10}\n")
                    for distance, neighbor_symbol, neighbor_frac_position, neighbor_index in closest_atoms:
                        frac_pos_str = ' '.join(f"{coord:.6f}" for coord in neighbor_frac_position)
                        file.write(f"{neighbor_index:<10} {neighbor_symbol:<10} {frac_pos_str:<30} {distance:<10.4f}\n")

            if susbstitutional:
                for new_symbol, old_symbol, frac_position, old_index, new_index in susbstitutional:
                    file.write("\n##################################################################\n")
                    file.write(f"\nSubstitutional: {new_symbol}_{old_symbol}\n")
                    file.write(f"Index in ../perfect/POSCAR: {new_index}\n")
                    file.write(f"Index in {folder_name}/POSCAR: {old_index}\n")
                    file.write(f"Position: {frac_position}\n")

                    closest_atoms = self.find_closest_atoms(frac_position)
                    for missed_atom in vacancies:
                        missed_symbol, missed_position, missed_index = missed_atom
                        if self.cartesian_distance(frac_position, missed_position) < self.tolerance:
                            closest_atoms.append((self.cartesian_distance(frac_position, missed_position), missed_symbol, missed_position, missed_index))

                    if closest_atoms:
                        closest_distance = closest_atoms[0][0]
                        same_distance_atoms = [atom for atom in closest_atoms if np.isclose(atom[0], closest_distance)]
                        same_distance_atoms.sort(key=lambda x: x[0])

                        file.write(f"\nClosest neighbors to the {new_symbol}_{old_symbol} defect in {folder_name}/POSCAR:\n")
                        file.write(f"{'Index':<10} {'Atom':<10} {'Position':<30} {'Distance (A)':<10}\n")
                        for distance, neighbor_symbol, neighbor_frac_position, neighbor_index in same_distance_atoms:
                            frac_pos_str = ' '.join(f"{coord:.6f}" for coord in neighbor_frac_position)
                            file.write(f"{neighbor_index:<10} {neighbor_symbol:<10} {frac_pos_str:<30} {distance:<10.4f}\n")

            if interstitial:
                for symbol, frac_position, index in interstitial:

                    file.write("\n##################################################################\n")
                    file.write(f"Interstitial: {symbol}_i\n")
                    file.write(f"Index in {folder_name}/POSCAR: {index}\n")
                    file.write(f"Position: {frac_position}\n")

                    closest_atoms = self.find_closest_atoms(frac_position)
                    file.write(f"\nClosest neighbors to the {symbol}_i defect in {folder_name}/POSCAR:\n")
                    file.write(f"{'Index':<10} {'Atom':<10} {'Position':<30} {'Distance (A)':<10}\n")
                    for distance, neighbor_symbol, neighbor_frac_position, neighbor_index in closest_atoms:
                        frac_pos_str = ' '.join(f"{coord:.6f}" for coord in neighbor_frac_position)
                        file.write(f"{neighbor_index:<10} {neighbor_symbol:<10} {frac_pos_str:<30} {distance:<10.4f}\n")

        print("\nDefect information was saved in neighbor_atoms.dat")
