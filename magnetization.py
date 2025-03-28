#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2024-10

import os
from collections import defaultdict

def get_magnetization(file_path):
    magnetizations = []

    with open(file_path, 'r') as file:
        for line in file:
            if "magnetization" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "magnetization" and i + 1 < len(parts):
                        try:
                            magnetizations.append(float(parts[i + 1]))
                        except ValueError:
                            continue
    # print the second-to-last one
    if len(magnetizations) >= 2:
        return magnetizations[-2]
    return None

def process_outcar_files(base_directory):
    results = []
    
    for root, _, files in os.walk(base_directory):
        if 'OUTCAR' in files:
            outcar_path = os.path.join(root, 'OUTCAR')
            magnetization_value = get_magnetization(outcar_path)
            folder_name = os.path.basename(root)
            results.append((folder_name, magnetization_value))

    # Grouping folders with similar names
    grouped_results = defaultdict(list)
    for folder, magnetization in results:
        prefix = ''.join([char for char in folder if char.isalpha()]) 
        grouped_results[prefix].append((folder, magnetization))

    # Print results in a grouped tabular format
    print(f"{'Folder name':<14} {'Magnetization(μB)':<18} {'Spin state'}")
    for prefix, group in sorted(grouped_results.items()):
#        print(f"\nGroup: {prefix}")
        print("-" * 44)
        for folder, magnetization in sorted(group):
            magnetization_show = f"{magnetization:.7f}" if magnetization is not None else "Nan"
            spin_state = round(float(magnetization_show) / 2, 1) if magnetization is not None else "Nan"
            print(f"{folder:<14} {magnetization_show:<18} {spin_state}")

base_directory = '.'  
process_outcar_files(base_directory)
