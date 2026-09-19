--- 
# Examples of how to use the extra scripts stored in the [Extra-scripts](https://github.com/JosephPVera/Localized-States/tree/main/Extra-scripts) folder
---

So far, these scripts have not been implemented in the LSPD package.

**WARNING:** Only the inputs and outputs necessary to analyze the results are included here. If you wish to follow the steps required to perform the calculations, please check the [Guide-for-DFT-calculations](https://github.com/JosephPVera/Guide-for-DFT-calculations) repository.

---
# 1. VASP software
---

## 1.1. Folder tree
![Alt text](https://github.com/JosephPVera/Localized-States/blob/main/Examples/Figures/vasp.png)

## 1.2. PD folder

### 1.2.1. Primitive folder
After performing all the calculations, use the [primitive.py](https://github.com/JosephPVera/Localized-States/blob/main/Extra-scripts/primitive.py) script to extract the **VBM**, **CBM**, **gap**, and **dielectric tensor contributions**. This information is saved in a **primitive.json** file.

### 1.2.2. Competing phases (cpd) folder
After performing all the calculations, use the [chem_pot.py](https://github.com/JosephPVera/Localized-States/blob/main/Extra-scripts/chem_pot.py) script to extract the **compound**, **total energy**, **number of atoms**, and **total energy per atom**. This information is saved in a **chem_pot.json** file.

### 1.2.3. Defect folder
After performing all the calculations, the following steps must be carried out for each folder (**N_C-V_C_-3**, **N_C-V_C_-2**, **N_C-V_C_-1**, **N_C-V_C_0**, **N_C-V_C_1**, and **N_C-V_C_2**), except for the **perfect** folder. For example, in the [N_C-V_C_-1](https://github.com/JosephPVera/Localized-States/tree/main/Examples/VASP/diamond/PBE/PD/defect/N_C-V_C_-1) folder:

- Use the [corrections.py](https://github.com/JosephPVera/Localized-States/blob/main/Extra-scripts/corrections.py) script to extract the **charge state**, **lattice parameters**, **dielectric tensor**, **defect coordinates**, **point group for the defect**, **energy corrections**, and **potentials**. This information is saved in a **correction.json** file.
- Use the [corrections_plot.py](https://github.com/JosephPVera/Localized-States/blob/main/Extra-scripts/corrections_plot.py) script to plot the potential alignment.

  ![Alt text](https://github.com/JosephPVera/Localized-States/blob/main/Examples/VASP/diamond/PBE/PD/defect/N_C-V_C_-1/correction_plot.png)
  
- Use the [eigenplot.py](https://github.com/JosephPVera/Localized-States/blob/main/eigenplot.py) script to plot the Kohn-Sham level diagram.
  
  ![Alt text](https://github.com/JosephPVera/Localized-States/blob/main/Examples/VASP/diamond/PBE/PD/defect/N_C-V_C_-1/kohn-sham-states.png)
  
- Use the [locplot.py](https://github.com/JosephPVera/Localized-States/blob/main/locplot.py) script to plot the degree of localization using the Projected Density of States (PDOS).

  ![Alt text](https://github.com/JosephPVera/Localized-States/blob/main/Examples/VASP/diamond/PBE/PD/defect/N_C-V_C_-1/eigenplot_localization.png)

- If the **WAVECAR** have been saved, use the [ipr.py](https://github.com/JosephPVera/Localized-States/blob/main/ipr.py) script to plot the degree of localization using the Inverse Participation Ratio (IPR).

  ![Alt text](https://github.com/JosephPVera/Localized-States/blob/main/Examples/VASP/c-BN/HSE06/PD/defect/V_B-Si_B_-1/eigenplot_localization-IPR.png)

- If the **WAVECAR** have been saved, use the [ks-orbital.py](https://github.com/JosephPVera/Localized-States/blob/main/Extra-scripts/ks-orbital.py) script to extract the wavefunction of a desired band at a given k-point and spin channel. Check the example in the [V_B-Si_B_0](https://github.com/JosephPVera/Localized-States/tree/main/Examples/VASP/c-BN/HSE06/PD/defect/V_B-Si_B_0) folder, where the wavefunctions, such as **wfc_s2_k1_b430_r.vasp**, **wfc_s2_k1_b431_r.vasp**, and so on, are saved. These files can be used to determine the symmetry of the wavefunction for a given orbital with an assigned band number using the point group of the defect site, by inspecting how the isosurface transforms under the symmetry operations of that point group. For this purpose, the [point_groups.py](https://github.com/JosephPVera/Localized-States/blob/main/Extra-scripts/point_groups.py) script can be used, which works together with the [point_groups_lib.py](https://github.com/JosephPVera/Localized-States/blob/main/Extra-scripts/point_groups_lib.py) script.







---
# 2. Quantum ESPRESSO software
---

## 2.1. Folder tree
![Alt text](https://github.com/JosephPVera/Localized-States/blob/main/Examples/Figures/qe.png)
