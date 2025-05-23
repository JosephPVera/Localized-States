# Localized States in Point Defects (LSPD)
---- 

**Motivation**: Working with point defects involves using supercells (sometimes too larges), which can be a challenge in terms of information processing. Performing the analysis manually for each defect with several charges states is tedious (time consuming). Luckely, programming can help us with that.

----
The package contains a serie of scripts designed to perform a post-analysis when working with point defects in VASP. Their main functionalities are focused on:
- Plotting the Kohn-Sham states
- Plotting how localized the Kohn-Sham states are in the gap either using the Projected DOS onto s, p, and d orbitals or the Inverse Participation Ratio (IPR).
- Review which atoms (indexs) contribute to this high localization. 
- Additionally, other useful scripts are included.
  
## 1. Guide to reading files in relation to figures
### 1.1. Kohn-Sham states
![Alt text](https://github.com/JosephPVera/Localized-States/blob/main/tests/Guide/Figures/guide.png)

### 1.2. Localized states
![Alt text](https://github.com/JosephPVera/Localized-States/blob/main/tests/Guide/Figures/Spin_up-kpoint_1.png)
![Alt text](https://github.com/JosephPVera/Localized-States/blob/main/tests/Guide/Figures/Spin_down-kpoint_4.png)

## 2. Usage
### 2.1. Kohn-Sham states
Plot the Kohn-Sham states with [eigenplot.py](https://github.com/JosephPVera/Localized-States/blob/main/eigenplot.py). Check the script to use rescaling, it can be changed to **res = 0** or **res = vbm**.
![Alt text](https://github.com/JosephPVera/Localized-States/blob/main/tests/kohn-sham-states.png)

Use the **--band** tag to include band numbers in the gap.
![Alt text](https://github.com/JosephPVera/Localized-States/blob/main/tests/kohn-sham-states-band.png)

Use the **--split** tag to split the degenerate states, this only works for the Gamma point.
![Alt text](https://github.com/JosephPVera/Localized-States/blob/main/tests/kohn-sham-states-degenerate.png)

### 2.2. Visualize the localized states
Plot the below figures with [locplot.py](https://github.com/JosephPVera/Localized-States/blob/main/locplot.py). The **locplot.py** script takes the sum of the 5 heaviest values (most contribution) ​​in each band per k-point (Energy versus sum), it also can be change for check the total contribution (tot) for each band per k-point (Energy versus tot) using the **--tot** tag. Once again, you can change the scale via **res = 0** or **res = vbm**.
1. Spin up
![Alt text](https://github.com/JosephPVera/Localized-States/blob/main/tests/Spin_up-kpoint_1.png)
2. Spind down
![Alt text](https://github.com/JosephPVera/Localized-States/blob/main/tests/Spin_down-kpoint_1.png)

Use the **--band** tag to include band numbers in the gap.
1. Spin up
![Alt text](https://github.com/JosephPVera/Localized-States/blob/main/tests/Spin_up-kpoint_1-band.png)
2. Spind down
![Alt text](https://github.com/JosephPVera/Localized-States/blob/main/tests/Spin_down-kpoint_1-band.png)

### 2.3. IPR
Inverse Participation Ratio (IPR) can also be used to plot the localized states. To obtain the plots, the **WAVECAR** is required. It is important to highlight that the script for this subsection is used from [VaspBandUnfolding](https://github.com/QijingZheng/VaspBandUnfolding/blob/master/vaspwfc.py), so it must download from there. **Note**: Simply download and copy it into the LSPD module; the other processing and plotting functions are already adapted.

Once again, you can change the scale via **res = 0** or **res = vbm**, and print the band index using **--band** tag. On the other hand, If the calculations were performed using multiple k-points, the [ipr.py](https://github.com/JosephPVera/Localized-States/blob/main/ipr.py) script works by default. However, if the calculations were performed using only the gamma point, the **--gamma** tag must be used.
1. Spin up
![Alt text](https://github.com/JosephPVera/Localized-States/blob/main/tests/IPR-Spin_up-kpoint_1.png)   
2. Spin down
![Alt text](https://github.com/JosephPVera/Localized-States/blob/main/tests/IPR-Spin_down-kpoint_1.png)

### 2.4. Localized states
Review in detail which atoms (indexes) contribute to the high localization of states at each band number in the gap with [localized.py](https://github.com/JosephPVera/Localized-States/blob/main/localized.py) (it can be extended by modifying VBM and CBM). Check the example [localized_Va_N1_2.dat](https://github.com/JosephPVera/Localized-States/blob/main/tests/localized_Va_N1_2.dat) file.
   ```bash
   Defect: Va_N1_2

   VBM = 7.2945 eV
   CBM = 11.7449 eV


   ###########################################################
              vasprun.xml file (EIGENVAL information)                            
   ###########################################################
   ###########################################################
                              SPIN UP                     
   ###########################################################
   ###########################################################
                             kpoint 1                   
   ###########################################################
   Band       Energy         Occ        Occupancy
   429        7.943600       0.506200   Partially Occupied
   430        11.375200      0.000000   Unoccupied
   431        11.447100      0.000000   Unoccupied
   432        11.447100      0.000000   Unoccupied
                             ...
                             ...
                             ...
                             ...
   ########################################################################
                     vasprun.xml file (PROCAR information)
   ########################################################################
   ########################################################################
                                  SPIN UP                                  
   ########################################################################

   ########################################################################
                                  KPOINT 1                             
   ########################################################################

   Information of band 429:
   index  s          p          d          tot       

   Information of band 430:
   index  s          p          d          tot       
   41     0.014      0.247      0.000      0.261     

   Information of band 431:
   index  s          p          d          tot       
   26     0.013      0.232      0.000      0.245     

   Information of band 432:
   index  s          p          d          tot       
   71     0.010      0.174      0.000      0.184     
   104    0.010      0.174      0.000      0.184
                             ...
                             ...
                             ...
                             ...
   ```

### 2.5. Neighbors of the defect
Use the [defects.py](https://github.com/JosephPVera/Localized-States/blob/main/defects.py) script to checks if the localized states belong to ions that are close to or neighboring the defect. Check the example [neighbor_atoms.dat](https://github.com/JosephPVera/Localized-States/blob/main/tests/neighbor_atoms.dat) file.
   ```bash
   Vacancy: V_N
   Index in ../perfect/POSCAR: 149
   Position: [0.58333333 0.58333333 0.41666667]

   Closest neighbors to the V_N defect in Va_N1_2/POSCAR:
   Index      Atom       Position                       Distance (A)
   41         B          0.500000 0.500000 0.333333     1.5699    
   71         B          0.500000 0.666667 0.500000     1.5699    
   104        B          0.666667 0.500000 0.500000     1.5699    
   26         B          0.666667 0.666667 0.333333     1.5699 
   ```


---
# Enjoy your outcomes
---
# Disfruta tus resultados
---
