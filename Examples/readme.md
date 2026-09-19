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
After performing all the calculations, use the [primitive.py](https://github.com/JosephPVera/Localized-States/blob/main/Extra-scripts/primitive.py) script to extract the **VBM**, **CBM**, **gap**, and **dielectric tensor contributions**. This information is saved in a primitive.json file.


---
# 2. Quantum ESPRESSO software
---

## 2.1. Folder tree
![Alt text](https://github.com/JosephPVera/Localized-States/blob/main/Examples/Figures/qe.png)
