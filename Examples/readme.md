--- 
# Examples of how to use the extra scripts stored in the [Extra-scripts](https://github.com/JosephPVera/Localized-States/tree/main/Extra-scripts) folder
---

So far, these scripts have not been implemented in the LSPD package.

**WARNING:** Only the inputs and outputs necessary to analyze the results are included here. If you wish to follow the steps required to perform the calculations, please check the [Guide-for-DFT-calculations](https://github.com/JosephPVera/Guide-for-DFT-calculations) repository.

---
# 1. VASP software
---

## 1.1. Tree folder
![Alt text](https://github.com/JosephPVera/Guide-for-DFT-calculations/blob/main/Quantum-ESPRESSO/Primitive/Figures/QE_workflow_pbe.png)


## 1.2. Convergence tests
The first step in obtaining accurate results is to perform convergence tests for parameters such as the **energy cutoff for wavefunctions**, **energy cutoff for charge density**, and **k-point 
mesh**. These parameters can be modified through the **ecutwfc** and **ecutrho** tags, as well as the **K_POINTS** section.  


---
# 2. Quantum ESPRESSO software
---
