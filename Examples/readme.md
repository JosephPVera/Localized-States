--- 
# Examples to use the extra scripts store in the [Extra-scripts](https://github.com/JosephPVera/Localized-States/tree/main/Extra-scripts) folder
---

Steps for Quantum ESPRESSO calculations using **PBE** and **HSE06** functionals.

A guide to installing quantum ESPRESSO can be found in the [Quantum ESPRESSO repository](https://github.com/JosephPVera/Quantum_espresso_software).

---
# 1. PBE functional
---

## 1.1. Workflow
![Alt text](https://github.com/JosephPVera/Guide-for-DFT-calculations/blob/main/Quantum-ESPRESSO/Primitive/Figures/QE_workflow_pbe.png)


## 1.2. Convergence tests
The first step in obtaining accurate results is to perform convergence tests for parameters such as the **energy cutoff for wavefunctions**, **energy cutoff for charge density**, and **k-point 
mesh**. These parameters can be modified through the **ecutwfc** and **ecutrho** tags, as well as the **K_POINTS** section.  
