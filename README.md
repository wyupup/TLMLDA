# TLMLDA
## Ablation experiments of TLMLDA
This project currently consists of two ablation experiments (TLMLDA_α & TLMLDA_M), complete experiments and more details coming soon.

TLMLDA_α: TLMLDA is only trained using the fixed hyperparameters in the dynamic weight factor.

TLMLDA_M: TLMLDA uses MMD for feature migration.

The results of the ablation experiments are as follows：
Algorithm | E—>B | B—>E | E—>C | C—>E | B—>C | C—>B  
:-----:|:-----:|:-----:|:----------:|:----:|:-----:|:--------:|
TLMLDA_α | 54.40 | 34.42 | 31.60 | 30.60 | 32.70 | 52.21 
TLMLDA_M | 57.04 | 38.34 | 34.90 | 29.29 | 32.70 | 56.84 
TLMLDA  | **59.73** | **43.19** | **35.40** | **32.77** | **41.00** | **58.58** 

