# ANN-WQI-ANN-Project
ANN-based Water Quality Index (WQI) prediction using LM, SCG, and BR algorithms with performance analysis
# Water Quality Prediction using ANN (WQI)

##  1. Paper Details

**Title:** Prediction of Water Quality Index using Artificial Neural Network  
**Based on:** Artificial Neural Network (ANN)  
**Journal:** Materials Today: Proceedings  
**Publisher:** Elsevier  
**Year:** 2023 (Online available: 2021)  

---

##  2. What does this paper do?

This paper uses Artificial Neural Network (ANN) to predict Water Quality Index (WQI) in a fast and accurate way and provides an alternative to traditional methods.

###  Goal:
- Predict water quality efficiently  
- Reduce complexity of traditional WQI calculation  
- Improve accuracy using ANN  

---

##  3. Overview (Pipeline)

###  Step 1: Data Loading
- Water quality data collected from Godavari River  
- 672 samples from 14 stations  

###  Step 2: Data Preprocessing
- Cleaning data  
- Normalization  
- Handling missing values  

###  Step 3: Imbalance Handling
- Dataset balanced (if needed)

###  Step 4: Model Building
- ANN model created  
- Hidden neurons tested (5–15)

###  Step 5: Algorithms Used
- Levenberg–Marquardt (LM)  
- Scaled Conjugate Gradient (SCG)  
- Bayesian Regularization (BR)

###  Step 6: Evaluation
- MSE (Mean Squared Error)  
- Correlation Coefficient (Cc)

---

##  4. Results Summary

| Parameter | Best Value |
|----------|-----------|
| Best Algorithm | LM |
| Epochs | 1000 |
| Data Split | 75% Train / 15% Validation / 10% Test |
| Hidden Neurons | 10 |
| Best MSE | ~1.11 |
| Correlation | ~0.998 |

 **Conclusion:** LM algorithm performed best.

---

##  5. Code (Step-wise Example)

# Step 1: Import Libraries
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
