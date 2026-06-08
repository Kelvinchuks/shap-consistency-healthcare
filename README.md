# Same Patient, Different Explanations? 
**A Comparative SHAP Study Across Models and Datasets in Healthcare Risk Prediction**

**Authors:** Sakshi Saggam & Toochukwu Okutalukwe  
**Domain:** Transparency, Explainability, and Accountability in AI (Data and Algorithmic Governance)

##  Project Objectives
As machine learning models move into high-stakes domains like healthcare, making predictions transparent and explainable is critical. This project provides a systematic, empirical test to evaluate **SHAP's (SHapley Additive exPlanations) consistency across ML models in healthcare**. 

**Primary Research Question:** When two structurally different machine learning models are trained on the same healthcare dataset and explained with SHAP, do they provide the same account of why a given prediction was made — and does any observed divergence replicate across different healthcare prediction tasks?

**Sub-Questions:**
1. **Within-dataset:** Do the models agree on which features drive predictions, both globally and locally?
2. **Cross-dataset (within disease):** Does the pattern of agreement/divergence replicate across different datasets for the same disease? 
3. **Cross-disease:** Is the divergence pattern consistent across disease domains (diabetes vs. heart disease), or is it disease-specific?
4. **Governance implications:** If explanations diverge, how must we re-evaluate the definition of “regulatory transparency” in clinical AI deployment?

## Architecture & Methodology
We evaluate explanation consistency by comparing two structurally distinct models:
* **XGBoost** (Tree-based ensemble)
* **Multilayer Perceptron (MLP)** (Neural Network built with PyTorch)

These models are trained on four healthcare datasets spanning two diseases to determine whether any explanation divergence is a property of the model architecture, the dataset, or both.

### Datasets Used
The analysis is conducted on the following datasets:
1. Pima Indians Diabetes Dataset
2. Diabetes Prediction Dataset 
3. UCI Heart Disease Dataset
4. Heart Disease Prediction Dataset (Framingham)

##  Prerequisites & Dependencies
The project is built using Python 3. To run the notebook and scripts, you will need the following libraries:
* `torch`
* `shap`
* `xgboost`
* `scikit-learn`
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scipy`

##  Setup and Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/shap-consistency-healthcare.git](https://github.com/your-username/shap-consistency-healthcare.git)
   cd shap-consistency-healthcare