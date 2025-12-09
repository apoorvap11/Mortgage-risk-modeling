

# Mortgage Risk Modeling  
### Interpretable Machine Learning • Responsible AI • SHAP • Partial Dependence • Credit Risk Analytics

This project builds and explains machine learning models that predict **high-priced mortgage loans**, a critical factor in assessing borrower risk, fairness, and financial outcomes.  
The focus is not only on predictive performance but on **transparency, interpretability, and responsible ML practices**, skills essential for real-world credit risk modeling.

Originally developed as part of my graduate coursework, this repo has been fully refined into a **professional, portfolio-quality project** demonstrating end-to-end responsible machine learning.

---

## Key Highlights

- **Interpretable Models**: Elastic Net Logistic Regression, Explainable Boosting Machine (EBM), Monotonic XGBoost  
- **Model Explainability**: SHAP values, feature scores, partial dependence plots  
- **Responsible ML**: Transparent modeling, monotonic constraints, fairness-aware interpretations  
- **End-to-End Workflow**: Data exploration → model training → explainability → insights  
- **Regulator-Aligned Analysis**: Methods suitable for credit decisioning, model audits, and compliance teams  

---

## Project Objective

Predict the likelihood that a mortgage is **high-priced**, defined as a loan with an annual percentage rate significantly above comparable market rates.  
This prediction task matters because high-priced loans disproportionately affect vulnerable groups, making interpretability and fairness essential.

This project answers:
- **What features most influence mortgage pricing risk?**  
- **How do different models behave for different borrowers?**  
- **Are the model explanations consistent with financial intuition?**

---

## Tools & Technologies

- **Python**
- **h2o** (Elastic Net GLM)
- **XGBoost** with monotonic constraints
- **InterpretML** (Explainable Boosting Machine)
- **SHAP**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Jupyter Notebook**

---

## 📂 Repository Structure


---

# 📊 Models Trained

### **1. Elastic Net Logistic Regression (h2o)**
- Highly interpretable
- Provides coefficient-based global importance
- Strong performance benchmark

### **2. Explainable Boosting Machine (EBM)**
- Transparent additive model
- Provides intuitive feature shape functions
- Excellent for compliance-driven modeling

### **3. Monotonic Gradient Boosting (XGBoost)**
- Enforces domain-informed monotonic relationships  
- Provides SHAP explanations for both global and local importance  
- Stronger predictive performance with interpretable constraints  

Each model outputs a **probability score (`phat`)** representing whether a loan is likely to be high-priced.

---

# 📈 Global Feature Importance

Techniques used:
- Regression coefficients  
- SHAP global importance  
- EBM main effect magnitudes  

These methods reveal **which borrower and loan characteristics drive higher pricing risk**.

📁 *See reports/global_feature_importance.png*

---

# 📍 Local Feature Importance  
Explained predictions for individuals at:
- 10th percentile risk  
- 50th percentile risk  
- 90th percentile risk  

Tools used:
- SHAP local explanations  
- EBM local feature scores  
- Logistic regression marginal contributions  

This reveals how features shift risk **for different borrower profiles**.

📁 *See reports/local_feature_importance.png*

---

# 🔍 Feature Behavior Analysis

To understand **how features affect predictions**, the project includes:
- Partial Dependence Plots (PDP)
- EBM shape functions  
- Cross-model behavioral comparisons  

These plots help validate whether model behavior aligns with **financial intuition**.

📁 *See reports/partial_dependence_plots/*

---

# ▶️ How to Run This Project

1. Install dependencies:
```bash
pip install -r requirements.txt


