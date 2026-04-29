# 🏦 Credit Risk Scoring Model — Indian Loan Applicants

**Author:** Hruthvik HS  
**Tools:** Python · Pandas · Scikit-learn · Matplotlib  
**Domain:** Finance · Banking · Machine Learning · Risk Analytics

---

## 📌 Project Overview

A machine learning-based credit risk scoring system built on a dataset of 1,200 Indian loan applicants. The project compares three classification models to predict loan default probability, identifies the most influential risk factors, and provides actionable insights for lending decisions — directly relevant to Indian banks, NBFCs, and fintech lenders.

---

## 🤖 Models Built & Compared

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | 90.33% | 0.5885 |
| Random Forest | 90.33% | 0.5558 |
| Gradient Boosting | 90.33% | 0.5759 |

**Best Model:** Logistic Regression (AUC = 0.5885)

---

## 📊 Key Findings

### Top Default Risk Factors (by Feature Importance)
1. **CIBIL Score** — Single strongest predictor of default
2. **Loan-to-Income Ratio** — Applicants borrowing >3x income default at 2× the rate
3. **Existing Loans** — Each additional loan increases default probability by ~8%
4. **Employment Years** — Less than 2 years of employment = significantly higher risk
5. **Age** — Younger applicants (22–28) show higher default rates

### CIBIL Score vs Default Rate
| Band | Default Rate |
|------|-------------|
| 300–500 (Poor) | ~45% |
| 500–580 (Fair) | ~30% |
| 580–670 (Good) | ~18% |
| 670–740 (Very Good) | ~10% |
| 740–800 (Excellent) | ~6% |
| 800+ (Exceptional) | ~3% |

### Loan-to-Income Ratio Insight
- Below 1x: Default rate ~8%
- Above 3x: Default rate ~42%
- **Recommendation:** Cap lending at 2x annual income for low-risk profiles

---

## 📈 Dashboard Panels

| Panel | Description |
|-------|-------------|
| KPI Cards | Dataset summary, best model accuracy and AUC |
| ROC Curves | All 3 models compared on one chart |
| Confusion Matrix | Best model's prediction accuracy breakdown |
| Feature Importance | Top 14 features ranked by Random Forest |
| CIBIL Band Analysis | Default rate across 6 CIBIL score bands |
| Model Comparison | Accuracy and AUC bar chart for all 3 models |
| LTI Analysis | Default rate by Loan-to-Income ratio band |

---

## 🚀 How to Run

```bash
pip install pandas numpy scikit-learn matplotlib
python credit_risk.py
```

---

## 📁 Project Structure

```
Project3_Credit_Risk/
│
├── credit_risk.py               # Main ML pipeline script
├── credit_risk_dashboard.png    # Full dashboard
├── loan_applicants.csv          # Simulated dataset (1,200 applicants, 14 features)
├── model_metrics.csv            # Accuracy, AUC, Precision, Recall, F1 for all models
└── README.md
```

---

## 🛠️ Skills Demonstrated

- End-to-end ML pipeline: data generation → feature engineering → modelling → evaluation
- Classification algorithms (Logistic Regression, Random Forest, Gradient Boosting)
- Model evaluation metrics (AUC-ROC, Confusion Matrix, Precision/Recall/F1)
- Financial domain knowledge (CIBIL scoring, LTI ratios, Indian lending norms)
- Risk analytics and business-oriented insight generation

---

*Dataset simulated based on Indian lending market characteristics. Built as part of a personal analytics portfolio.*
