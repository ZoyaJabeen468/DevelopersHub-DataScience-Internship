# DevelopersHub Data Science Internship — Phase 1

**Intern:** Zoya Jabeen
**Organization:** DevelopersHub Corporation
**Due Date:** 26th June, 2026

---

## Tasks Completed (5/5)

### Task 1 — Exploring and Visualizing a Simple Dataset
**Objective:** Understand how to read, summarize, and visualize a dataset.

**Approach:**
- Loaded the Iris dataset using pandas and inspected structure with `.shape`, `.columns`, `.head()`
- Checked for missing values and class balance
- Created scatter plots, histograms, box plots, and a pair plot using matplotlib and seaborn

**Results & Insights:**
- No missing values found — dataset is clean
- Petal length and petal width almost perfectly separate the three species
- Setosa is clearly distinct; Virginica has the largest measurements overall

---

### Task 2 — Credit Risk Prediction
**Objective:** Predict whether a loan applicant is likely to default on a loan.

**Approach:**
- Handled missing values: mode for categorical columns, median for LoanAmount
- Visualized loan amount, education, income, credit history, and property area
- Engineered Total Income feature and applied log transforms to reduce skewness
- Trained Logistic Regression and Decision Tree classifiers
- Applied `class_weight='balanced'` to fix class imbalance bias

**Results & Insights:**
- Logistic Regression: 86.18% accuracy (baseline), Rejection Recall improved 0.58 → 0.63 after balancing
- Credit History is the single most important feature (~80% approval rate for good credit)
- Without balancing, models tend to ignore the minority class (Rejected loans)

---

### Task 3 — Customer Churn Prediction (Bank Customers)
**Objective:** Identify customers who are likely to leave the bank.

**Approach:**
- Dropped identifier columns (RowNumber, CustomerId, Surname)
- Label Encoded Gender (binary); One-Hot Encoded Geography (no ordinal relationship)
- Trained Logistic Regression and Random Forest with and without `class_weight='balanced'`
- Analyzed feature importance from the Random Forest model

**Results & Insights:**
- Random Forest Balanced: 81.9% accuracy, Churn Recall improved from 0.19 → 0.71
- Top churn drivers: Age, NumOfProducts, IsActiveMember, Balance, Geography (Germany)
- Without balancing, models missed 80% of churners despite showing ~80% accuracy

---

### Task 4 — Predicting Insurance Claim Amounts
**Objective:** Estimate the medical insurance claim amount based on personal data.

**Approach:**
- Label Encoded binary columns (sex, smoker); One-Hot Encoded region
- Visualized how BMI, age, and smoking status impact charges via scatter plots and box plots
- Trained a Linear Regression model and plotted actual vs predicted values and residuals

**Results & Insights:**
- MAE: $4,181 | RMSE: $5,796 | R² Score: 0.78
- Smoking status is by far the strongest predictor of insurance charges
- High-BMI smokers form a separate high-cost cluster — BMI and smoking interact, not just add up
- RMSE > MAE indicates high-cost smokers are the hardest cases to predict

---

### Task 5 — Personal Loan Acceptance Prediction
**Objective:** Predict which customers are likely to accept a personal loan offer.

**Approach:**
- Explored age, job, marital status, education, and campaign history
- Identified and removed `duration` column (data leakage — only known after the call)
- Trained Logistic Regression, Decision Tree, and Random Forest without leakage
- Ran a separate comparison with duration included to quantify the leakage impact
- Extracted feature importance and acceptance rates by customer segment

**Results & Insights:**
- Random Forest (realistic, no leakage): 72.5% accuracy
- Including `duration` artificially inflates accuracy to 84.6% — a 12-point misleading gain
- Previous campaign success is the strongest predictor (91.3% acceptance rate)
- Students and retirees accept most often; blue-collar and entrepreneur segments least

---

## Key Learnings

### 1. Class imbalance is a silent model killer
High accuracy can hide a failing model. In Tasks 2 and 3, models looked good at
80–86% accuracy but were missing 75–80% of the minority class. Fixed using
`class_weight='balanced'`, raising churn recall from 0.19 → 0.71.

### 2. Data leakage gives fake accuracy
In Task 5, the `duration` column (call length) inflated accuracy by 12 points.
It is only known after a call ends, so it cannot predict who to call. Removing it
gives a lower but honest and deployable model.

### 3. Choosing the right metric for the problem
- Imbalanced classification → Recall and F1-score matter more than accuracy
- Regression → MAE, RMSE, and R² tell the real story
- Business context → ask which mistake is more costly, then optimize for that

### 4. Feature engineering adds real value
Combining ApplicantIncome and CoapplicantIncome into Total_Income and applying
log transforms reduced skewness and improved model learning in Task 2.

### 5. Models should produce business recommendations
Feature importance and coefficient analysis (Tasks 3, 4, 5) were translated into
specific, actionable recommendations — which customer segments to target and why.

---

## Tools & Libraries
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

## Datasets
- Task 1: Iris Dataset (via seaborn)
- Task 2: Loan Prediction Dataset (Kaggle)
- Task 3: Churn Modelling Dataset
- Task 4: Medical Cost Personal Dataset
- Task 5: Bank Marketing Dataset (UCI Machine Learning Repository)
