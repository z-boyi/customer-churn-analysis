# Customer Churn Analysis

## Objective
The goal of this project is to analyze customer behavior and identify key drivers of customer churn using data analytics, statistical testing, and interpretable modeling. The project focuses on translating data insights into actionable business recommendations.

---

## Dataset
- **Source**: Telco Customer Churn dataset
- **Observations**: ~7,000 customers
- **Features**: Demographics, service usage, billing information
- **Target variable**: Churn (1 = Yes, 0 = No)

---

## Tools & Technologies
- Python
- pandas, numpy
- matplotlib, seaborn
- scipy, statsmodels
- scikit-learn
- Jupyter Notebook

---

## Project Structure
```
.
├── data
│   ├── clean_churn.csv
│   └── raw_churn.csv
├── figures
│   ├── churn_by_constract.png
│   ├── churn_by_internet_service_type.png
│   ├── churn_by_payment_methods.png
│   ├── churn_by_tenure_group.png
│   ├── correlation_matrix.png
│   ├── monthly_charges_by_churn_status.png
│   └── roc_curve.png
├── notebooks
│   ├── 01_load_and_inspect.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_statistical_analysis.ipynb
│   └── 05_modeling.ipynb
├── README.md
├── requirements.txt
└── src

```

---

## Methodology
1. **Data Cleaning**
   - Converted `TotalCharges` to numeric and handled missing values
   - Removed non-informative identifiers
   - Engineered tenure-based features

2. **Exploratory Data Analysis (EDA)**
   - Analyzed churn patterns by contract type, tenure, payment method, and pricing
   - Visualized key relationships using bar charts and boxplots

3. **Statistical Analysis**
   - Chi-square tests for categorical variables
   - Two-sample t-tests for pricing differences
   - Logistic regression for statistical interpretation of churn drivers

4. **Modeling**
   - Built an interpretable logistic regression model
   - Evaluated performance using confusion matrix and ROC–AUC

---

## Key Findings
- Month-to-month contract customers have significantly higher churn rates
- Customers with tenure under 12 months are at highest risk of churn
- Higher monthly charges are associated with increased churn
- Automatic payment methods are linked to lower churn probability

---

## Business Recommendations
- Incentivize customers to switch from month-to-month to long-term contracts
- Implement early engagement programs for new customers
- Offer targeted pricing or discounts to high-risk segments

---

## Limitations
- No advanced hyperparameter tuning
- Class imbalance addressed implicitly
- Focus on interpretability over maximum predictive accuracy

---

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Run notebooks from `01_load_and_inspect.ipynb` to `05_modeling.ipynb`.
