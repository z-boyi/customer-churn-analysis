# Customer Churn Analysis

## Project Overview
This project analyzes customer churn behavior using a real-world telecommunications dataset.  
The goal is to identify key factors associated with customer churn and translate data insights into actionable business recommendations.

The project follows a structured data analytics workflow:
1. Data loading and inspection
2. Data cleaning and preprocessing
3. Exploratory data analysis (EDA)
4. Statistical analysis
5. Predictive modeling
6. Business insights and recommendations

---

## Dataset
- **Name**: Telco Customer Churn
- **Source**: IBM Sample Data / Kaggle
- **Size**: ~7,000 customer records
- **Description**:  
  The dataset contains customer demographics, service usage, contract details, and billing information, along with a binary churn indicator.

---

## Tools and Libraries
- Python
- pandas and NumPy for data manipulation
- matplotlib and seaborn for visualization
- scipy and statsmodels for statistical analysis
- Jupyter Notebook for analysis and documentation
- Git and GitHub for version control

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
│   └── monthly_charges_by_churn_status.png
├── notebooks
│   ├── 01_load_and_inspect.ipynb
│   ├── 02_data_cleaning.ipynb
│   └── 03_eda.ipynb
├── README.md
├── requirements.txt
└── src
```

---

## Workflow
1. **Data Loading & Inspection**
   The initial stage of the project focused on loading the raw dataset and performing a comprehensive data inspection to understand its structure and quality.

    Key steps included:
    - Loading the raw Telco Customer Churn dataset into Python using pandas
    - Examining dataset dimensions, column names, and data types
    - Identifying categorical and numerical variables
    - Inspecting the target variable distribution to assess class balance
    - Checking for missing values and duplicate records

    Initial inspection showed that the dataset contains approximately 7,000 customer records with a mix of demographic, service, and billing features. The churn variable is moderately imbalanced, with roughly one quarter of customers having churned. Most features are categorical, indicating the need for encoding and careful preprocessing in later stages. One billing variable, `TotalCharges`, was identified as requiring further cleaning due to incorrect data type representation.

2. **Data Cleaning**
   Key data preparation steps included:
    - Converting `TotalCharges` from string to numeric format
    - Removing invalid records with missing total charges
    - Encoding the churn variable as a binary indicator
    - Standardizing categorical variables
    - Creating tenure groups for customer segmentation

    The cleaned dataset is saved as `clean_churn.csv` for reproducibility.


3. **Exploratory Data Analysis**
   Exploratory analysis revealed several strong churn patterns:

    - Customers on month-to-month contracts exhibit significantly higher churn
    - New customers with tenure below 12 months are most likely to churn
    - Higher monthly charges are associated with increased churn risk
    - Customers using automatic payment methods tend to churn less
    - Fiber optic service customers show higher churn than DSL users

    These findings highlight both behavioral and pricing factors related to customer retention.


4. **Statistical Analysis & Modeling (next)**
   - Hypothesis testing
   - Logistic regression
   - Model interpretation

---

## Key Data Cleaning Decisions
- Rows with invalid `TotalCharges` values were removed due to minimal impact (<0.2%)
- Customer identifiers were excluded from analysis
- Tenure was grouped to support customer segmentation

---

## Status
🚧 In progress  
- Data cleaning completed  
- EDA and modeling in progress