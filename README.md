Bank Loan Approval Prediction
A machine learning project that predicts whether a bank loan application
is likely to be approved or rejected based on applicant and
loan-related information.
The project is implemented as a Jupyter Notebook and uses a dataset
containing 1,000 loan applications. It demonstrates a complete
beginner-to-intermediate machine learning workflow: data inspection,
cleaning, exploratory data analysis, preprocessing, model training, and
model evaluation.
> **Note:** This project is educational/demo-oriented. The model results
> should not be treated as a production credit-decision system.
Project Overview
The goal is to build classification models that learn from historical
loan application data and predict the `Loan_Status` of an application.
The notebook compares four machine learning algorithms:
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
K-Nearest Neighbors (KNN)
The models are evaluated using:
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Dataset
File: `Loan_Applicants_1000.xlsx`
The dataset contains 1,000 records and 13 columns.
Features
Column                Description
---
`Loan_ID`             Unique identifier for the loan application
`Gender`              Applicant gender
`Married`             Whether the applicant is married
`Dependents`          Number of dependents; `3+` is converted to `3`
`Education`           Graduate or Not Graduate
`Self_Employed`       Whether the applicant is self-employed
`ApplicantIncome`     Applicant's income
`CoapplicantIncome`   Co-applicant's income
`LoanAmount`          Requested loan amount
`Loan_Amount_Term`    Loan repayment term
`Credit_History`      Credit history indicator
`Property_Area`       Rural, Semiurban, or Urban
`Loan_Status`         Target: approved (`Y`) or rejected (`N`)
Dataset characteristics
Rows: 1,000
Columns: 13
Numeric columns: 5
Categorical/object columns: 8
Missing values in the supplied dataset: 0
Duplicate rows reported by the notebook: 0
IQR outliers reported by the notebook for `ApplicantIncome`,
`CoapplicantIncome`, and `LoanAmount`: 0
The supplied dataset is also strongly skewed toward approved
applications. The notebook's grouped analysis shows 833 approved and 5
rejected applications among records with `Credit_History = 1`, compared
with 126 approved and 36 rejected among records with
`Credit_History = 0`. This imbalance is important when interpreting
model performance.
Project Structure
``` text
Bank-Loan-Approval-Prediction/
│
├── BankLoanApprovalPrediction_1.ipynb
├── Loan_Applicants_1000.xlsx
└── README.md
```
Technologies Used
Python 3
Jupyter Notebook
Pandas --- data loading and manipulation
Matplotlib --- visualization
Seaborn --- statistical visualization
Plotly --- interactive visualization
Scikit-learn --- preprocessing, model training, and evaluation
OpenPyXL --- Excel file support through Pandas
Machine Learning Workflow
1. Import libraries
The notebook imports the required data-science, visualization,
preprocessing, model-training, and evaluation libraries.
2. Load the dataset
The Excel dataset is loaded using:
``` python
data = pd.read_excel("Loan_Applicants_1000.xlsx")
```
3. Inspect the data
The notebook checks:
Dataset dimensions and column names
Data types
Non-null values
Missing values
Duplicate rows
4. Data cleaning
The notebook prepares the dataset by:
Filling missing numeric values with the median
Filling missing categorical values with the mode
Converting `Dependents = "3+"` to `3`
Removing `Loan_ID` because it is an identifier rather than a
predictive feature
Checking for duplicate rows
5. Outlier handling
The notebook uses the Interquartile Range (IQR) method for:
`ApplicantIncome`
`CoapplicantIncome`
`LoanAmount`
Values outside the IQR-based limits are capped using `clip()` rather
than removing complete records.
For the supplied dataset, the notebook reported no IQR outliers in these
columns.
6. Categorical encoding
Categorical columns are converted into numerical values using
`LabelEncoder`.
Encoded columns include:
`Gender`
`Married`
`Education`
`Self_Employed`
`Property_Area`
`Loan_Status`
7. Feature scaling
The notebook applies `StandardScaler` to:
`ApplicantIncome`
`CoapplicantIncome`
`LoanAmount`
`Loan_Amount_Term`
8. Exploratory Data Analysis
The notebook includes:
Descriptive statistics
Approval/rejection distribution
Property-area approval proportions
Credit-history vs loan-status analysis
Correlation heatmap
Applicant-income boxplot
Loan-amount histogram
Interactive Plotly scatter plot of applicant income vs. loan amount
9. Train/test split
The processed dataset is split into:
80% training data
20% testing data
with:
``` python
random_state=42
```
10. Model training
Four classifiers are trained:
``` text
Logistic Regression
Decision Tree
Random Forest
K-Nearest Neighbors
```
The Random Forest model uses 100 decision trees, while KNN uses 5
neighbors.
Model Results
The notebook produced the following results on its 20% test split:
Model                   Accuracy   Precision   Recall   F1 Score   ROC-AUC
---
Logistic Regression       0.9900      0.9897   1.0000     0.9948    0.8750
Random Forest             0.9750      0.9746   1.0000     0.9871    0.6875
Decision Tree             0.9650      0.9843   0.9792     0.9817    0.8021
K-Nearest Neighbors       0.9650      0.9695   0.9948     0.9820    0.6224
These are the results recorded in the notebook for the supplied dataset
and split. They should not be interpreted as evidence that the same
performance will generalize to new or real-world loan applications.
Important Technical Notes
There are a few implementation details worth improving before treating
this as a production-quality ML pipeline:
1. Data leakage during scaling
The notebook scales the complete dataset before `train_test_split`. In a
rigorous ML workflow, the scaler should be fitted only on the training
data and then used to transform the test data.
A better approach is:
``` python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler.fit(X_train[numeric_features])
X_train[numeric_features] = scaler.transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])
```
A `Pipeline` is even safer because it keeps preprocessing and modeling
together.
2. Duplicate removal is not reassigned
The notebook contains:
``` python
data.drop_duplicates()
```
This returns a modified DataFrame but does not update `data` unless the
result is reassigned or `inplace=True` is used.
Prefer:
``` python
data = data.drop_duplicates()
```
or:
``` python
data.drop_duplicates(inplace=True)
```
The same issue appears in the duplicate-verification section.
3. Missing-value filling should be explicit
The notebook uses `fillna()` without assigning the returned Series back
for numeric columns. A robust version is:
``` python
for col in num_cols:
    data[col] = data[col].fillna(data[col].median())
```
The categorical section already assigns the result back to the column.
4. ROC-AUC calculation
The notebook calculates ROC-AUC using predicted class labels:
``` python
roc_auc_score(y_test, y_pred)
```
For a more meaningful ROC-AUC measurement, use prediction probabilities
or decision scores:
``` python
y_prob = model.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_prob)
```
This is particularly important when comparing classifiers.
5. LabelEncoder for input features
`LabelEncoder` is mainly intended for encoding target labels. For
nominal input features such as `Property_Area`, one-hot encoding is
generally more appropriate because the encoded numbers should not
accidentally suggest an ordering.
For example:
``` python
pd.get_dummies(data, columns=["Property_Area"], drop_first=True)
```
or use `OneHotEncoder` inside a scikit-learn `ColumnTransformer`.
6. Class imbalance
The dataset contains substantially more approved than rejected
applications. Accuracy alone can therefore be misleading.
For a stronger version of the project, consider:
Stratified train/test splitting
Confusion matrices
Precision-recall analysis
Class weights
Resampling techniques when appropriate
Cross-validation
ROC-AUC based on probability scores
PR-AUC / Average Precision
How to Run
1. Clone or download the project
Place the notebook and Excel dataset in the same directory.
2. Install dependencies
``` bash
pip install pandas matplotlib seaborn plotly scikit-learn openpyxl jupyter
```
3. Start Jupyter Notebook
``` bash
jupyter notebook
```
4. Open the notebook
Open:
``` text
BankLoanApprovalPrediction_1.ipynb
```
5. Run the cells
Run the notebook from top to bottom. Make sure
`Loan_Applicants_1000.xlsx` is available in the notebook's working
directory.
Example Prediction Workflow
After training a selected model, a new application can conceptually be
processed through the same preprocessing pipeline and passed to the
classifier:
``` python
prediction = model.predict(new_application)
```
For a real application, the preprocessing steps used during training
must be saved and applied identically to new data.
Future Improvements
Possible extensions include:
Build a reusable `Pipeline`
Use `ColumnTransformer` for mixed numerical/categorical
preprocessing
Add cross-validation
Tune hyperparameters with `GridSearchCV` or `RandomizedSearchCV`
Add confusion-matrix visualizations for every model
Calculate ROC-AUC from probability scores
Add PR-AUC
Investigate class imbalance
Compare additional models such as Gradient Boosting or XGBoost
Save the trained model with `joblib`
Create a Flask/FastAPI prediction API
Build a Streamlit web interface
Add automated tests
Add experiment tracking
Package the project with a `requirements.txt`
Separate data preprocessing, training, evaluation, and inference
into Python modules
Disclaimer
This repository is intended for learning and demonstration purposes.
Loan approval is a high-impact financial decision, and a real lending
system requires appropriate data governance, validation, fairness
assessment, explainability, security, regulatory compliance, human
oversight, and production monitoring.
The model's predictions should not be used as the sole basis for actual
lending decisions.
Author
Bank Loan Approval Prediction --- Machine Learning Project
Built as a practical machine-learning project demonstrating the
end-to-end workflow for a binary classification problem.
