# Employee Attrition Prediction

Machine Learning model to predict which employees are more likely to leave a company based on demographic, work-related, and satisfaction variables.

## Table of Contents
- [Business Problem](#business-problem)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Conclusions](#conclusions)

## Business Problem

- Hiring processes are difficult and costly
- Recruiting new personnel requires capital, time, and specialized skills
- Employee turnover generates economic and knowledge losses for the company

**Objective:** Develop a predictive model to identify which employees are more likely to leave the company.

**Target variable (Attrition):**
- 1: Employee left the company
- 0: Employee stayed with the company

## Technologies Used

### Languages and Libraries
- **Python 3.11**
- **Data Analysis:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn
- **Deep Learning:** TensorFlow/Keras
- **Imbalance Handling:** imbalanced-learn (SMOTE)

### Specific Libraries
```python
- sklearn.preprocessing: LabelEncoder, OneHotEncoder, MinMaxScaler
- sklearn.model_selection: train_test_split
- sklearn.linear_model: LogisticRegression
- sklearn.ensemble: RandomForestClassifier
- sklearn.metrics: accuracy_score, confusion_matrix, classification_report
- tensorflow.keras: Sequential, Dense layers
- imblearn.over_sampling: SMOTE
```

## Dataset

**Source:** Human Resources dataset (`Human_Resources.csv`)  
**Size:** 1,470 employees  
**Class distribution:**
- Employees who left: 237 (16.12%)
- Employees who stayed: 1,233 (83.88%)

### Available Variables
The dataset includes 35 variables covering different aspects:

**Demographics:** Age, Gender, MaritalStatus, Education, EducationField

**Work-Related:** Department, JobRole, JobLevel, JobInvolvement, BusinessTravel

**Compensation:** MonthlyIncome, DailyRate, HourlyRate, MonthlyRate, PercentSalaryHike, StockOptionLevel

**Satisfaction and Balance:** JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance

**Experience:** TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager

**Other:** DistanceFromHome, OverTime, PerformanceRating, TrainingTimesLastYear, NumCompaniesWorked

## Methodology

### 1. Data Exploration and Cleaning
- Null value analysis (0 missing values found)
- Identification of constant or irrelevant variables
- Removal of columns without variability: `EmployeeCount`, `StandardHours`, `Over18`, `EmployeeNumber`
- Conversion of binary categorical variables to numeric format

### 2. Exploratory Data Analysis (EDA)

#### Key Analysis Findings

**Characteristics of employees who left vs. stayed:**
- **Age:** Average age of employees who stayed is higher
- **Distance from work:** Employees who stayed live closer to work
- **Satisfaction:** Higher job and environmental satisfaction in employees who stayed
- **Stock options:** Employees who stayed have higher stock option levels
- **Marital status:** Single employees tend to leave more than married or divorced ones
- **Job role:** Sales Representatives have higher attrition rates
- **Involvement:** Lower job involvement correlates with higher attrition
- **Experience level:** Employees with low job level (less experience) tend to leave more

**Important Correlations:**
- Job level strongly correlates with total working hours
- Monthly income strongly correlates with job level
- Monthly income strongly correlates with total working hours
- Age correlates with monthly income

### 3. Data Preprocessing

**Categorical variable transformation:**
- Application of One-Hot Encoding for text categorical variables:
  - `BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`

**Normalization:**
- Application of MinMaxScaler to scale all features to the range [0, 1]
- Total features after preprocessing: 50

**Data split:**
- 75% training set
- 25% test set

## Models Implemented

### 1. Logistic Regression
Classic linear model for binary classification.

**Configuration:**
- Standard scikit-learn implementation
- No initial hyperparameter tuning

### 2. Random Forest Classifier
Ensemble model based on decision trees.

**Configuration:**
- Standard scikit-learn implementation
- Uses multiple decision trees to improve generalization

### 3. Deep Neural Network (Deep Learning)
Dense neural network implemented with TensorFlow/Keras.

**Architecture:**
```
- Input Layer: 50 features
- Hidden Layer 1: 500 neurons, ReLU activation
- Hidden Layer 2: 500 neurons, ReLU activation
- Output Layer: 1 neuron, Sigmoid activation
- Total parameters: 276,501
```

**Training configuration:**
- Optimizer: Adam
- Loss function: Binary Crossentropy
- Epochs: 100
- Batch size: 50
- Validation split: 20%

**Balancing technique applied:**
- SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance

## Results

### Model Comparison

| Model | Accuracy | Precision (Class 0) | Recall (Class 0) | F1-Score (Class 0) | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|--------|----------|---------------------|------------------|--------------------|--------------------|------------------|-------------------|
| **Logistic Regression** | **89.40%** | 0.90 | 0.99 | **0.94** | 0.85 | 0.40 | 0.54 |
| Random Forest | 86.68% | 0.87 | 1.00 | 0.93 | 0.91 | 0.17 | 0.29 |
| Deep Learning (without SMOTE) | 86.96% | 0.90 | 0.95 | 0.92 | 0.62 | 0.45 | 0.52 |
| Deep Learning (with SMOTE) | 84.78% | 0.90 | 0.92 | 0.91 | 0.52 | 0.47 | 0.49 |

**Class 0:** Employees who stayed  
**Class 1:** Employees who left

### Results Analysis

**Best model: Logistic Regression**
- Highest overall accuracy (89.40%)
- Best F1-Score for class 0 (0.94)
- Good balance between precision and recall for the majority class

**Identified challenges:**
- **Class imbalance:** The significant difference in the number of observations (83.88% vs 16.12%) generates bias towards the majority class
- **Minority class performance:** All models show difficulty in correctly predicting class 1 (employees who leave)
- **Deep Learning overfitting:** The neural network model showed signs of overfitting (very low training loss, but increasing validation loss)
- **SMOTE did not improve results:** The oversampling technique did not improve the deep learning model's performance

## Conclusions

1. **Recommended model:** Logistic Regression presents the best overall performance considering accuracy and F1-Score, while also being more interpretable and computationally efficient.

2. **Class imbalance:** The main challenge is the dataset imbalance (16% vs 84%), which affects the ability to correctly predict the minority class (employees who leave).

3. **Business insights:** Exploratory analysis revealed key retention factors:
   - Job and environmental satisfaction
   - Proximity to workplace
   - Level of job involvement
   - Stock options and compensation
   - Marital status and role within the company

4. **Future improvements:**
   - Implement regularization techniques (L1/L2) or early stopping
   - Test other class balancing methods (undersampling, class weights)
   - Feature engineering to create new predictive variables
   - Hyperparameter tuning via Grid Search or Random Search
   - Explore more sophisticated ensemble models (XGBoost, LightGBM)

5. **Practical application:** The model can help identify employees at risk of leaving, allowing HR to implement proactive retention strategies and reduce turnover costs.
