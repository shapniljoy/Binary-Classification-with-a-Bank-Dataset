# Binary Classification with a Bank Dataset

## 📌 Project Overview
This project focuses on predicting whether a bank client will subscribe to a term deposit based on marketing campaign data. It involves an end-to-end machine learning pipeline starting from extensive Exploratory Data Analysis (EDA) and Feature Engineering to Model Selection. The project culminates in a highly optimized `CatBoostClassifier` tuned using Optuna, delivering excellent predictive performance.

## 📂 Project Structure
The repository consists of the following Jupyter Notebooks, meant to be executed sequentially:

1. **`EDA and Feature_Engineering.ipynb`**
   - Performed thorough Exploratory Data Analysis (EDA) to understand feature distributions and target relationships.
   - Handled missing values, outliers, and data inconsistencies.
   - Engineered new features such as binning continuous variables (e.g., `age`, `balance`), generating time-based features (`week`, `qtr`), and creating categorical flags (e.g., `pdays_cat`, `is_q2_calls`).
   - Saved the preprocessed and transformed dataset to `final_data.csv`.

2. **`Feature selection.ipynb`**
   - Analyzed feature importance and correlation.
   - Selected the most relevant features to reduce dimensionality, mitigate noise, and improve model efficiency.

3. **`Model_selection.ipynb`**
   - Built a robust preprocessing pipeline using Scikit-Learn's `ColumnTransformer` (incorporating `PowerTransformer`, `StandardScaler`, `OneHotEncoder`, and `OrdinalEncoder`).
   - Evaluated a wide range of machine learning algorithms:
     - Logistic Regression, SVC, Decision Tree, Random Forest, Gradient Boosting, GaussianNB, XGBoost, LightGBM, and CatBoost.
   - Handled class imbalance using automated class weight balancing techniques.
   - **Result**: `CatBoostClassifier` emerged as the best-performing model based on ROC-AUC scores (~0.930).

4. **`CatBoost_Optuna.ipynb`**
   - Fine-tuned the hyperparameters of the chosen `CatBoostClassifier` using the **Optuna** framework.
   - Achieved a highly robust model with optimized depth, learning rate, and L2 regularization.

## 📊 Dataset
- **`bank-full.csv`**: The original raw bank marketing dataset.
- **`final_data.csv`**: The cleaned, engineered dataset ready for modeling.

## 🚀 Technologies & Libraries
- **Data Manipulation**: `pandas`, `numpy`
- **Data Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Machine Learning**: `scikit-learn` (Pipelines, Preprocessing, Models), `xgboost`, `lightgbm`, `catboost`
- **Hyperparameter Tuning**: `optuna`

## 📈 Key Results
The final optimized `CatBoostClassifier` achieved exceptional predictive performance:
- **Cross-Validation ROC-AUC**: `0.9296`
- **Test ROC-AUC Score**: `0.9294`
