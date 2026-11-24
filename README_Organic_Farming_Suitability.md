# Organic Farming Suitability Prediction Using Machine Learning

## 1. Project Overview

This project uses machine learning classification algorithms to predict whether a soil sample is **Suitable** or **Not Suitable** for organic farming. It analyzes soil properties, environmental conditions, crop information, and previous fertilizer and pesticide usage.

The project is implemented in a Jupyter Notebook using Python and scikit-learn.

> **Important:** This is a machine-learning project for academic and exploratory use. Its predictions should not replace laboratory soil testing or advice from qualified agricultural experts.

## 2. Objectives

- Explore and understand the soil dataset.
- Clean and prepare the data for machine learning.
- Analyze numerical and categorical soil-related features.
- Train and compare four classification algorithms.
- Evaluate model performance using standard classification metrics.
- Identify influential features.
- Save the selected trained model and use it to predict a new soil sample.

## 3. Dataset

The notebook was run with a CSV dataset containing **1,000 records and 24 columns**.

**CSV filename used in the notebook:**

`organic_farming_suitability_1000_source_calibrated.csv`

The target column is:

`Organic_Farming_Suitability`

The target classes found in the executed notebook were:

| Class | Records | Percentage |
|---|---:|---:|
| Not Suitable | 577 | 57.7% |
| Suitable | 423 | 42.3% |
| **Total** | **1,000** | **100%** |

### Input features

The notebook uses the following input features:

- `State`
- `Soil_Type`
- `Crop`
- `pH`
- `Organic_Carbon_pct`
- `Available_N_kg_ha`
- `Available_P_kg_ha`
- `Available_K_kg_ha`
- `Available_S_mg_kg`
- `Available_Zn_mg_kg`
- `Available_Fe_mg_kg`
- `Available_Mn_mg_kg`
- `Available_Cu_mg_kg`
- `Available_B_mg_kg`
- `EC_dS_m`
- `Annual_Rainfall_mm`
- `Mean_Temperature_C`
- `Soil_Moisture_pct`
- `Previous_Fertilizer_Usage`
- `Previous_Pesticide_Usage`
- `Years_Chemical_Input`

`Sample_ID` and `Suitability_Score` are excluded from the model inputs. The target column is also kept separate from the input features.

**Dataset setup:** The CSV file is not embedded in this notebook. Keep the CSV in your project folder or update the `file_path` variable in the notebook to point to its location.

## 4. Technologies and Libraries

- Python
- Jupyter Notebook
- pandas — data loading and manipulation
- NumPy — numerical operations
- Matplotlib and Seaborn — visualizations
- scikit-learn — preprocessing, model training, and evaluation
- joblib — saving and loading the trained model

## 5. Project Workflow

1. **Import libraries:** Load the Python libraries required for data analysis and machine learning.
2. **Load the dataset:** Read the CSV file using pandas.
3. **Understand the data:** Inspect sample rows, dataset dimensions, data types, descriptive statistics, and feature names.
4. **Clean the data:** Check missing values and duplicates, remove duplicate rows, and replace infinite values with missing values.
5. **Exploratory Data Analysis (EDA):** Examine class distribution, feature distributions, box plots, and numerical correlations.
6. **Separate features and target:** Create `X` from the input columns and `y` from `Organic_Farming_Suitability`.
7. **Split the data:** Use an 80:20 train-test split with stratification and `random_state=42`.
8. **Preprocess features:** Impute missing numerical values with the median, scale numerical features with `StandardScaler`, impute categorical values with the most frequent category, and encode categories with `OneHotEncoder`.
9. **Train models:** Fit four classification pipelines using the training data.
10. **Evaluate models:** Compare accuracy, precision, recall, F1 score, ROC-AUC, classification reports, confusion matrices, and ROC curves.
11. **Interpret features:** Review Random Forest feature importance and permutation importance for the selected model.
12. **Save and test the model:** Save the selected pipeline as a `.pkl` artifact, reload it, and predict the class for a sample soil record.

## 6. Machine Learning Algorithms

The notebook trains and compares these four classifiers:

| Algorithm | Configuration in notebook |
|---|---|
| Logistic Regression | `max_iter=2000` |
| Decision Tree | `max_depth=8`, `min_samples_split=5` |
| K-Nearest Neighbors (KNN) | `n_neighbors=5` |
| Random Forest | `n_estimators=300`, `class_weight="balanced"` |

Each classifier is placed in a scikit-learn `Pipeline` with the same preprocessing steps.

The model is selected by sorting the evaluation results by **F1 Score**, then **ROC-AUC**, then **Accuracy**, in descending order.

## 7. Model Evaluation Results

The following values are from the saved outputs in the supplied notebook. They are results from its particular train-test split, not a guarantee of performance on new data.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8050 | 0.7875 | 0.7412 | 0.7636 | 0.8891 |
| Random Forest | 0.7850 | 0.7442 | 0.7529 | 0.7485 | 0.8836 |
| K-Nearest Neighbors | 0.7650 | 0.7714 | 0.6353 | 0.6968 | 0.8144 |
| Decision Tree | 0.7400 | 0.6941 | 0.6941 | 0.6941 | 0.7508 |

**Selected model:** Logistic Regression

Its recorded test accuracy is **80.50%**, and its ROC-AUC is **0.8891**.

### Meaning of the evaluation metrics

- **Accuracy:** The proportion of all test predictions that are correct.
- **Precision:** Of the samples predicted as Suitable, the proportion that are actually Suitable.
- **Recall:** Of the actually Suitable samples, the proportion correctly identified.
- **F1 Score:** The harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well the model distinguishes between the two classes across classification thresholds.

## 8. Example Prediction

The notebook demonstrates prediction using a sample with these details:

- State: West Bengal
- Soil type: Loamy
- Crop: Tomato
- pH: 6.5
- Previous fertilizer usage: Organic
- Previous pesticide usage: Low

The executed notebook output for this sample is:

- **Predicted class:** Not Suitable
- **Estimated probability of Suitable:** 14.02%

This is an example output from the notebook's saved model and sample values. It is not an agricultural recommendation.

## 9. Installation and Setup

### Requirements

Use Python 3 and install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

The notebook uses `OneHotEncoder(sparse_output=False)`, which requires a compatible scikit-learn version (scikit-learn 1.2 or newer).

### Run the project

1. Download or clone the project repository.
2. Place the CSV dataset in an accessible folder.
3. Open the notebook in Jupyter Notebook or JupyterLab.
4. Update the CSV path in the dataset-loading cell:

```python
file_path = "organic_farming_suitability_1000_source_calibrated.csv"
df = pd.read_csv(file_path)
```

5. Run the notebook cells from top to bottom.
6. After training and evaluation, the notebook saves the selected model as:

```text
best_organic_farming_suitability_model.pkl
```

### Suggested project folder structure

```text
Organic-Farming-Suitability-Prediction/
│
├── Organic Farming Suitability Prediction_ML Project.ipynb
├── organic_farming_suitability_1000_source_calibrated.csv
├── best_organic_farming_suitability_model.pkl
└── README.md
```

The CSV and model artifact are generated or supplied separately; they are not included inside the notebook file itself.

## 10. Saved Model

The notebook saves a dictionary using `joblib`. It contains:

- The fitted model pipeline
- Selected model name
- Required feature column names
- Target column name
- Positive class (`Suitable`)
- Excluded columns

To load the saved artifact:

```python
import joblib

saved_artifact = joblib.load("best_organic_farming_suitability_model.pkl")

loaded_model = saved_artifact["model"]
feature_columns = saved_artifact["feature_columns"]
```

For a new prediction, create a pandas DataFrame with all required feature columns and arrange them in the order stored in `feature_columns`:

```python
new_soil_df = pd.DataFrame([new_soil_data])
new_soil_df = new_soil_df[feature_columns]

prediction = loaded_model.predict(new_soil_df)[0]
probabilities = loaded_model.predict_proba(new_soil_df)[0]

print("Predicted suitability:", prediction)
print("Class probabilities:", dict(zip(loaded_model.classes_, probabilities)))
```

Use the same feature names and compatible value types as the training data.

## 11. Limitations

- Model performance is measured on one stratified 80:20 train-test split.
- The results depend on the quality, coverage, and representativeness of the supplied dataset.
- A model probability is not a guarantee that a field is suitable for organic farming.
- The notebook does not establish that its predictions are validated through field trials.
- The model should be tested on independent real-world soil samples before practical use.

## 12. Future Improvements

- Validate the model using cross-validation and an independent dataset.
- Tune model hyperparameters systematically.
- Add more geographically diverse and field-verified soil samples.
- Review class-specific errors with agricultural domain experts.
- Build a simple web application for entering soil details and viewing predictions.
- Add clear guidance explaining that predictions are decision-support estimates, not a substitute for soil testing.

## 13. Conclusion

This project demonstrates an end-to-end machine learning workflow for classifying organic farming suitability from soil and environmental information. It includes data preparation, exploratory analysis, model comparison, evaluation, feature-importance analysis, model saving, and prediction on a new sample. In the executed notebook, Logistic Regression achieved the highest F1 Score and was selected as the final model.

---

**Project type:** Machine Learning / Classification  
**Notebook:** `Organic Farming Suitability Prediction_ML Project.ipynb`  
**Target:** `Organic_Farming_Suitability`  
**Selected model in recorded run:** Logistic Regression
