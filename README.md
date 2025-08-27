# ⚡ Household Power Consumption — Streamlit ML App

**Overview**  
This Streamlit application trains a Random Forest regressor on the [Household Power Consumption] dataset and provides model metrics, visualizations, and an interactive runtime prediction form.

---

## Features
- Upload dataset (`household_power_consumption.txt` or CSV)
- Preprocesses data (combines `Date` + `Time`, handles missing values)
- Trains a `RandomForestRegressor`
- Displays performance metrics: **MSE**, **MAE**, **R²**
- Visualizations:
  - Distribution of `Global_active_power`
  - Actual vs Predicted scatter plot
- Runtime prediction via a Streamlit form (enter all features)

---

## Requirements

- Python 3.8+ recommended  
- Packages:
  - streamlit
  - pandas
  - numpy
  - matplotlib
  - scikit-learn

Install with:

```bash
pip install streamlit pandas numpy matplotlib scikit-learn
```
Run
```bash
From the project directory:
```

Dataset / Expected Input

This app expects the Household Power Consumption dataset in the original format (semicolon-separated). Typical file name: household_power_consumption.txt.

Expected columns (at minimum):

Date (format dd/mm/yyyy)

Time (format HH:MM:SS)

Global_active_power (target)

plus the other feature columns commonly present in the dataset, such as:

Global_reactive_power

Voltage

Global_intensity

Sub_metering_1

Sub_metering_2

Sub_metering_3

## Notes:

The uploader reads CSV with sep=';' and treats ? as NaN.

Date + Time are combined into a datetime column and original Date, Time columns are dropped.

Rows with missing values are removed (df.dropna()).

## Model & Preprocessing Details

Train/test split: 80% train / 20% test (random_state=42)

Feature scaling: StandardScaler (fit on training features and applied to test and runtime inputs)

Model: RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)

## Metrics shown:

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

R² Score

## Visuals:

Histogram (distribution) of Global_active_power

Scatter plot of actual vs predicted values on the test set

## Runtime predictions:

App generates number inputs for all model feature columns in the same order as the training features; values are scaled using the same StandardScaler before prediction.

## Known issues & suggestions

The app uses df.dropna() — if dataset has many missing rows you may lose much data. Consider more careful imputation (SimpleImputer, forward-fill, etc.).

StandardScaler is used on all features. If any non-numeric columns exist, convert/encode them first.

Current model hyperparameters are simple; consider hyperparameter tuning (GridSearchCV or RandomizedSearchCV) and cross-validation.

Consider saving the trained model and scaler (e.g., with joblib) for faster reuse and to avoid retraining each run.

Add time-based features (hour, day-of-week, month) extracted from datetime to potentially improve performance.

Add caching (@st.cache_data / @st.cache_resource) in Streamlit to speed up repeated runs.

Add better error handling for unexpected file formats.

## Example improvements to the code

Persist model & scaler to disk after training for reuse.

Use st.sidebar for upload / hyperparameter controls.

Add logging and progress indicators during training.

Provide options for different models and hyperparameter tuning.


