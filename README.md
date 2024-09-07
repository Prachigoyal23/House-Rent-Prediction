# House-Rent-Prediction

## Overview

This project focuses on predicting house rent prices using machine learning algorithms. The goal is to build a model that can estimate rental prices based on various features of the houses. The project employs Python and several machine learning libraries to preprocess data, train models, and evaluate their performance.

## Features

- Data visualization and exploration
- Data preprocessing and feature engineering
- Model training using linear regression and decision tree algorithms
- Evaluation of model performance
- Saving and loading models for future use

## Libraries Used

- `matplotlib`: For creating visualizations
- `numpy`: For numerical operations
- `pandas`: For data manipulation and analysis
- `scipy`: For scientific and technical computing
- `scikit-learn`: For machine learning algorithms and tools
- `joblib`: For saving and loading models

## Data

The dataset used in this project is `data.csv`, which contains various features related to house properties and their rental prices. 

## Steps

1. **Data Loading and Visualization**

   Load the dataset and visualize it using histograms to understand the distribution of features.

   ```python
   housing = pd.read_csv("data.csv")
   housing.hist(bins=50, figsize=(20, 15))
   ```

2. **Data Splitting**

   Split the data into training and test sets. The project uses both a custom split function and Scikit-Learn's `train_test_split` and `StratifiedShuffledSplit` for comparison.

   ```python
   from sklearn.model_selection import train_test_split, StratifiedShuffledSplit
   
   # Custom split
   train_set, test_set = split_train_test(housing, 0.2)
   
   # Scikit-Learn split
   train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
   
   # Stratified split
   split = StratifiedShuffledSplit(n_splits=1, test_size=0.2, random_state=42)
   for train_index, test_index in split.split(housing, housing['CHAS']):
       strat_train_set = housing.loc[train_index]
       strat_test_set = housing.loc[test_index]
   ```

3. **Feature Engineering**

   Create new features and handle missing values.

   ```python
   # Feature engineering
   housing["TAXRM"] = housing['TAX'] / housing['RM']
   
   # Handling missing values
   from sklearn.impute import SimpleImputer
   imputer = SimpleImputer(strategy="median")
   imputer.fit(housing)
   housing_tr = pd.DataFrame(imputer.transform(housing), columns=housing.columns)
   ```

4. **Pipeline Creation**

   Set up a pipeline for data preprocessing, including imputation and standard scaling.

   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   
   my_pipeline = Pipeline([
       ('imputer', SimpleImputer(strategy="median")),
       ('std_scaler', StandardScaler()),
   ])
   
   housing_num_tr = my_pipeline.fit_transform(housing)
   ```

5. **Model Training**

   Train the model using Linear Regression and Decision Tree Regressor.

   ```python
   from sklearn.linear_model import LinearRegression
   model = LinearRegression()
   model.fit(housing_num_tr, housing_labels)
   ```

6. **Model Evaluation**

   Evaluate the model's performance using mean squared error and root mean squared error.

   ```python
   from sklearn.metrics import mean_squared_error
   housing_predictions = model.predict(housing_num_tr)
   lin_mse = mean_squared_error(housing_labels, housing_predictions)
   lin_rmse = np.sqrt(lin_mse)
   ```

7. **Model Persistence**

   Save the trained model using `joblib` for future use.

   ```python
   from joblib import dump
   dump(model, 'Prediction.joblib')
   ```

8. **Testing the Model**

   Load the model and test its performance on the test set.

   ```python
   from joblib import load
   model = load('Prediction.joblib')
   x_test = strat_test_set.drop("MEDV", axis=1)
   y_test = strat_test_set["MEDV"].copy()
   x_test_prepared = my_pipeline.transform(x_test)
   final_prediction = model.predict(x_test_prepared)
   final_mse = mean_squared_error(y_test, final_prediction)
   final_rmse = np.sqrt(final_mse)
   ```

## Future Work

- Experiment with additional features and more advanced models.
- Perform hyperparameter tuning to improve model performance.
- Explore other machine learning algorithms for comparison.

## Contact

For any questions or suggestions, feel free to reach out to goyalprachi2324@gmail.com or open an issue on this repository.

---

Feel free to modify any sections according to your project's specifics or personal preferences.
