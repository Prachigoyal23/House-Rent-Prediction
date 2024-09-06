# Import necessary libraries
import matplotlib.pyplot as plt  # For creating visualizations
import numpy as np               # For numerical operations
import pandas as pd              # For data manipulation and analysis
import scipy as sp              # For scientific and technical computing
import sklearn                   # For machine learning

housing = pd.read_csv("data.csv")
housing.hist(bins = 50, figsize = (20, 15))

# Now we will separate some into for the test set it will be use at the time of testing 

import numpy as np
def split_train_test(data, test_ratio):
  shuffled = np.random.permutation(len(data))
  test_set_size = int(len(data) * test_ratio)
  test_indices = shuffled[:test_set_size]
  train_indices = shuffled[test_set_size:]
  return data.iloc(train_indices], data.iloc[test_indices]

                   train_set, test_set = split_train_test(housing, 0.2)
                   print(f"Rows in train set : {len(train_set}\ n Rows in test set: {len(test_set}\n")

                  from sklearn.model_selection import train_test_split
  train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

from sklearn.model_selection import StratifiedShuffledSplit
Split = StratifiedShuffledSplit(n_Splits = 1, test_size = 0.2, random_state = 42)

for train_index, test_index in split.split(housing, housing['CHAS']):
  strat_train_set = housing.loc[train_index]
  strat_test_set = housing.loc[test_index]
  strat_train_set['CHAS'].value_counts()

# Correlations

corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending = False)

from pandas.plotting import scatter_matrix
attributes = ["RM", "ZN", "MEDV", "LSTAT"]
scatter_matrix(housing[attributes], figsize = (12, 8))

housing.plot(kind = "scatter", x = "RM", y = "MEDV", alpha = 0.1)

# Attribute Combinations

housing["TAXRM"] = housing['TAX']/housing['RM']
housing["TAXRM"]
housing.head()
housing = strat_train_set.drop("MEDV", axis = 1)
housing_labels = strat_train_set["MEDV'].copy()

# Missing Attributes

housing.drop("RM", axis = 1).shape

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)

x = imputer.transform(housing)

housing_tr = pd.DataFrame(x, columns = housing.columns)
housing_tr.describe()

# Creating a pipeline
from sklearn.pipeline import pipeline
from sklearn.preprocessing  import StandardScaler
my_pipeline = pipeline([
  ('imputer', SimpleImputer(strategy = "median")),
  ('std_scaler', StandardScaler()),
])

housing_num_tr = my_pipeline.fit_transform(housing)
housing_num_tr

# Selecting a Desired Model
from sklearn.tree import Decision Tree Regressor
model = Decision Tree Regressor()
from sklearn.linear model import LinearRegression
model = LinearRegression()
model.fit(housing_num_tr, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]

prepared_data = my_pipeline.transform(some_data)

model.predict(prepared_data)
list(some_lables)

# Evalating the model

from sklearn.metrics import mean_squared_error
housing_prediction = model.predict(housing_num_tr)
lin_mse = mean_squared_error(housing_lables, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_mse

# Now we will launch our model

from joblib import dump, load
dump(model, 'Prediction.joblib')

# Testing the model on test data

x_test = strat_test_set.drop("MEDV", axis = 1)
y_test = strat_test_set["MEDV"].copy()
x_test_prepared = my_pipeline.transform(x_test)
final_prediction = model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_prediction)
final_rmse = np.sqrt(final_mse)
final_rmse







                   
