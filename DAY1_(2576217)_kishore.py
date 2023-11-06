#!/usr/bin/env python
# coding: utf-8

# Predicting House Prices
# 
# Day 1
# 
# You are working for a real estate company, and your goal is to build a predictive model to estimate house prices based on various features. You have a dataset containing information about houses, such as square footage, number of bedrooms, number of bathrooms, and other relevant attributes. You are tasked with the following:
# 
# Dataset: You can choose / download the dataset from Kaggle/ UCI Repository or any other medium.
# 
# 1. Data Preparation:
# 
# a. Load the dataset using pandas.
# 
# b. Explore and clean the data. Handle missing values and outliers.
# 
# c. Split the dataset into training and testing sets.
# 
# 2. Implement Simple Linear Regression:
# 
# a. Choose a feature (e.g., square footage) as the independent variable (X) and house prices as the dependent variable (y).
# b. Implement a simple linear regression model using sklearn to predict house prices based on the selected feature.
# 
# c. Visualize the data and the regression line.
# 
# 3. Evaluate the Simple Linear Regression Model:
# 
# a. Use scikit-learn to calculate the R-squared value to assess the goodness of fit.
# 
# b. Interpret the R-squared value and discuss the model's performance.
# 
# 4. Implement Multiple Linear Regression:
# 
# a. Select multiple features (e.g., square footage, number of bedrooms, number of bathrooms) as independent variables (X) and house prices as the dependent variable (y).
# 
# b. Implement a multiple linear regression model using scikit-learn to predict house prices based
# 
# on the selected features.
# 
# 5. Evaluate the Multiple Linear Regression Model:
# 
# a. Calculate the Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) to assess the model's accuracy.
# 
# b. Discuss the advantages of using multiple features in regression analysis.
# 6. Model Comparison:
# 
# a. Compare the results of the simple linear regression and multiple linear regression models.
# 
# Day 1 Linear Regress.pdf
# 
# MLA-AIML Batch
# 
# Mphasis
# 
# b. Discuss the advantages and limitations of each model.
# 
# 7. Model Improvement:
# 
# a. Explore potential model improvements, such as feature selection, feature engineering, or hyperparameter tuning, and describe how they could enhance the model's performance.
# 
# 8. Conclusion:
# 
# a. Summarize the findings and provide insights into how this predictive model can be used to assist the real estate company in estimating house prices. 1
# 
# 9. Presentation:
# 
# a. Prepare a presentation or report summarizing your analysis, results, and recommendations.
# In this case study, you are required to demonstrate your ability to preprocess data, implement both simple and multiple linear regression models, evaluate their performance, and make recommendations for improving the models. This case study should assess your knowledge of using Python libraries like NumPy, pandas, and scikit-learn for linear regression tasks and your understanding of model evaluation techniques.

# In[43]:


import pandas as pd


# Load the dataset from Housing.csv
data = pd.read_csv('Housing.csv')

# Display the first few rows of the dataset to get an overview
print(data.head())


# In[37]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset from Housing.csv
data = pd.read_csv('Housing.csv')

# Define the feature (independent variable) and target (dependent variable)
X = data[['area']]  # Independent variable
y = data['price']  # Dependent variable

# Split the dataset into a training set (80%) and a testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Visualize the data and regression line
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.title('Simple Linear Regression')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()
plt.show()

# Coefficients and intercept of the linear regression model
slope = model.coef_[0]
intercept = model.intercept_

print("Linear Regression Equation: Price = {} * Area + {}".format(slope,intercept))


# In[16]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset from Housing.csv
data = pd.read_csv('Housing.csv')

# Define the feature (independent variable) and target (dependent variable)
X = data[['area']]  # Independent variable
y = data['price']  # Dependent variable

# Split the dataset into a training set (80%) and a testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the R-squared value
r_squared = r2_score(y_test, y_pred)

print("R-squared value:", r_squared)

# Interpret the R-squared value and discuss the model's performance
if r_squared == 1:
    print("The model perfectly fits the data.")
elif r_squared > 0.7:
    print("The model has a strong fit.")
elif r_squared > 0.5:
    print("The model has a moderate fit.")
else:
    print("The model has a weak fit. It may not be a good predictor of house prices.")


# In[38]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

# Load your dataset into a Pandas DataFrame
# Assuming you have 'X' as the DataFrame of features and 'y' as the target variable (house prices).

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r_squared = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Value:",r_squared)


# In[24]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the dataset from Housing.csv
data = pd.read_csv('Housing.csv')

# Define the features (independent variables) and target (dependent variable)
X = data[['area', 'bedrooms', 'bathrooms']]  # Independent variables
y = data['price']  # Dependent variable

# Split the dataset into a training set (80%) and a testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Multiple Linear Regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics: MAE, MSE, RMSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Advantages of using multiple features in regression analysis
print("Advantages of using multiple features:")
print("1. Improved model accuracy: Multiple features can capture complex relationships in the data that a single feature may miss.")
print("2. Better predictive power: More features can provide a more comprehensive view of the factors influencing the target variable.")
print("3. Reduced bias: Multiple features help reduce bias in the model, leading to more accurate predictions.")
print("4. Enhanced model interpretability: Including domain-specific features can lead to more interpretable models.")


# In[25]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset from Housing.csv
data = pd.read_csv('Housing.csv')

# Simple Linear Regression
X_simple = data[['area']]
y_simple = data['price']
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train_simple)
y_pred_simple = simple_model.predict(X_test_simple)

# Multiple Linear Regression
X_multi = data[['area', 'bedrooms', 'bathrooms']]
y_multi = data['price']
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

multi_model = LinearRegression()
multi_model.fit(X_train_multi, y_train_multi)
y_pred_multi = multi_model.predict(X_test_multi)

# Evaluate Simple Linear Regression
mse_simple = mean_squared_error(y_test_simple, y_pred_simple)
r2_simple = r2_score(y_test_simple, y_pred_simple)

# Evaluate Multiple Linear Regression
mse_multi = mean_squared_error(y_test_multi, y_pred_multi)
r2_multi = r2_score(y_test_multi, y_pred_multi)

# Display the evaluation results
print("Simple Linear Regression Results:")
print("Mean Squared Error (MSE):", mse_simple)
print("R-squared (R2):", r2_simple)

print("\nMultiple Linear Regression Results:")
print("Mean Squared Error (MSE):", mse_multi)
print("R-squared (R2):", r2_multi)

# Discuss the advantages and limitations of each model
print("\nAdvantages of Simple Linear Regression:")
print("- Simplicity: Easy to understand and implement.")
print("- High interpretability: Clear understanding of the relationship between a single predictor and the target variable.")
print("- Lower risk of overfitting: Simpler models are less prone to overfitting.")

print("\nAdvantages of Multiple Linear Regression:")
print("- Improved accuracy: Can capture complex relationships involving multiple predictors.")
print("- Better predictive power: Utilizes multiple variables for a more comprehensive view.")
print("- Enhanced interpretability: Can accommodate domain-specific factors and interactions.")

print("\nLimitations of Simple Linear Regression:")
print("- Limited explanatory power: Not suitable for complex relationships.")
print("- Restricted use: Only applicable when there is a clear, strong predictor.")

print("\nLimitations of Multiple Linear Regression:")
print("- Complexity: More challenging to understand and explain due to multiple predictors.")
print("- Overfitting risk: Prone to overfitting when dealing with many predictors without proper feature selection.")


# In[52]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

# Load the dataset from Housing.csv
data = pd.read_csv('Housing.csv')

# Define the features (independent variables) and target (dependent variable)
X = data[['area', 'bedrooms', 'bathrooms']]  # Independent variables
y = data['price']  # Dependent variable

# Split the dataset into a training set (80%) and a testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random


# In[ ]:


from sklearn.feature_selection import SelectKBest, f_regression

# Select the top k best features
k_best = SelectKBest(score_func=f_regression, k=2)  # Select the top 2 features
X_train_new = k_best.fit_transform(X_train, y_train)
X_test_new = k_best.transform(X_test)


# In[ ]:


# Example of creating a new feature by combining existing ones
data['bed_bath_ratio'] = data['bedrooms'] / data['bathrooms']


# In[54]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

param_grid = {
    'normalize': [True, False],
    'fit_intercept': [True, False]
}

# Create a GridSearchCV object with cross-validation (cv=5)
grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_


# In[42]:


from sklearn.linear_model import Lasso, Ridge

lasso_model = Lasso(alpha=0.01)
ridge_model = Ridge(alpha=0.01)

lasso_model.fit(X_train,y_train)
ridge_model.fit(X_train,y_train)


# In[53]:


import pandas as pd
from sklearn.model_selection import train_test_split

# a. Load the dataset using pandas
# Replace 'your_dataset.csv' with the actual dataset file path
data = pd.read_csv('Housing.csv')

# b. Explore and clean the data
# Examine the first few rows of the dataset
print(data.head())

# Get a summary of the dataset's structure
print(data.info())

# Generate descriptive statistics for numerical columns
print(data.describe())

# Handle missing values (e.g., fill with mean, median, or drop)
data.fillna(data.mean(), inplace=True)  # Replace missing values with the mean

# Address outliers (you can visualize them to make decisions)
# For visualization, you can use libraries like Matplotlib or Seaborn

# c. Split the dataset into training and testing sets
X = data.drop('parking', axis=1)  # Features
y = data['parking']  # Target variable

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


