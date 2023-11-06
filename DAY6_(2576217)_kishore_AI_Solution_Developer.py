#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Load the dataset (assuming it's in a CSV file)"C://Users//kishore//train_u6lujuX_CVtuZ9i.csv"
credit_data = pd.read_csv("C://Users//kishore//train_u6lujuX_CVtuZ9i.csv")

# Display the first few rows of the dataset to understand its structure
print(credit_data.head())

# Describe the features, target variable, and data distribution
# Features are the input variables, and the target variable is the credit risk outcome.

# Features
features = credit_data.drop('Loan_Status', axis=1)
print("Features:")
print(features.head())

# Target variable
target = credit_data['Loan_Status']
print("\nTarget Variable:")
print(target.head())

# Data distribution
print("\nData Distribution:")
print(credit_data['Loan_Status'].value_counts())


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load your dataset
# Replace 'your_dataset.csv' with the actual file path or URL
dataset = pd.read_csv("C://Users//kishore//train_u6lujuX_CVtuZ9i.csv")


# In[36]:


# Preprocess categorical data (if necessary)
# You can use LabelEncoder or OneHotEncoder

# For example, using LabelEncoder for Gender column:
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
dataset['Gender'] = label_encoder.fit_transform(dataset['Gender'])

# Define your features (X) and target variable (y)
X = dataset.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = dataset['Loan_Status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear SVM model
linear_svm_model = SVC(kernel='linear')
linear_svm_model.fit(X_train, y_train)


# In[25]:





# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset (replace 'your_dataset.csv' with the actual file path or URL)
dataset = pd.read_csv("C://Users//kishore//train_u6lujuX_CVtuZ9i.csv")


# In[27]:


# Define your features (X) and target variable (y) based on your dataset
X = dataset.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = dataset['Loan_Status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the non-linear SVM model with a kernel (e.g., Radial Basis Function)
non_linear_svm_model = SVC(kernel='rbf')
non_linear_svm_model.fit(X_train, y_train)


# In[28]:


# Make predictions on the test set
y_pred = non_linear_svm_model.predict(X_test)

# Evaluate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display the performance metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


# In[34]:


# Make predictions on the test set
y_pred = non_linear_svm_model.predict(X_test)

# Evaluate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display the performance metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


# In[38]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the dataset
df = pd.read_csv("C://Users//kishore//train_u6lujuX_CVtuZ9i.csv")  # Replace with the actual file path and name of your dataset

# Step 2: Split the dataset into features (X) and target variable (y)
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the linear SVM model
linear_model = SVC(kernel='linear')
linear_model.fit(X_train, y_train)

# Step 5: Make predictions on the test set using the linear model
y_pred_linear = linear_model.predict(X_test)

# Step 6: Evaluate the linear SVM model's performance
accuracy_linear = accuracy_score(y_test, y_pred_linear)
precision_linear = precision_score(y_test, y_pred_linear)
recall_linear = recall_score(y_test, y_pred_linear)
f1_linear = f1_score(y_test, y_pred_linear)

print('Linear SVM Model Performance:')
print('Accuracy:', accuracy_linear)
print('Precision:', precision_linear)
print('Recall:', recall_linear)
print('F1-score:', f1_linear)

# Step 7: Train the non-linear SVM model
non_linear_model = SVC(kernel='rbf')  # You can choose a different non-linear kernel if desired
non_linear_model.fit(X_train, y_train)

# Step 8: Make predictions on the test set using the non-linear model
y_pred_non_linear = non_linear_model.predict(X_test)

# Step 9: Evaluate the non-linear SVM model's performance
accuracy_non_linear = accuracy_score(y_test, y_pred_non_linear)
precision_non_linear = precision_score(y_test, y_pred_non_linear)
recall_non_linear = recall_score(y_test, y_pred_non_linear)
f1_non_linear = f1_score(y_test, y_pred_non_linear)

print('Non-Linear SVM Model Performance:')
print('Accuracy:', accuracy_non_linear)
print('Precision:', precision_non_linear)
print('Recall:', recall_non_linear)
print('F1-score:',f1_non_linear)


# In[42]:


import pandas as pd

# Load the dataset
df = pd.read_csv("C://Users//kishore//train_u6lujuX_CVtuZ9i.csv")

# Display the first few rows of the dataset to get an overview
print(df.head())

# Get information about the dataset, including data types and non-null counts
print(df.info())

# Summary statistics of numerical columns
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Data distribution for categorical variables
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area', 'Loan_Status']
for col in categorical_cols:
    print(f"Distribution for {col}:")
    print(df[col].value_counts())
    print("\n")
    


# In[44]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer  # Import the imputer

# Load the dataset
df = pd.read_csv("C://Users//kishore//train_u6lujuX_CVtuZ9i.csv")

# Data preprocessing: Encode categorical variables
label_encoder = LabelEncoder()
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Handle missing values with imputation
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
imputer = SimpleImputer(strategy='mean')  # You can choose a different strategy, such as median or mode
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# Define features (X) and target variable (y)
X = df.drop(columns=['Loan_ID', 'Loan_Status'])
y = df['Loan_Status']

# Split the dataset into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision
precision = precision_score(y_test, y_pred)

# Calculate recall
recall = recall_score(y_test, y_pred)

# Calculate F1-score
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:",f1)


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer  # Import the imputer

# Load the dataset
df = pd.read_csv("C://Users//kishore//train_u6lujuX_CVtuZ9i.csv")

# Data preprocessing: Encode categorical variables
label_encoder = LabelEncoder()
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Handle missing values with imputation
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
imputer = SimpleImputer(strategy='mean')  # You can choose a different strategy, such as median or mode
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# Define features (X) and target variable (y)
X = df.drop(columns=['Loan_ID', 'Loan_Status'])
y = df['Loan_Status']

# Split the dataset into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the non-linear SVM model with a kernel (e.g., RBF kernel)
svm_model = SVC(kernel='rbf')  # You can change the kernel as needed (e.g., 'poly' for Polynomial kernel)
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision
precision = precision_score(y_test, y_pred)

# Calculate recall
recall = recall_score(y_test, y_pred)

# Calculate F1-score
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:",f1)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder

# Create a synthetic dataset for visualization
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, random_state=1)

# Create a linear SVM model
linear_svm = SVC(kernel='linear')
linear_svm.fit(X, y)

# Create a non-linear SVM model (e.g., RBF kernel)
rbf_svm = SVC(kernel='rbf', gamma='auto')
rbf_svm.fit(X, y)

# Visualize decision boundaries for linear SVM
def plot_decision_boundary(model, ax, title):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plot_decision_boundary(linear_svm, axes[0], "Linear SVM Decision Boundary")
plot_decision_boundary(rbf_svm, axes[1], "RBF SVM Decision Boundary")

plt.show()


# In[51]:


# Assuming you have already trained linear_svm and rbf_svm models as mentioned earlier

# Get support vectors for the linear SVM
linear_support_vectors = linear_svm.support_vectors_

# Get support vectors for the non-linear SVM (RBF kernel)
rbf_support_vectors = rbf_svm.support_vectors_

# Count the number of support vectors
num_linear_support_vectors = len(linear_support_vectors)
num_rbf_support_vectors = len(rbf_support_vectors)

print(f"Number of support vectors for Linear SVM: {num_linear_support_vectors}")
print(f"Number of support vectors for RBF SVM: {num_rbf_support_vectors}")


# In[50]:


# Now, you can evaluate the models as previously shown
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming you have already trained linear_svm and rbf_svm models as mentioned earlier
# Replace X_test and y_test with your actual testing data
X_test, y_test = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Make predictions using both models
linear_predictions = linear_svm.predict(X_test)
rbf_predictions = rbf_svm.predict(X_test)

# Evaluate the linear SVM model
linear_accuracy = accuracy_score(y_test, linear_predictions)
linear_precision = precision_score(y_test, linear_predictions)
linear_recall = recall_score(y_test, linear_predictions)
linear_f1 = f1_score(y_test, linear_predictions)

# Evaluate the non-linear SVM model (RBF kernel)
rbf_accuracy = accuracy_score(y_test, rbf_predictions)
rbf_precision = precision_score(y_test, rbf_predictions)
rbf_recall = recall_score(y_test, rbf_predictions)
rbf_f1 = f1_score(y_test, rbf_predictions)

# Compare the performance metrics
print("Linear SVM:")
print("Accuracy:", linear_accuracy)
print("Precision:", linear_precision)
print("Recall:", linear_recall)
print("F1-score:", linear_f1)

print("\nNon-Linear SVM (RBF Kernel):")
print("Accuracy:", rbf_accuracy)
print("Precision:", rbf_precision)
print("Recall:", rbf_recall)
print("F1-score:",rbf_f1)


# In[46]:


import pandas as pd

# Load the credit risk dataset
df = pd.read_csv("C://Users//kishore//train_u6lujuX_CVtuZ9i.csv")

# Display the first few rows of the dataset to understand its structure
print(df.head())

# You can explore the dataset further, e.g., check for missing values, statistics, and more
# Example: Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)

# Example: Get summary statistics of numerical columns
summary_stats = df.describe()
print("Summary Statistics:")
print(summary_stats)


# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid for linear SVM
linear_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100]
}

# Define the hyperparameter grid for non-linear SVM with RBF kernel
rbf_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

# Create the GridSearchCV objects
linear_grid_search = GridSearchCV(SVC(kernel='linear'), param_grid=linear_param_grid, cv=5)
rbf_grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid=rbf_param_grid, cv=5)

# Fit the models
linear_grid_search.fit(X_train, y_train)
rbf_grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_linear_params = linear_grid_search.best_params_
best_rbf_params = rbf_grid_search.best_params_

# Train models with the best hyperparameters
best_linear_model = SVC(kernel='linear', **best_linear_params)
best_rbf_model = SVC(kernel='rbf', **best_rbf_params)

# Evaluate the models as before
best_linear_model.fit(X_train, y_train)
best_rbf_model.fit(X_train, y_train)
linear_accuracy = best_linear_model.score(X_test, y_test)
rbf_accuracy = best_rbf_model.score(X_test, y_test)

print("Best Linear SVM Accuracy:", linear_accuracy)
print("Best RBF SVM Accuracy:",rbf_accuracy)


# In[ ]:




