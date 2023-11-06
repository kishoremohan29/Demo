#!/usr/bin/env python
# coding: utf-8
# day3
# In[71]:


import pandas as pd

# Load the dataset (replace 'dataset.csv' with the actual file path)
data = pd.read_csv("C://Users//kishore//creditcard.csv")

# Display the first few rows of the dataset to get an overview of the features
print(data.head())

# Check the column names and data types
print(data.info())


# In[69]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset (replace 'creditcard.csv' with the actual file path)
data = pd.read_csv("C://Users//kishore//creditcard.csv")

# Define your features (independent variables) and target (dependent variable)
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print("Model Performance Metrics:")
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))

print("\nConfusion Matrix:")
print(conf_matrix)


# In[72]:


from sklearn.preprocessing import StandardScaler

# Standardize numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[74]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


# In[75]:


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, drop='first')  # Drop the first category to avoid multicollinearity
X_train_encoded = encoder.fit_transform(X_train_categorical)
X_test_encoded = encoder.transform(X_test_categorical)


# In[77]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the feature-engineered dataset
data = pd.read_csv("C://Users//kishore//creditcard.csv")  # Replace with your actual file path

# Define your features (independent variables) and target (dependent variable)
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Classification report
class_report = classification_report(y_test, y_pred)

# Print the results
print("Model Performance Metrics:")
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)


# In[78]:


# Assuming you have already fitted a logistic regression model (model) and have the feature names
coefficients = model.coef_[0]
feature_names = X.columns

# Create a dataframe to display the feature names and their corresponding coefficients
coeff_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort the dataframe by the absolute magnitude of the coefficients
coeff_df['Abs_Coefficient'] = abs(coeff_df['Coefficient'])
sorted_coeff_df = coeff_df.sort_values(by='Abs_Coefficient', ascending=False)

# Display the sorted coefficients
print(sorted_coeff_df)


# In[79]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Initial Logistic Regression Model
y_pred_initial = model_initial.predict(X_test)

# Performance metrics for the initial model
accuracy_initial = accuracy_score(y_test, y_pred_initial)
precision_initial = precision_score(y_test, y_pred_initial)
recall_initial = recall_score(y_test, y_pred_initial)
f1_initial = f1_score(y_test, y_pred_initial)
roc_auc_initial = roc_auc_score(y_test, model_initial.predict_proba(X_test)[:,1])

# Feature-Engineered and Balanced Data Model
y_pred_feature_engineered = model_feature_engineered.predict(X_test)

# Performance metrics for the feature-engineered model
accuracy_feature_engineered = accuracy_score(y_test, y_pred_feature_engineered)
precision_feature_engineered = precision_score(y_test, y_pred_feature_engineered)
recall_feature_engineered = recall_score(y_test, y_pred_feature_engineered)
f1_feature_engineered = f1_score(y_test, y_pred_feature_engineered)
roc_auc_feature_engineered = roc_auc_score(y_test, model_feature_engineered.predict_proba(X_test)[:,1])

# Compare the performance metrics
print("Initial Model Performance:")
print("Accuracy: {:.2f}".format(accuracy_initial))
print("Precision: {:.2f}".format(precision_initial))
print("Recall: {:.2f}".format(recall_initial))
print("F1 Score: {:.2f}".format(f1_initial))
print("AUC-ROC: {:.2f}".format(roc_auc_initial))

print("\nFeature-Engineered Model Performance:")
print("Accuracy: {:.2f}".format(accuracy_feature_engineered))
print("Precision: {:.2f}".format(precision_feature_engineered))
print("Recall: {:.2f}".format(recall_feature_engineered))
print("F1 Score: {:.2f}".format(f1_feature_engineered))
print("AUC-ROC: {:.2f}".format(roc_auc_feature_engineered))


# In[82]:


import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load your dataset (replace 'creditcard.csv' with your dataset)
data = pd.read_csv("C://Users//kishore//creditcard.csv")

# Define your features (independent variables) and target (dependent variable)
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose the class imbalance strategy

# 1. Oversampling the Minority Class using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)

# 2. Undersampling the Majority Class
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_train_undersampled, y_train_undersampled = undersampler.fit_resample(X_train, y_train)

# 3. Using a combination of SMOTE and Tomek links
smote_tomek = SMOTETomek(sampling_strategy='auto', random_state=42)
X_train_smote_tomek, y_train_smote_tomek = smote_tomek.fit_resample(X_train, y_train)

# Train a logistic regression model on each of the balanced datasets
model_oversampled = LogisticRegression()
model_oversampled.fit(X_train_oversampled, y_train_oversampled)

model_undersampled = LogisticRegression()
model_undersampled.fit(X_train_undersampled, y_train_undersampled)

model_smote_tomek = LogisticRegression()
model_smote_tomek.fit(X_train_smote_tomek, y_train_smote_tomek)

# Evaluate the models on the test set
y_pred_oversampled = model_oversampled.predict(X_test)
y_pred_undersampled = model_undersampled.predict(X_test)
y_pred_smote_tomek = model_smote_tomek.predict(X_test)

# Now, you can evaluate the performance of these models using various metrics.





