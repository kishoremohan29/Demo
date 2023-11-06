#!/usr/bin/env python
# coding: utf-8
# Day 5
# # 1


# In[11]:


import pandas as pd

# Load the dataset
df = pd.read_csv('train.csv')

# Explore the dataset structure
print("Dataset Info:")
print(df.info())

# Display the first few rows of the dataset
print("\nFirst 5 Rows:")
print(df.head())

# Descriptive statistics of the dataset
print("\nDescriptive Statistics:")
print(df.describe())

# Data distribution of categorical features
print("\nValue Counts for Categorical Features:")
categorical_features = ['Gender','Customer Type','Type of Travel','Class', 'satisfaction']
for feature in categorical_features:
    print(f"\n{feature}:\n{df[feature].value_counts()}")

# Data distribution of numerical features
# You can use histograms or box plots to visualize numerical data distribution
import matplotlib.pyplot as plt

numerical_features = ['Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes']
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    df[feature].plot(kind='hist', bins=20, title=feature)
    plt.xlabel(feature)
    plt.show()


# ## 2

# In[19]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('train.csv')

# Select features and target variable
features = ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
            'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
            'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
            'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
target = 'satisfaction'

X = df[features]
y = df[target]

# Split the dataset into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
classifier = DecisionTreeClassifier()

# Train the model on the training data
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Evaluate the classification model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Display the classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=df[target].unique()))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


# ### 3

# In[20]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('train.csv')

# Select features and target variable
features = ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
            'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
            'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
            'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
target = 'satisfaction'

X = df[features]
y = df['Age']  # Assuming 'Sales' is the target variable for the regression task

# Split the dataset into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Regressor
regressor = DecisionTreeRegressor()

# Train the model on the training data
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Evaluate the regression model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")


# # 4

# In[21]:


pip install scikit-learn graphviz pydotplus


# In[22]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Load a sample dataset for classification
iris = load_iris()
X, y = iris.data, iris.target

# Create and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()


# In[23]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Generate sample data for regression
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.1, 100)

# Create and train the decision tree regressor
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(regressor, filled=True, feature_names=['X'])
plt.show()


# #### 5

# In[24]:


# Get feature importances from the classification decision tree
feature_importances = classifier.feature_importances_

# Create a DataFrame to associate feature names with their importance scores
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Sort the features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the top N most important features
top_n_features = 10  # You can change this value
print(f"Top {top_n_features} Most Important Features for Classification:")
print(feature_importance_df.head(top_n_features))


# In[35]:


# Assuming 'regressor' is your trained regression model and 'data' is your DataFrame
features = data.columns.tolist()
features.remove('Age')  # Replace 'target_variable_name' with the actual name of your target variable


# In[36]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('train.csv')

# Drop any rows with missing values
df = df.dropna()

# Encode categorical features
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

# Define features (X) and target variable (y)
X = df.drop('satisfaction', axis=1)  # Assuming 'satisfaction' is the target column
y = df['satisfaction']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Print feature importances
feature_importances = clf.feature_importances_
feature_names = X.columns

# Zip and sort the features based on importance
feature_importance_dict = dict(zip(feature_names, feature_importances))
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Print the sorted feature importance
for feature, importance in sorted_feature_importance:
    print(f'{feature}: {importance}')


# In[27]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create decision tree classifier with pruning
clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predict the test set
y_pred = clf.predict(X_test)

# Print the accuracy of the model
print("Accuracy: ", accuracy_score(y_test,y_pred))


# # 7

# In[38]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load your sales data (e.g., sales and relevant features) into a DataFrame
data = pd.read_csv('train.csv')

# Prepare the data
X = data[['Age', 'Inflight wifi service']]  # Input features
y = data['Flight Distance']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the mean squared error as a measure of prediction accuracy
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# # 8

# In[32]:


from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Instantiate the imputer
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on your entire dataset and transform it
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Classification Accuracy: {accuracy}")


# # 9

# In[33]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('train.csv')

# Perform one-hot encoding for categorical variables
data = pd.get_dummies(data, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'])

# Split the data into features and target variable
X = data.drop('satisfaction', axis=1)
y = data['satisfaction']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Classification Accuracy: {accuracy}")





