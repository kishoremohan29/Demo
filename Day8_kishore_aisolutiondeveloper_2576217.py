#!/usr/bin/env python
# coding: utf-8

# In[8]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


df = pd.read_csv("C://Users//kishore//Wine.csv")
df


# In[11]:


df.columns


# In[12]:


from sklearn.model_selection import train_test_split
X=df.drop("Customer_Segment",axis=1).values
y=df["Customer_Segment"].values


# In[13]:


X_train, X_test, y_train,y_test =train_test_split(X,y, test_size=0.2,random_state=42)


# In[14]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[15]:


from sklearn.decomposition import PCA
pca= PCA(n_components=2)# we make an instance of PCA and decide how many components we want to have
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[16]:


print(X_train.shape) # As we can see, we have reduced feature into 2 main features
print(X_test.shape)


# In[17]:


plt.figure(figsize=(15,10))
plt.scatter(X_train[:,0],X_train[:,1],cmap="plasma")
plt.xlabel("The First Principal Component")
plt.ylabel("The Second Principal Component")
#Here we plot all the rows of columns 1 and column 2 in a scatterplot


# In[18]:


pca.components_


# In[19]:


df_comp=pd.DataFrame(pca.components_)
df_comp


# In[20]:


plt.figure(figsize=(15,10))
sns.heatmap(df_comp,cmap="magma")


# In[21]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)


# In[22]:


from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
#We have %100 procent accuracy although we have just used the main components of the data


# In[23]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
#This is performance of the algorithm with training set


# In[24]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
#This is visualization of performance of the algorithm with test set


# # 2. Linear Discriminant Analysis (LDA):

# In[27]:


plt.figure(figsize=(12,10))
plt.imshow(plt.imread("C://Users//kishore//Pictures//Screenshots//v2.png"))


# In[28]:


X_train, X_test, y_train,y_test =train_test_split(X,y, test_size=0.2,random_state=42)
print(X_train.shape)
print(X_test.shape)


# In[29]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda =LDA(n_components=2) # we select the same number of components
X_train = lda.fit_transform(X_train,y_train) # we have to write both X_train and y_train
X_test = lda.transform(X_test)


# In[30]:


print(X_train.shape)
print(X_test.shape)


# In[31]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)


# In[32]:


from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[33]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('cyan', 'purple', 'white')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('cyan', 'purple', 'white'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()


# In[35]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()


# # GridSearchCV + KFold CV: The Right Way

# In[36]:


from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="ticks", context="talk")


# # Import Libraries

# In[37]:


import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from collections import Counter


# # 2.2 Import Data

# In[38]:


try:
    raw_df = pd.read_csv("C://Users//kishore//heart_failure_clinical_records_dataset.csv")
except:
    raw_df = pd.read_csv('heart.csv')


# In[39]:


raw_df.head()


# # 3. Data pre-processing

# In[44]:


import pandas as pd
import plotly.express as px

# Load the heart disease dataset
df = pd.read_csv("C://Users//kishore//heart (1).csv")

# Check for data imbalance
labels = ["Healthy", "Heart Disease"]
values = df['HeartDisease'].value_counts().tolist()

# Create a pie chart to visualize the data imbalance
fig = px.pie(values=values, names=labels, width=700, height=400, color_discrete_sequence=["skyblue", "black"],
             title="Healthy vs Heart Disease")
fig.show()


# # Checking for outliers

# # Creating dummies

# In[55]:


df = pd.get_dummies(df, drop_first=True)



# In[56]:


X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']


# In[69]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.3, random_state = 42)


# # Feature scaling

# In[58]:


from sklearn.preprocessing import StandardScaler

# Creating function for scaling
def Standard_Scaler (df, col_names):
    features = df[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    df[col_names] = features
    
    return df


# In[59]:


col_names = numerical_columns
X_train = Standard_Scaler (X_train, col_names)
X_test = Standard_Scaler (X_test, col_names)


# # 4. Model building

# In[60]:


from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

#We are going to ensure that we have the same splits of the data every time. 
#We can ensure this by creating a KFold object, kf, and passing cv=kf instead of the more common cv=5.

kf = KFold(n_splits=5, shuffle=False)


# In[61]:


rf = RandomForestClassifier(n_estimators=50, random_state=13)
rf.fit(X_train, y_train)


# In[62]:


y_pred = rf.predict(X_test)


# In[63]:


from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
cm = confusion_matrix(y_test, y_pred)

rf_Recall = recall_score(y_test, y_pred)
rf_Precision = precision_score(y_test, y_pred)
rf_f1 = f1_score(y_test, y_pred)
rf_accuracy = accuracy_score(y_test, y_pred)

print(cm)


# # K-Fold Cross-validation

# In[64]:


from statistics import stdev
score = cross_val_score(rf, X_train, y_train, cv=kf, scoring='recall')
rf_cv_score = score.mean()
rf_cv_stdev = stdev(score)
print('Cross Validation Recall scores are: {}'.format(score))
print('Average Cross Validation Recall score: ', rf_cv_score)
print('Cross Validation Recall standard deviation: ', rf_cv_stdev)


# In[67]:


from statistics import stdev
score = cross_val_score(rf, X_train, y_train, cv=kf, scoring='recall')
rf_cv_score = score.mean()
rf_cv_stdev = stdev(score)
print('Cross Validation Recall scores are: {}'.format(score))
print('Average Cross Validation Recall score: ', rf_cv_score)
print('Cross Validation Recall standard deviation: ',rf_cv_stdev)


# # Hyperparameter Tuning Using GridSearchCV

# In[ ]:


from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [130, 150, 170, 190, 200],
    'max_depth': [8, 10, 12, 14],
    'min_samples_split': [3, 4, 5, 6],
    'min_samples_leaf': [1, 2, 3],
    'random_state': [13]
}

grid_rf = GridSearchCV(rf, param_grid=params, cv=kf, scoring='recall').fit(X_train, y_train)


# In[ ]:


print('Best parameters:', grid_rf.best_params_)
print('Best score:', grid_rf.best_score_)


# In[ ]:


y_pred = grid_rf.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)

grid_rf_Recall = recall_score(y_test, y_pred)
grid_rf_Precision = precision_score(y_test, y_pred)
grid_rf_f1 = f1_score(y_test, y_pred)
grid_rf_accuracy = accuracy_score(y_test, y_pred)

print(cm)


# # 4.4 K-Fold Cross-validation after tuning hyperparameters

# In[ ]:


score2 = cross_val_score(grid_rf, X_train, y_train, cv=kf, scoring='recall')


# In[ ]:


grid_cv_score = score2.mean()
grid_cv_stdev = stdev(score2)

print('Cross Validation Recall scores are: {}'.format(score2))
print('Average Cross Validation Recall score: ', grid_cv_score)
print('Cross Validation Recall standard deviation: ', grid_cv_stdev)


# In[ ]:


ndf2 = [(grid_rf_Recall, grid_rf_Precision, grid_rf_f1, grid_rf_accuracy, grid_cv_score, grid_cv_stdev)]

grid_score = pd.DataFrame(data = ndf2, columns=
                        ['Recall','Precision','F1 Score', 'Accuracy', 'Avg CV Recall', 'Standard Deviation of CV Recall'])
grid_score.insert(0, 'Random Forest', 'After tuning hyperparameters')
grid_score


# In[ ]:


predictions = pd.concat([rf_score, grid_score], ignore_index=True, sort=False)
predictions.sort_values(by=['Avg CV Recall'], ascending=False)


# In[ ]:


from sklearn.metrics import roc_auc_score
ROCAUCscore = roc_auc_score(y_test, y_pred)
print(f"AUC-ROC Curve for Random Forest with tuned hyperparameters: {ROCAUCscore:.4f}")


# In[ ]:


y_proba = grid_rf.predict_proba(X_test)

from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
def plot_auc_roc_curve(y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    roc_display.figure_.set_size_inches(5,5)
    plt.plot([0, 1], [0, 1], color = 'g')
# Plots the ROC curve using the sklearn methods - Good plot
plot_auc_roc_curve(y_test, y_proba[:, 1])
# Plots the ROC curve using the sklearn methods - Bad plot
#plot_sklearn_roc_curve(y_test, y_pred)


# # LOOCV

# In[34]:


from numpy import mean
from numpy import std
from sklearn.datasets import make_blobs
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Create dataset
x, y = make_blobs(n_samples=100, random_state=0)

# Create LOOCV procedure
cv = LeaveOneOut()

# Create model
model = RandomForestClassifier(random_state=1)

# Evaluate model using LOOCV
scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)

# Report performance
print(f"Accuracy: {mean(scores):.3f} ({std(scores):.3f})")


# In[2]:





# In[3]:


iris_ds= pd.read_csv("C://Users//kishore//Iris.csv")
iris_ds.columns,iris_ds.shape
#print(iris_ds.shape)


# In[4]:


iris_ds.head()


# In[5]:


iris_ds.tail()


# In[6]:


iris_ds.isnull().sum()


# In[7]:


iris_ds.dtypes


# In[9]:


iris_ds.Species.value_counts()


# In[10]:


iris_ds.columns.values


# In[11]:


iris_ds.hist(figsize=(20,10))


# In[12]:


sns.pairplot(iris_ds,hue='PetalWidthCm')


# In[13]:


sns.boxplot(data=iris_ds)


# In[14]:


iris_ds.describe()


# In[15]:


iris_ds.info()


# In[17]:


iris_ds.Target = iris_ds.Species.astype('category')


# In[18]:


iris_ds.info()


# In[19]:


iris_ds.Target.cat.codes.head()


# In[24]:


iris_ds.Target.tail()


# In[25]:


iris_ds.columns.values


# In[27]:


X = iris_ds[['SepalWidthCm','PetalLengthCm']]
y = iris_ds.Species


# In[31]:


def normal_prediction():
    logis = LogisticRegression()
    logis.fit(x_train,y_train)
    print("logistic regression::\n",confusion_matrix(y_test,logis.predict(x_test)),"\n")
    
    svm = SVC()
    svm.fit(x_train,y_train)
    print("SVM ::\n",confusion_matrix(y_test,logis.predict(x_test)),"\n")
    
    knn = KNeighborsClassifier()
    knn.fit(x_train,y_train)
    print("KNN ::\n",confusion_matrix(y_test,knn.predict(x_test)),"\n")
    
    dTmodel = DecisionTreeClassifier()
    dTmodel.fit(x_train,y_train)
    print("DecisionTree ::\n",confusion_matrix(y_test,dTmodel.predict(x_test)),"\n")
    
    rForest = RandomForestClassifier()
    rForest.fit(x_train,y_train)
    print("RandomForest ::\n",confusion_matrix(y_test,rForest.predict(x_test)),"\n")

    grBoosting = GradientBoostingClassifier()
    grBoosting.fit(x_train,y_train)
    print("GradientBoosting ::\n",confusion_matrix(y_test,grBoosting.predict(x_test)),"\n")


# In[ ]:





# In[ ]:




