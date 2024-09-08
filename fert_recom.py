import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
Data_path = "D:/Python Projects/Crop-Prediction-Django-master/Fertilizer Prediction.csv"
df = pd.read_csv(Data_path)
df.head()
# Statistical info
df.describe()
# Datatypes of Attributes
df.info()

# Check the unique values in dataset
df.apply(lambda x: len(x.unique()))
df.isnull().sum()
# check for categorical attributes
cat_col = []
for x in df.dtypes.index:
    if df.dtypes[x] == 'object':
        cat_col.append(x)
cat_col

# print the categorical columns
for col in cat_col:
    print(col)
    print(df[col].value_counts())
    print()
    sns.countplot(df['Soil Type'])

plt.xticks(rotation=90)
sns.countplot(df['Crop Type'])

plt.xticks(rotation=90)
sns.countplot(df['Fertilizer Name'])


# Defining function for Continuous and Catogorical variable

def plot_conti(x):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), tight_layout=True)
    axes[0].set_title('Distribution')
    sns.distplot(x, ax=axes[0])
    axes[1].set_title('Checking Outliers')
    sns.boxplot(x, ax=axes[1])
    axes[2].set_title('Relation with Output Variable')
    sns.boxplot(y=x, x=df['Fertilizer Name'])


def plot_cato(x):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), tight_layout=True)
    axes[0].set_title('Count Plot')
    sns.countplot(x, ax=axes[0])
    axes[1].set_title('Relation with Output Variable')
    sns.countplot(x=x, hue=df['Fertilizer Name'], ax=axes[1])
# EDA - Temparature variable
plot_conti(df['Temparature'])

# EDA - Humidity variable
plot_conti(df['Humidity '])

# EDA - Moisture variable
plot_conti(df['Moisture'])

plot_cato(df['Soil Type'])

# Relation of Soil Type with Temperature
plt.xticks(rotation=90)
sns.boxplot(x=df['Soil Type'],y=df['Temparature'])

# Relation of Soil Type and Temperature with Output Variable
plt.figure(figsize=(15,6))
sns.boxplot(x=df['Soil Type'],y=df['Temparature'],hue=df['Fertilizer Name'])

# EDA - Crop Type variable
plot_cato(df['Crop Type'])

# Relation of Crop Type with temperature
plt.xticks(rotation=90)
sns.boxplot(x=df['Crop Type'],y=df['Temparature'])
# Relation of Crop Type with Humidity
plt.figure(figsize=(15,8))
sns.boxplot(x=df['Crop Type'],y=df['Humidity '])

# EDA - Nitrogen variable
plot_conti(df['Nitrogen'])

# Relation of Nitrogen wrt to Crop Type
plt.figure(figsize=(15,8))
sns.boxplot(x=df['Crop Type'],y=df['Nitrogen'])

# EDA - Potassium variable
plot_conti(df['Potassium'])

# EDA - Phosphorous variable
plot_conti(df['Phosphorous'])
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# Check the number of zeroes in a Column
# Delete or Drop columns if zero values are more than 50%

sum(df['Potassium'] == 0)

X = df.drop(columns = ['Fertilizer Name', 'Potassium', 'Temparature'], axis=1)
y = df['Fertilizer Name']

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2,3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Initializing empty lists to append all model's name and corresponding name
accuracy = []
model = []

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_knn = sc.fit_transform(X_train[:, 16:])
X_test_knn = sc.transform(X_test[:, 16:])

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(X_train_knn, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_knn)

# Making the Confusion Matrix and Calculating the Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

acc1 = accuracy_score(y_test, y_pred)
accuracy.append(acc1)
model.append('K-Nearest Neighbors')
print("K-Nearest Neighbours's Accuracy :", acc1)

# Cross validation score
from sklearn.model_selection import cross_val_score
score = cross_val_score(classifier, X, y, cv=5)
print("Cross-Validation Score:", score)

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix and Calculating the Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

acc2 = accuracy_score(y_test, y_pred)
accuracy.append(acc2)
model.append('Kernel SVM')
print("Kernel SVM's Accuracy :", acc2)

# Cross validation score
from sklearn.model_selection import cross_val_score
score = cross_val_score(classifier, X, y, cv=5)
print("Cross-Validation Score:", score)

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix and Calculating the Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

acc3 = accuracy_score(y_test, y_pred)
accuracy.append(acc3)
model.append('Naive Bayes')
print("Naive Bayes's Accuracy :", acc3)

# Cross validation score
from sklearn.model_selection import cross_val_score
score = cross_val_score(classifier, X, y, cv=5)
print("Cross-Validation Score:", score)

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix and Calculating the Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

acc4 = accuracy_score(y_test, y_pred)
accuracy.append(acc4)
model.append('Decision Tree Classification')
print("Decision Tree Classification's Accuracy :", acc4)

# Cross validation score
from sklearn.model_selection import cross_val_score
score = cross_val_score(classifier, X, y, cv=5)
print("Cross-Validation Score:", score)

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 100, criterion = 'gini' , random_state= 42)
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix and Calculating the Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

acc5 = accuracy_score(y_test, y_pred)
accuracy.append(acc5)
model.append('Random Forest Classification')
print("Random Forest Classification's Accuracy:", acc5)

# Cross validation score
from sklearn.model_selection import cross_val_score
score = cross_val_score(classifier, X, y, cv=5)
print("Cross-Validation Score:", score)

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

plt.xticks(rotation=90)
sns.barplot(x = model, y = accuracy, palette ='dark')



