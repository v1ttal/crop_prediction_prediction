
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
crop_data=pd.read_csv("Crop_recommendation.csv")
crop_data
crop_data.shape

#rows X columns
crop_data.info()
# dataset columns
crop_data.columns
crop_data.rename(columns = {'label':'Crop'}, inplace = True)
crop_data
# statistical inference of the dataset

crop_data.describe()
# Checking missing values of the dataset in each column
crop_data.isnull().sum()
# Dropping missing values
crop_data = crop_data.dropna()
crop_data
#checking
crop_data.isnull().values.any()
# Visualizing the features

ax = sns.pairplot(crop_data)
ax
crop_data.Crop.unique()
# get top 5 most frequent growing crops
n = 5
crop_data['Crop'].value_counts()[:5].index.tolist()
sns.barplot(crop_data["Crop"], crop_data["temperature"])
plt.xticks(rotation = 90)
sns.barplot(crop_data["Crop"], crop_data["ph"])
plt.xticks(rotation = 90)
sns.barplot(crop_data["Crop"], crop_data["humidity"])
plt.xticks(rotation = 90)
sns.barplot(crop_data["Crop"], crop_data["rainfall"])
plt.xticks(rotation = 90)
crop_data.corr()
sns.heatmap(crop_data.corr(), annot =True)
plt.title('Correlation Matrix')

# Shuffling data to remove order effects

# shuffling the dataset to remove order
from sklearn.utils import shuffle

df  = shuffle(crop_data,random_state=5)
df.head()

# Selection of Feature and Target variables.

x = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['Crop']
# Encoding target variable

y = pd.get_dummies(target)
y
# Splitting data set - 25% test dataset and 75%


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state= 0)

print("x_train :",x_train.shape)
print("x_test :",x_test.shape)
print("y_train :",y_train.shape)
print("y_test :",y_test.shape)
# Importing necessary libraries for multi-output classification

from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
# Training

forest = RandomForestClassifier(random_state=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
multi_target_forest.fit(x_train, y_train)
# Predicting test results

forest_pred = multi_target_forest.predict(x_test)
forest_pred
# Calculating Accuracy

from sklearn.metrics import accuracy_score
a1 = accuracy_score(y_test, forest_pred)
print('Accuracy score:', accuracy_score(y_test, forest_pred))
from sklearn.model_selection import cross_val_score
score = cross_val_score(multi_target_forest,X = x_train, y = y_train,cv=5)
score
b1 = "{:.2f}".format(score.mean()*100)
b1 = float(b1)
b1
c1 = (score.std()*100)
c1
print("Accuracy : {:.2f}%".format (score.mean()*100))
print("Standard Deviation : {:.2f}%".format(score.std()*100))
# Training
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=6)
multi_target_decision = MultiOutputClassifier(clf, n_jobs=-1)
multi_target_decision.fit(x_train, y_train)
# Predicting test results

decision_pred = multi_target_decision.predict(x_test)
decision_pred
# Calculating Accuracy

from sklearn.metrics import accuracy_score
a2 = accuracy_score(y_test,decision_pred)
print('Accuracy score:', accuracy_score(y_test,decision_pred))
a2
from sklearn.model_selection import cross_val_score
score = cross_val_score(multi_target_decision,X = x_train, y = y_train,cv=7)
score
b2 = "{:.2f}".format(score.mean()*100)
b2 = float(b2)
b2
c2 = (score.std()*100)
c2
from sklearn.neighbors import KNeighborsClassifier

knn_clf=KNeighborsClassifier()
model = MultiOutputClassifier(knn_clf, n_jobs=-1)
model.fit(x_train, y_train)
knn_pred = model.predict(x_test)
knn_pred
# Calculating Accuracy

from sklearn.metrics import accuracy_score
a3 = accuracy_score(y_test,knn_pred)
print('Accuracy score:', accuracy_score(y_test,knn_pred))
a3
from sklearn.model_selection import cross_val_score
score = cross_val_score(model,X = x_train, y = y_train,cv=7)
score
b3 = "{:.2f}".format(score.mean()*100)
b3 = float(b3)
b3
c3 = (score.std()*100)
c3
import pandas as pd

# initialise data of lists.
data = {'Algorithms': ['Random Forest', 'Decision-tree', 'KNN Classifier'],
        'Accuracy': [b1, b2, b3],
        'Standard Deviation': [c1, c2, c3]}
# Creates pandas DataFrame.
df = pd.DataFrame(data)

# print the data
df
import numpy as np
import matplotlib.pyplot as plt

# create a dataset
Algorithms = ['Random Forest', 'Decision-tree', 'KNN Classifier']
Accuracy = [b1, b2, b3]

x_pos = np.arange(len(Accuracy))

# Create bars with different colors
plt.bar(x_pos, Accuracy, color=['#488AC7', '#ff8c00', '#009150'])

# Create names on the x-axis
plt.xticks(x_pos, Algorithms)
plt.ylabel('Accuracy(in %)')
plt.xlabel('Machine Learning Classifying Techniques')

# Show graph
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# create a dataset
Algorithms = ['Random Forest', 'Decision-tree', 'KNN Classifier']
Accuracy = [b1, b2, b3]

x_pos = np.arange(len(Accuracy))

# Create bars with different colors
plt.bar(x_pos, Accuracy, color=['#488AC7', '#ff8c00', '#009150'])

# Create names on the x-axis
plt.xticks(x_pos, Algorithms)
plt.ylabel('Accuracy(in %)')
plt.xlabel('Machine Learning Classifying Techniques')

# Show graph
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# create a dataset
Algorithms = ['Random Forest', 'Decision-tree', 'KNN']
Accuracy = [c1, c2, c3]

x_pos = np.arange(len(Accuracy))

# Create bars with different colors
plt.bar(x_pos, Accuracy, color=['#488AC7', '#ff8c00', '#009150'])

# Create names on the x-axis
plt.xticks(x_pos, Algorithms)
plt.ylabel('Standard Deviation(in %)')
plt.xlabel('Machine Learning Classifying Techniques')

# Show graph
plt.show()


def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha='center')


if __name__ == '__main__':
    # creating data on which bar chart will be plot
    x = ["Random Forest", "Decision tree", "KNN"]
    y = [b1, b2, b3]

    x_pos = np.arange(len(y))

    # Create bars with different colors
    plt.bar(x_pos, y, color=['#A52A2A', '#00008B', '#2E8B57'])

    # calling the function to add value labels
    addlabels(x, y)

    # giving X and Y labels
    plt.xlabel("Machine Learning Classifying Algorithms")
    plt.ylabel("Accuracy (in %)")

    # visualizing the plot
    plt.show()
    plt.bar(df['Algorithms'], df['Accuracy'], color=['#A52A2A', '#00008B', '#2E8B57'])
    fig = plt.figure(figsize=(15, 10))
    plt.title('Crop Suggestion Model')

    # Show Plot
    plt.show()

    import numpy as np
    import matplotlib.pyplot as plt

    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(10, 6))

    # set height of bar
    Algorithms = ['Random Forest', 'Decision-tree', 'KNN Classifier']
    Accuracy = [b1, b2, b3]
    Standard_Deviation = [c1, c2, c3]

    # Set position of bar on X axis
    br1 = np.arange(len(Accuracy))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, Accuracy, color='blue', width=barWidth,
            edgecolor='grey', label='Accuracy')
    plt.bar(br2, Standard_Deviation, color='maroon', width=barWidth,
            edgecolor='grey', label='Standard Devation')

    # Adding Xticks
    plt.xlabel('Algorithms', fontweight='bold', fontsize=10)
    plt.ylabel('Accuracy (in %)', fontweight='bold', fontsize=10)
    plt.xticks([r + barWidth for r in range(len(Accuracy))],
               Algorithms)

    plt.legend()
    plt.show()
    # Saving the trained Random Forest model
    import pickle

    # Dump the trained Naive Bayes classifier with Pickle
    RF_pkl_filename = 'RandomForest.pkl'
    # Open the file to save as pkl file
    RF_Model_pkl = open(RF_pkl_filename, 'wb')
    pickle.dump(multi_target_forest, RF_Model_pkl)
    # Close the pickle instances
    RF_Model_pkl.close()

