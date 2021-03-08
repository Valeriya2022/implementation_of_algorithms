#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\valeriya.nikiforova\Documents\UCA\Junior\semester 2\Machine Learning\week_1\implementation_of_algorithms\naiveBayesClassifier\KNN_Data.csv')
X = dataset.iloc[:, [0,1]].values
y = dataset.iloc[:, 2].values

#Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size= 0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


from sklearn.naive_bayes import BernoulliNB
classifer=BernoulliNB()
classifer.fit(X_train,y_train)

y_pred = classifer.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm,annot=True)
plt.savefig('h.png')
print(cm)

from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)

from sklearn.metrics import classification_report
print(classification_report(y_pred, y_test))