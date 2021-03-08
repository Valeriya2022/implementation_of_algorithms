

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#Import the data set from Desktop
dataset = pd.read_csv(r"C:\Users\valeriya.nikiforova\Documents\UCA\Junior\semester 2\Machine Learning\week_1\implementation_of_algorithms\K_nearest_neigbour\KNN_Data.csv")
X=dataset.iloc[:,[0,1]].values
y=dataset.iloc[:,2].values
#Training and Testing Data (divide the data into two part)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.30, random_state=0)


# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifer = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2 )
classifer.fit(X_train,y_train)


y_pred= classifer.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True)
plt.savefig('h.png')
print(cm)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
                            

