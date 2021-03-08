

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#Import the data set from Desktop
dataset = pd.read_csv(r'C:\Users\valeriya.nikiforova\Documents\UCA\Junior\semester 2\Machine Learning\week_1\implementation_of_algorithms\K_nearest_neigbour\Data_Confusion_Matrix.csv')
Actual = dataset.iloc[:,0].values
Predicted = dataset.iloc[:,1].values
# Actual=[1,1,0,0,1,1]
# Predicted= [1,1,1,0,0,1]
#Training and Testing Data (divide the data into two part)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Actual, Predicted)
sns.heatmap(cm,annot=True)
plt.savefig('h.png')
print(cm)

from sklearn.metrics import accuracy_score
accuracy_score(Actual,Predicted)

from sklearn.metrics import classification_report
print(classification_report(Actual, Predicted))
                            

