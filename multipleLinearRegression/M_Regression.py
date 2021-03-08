

#import the libraries
import numpy as py
import matplotlib.pyplot as plt
import pandas as pd

#Import the data set from Desktop
dataset = pd.read_csv(r"C:\Users\valeriya.nikiforova\Documents\UCA\Junior\semester 2\Machine Learning\week_1\implementation_of_algorithms\multipleLinearRegression\M_Regression.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.30, random_state=0)

#regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

#for predict the test values
y_prdict=reg.predict(X_test)
print(y_prdict);

#Visualize the Traing data




