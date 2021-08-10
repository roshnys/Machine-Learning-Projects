import csv
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

g_open = [[1389]] #open value of today of google
r_open = [[0.2017]] #open value of today of ripple

# Reading dataset of google stocks
dataset1 = pd.read_csv("GOOG.csv")
dataset1 = dataset1.dropna()
X1=dataset1.iloc[: , 1:2].values
y1=dataset1.iloc[: , 2:3].values

# Reading dataset of Ripple stocks
dataset2 = pd.read_csv("XRP-USD.csv")
dataset2 = dataset2.dropna()
X2=dataset2.iloc[: , 1:2].values
y2=dataset2.iloc[: , 2:3].values

#transforming dataset of google for Support Vector Regressor
sc_X1 = StandardScaler()
sc_y1 = StandardScaler()
X1 = sc_X1.fit_transform(X1)
y1 = y1.reshape(-1,1)
y1 = sc_y1.fit_transform(y1)

#transforming dataset ripple for Support Vector Regressor
sc_X2 = StandardScaler()
sc_y2 = StandardScaler()
X2 = sc_X2.fit_transform(X2)
y2 = y2.reshape(-1,1)
y2 = sc_y2.fit_transform(y2)

#Support Vector Regressor for google
regressor1 = SVR(kernel='rbf', C=450, gamma=0.1, epsilon=.1)
regressor1.fit(X1,y1)
y_pred1 = regressor1.predict([[6.5]])
y_pred1 = sc_y1.inverse_transform(y_pred1)
print(y_pred1)

#Support Vector Regressor for ripple
regressor2 = SVR(kernel='rbf', C=250, gamma=0.5, epsilon=.1)
regressor2.fit(X2,y2)
#y_pred2 = regressor2.predict(r_open)
y_pred2 = regressor2.predict([[6.5]])
y_pred2 = sc_y2.inverse_transform(y_pred2)
print(y_pred2)
