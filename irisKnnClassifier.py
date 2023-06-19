# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:48:53 2023

@author: tamer
"""

#1. Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2. Data Preprocessing
data = pd.read_csv("iris.csv")

x = data.drop(["variety"],axis=1).values
y = data.variety.values

#2.1 Transforming String Values to Numeric Values
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)

#3. Training Section
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#4. Classifier Applying
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

#4.1 Predictions
y_pred = knn.predict(x_test)

#5. Scores and Confusion Matrix of Predictions
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
    
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)



