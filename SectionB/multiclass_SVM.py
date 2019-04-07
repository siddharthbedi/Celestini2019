#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:58:44 2019

@author: siddharth
"""

import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#reading the file

df = pd.read_csv('zoo_data.csv')



#splitting into independent and dependent set
X = df.iloc[:,1:17].values
y = df.iloc[:,[17]].values



#Train-Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Fitting SVM to the Training set
# Run required classifier
#classifier = SVC(kernel = 'linear', random_state = 0)
#classifier = SVC(kernel = 'rbf', random_state = 0)
#classifier = SVC(kernel = 'poly', random_state = 0)
classifier = SVC(kernel = 'sigmoid', random_state = 0,probability = True)



#fitting the model
classifier.fit(X_train, y_train)



#reshaping the dataframe to process it in further stages
y_test = y_test.ravel()



#Predicting the Test set results
y_pred = classifier.predict(X_test)



#applying k-fold validation for K = 5
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train , y = y_train ,cv = 5 )


#gives accuracy score as the output
accuracy_score(y_test, y_pred)

#gives precision score as the output
precision_score(y_test, y_pred, average='macro') 



