# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:49:27 2018

@author: Nikhil Anand
"""

#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#DAtaframe
dataframe = pd.read_csv("data.csv")

#Input and Output
x = dataframe.iloc[:,0:9].values
y = dataframe.iloc[:,9].values

#Splitting the dataset
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.10,random_state=0)

#Scaling the input values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


#Logistics regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
""" 75% accuracy """

#K-NN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
""" 83.33% accuracy """

#SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
""" 75% accuracy """

#Kernel SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
""" 58.33% """

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
""" 75% """

#Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
""" 58.33% """

#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
""" 66.66% """ 

#scatter plot (Age vs Cancer)
plt.scatter(x_test[:,0],y_test, color='yellow')
plt.scatter(x_test[:,0],y_pred, color='red')
plt.xlabel('Age')
plt.ylabel('Breast Cancer')
plt.show()

#scatter plot (BMI vs Cancer)
plt.scatter(x_test[:,1],y_test, color='yellow')
plt.scatter(x_test[:,1],y_pred, color='green')
plt.xlabel('BMI')
plt.ylabel('Breast Cancer')
plt.show()

#scatter plot (Glucose vs Cancer)
plt.scatter(x_test[:,2],y_test, color='red')
plt.scatter(x_test[:,2],y_pred, color='maroon')
plt.xlabel('Glucose')
plt.ylabel('Breast Cancer')
plt.show()

#scatter plot (Insulin vs Cancer)
plt.scatter(x_test[:,3],y_test, color='cyan')
plt.scatter(x_test[:,3],y_pred, color='green')
plt.xlabel('Insulin')
plt.ylabel('Breast Cancer')
plt.show()

#scatter plot (HOMA vs Cancer)
plt.scatter(x_test[:,4],y_test, color='cyan')
plt.scatter(x_test[:,4],y_pred, color='blue')
plt.xlabel('HOMA')
plt.ylabel('Breast Cancer')
plt.show()

#scatter plot (Leptin vs Cancer)
plt.scatter(x_test[:,5],y_test, color='grey')
plt.scatter(x_test[:,5],y_pred, color='black')
plt.xlabel('Leptin')
plt.ylabel('Breast Cancer')
plt.show()

#Stack plot
labels = ["Age ", "BMI", "Glucose","Insulin","HOMA","Leptin","Adiponectin","Resistin","MCP.1"]
fig,ax = plt.subplots()
ax.stackplot(y,x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5],x[:,6],x[:,7],x[:,8],labels=labels)
ax.legend(loc=2)
plt.margins(0,0)
plt.show()  

#Stack plot
labels = ["Age ", "BMI", "Glucose","Insulin","HOMA","Leptin","Adiponectin","Resistin","MCP.1"]
fig,ax = plt.subplots()
ax.stackplot(y_test,x_test[:,0],x_test[:,1],x_test[:,2],x_test[:,3],x_test[:,4],x_test[:,5],x_test[:,6],x_test[:,7],x_test[:,8],labels=labels)
ax.legend(loc=2)
plt.margins(0,0)
plt.show()  

#Stack plot
labels = ["Age ", "BMI", "Glucose","Insulin","HOMA","Leptin","Adiponectin","Resistin","MCP.1"]
fig,ax = plt.subplots()
ax.stackplot(y_pred,x_test[:,0],x_test[:,1],x_test[:,2],x_test[:,3],x_test[:,4],x_test[:,5],x_test[:,6],x_test[:,7],x_test[:,8],labels=labels)
ax.legend(loc=2)
plt.margins(0,0)
plt.show()  













