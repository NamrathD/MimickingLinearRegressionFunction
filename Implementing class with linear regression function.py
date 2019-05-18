#In this program, I created a class that imitates the function of linear regression function (rather than using the library)
#linear.fit(X_train,Y_train)  y_pred=linear.predict(X_test)
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


class LinearRegression():
    para = []
    def __init__(self):
        pass
    def fit(X_train,y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        para = []
        for i in range(len(X_train[0])):
            para.append(0)
        m = len(y_train) # number of training examples
        iterations = 100 # number of times the program will repeat or iter to find the min
        LinearRegression.GD(X_train,y_train,para,m,iterations)
    def GD(X_train,y_train,para,m,iterations):
        tempPara = para
        for i in range(iterations):
            hypothesis = (X_train.dot(para))
            hypothesis = hypothesis.reshape(-1, 1)
            for j in range(len(para)):
                X_ji = X_train[:,j]
                X_ji= X_ji.reshape(-1,1)
                trust = ((hypothesis-y_train) *X_ji)
                trust = np.array(trust)
                tempPara[j] = para[j] - (0.1*(1/m)* (np.sum((hypothesis-y_train) *X_ji)))
            para = tempPara
            #CostFunction = (1/2*m)*(np.sum((hypothesis-y_train)**2))
            #graphCost.append(CostFunction)

            #print(para)
            LinearRegression.para = para # updating it to whole class variable, so that functions such as predict can use it later
    def predict(X_test):
        X_test = np.array(X_test)
        para = LinearRegression.para
        y_pred = (X_test.dot(para))
        return y_pred
    def score(X_test,y_test, sample_weight=None):
        y_pred = LinearRegression.predict(X_test)
        y_pred = np.array(y_pred)
        y_pred = y_pred.reshape(-1,1)
        y_test = np.array(y_test)
        u = np.sum((y_test - y_pred)**2)
        v = np.sum((y_test - np.mean(y_test))**2)
        score = 1 - (u/v)
        return score
linear = LinearRegression

#graphCost = [] #To graph the cost. length of # of iterations

X_train = np.array([[1],[2],[3],[4]])
y_train = [[3],[6],[9],[12]]

X_test = [[5],[6],[7],[8],[9],[10]]



linear.fit(X_train,y_train)
y_pred = linear.predict(X_test)
score = linear.score(X_train,y_train)
print(y_pred)
print("Score is:",score)
