# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 07:44:10 2017

@author: João Pedro Augusto Costa
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation
from sklearn import svm
import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import math

##################### GARCIARENA PAPER (2017) ####################
MDP = 0.07
NV = 3

def _random(bounds, samples = None):
    if(samples == None):
        return random.randint(bounds[0], bounds[1])
    else:
        numbers = []
        for x in range(samples):
            numbers.append(random.randint(bounds[0], bounds[1]))
        return numbers

def _minIndex(array):        
    return np.asscalar((np.where(array == min(array))[0][0]))

def _numObservations(data):
    return len(data)

def _numVariables(data):
    return len(data[0])

def _mcarGenerating(data):
    x = _numObservations(data)
    y = _numVariables(data)   
    
    for i in range(int(x * MDP)):
        pos1 = _random([0, x - 1])
        pos2 = _random([0, y - 1])
        data[pos1, pos2] = np.nan        
    return data

def _muovGenerating(data):
    x = _numObservations(data)
    y = _numVariables(data) 
    MDVariables = _random([0, y - 1], NV)
    observations = []
    
    for i in range(int((x * MDP) / NV)):
        observations.append(_random([0, x - 1]))
    
    for i in range(0, len(observations)):
        for j in range(0, len(MDVariables)):
            data[observations[i], MDVariables[j]] = np.nan
        
    return data

def _mivGenerating(data):
    x = _numObservations(data)
    y = _numVariables(data)    
    causatives = _random([0, y - 1], NV)
    
    for i in range(0, len(causatives)):
        observations = []
        aux = data[:, causatives[i]]
        for j in range(0, int((x * MDP) / NV)):
            observations.append(_minIndex(aux))
            aux[observations[j]] = float('inf')
        for j in range(0, len(observations)):
            data[observations[j], causatives[i]] = np.nan
    return data

def _imputationMean(data):
    means = []
    numCols = len(data[0])  
    for x in range(0, numCols):
        col = data[:, x]                 
        means.append(np.mean(col[~np.isnan(col)]))        
        col[np.isnan(col)] = means[x]  

def _imputationMedian(data):
    numCols = len(data[0])  
    for x in range(0, numCols):
        col = data[:, x]  
        sortedCol = np.sort(col)
        median = sortedCol[int(len(sortedCol)/2)]
        col[np.isnan(col)] = median
        
def _imputationMostFrequentValue(data):
    numCols = len(data[0])  
    for x in range(0, numCols):
        col = data[:, x]    
        unique,pos = np.unique(col,return_inverse=True)
        howMany = np.bincount(pos)
        mostFrequent = col[howMany.argmax()]
        col[np.isnan(col)] = float(mostFrequent)
    
#################################################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

#Simple normalization
def simpleNormalization(data):
    numCols = len(data[0])       
    for x in range(0, numCols):
        col = data[:, x]           
        lvMax = max(col)
        lvMin = min(col)
        for y in range(0, len(col)):              
            data[y,x] = (data[y,x] - lvMin) / (lvMax - lvMin)               

def plot(test_data, series):
    X = np.arange(len(test_data))
    plt.plot(X, test_data , 'bs')
    plt.show()
    for s in series:
        plt.plot(X, s , 'bs')
        plt.show()
    
def load():    
    with open('creditcard.csv') as csvfile:
        reader = csv.reader(csvfile)
        data = []
        for row in reader:
            data.append([float(x) for x in row if is_number(x)])        
        data = np.array(data[1:])     
        simpleNormalization(data)       
        return data

def runMlp(train_data, train_labels, test_data, test_labels):
    mlp = MLPClassifier(solver='sgd', learning_rate='adaptive', learning_rate_init=0.02, max_iter=10000,
                        activation='tanh', momentum=0.9, shuffle=True)
    mlp.fit(train_data, train_labels)
    prediction = mlp.predict(test_data)
    return prediction    

def runLabelSpreading(train_data, train_labels, test_data, test_labels):
    labelSpreading = LabelSpreading(kernel='knn')
    labelSpreading.fit(train_data, train_labels)
    prediction = labelSpreading.predict(test_data)
    return prediction

def runLabelPropagation(train_data, train_labels, test_data, test_labels):
    labelPropagation = LabelPropagation(kernel='knn')
    labelPropagation.fit(train_data, train_labels)
    prediction = labelPropagation.predict(test_data)
    return prediction

def runSVM(train_data, train_labels, test_data, test_labels):
    clf = svm.SVC()
    clf.fit(train_data, train_labels)
    prediction = clf.predict(test_data)
    return prediction

def main():
    dataset = load()   
    data = dataset[:, 0 : -1]
    labels = dataset[:, -1]  #last colum   
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels,
                                                                        train_size=0.8)
    _mivGenerating(train_data)
    _imputationMostFrequentValue(train_data)     
   
    prediction = runLabelPropagation(train_data, train_labels, test_data, test_labels)
    prediction2 = runLabelSpreading(train_data, train_labels, test_data, test_labels)
    prediction3 = runMlp(train_data, train_labels, test_data, test_labels)
    prediction4 = runSVM(train_data, train_labels, test_data, test_labels)    
    print((test_labels == 1).sum())
    print((prediction == 1).sum())
    print((prediction2 == 1).sum())
    print((prediction3 == 1).sum())
    print((prediction4 == 1).sum())
    plot(test_labels, [prediction, prediction2, prediction3, prediction4])

    #plot(labels, labels)
            
    
main()