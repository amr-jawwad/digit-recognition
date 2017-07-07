# -*- coding: utf-8 -*-
"""
Created on Fri 07 Jul 2017

@author: Amr
@email : amr.jawwad@outlook.com
"""
from sklearn import svm, metrics
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import sys

def BuildModel():
    #Load data
    try:
        mnist = fetch_mldata('MNIST original')
    except:
        print "Could not load MNIST data. Terminating program."
        sys.exit(0)
    
    X = mnist.data/255.0
    y = mnist.target
    
    #Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0,stratify=y)
    
    #Set to 1 if you want to search hyperparameters for best precision (takes time in the order of days)
    Get_Optimal_Model_Params = 0
    
    if Get_Optimal_Model_Params:
        #Hyperparameters search space
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.05, 1e-3, 1e-4], 'C': [1, 5, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 5, 10, 100, 1000]}]
        print "Started searching for optimal hyperparameters:"
        classifier = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5,scoring='precision_macro')
        print "Best parameters:"
        print classifier.best_params_
        print "Fitting model:"
        classifier.fit(X_train,y_train)
    else:
        #Nice hyperparameters
        classifier = svm.SVC(C=5,gamma=0.05)
        print "Fitting model:"
        classifier.fit(X_train, y_train)
    try:
        #Save model to file
        joblib.dump(classifier, 'SVM_model.pkl')
    except:
        print "Could not save model to file. The program will continue exection, however."
    
    #Evaluate model
    expected = y_test
    predicted = classifier.predict(X_test)
    
    print "Classification report for classifier:"
    print metrics.classification_report(expected, predicted)
    print "Confusion matrix:"
    print metrics.confusion_matrix(expected, predicted)
    print "Accuracy:"
    print metrics.accuracy_score(expected, predicted)

BuildModel()