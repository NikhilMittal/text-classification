#!/usr/bin/env python
# -*- coding: utf-8 -*-

from util import *
from linear_model import LinearModel
from naive_bayes import NaiveBayes
from svm import SVM

from sklearn import metrics

## To get data from util
train_X, valid_X, train_y, valid_y = get_data()

## Linear Model
print("\nExecuting Logistic Regression!")
model_LM = LinearModel()
predictions_LM = model_LM.get_predictions()
score_LM = model_LM.get_score(predictions_LM)

## Naive Bayes
print("\nExecuting Multinomial Naive Bayes Model!")
model_NB = NaiveBayes()
predictions_NB = model_NB.get_predictions()
score_NB = model_NB.get_score(predictions_NB)

## SVM
print("\nExecuting Support Vector Classifier!")
model_SV = SVM()
predictions_SV = model_SV.get_predictions()
score_SV = model_SV.get_score(predictions_SV)
		
## Voting Algorithm
## Idea:
## In case of majority among three model, we prefer to go with the majority. 
## In case of no majority, we prefer the best performing individual model
## which is Logistic Regression in this case.
print("\nExecuting Voting Algorithm!")
predictions_VA = []
for i, x in enumerate(valid_X) :
    if predictions_NB[i] == predictions_SV[i] :
        predictions_VA = np.append(predictions_VA, predictions_SV[i])
    else:
        predictions_VA = np.append(predictions_VA, predictions_LM[i])

## Score Voting Algorithm
score_VA = metrics.accuracy_score(predictions_VA, valid_y)

## Score Comparison
print("\nComparing Accuracy Scores: ")
print("Logistic Regression: ", score_LM)
print("Multinomial Naive Bayes Model: ", score_NB)
print("Linear Support Vector Classifier: ", score_SV)
print("Voting Algorithm: ", score_VA)
         