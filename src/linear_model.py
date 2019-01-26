#!/usr/bin/env python
# -*- coding: utf-8 -*-

from util import *

from sklearn import linear_model
from sklearn import metrics

## To get data from util
train_X, valid_X, train_y, valid_y = get_data()

## To get tf-idf feature vectors using util
train_X_vec, valid_X_vec = get_tfidf_feature_vectors()

## Logistic Regression Model
class LinearModel:
	def __init__(self):
		self.clf = linear_model.LogisticRegression()
		self.clf.fit(train_X_vec, train_y)
		
	def get_predictions(self):
		return self.clf.predict(valid_X_vec)
		
	def get_score(self, predictions):
		return metrics.accuracy_score(predictions, valid_y)

set_display_output = input('Do you want to run Linear Model separately - Yes/No? ')

if set_display_output == 'Yes':
	## Model execution
	model = LinearModel()
	print("Executing Logistic Regression Model")
	execute(model)