#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import csv

from sklearn import preprocessing
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

## Loading Data
abstracts_file = '../data/data_input.csv'
categories_file = '../data/data_output.csv'


abstracts_data = list(csv.DictReader(open(abstracts_file), delimiter=',', quotechar='"'))
categories_data = list(csv.DictReader(open(categories_file), delimiter=',', quotechar='"'))

abstracts = np.array([item['abstract'] for item in abstracts_data])
categories = np.array([item['category'] for item in categories_data])

## Label encoding for the categories
unique_categories = list(set(categories))
le = preprocessing.LabelEncoder()
le.fit(unique_categories)
y = le.transform(categories)
X = abstracts

## To split data for training and validation
train_X, valid_X, train_y, valid_y = model_selection.train_test_split(X, y)
def get_data():
	return train_X, valid_X, train_y, valid_y
	
## To convert text into feature vector using TF-IDF Vectorizer
vec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=10000)
vec.fit(X)
train_X_vec = vec.transform(train_X)
valid_X_vec = vec.transform(valid_X)
def get_tfidf_feature_vectors():
	return train_X_vec, valid_X_vec

## To execute a given model and print results
def execute_model(model):
	## Model execution
	predictions = model.get_predictions()
	score = model.get_score(predictions)
	
	## Print results
	print("\nCategories & Index for Performance Matrices:\n")
	print(unique_categories)
	print(le.transform(unique_categories))
	print("\nAccuracy Score: ", score)
	print("\nClassification Report:")
	print(metrics.classification_report(valid_y, predictions))
	print("\nConfusion Matrix:")
	print(metrics.confusion_matrix(valid_y, predictions))
