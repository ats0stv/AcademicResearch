#!/usr/bin/python

# Copyright (C) 2018, Arun Thundyill Saseendran {ats0stv@gmail.com, thundyia@tcd.ie},
# Viren Chhabria {chhabriv@tcd.ie, viren.chhabria@gmail.com},
# Debrup Chakraborty {chakrabd@tcd.ie, rupdeb@gmail.com},
# Lovish Setia {setial@tcd.ie}, Aneek Barman Roy {barmanra@tcd.ie}
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pandas as pd
import numpy as np
import os
#import matplotlib.pyplot as plt
os.chdir('C:\\Users\\lovishsetia\\Desktop\\Academic\\ML\\Group\\wine_dataset')
list_accuracy = []
list_f1 = []
list_precision = []

# F1
# PRESCISON
# CONFUSION MATRIX
#data = pd.read_csv('dataset_wine.csv')
# noisy_data0.csv is the original datset
for i in range(10):
    data = pd.read_csv('noisy_data' + str(i) + '.csv')

    data.head()

    # Labels are the values we want to predict
    labels = np.array(data['quality'])

    # Remove the labels from the features axis 1 refers to the columns
    data = data.drop('quality', axis=1)

    # Saving feature names for later use
    data_list = list(data.columns)

    # Convert to numpy array
    data = np.array(data)

    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.25, random_state=42)

    # Import the model we are using
    from sklearn.ensemble import RandomForestClassifier

    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)

    # Train the model on training data
    rf.fit(train_features, train_labels)

    pred = rf.predict(test_features)
    #print('the type of pred is:{}'.format(pred))
    #print('The pred :{} is:{}'.format(i,pred))

    from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix

    list_accuracy.append(accuracy_score(test_labels, pred))
    list_f1.append(f1_score(test_labels, pred, average=None))
    list_precision.append(precision_score(
        test_labels, pred, average='weighted'))
    # confusion matrix
    confusion_matrix(test_labels, pred)


# print(list_accuracy)
# print(list_precision)
# print(list_f1)
