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
os.chdir('U:\\noisy_data\\dataset_mobile')
list_accuracy = []
list_f1 = []
list_precision = []


labels = None

accuracy_per_additional_noise = {}


def train_my_model(data, labels, file_name):

    global dict_feature
    scores = []

    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    #train_features, test_features, train_labels, test_labels = train_test_split(data, labels, test_size = 0.20, random_state = 42)
    train_features, test_features, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.20)

    #from sklearn import preprocessing
    #train_features =  preprocessing.StandardScaler().fit_transform(train_features)
    # Import the model we are using
    from sklearn.ensemble import RandomForestClassifier
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)

    # Train the model on training data
    rf.fit(train_features, train_labels)

    if file_name == 'dataset_mobile.csv':
        from sklearn.cross_validation import cross_val_score
        scores = cross_val_score(
            rf, train_features, train_labels, cv=10, scoring='accuracy')
        print('the mean cross-validation score of the dataset is:{}'.format(sum(scores)/float(len(scores))))

    # Train the model on training data
    #rf.fit(train_features, train_labels);
    # predict the results
    pred = rf.predict(test_features)

    from sklearn.metrics import accuracy_score, precision_score, f1_score
    # Calculate the metrics
    accuracy_per_additional_noise[file_name] = f1_score(
        test_labels, pred, average='weighted')
    #accuracy_per_additional_noise.append(accuracy_score(test_labels, pred))
    #print('the accuracy value is:{}'.format(accuracy_score(test_labels, pred)))
    #print('The accuracy for the original dataset is:{}'.format(accuracy_score(test_labels, pred)))


def prepare_data(file_name):
    global labels

    #print('the length of precision_order is:{}'.format(len(precision_order)))

    data = pd.read_csv(file_name)
    data.head()
    labels = np.array(data['price_range'])
    data = data.drop('price_range', axis=1)
    # Remove the features which have low co-relation
    data = data.drop('blue', axis=1)
    data = data.drop('clock_speed', axis=1)
    data = data.drop('dual_sim', axis=1)
    data = data.drop('fc', axis=1)
    data = data.drop('four_g', axis=1)
    data = data.drop('n_cores', axis=1)
    data = data.drop('sc_h', axis=1)
    data = data.drop('talk_time', axis=1)
    data = data.drop('three_g', axis=1)
    data = data.drop('wifi', axis=1)

    # for j in range(len(precision_order) - (i+1)):
    #    data= data.drop(precision_order[j], axis = 1)

    data_list = list(data.columns)

    # Convert to numpy array
    data = np.array(data)

    train_my_model(data, labels, file_name)


print('starting.............')
data = prepare_data('dataset_mobile.csv')


for i in range(10, 103, 10):
    # for j in range(1,3):
    # print('processing:{}'.format('noisy_data_'+str(i)+'_'+str(j)+'.csv'))
    print('processing:{}'.format('noisy_data_'+str(i)+'_'+'1.csv'))
    data = prepare_data('noisy_data_'+str(i)+'_1.csv')


for i in range(1, 6):
    # for j in range(1,3):
    print('processing:{}'.format('noisy_data_feature_'+str(i)+'_1.csv'))
    data = prepare_data('noisy_data_feature_'+str(i)+'_1.csv')

# for i in range(1,6):
#    for j in range(1,3):
#        print('processing:{}'.format('noisy_data_feature_y_'+str(i)+'_'+str(j)+'.csv'))
#        data = prepare_data('noisy_data_feature_y_'+str(i)+'_'+str(j)+'.csv')


for key in accuracy_per_additional_noise.keys():
    print('{}:{}'.format(key, accuracy_per_additional_noise[key]))
