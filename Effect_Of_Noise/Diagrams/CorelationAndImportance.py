#!/usr/bin/env python3.6

# Copyright (C) 2018, Arun Thundyill Saseendran | ats0stv@gmail.com, thundyia@tcd.ie
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

"""     Script for correlation and importance based feature selection
"""
__author__ = "Arun Thundyill Saseendran"
__version__ = "0.0.1"
__maintainer__ = "Arun Thundyill Saseendran"
__email__ = "thundyia@tcd.ie"

import os
import sklearn
import logging
import colorsys
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier


# Log Configuration
logging.basicConfig(level=logging.DEBUG)

# Global Var
DATASET = 'bike'


INPUTDIR = '/Users/arun/git/ML1819-task-1--team-20/Effect_Of_Noise/BikeDataSet'
INPUTFILE = 'hour.csv'

if DATASET == 'bike':
    INPUTDIR = '/Users/arun/git/ML1819-task-1--team-20/Effect_Of_Noise/BikeDataSet'
    INPUTFILE = 'hour.csv'

if DATASET == 'wine':
    INPUTDIR = '/Users/arun/git/ML1819-task-1--team-20/Effect_Of_Noise/WineDataSet'
    INPUTFILE = 'dataset_wine.csv'

if DATASET == 'mobile':
    INPUTDIR = '/Users/arun/git/ML1819-task-1--team-20/Effect_Of_Noise/MobileDataSet'
    INPUTFILE = 'mobiledataset.csv'



COLS_TO_TRY_REMOVING = ['dteday', 'instant', 'Noise', 'id', 'registered', 'casual']
COLS_TO_TRY_REMOVING = ['instant','id','dteday']
COLS_TO_TRY_REMOVING = ['instant','id','dteday', 'casual', 'registered']
TARGETS = {'wine':'quality', 'bike':'cnt', 'mobile':'price_range'}
yLABELS = {'wine':'Quality of Wine', 'bike':'Count of Cycles', 'mobile':'Mobile Price Range'}

def init():
    print('')


def createDirIdenpotent(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)


def loadDataUsingPandas(inputDataFilepath):
    logging.info(f'Loading data from the path {inputDataFilepath}')
    dataFrame = pd.read_csv(inputDataFilepath)
    return dataFrame

def cleanData(data):
    logging.debug('Data Frame Keys = {}'.format(str(data.keys())))
    X = data
    for col in COLS_TO_TRY_REMOVING:
        try:
            logging.debug(f'Trying to remove the column {col}')
            X = X.drop(col,axis=1)
        except Exception as e:
            print('') # Do Nothing
    logging.debug('Feature Frame Keys = {}'.format(str(X.keys())))
    return X

def createCorrelationPlot(dataFrame):
    fig, ax = plt.subplots(figsize=(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    corr = dataFrame.corr()
    sns.heatmap(corr, cmap='plasma', annot=True, fmt=".2f")
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.savefig('corr-{}.png'.format(INPUTFILE), bbox_inches='tight', pad_inches=0.2)

def plotVariableImportance(data, target):
    fig, ax = plt.subplots(figsize=(10, 10))
    rfc = RandomForestClassifier(n_estimators=10, n_jobs=10, random_state=42)
    targetData = data[target]
    data = data.drop(target, axis = 1)
    rfc.fit(data,targetData)
    importances = rfc.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
                  axis=0)
    indices = np.argsort(importances)

    features = data.keys()
    
    
    plt.title('Feature Importances')
    plt.bar(range(data.shape[1]), importances[indices],
            color="b", yerr=std[indices], align="center")
    plt.xticks(range(data.shape[1]), [features[i] for i in indices], rotation=90)
    plt.xlim([-1, data.shape[1]])
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    # plt.yticks(range(len(indices)), )
    # plt.xlabel('Relative Importance')
    plt.savefig('importance-{}.png'.format(INPUTFILE), bbox_inches='tight', pad_inches=0.2)


def outlierDetection(data,target):
    temp = data
    temp.drop(target,axis=1)
    temp = RobustScaler().fit_transform(temp)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.boxplot(data=temp, ax=ax)
    print(temp)
    logging.info('Saving box plot')
    plt.xticks(range(temp.shape[1]), [key for key in data.keys()], rotation=90)
    ax.set_xlabel('Features')
    ax.set_ylabel('Scaled Values')
    plt.savefig('box-{}.png'.format(INPUTFILE), bbox_inches='tight', pad_inches=0.2)
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3-q1
    print(iqr)
    data_out = data[~((data < (q1 - 1.5 * iqr)) |(data > (q3 + 1.5 * iqr))).any(axis=1)]
    print(data.shape)
    print(data_out.shape)


def histogram(data, target):
    fig, ax = plt.subplots(figsize=(10, 10))
    n, bins, patches = plt.hist(x=data[target], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    # n, bins, patches = plt.hist(x=data[target], bins='auto', color='#0504aa')
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(target)
    plt.ylabel(yLABELS[DATASET])
    plt.title('Histogram of Target Variable')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    logging.info(f'Histogram created for {target} variable of the {DATASET} dataset')
    plt.savefig('histogram-{}.png'.format(INPUTFILE), bbox_inches='tight', pad_inches=0.2)



    # size, scale = 1000, 10
    
    # data[target].plot.hist(grid=True, bins=4, rwidth=0.9,
    #                 color='#0504aa')
    # plt.title('Histogram of Target Variable')
    # plt.xlabel(target)
    # plt.ylabel(yLABELS[DATASET])
    # plt.grid(axis='y', alpha=0.75)
    # plt.savefig('histogram-{}.png'.format(INPUTFILE), bbox_inches='tight', pad_inches=0.2)

def main():
    init()
    dataFrame = loadDataUsingPandas(os.path.join(INPUTDIR, INPUTFILE))
    logging.debug('Loaded data')
    cleanedData = cleanData(dataFrame)
    logging.debug('Got cleaned Data')

    # outlierDetection(cleanedData, TARGETS[DATASET])
    # createCorrelationPlot(cleanedData)
    plotVariableImportance(cleanedData, TARGETS[DATASET])
    # plotBoxPlot(cleanedData)
    # histogram(cleanedData,TARGETS[DATASET])

if __name__ == '__main__':
  main()