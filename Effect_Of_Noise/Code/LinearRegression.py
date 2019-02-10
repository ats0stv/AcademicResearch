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

"""     Script to created a linear regression model on Bike
        Dataset and also to calculate and plot it various metrics
"""

import os
import sklearn
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn import datasets
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from prettytable import PrettyTable
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             explained_variance_score, mean_squared_log_error)


# Log Configuration
logging.basicConfig(level=logging.DEBUG)

# Global Var
DATA = None
GRAPHS_DIR = './LinearRegressionNewFeatureGraphs'
DATA_NOISE_INPUT_DIR = '/Users/arun/git/ML1819-task-1--team-20/Effect_Of_Noise/dataset_bike'
FEATURE_NOISE_INPUT_DIR = '/Users/arun/git/ML1819-task-1--team-20/Effect_Of_Noise/dataset_bike'
# INPUT_DATA_FILEPATH = os.path.join(INPUT_DIR, 'day.csv')
DATA_NOISE_FILE_TEMPLATE = 'noisy_data_{}_1.csv'
DATA_FEATURE_FILE_TEMPLATE = 'noisy_data_feature_{}_1.csv'
CLEAN_DATA = 'hour_removed_features.csv'
DATA_NOISE_INPUT_DATA_FILEPATHS = ['hour_removed_features.csv', 'noisy_data1.csv', 'noisy_data2.csv', 'noisy_data3.csv', 'noisy_data4.csv',
                                   'noisy_data5.csv', 'noisy_data6.csv', 'noisy_data7.csv', 'noisy_data8.csv',
                                   'noisy_data9.csv', 'noisy_data10.csv']
# DATA_NOISE_INPUT_DATA_FILEPATHS = ['hour.csv']
FEATURE_NOISE_INPUT_DATA_FILEPATHS = ['hour.csv', 'noisy_feature1.csv', 'noisy_feature2.csv', 'noisy_feature3.csv',
                                      'noisy_feature4.csv', 'noisy_feature5.csv']

TEST_SET_SIZE = 0.2
POLY_DEGREE = 3
OPACITY = 0.2
XLIM_L = 0
XLIM_R = 1100
YLIM_B = 0
YLIM_T = 1000
APPLY_LIMIT = True
TABLE = PrettyTable()
TABLEPOLY = PrettyTable()
SETCOUNT = 0
DATANOISE = True  # If Data Noise = True then Sample Noise specs are chosen
SIZE = 2
PREFIX = 'DNSet'
if not DATANOISE:
    PREFIX = 'FNSet'
ALPHA = 0.1  # Ridge Alpha
ALPHA1 = 1  # Ridge Alpha
ALPHA2 = 10  # Ridge Alpha
ALPHA3 = 10  # Ridge Alpha

# NP Seed
np.random.seed(19680801)


def init():
    createDirIdenpotent(GRAPHS_DIR)


def createDirIdenpotent(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)


def loadDataUsingPandas(inputDataFilepath):
    dataFrame = pd.read_csv(inputDataFilepath)
    return dataFrame


def plotPredictionData(y, pred, nameOfGraph):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.scatter(y, pred, alpha=OPACITY)
    ax.set(xlabel='Count of Cycles: $Y_i$',
           ylabel='Predicted Count of Cycles: $\hat{Y}_i$')
    graphPath = os.path.join(GRAPHS_DIR, nameOfGraph)
    logging.info('Graph saved in the path {}'.format(str(graphPath)))
    fig.savefig(graphPath)


def plotPredictionDataModified(XTest, yTest, yPred, nameOfGraph, limit=False, poly=False):
    logging.debug('Size of XTest = {}, yTest = {}'.format(
        str(len(XTest)), str(len(yTest))))
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.scatter(yTest, yPred, color='b', alpha=OPACITY, s=SIZE)
    if poly:
        ax.plot(np.unique(yTest), np.poly1d(np.polyfit(yTest, yPred, POLY_DEGREE))(
            np.unique(yTest)), '-', color='r', linewidth=3)
    else:
        ax.plot(np.unique(yTest), np.poly1d(np.polyfit(yTest, yPred, 1))
                (np.unique(yTest)), '-', color='r', linewidth=3)
    ax.set(xlabel='Count of Cycles: $Y_i$',
           ylabel='Predicted Count of Cycles: $\hat{Y}_i$')
    if limit:
        ax.set_xlim(XLIM_L, XLIM_R)
        ax.set_ylim(YLIM_B, YLIM_T)
    graphPath = os.path.join(GRAPHS_DIR, PREFIX+str(SETCOUNT)+'-'+nameOfGraph)
    logging.info('Graph saved in the path {}'.format(str(graphPath)))
    fig.savefig(graphPath)


def plotResidualGraph(yTrain, yTest, predTrain, predTest, nameOfGraph):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.scatter(predTrain, predTrain - yTrain, c='b',
               s=40, alpha=OPACITY, label='Train')
    ax.scatter(predTest, predTest - yTest, c='g', s=40, label='Test')
    ax.hlines(y=0, xmin=0, xmax=250)
    ax.set(xlabel='Residual', ylabel='Count of Cycles')
    graphPath = os.path.join(GRAPHS_DIR, nameOfGraph)
    logging.info('Graph saved in the path {}'.format(str(graphPath)))
    fig.savefig(graphPath)


def cleanAndCreateTestTrainSets(data):
    logging.debug('Data Frame Keys = {}'.format(str(data.keys())))
    logging.info('Dropping Targets')
    logging.debug('Dropping cnt')
    X = data.drop('cnt', axis=1)
    X = X.drop('atemp', axis=1)
    logging.info('Cleaning Features')
    logging.debug('Dropping instant')
    X = X.drop('instant', axis=1)

    try:
        logging.debug('Dropping Noise Label')
        X = X.drop('Noise', axis=1)
    except Exception as e:
        print('')  # Do Nothing

    logging.debug('Feature Frame Keys = {}'.format(str(X.keys())))

    logging.info('Splitting data for training and testing')
    XTrain, XTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, data.cnt,
                                                                            test_size=TEST_SET_SIZE,
                                                                            random_state=5)
    return X, data.cnt, XTrain, XTest, yTrain, yTest


def printMetics(yTest, predTest, scores, poly=False):
    logging.info('Calculating Metrics for set {}'.format(str(SETCOUNT)))
    global TABLE
    global TABLEPOLY
    print()
    setName = PREFIX+str(SETCOUNT)
    mse = np.sqrt(mean_squared_error(yTest, predTest)/len(yTest))
    if poly:
        if SETCOUNT == 0:
            TABLE.add_column('Metric (Poly)', ['Variance score (R Squared Score)',
                                               'Explained Variance Score', 'Mean Absolute Error',
                                               'Root Mean Square Error', 'CV - Accuracy'])
            TABLE.add_column(setName, ['%.2f' % r2_score(yTest, predTest),
                                       '%.2f' % explained_variance_score(
                                           yTest, predTest),
                                       '%.2f' % mean_absolute_error(
                                           yTest, predTest),
                                       '%.2f' % mse, '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)])
        else:
            TABLE.add_column(setName, ['%.2f' % r2_score(yTest, predTest),
                                       '%.2f' % explained_variance_score(
                                           yTest, predTest),
                                       '%.2f' % mean_absolute_error(
                                           yTest, predTest),
                                       '%.2f' % mse, '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)])
    else:
        if SETCOUNT == 0:
            TABLEPOLY.add_column('Metric', ['Variance score (R Squared Score)', 'Explained Variance Score',
                                            'Mean Absolute Error', 'Root Mean Square Error', 'CV - Accuracy'])
            TABLEPOLY.add_column(setName, ['%.2f' % r2_score(yTest, predTest), '%.2f' % explained_variance_score(yTest, predTest), '%.2f' % mean_absolute_error(yTest, predTest),
                                           '%.2f' % mse, '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)])
        else:
            TABLEPOLY.add_column(setName, ['%.2f' % r2_score(yTest, predTest), '%.2f' % explained_variance_score(yTest, predTest), '%.2f' % mean_absolute_error(yTest, predTest),
                                           '%.2f' % mse, '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)])
    print(TABLE)
    print(TABLEPOLY)
    print()
    printMetricsAsCSV(str(TABLE))
    printMetricsAsCSV(str(TABLEPOLY))


def printMetricsAsCSV(prettyTable):
    print()
    print('CSV Formatted Metrics')
    result = []
    for line in prettyTable.splitlines():
        splitdata = line.split("|")
        if len(splitdata) == 1:
            continue  # skip lines with no separators
        linedata = []
        for field in splitdata:
            field = field.strip()
            if field:
                linedata.append(field)
        result.append(linedata)

    for line in result:
        print(','.join(line))
    print()


def performLinearRegressionAndPlot(data):
    logging.info('Cleaning and Creating Train and Test Data')
    X, y, XTrain, XTest, yTrain, yTest = cleanAndCreateTestTrainSets(data)

    logging.info(f'Shape of Xtrain is {XTrain.shape} and yTrain is {yTrain.shape}')
    logging.info(f'Adding polynomial features of degree {POLY_DEGREE}')
    poly = PolynomialFeatures(degree=POLY_DEGREE)
    XTrain_ = poly.fit_transform(XTrain)
    XTest_ = poly.fit_transform(XTest)
    logging.info('Start Fitting')

    linearModelPoly = Ridge(alpha=ALPHA3)
    # linearModelPoly = RidgeCV(alphas=(ALPHA,ALPHA1,ALPHA2,ALPHA3))
    linearModel = RidgeCV(alphas=(ALPHA, ALPHA1, ALPHA2, ALPHA3))
    linearModelPoly.fit(XTrain_, yTrain)
    linearModel.fit(XTrain, yTrain)

    logging.info('Estimated Intercept Coefficient = {}'.format(
        str(linearModel.intercept_)))
    logging.info('Number of Coefficients = {}'.format(
        str(len(linearModel.coef_))))

    linearModelCross = make_pipeline(preprocessing.RobustScaler(), RidgeCV(
        alphas=(ALPHA, ALPHA1, ALPHA2, ALPHA3)))
    scores = cross_val_score(
        RidgeCV(alphas=(ALPHA, ALPHA1, ALPHA2, ALPHA3)), XTrain, yTrain, cv=10)

    logging.info('Prediction using Test Data')
    predTest_ = np.round(linearModelPoly.predict(XTest_))
    predTest = np.round(linearModel.predict(XTest))

    scoresPoly = cross_val_score(Ridge(alpha=ALPHA3), XTrain_, yTrain, cv=10)
    logging.info('Printing Linear Regression Metrics')
    printMetics(yTest, predTest, scores)

    logging.info('Printing Polynomial Regression Metrics')
    printMetics(yTest, predTest_, scoresPoly, poly=True)

    logging.info('Plotting the XY Test Poly Prediction data')
    plotPredictionDataModified(
        XTest_, yTest, predTest_, 'LRXYTestPolyPred.png', limit=APPLY_LIMIT, poly=True)

    logging.info('Plotting the XY Test Linear Prediction data')
    plotPredictionDataModified(
        XTest, yTest, predTest, 'LRXYTestLinearPred.png', limit=APPLY_LIMIT)


def calculateBaseline(data):
    logging.info('Cleaning and Creating Train and Test Data')
    X, y, XTrain, XTest, yTrain, yTest = cleanAndCreateTestTrainSets(data)
    # mean = np.round(np.mean(y))
    yBar = pd.DataFrame(np.round(np.mean(y)).astype(
        'int64'), index=np.arange(len(y), dtype="int64"), columns=['cnt'])
    # yBar.fill(mean)
    print(y)
    print(yBar)
    scores = cross_val_score(
        RidgeCV(alphas=(ALPHA, ALPHA1, ALPHA2, ALPHA3)), XTrain, yTrain, cv=10)
    printMetics(y, yBar, scores)


def calculateConfidenceInterval(data):
    logging.info('Cleaning and Creating Train and Test Data')
    X, y, XTrain, XTest, yTrain, yTest = cleanAndCreateTestTrainSets(data)
    mod = sm.OLS(y, X)
    res = mod.fit()
    print("Confidence Interval")
    print(res.conf_int(0.05))


def main():
    init()
    global SETCOUNT
    noisePercent = 0
    if DATANOISE:
        tempdf = None
        for i in range(11):
            filename = ''
            if noisePercent == 0:
                filename = CLEAN_DATA
            else:
                filename = DATA_NOISE_FILE_TEMPLATE.format(str(noisePercent))
            logging.info(f'Using the dataset {filename}')
            dataFrame = loadDataUsingPandas(
                os.path.join(DATA_NOISE_INPUT_DIR, filename))
            if noisePercent == 0:
                tempdf = dataFrame
            noisePercent = noisePercent + 10
            performLinearRegressionAndPlot(dataFrame)

            SETCOUNT = SETCOUNT + 1
        calculateBaseline(tempdf)
        # calculateConfidenceInterval(tempdf)
    else:
        for i in range(6):
            filename = ''
            if i == 0:
                filename = CLEAN_DATA
            else:
                filename = DATA_FEATURE_FILE_TEMPLATE.format(str(i))
            logging.info(f'Using the dataset {filename}')
            dataFrame = loadDataUsingPandas(
                os.path.join(FEATURE_NOISE_INPUT_DIR, filename))
            performLinearRegressionAndPlot(dataFrame)
            SETCOUNT = SETCOUNT + 1


if __name__ == '__main__':
    main()
