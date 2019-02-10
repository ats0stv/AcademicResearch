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

"""     Script to plot various metrics of  Bike Dataset 
"""


import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import datestr2num
from mpl_toolkits.mplot3d import axes3d, Axes3D

# Log Configuration
logging.basicConfig(level=logging.DEBUG)

# Global Var
GRAPHS_DIR = './NewWineDataGraphs'
INPUT_DIR = '/Users/arun/git/ML1819-task-1--team-20/Effect_Of_Noise/dataset_wine_noise'
RAW_DATA_PLOT_FILEPATH = os.path.join(GRAPHS_DIR, 'rawDataPlotgraph.png')
INPUT_DATA_FILEPATH = os.path.join(INPUT_DIR, 'noisy_data_100_1.csv')
DATA = None
XMAX = 300
OPACITY = 0.3
SIZE = 2
# NP Seed
np.random.seed(19680801)


class SecondsSinceFirst:
    def __init__(self, default=float('nan')):
        self.first = None
        self.default = default

    def __call__(self, value):
        value = value.decode('ascii', 'ignore').strip()
        if not value:  # no value
            return self.default
        try:  # specify input format here
            current = datetime.strptime(value, "%Y-%m-%d")
        except ValueError:  # can't parse the value
            return self.default
        else:
            if self.first is not None:
                return (current - self.first).total_seconds()
            else:
                self.first = current
                return 0.0


def loadDataUsingPandas(inputDataFilepath):
    dataFrame = pd.read_csv(inputDataFilepath)
    return dataFrame


def createDirIdenpotent(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)


def plotLineGraph(X=None, yArray=None, outputFileName='line.png', xAxes=[], yAxes=[], xMax=0, transpose=True):
    logging.debug('Plotting line graph - begin')
    countOfSubPlots = 0
    if X == None or y == None:
        X = np.arange(0.0, 2.0, 0.01)
        y = 1 + np.sin(2 * np.pi * X)
        yArray = [y]

    if yArray != None:
        countOfSubPlots = len(yArray)
    fig, ax = plt.subplots(countOfSubPlots, 1, figsize=(12, 8))

    if transpose:
        ax.plot(yArray[0], X)
    else:
        ax.plot(X, yArray[0])
    if len(xAxes) > 0:
        ax.set(xlabel=xAxes[0])
    if len(yAxes) > 0:
        ax.set(ylabel=yAxes[0])
    if xMax > 0:
        ax.set_xlim(0, xMax)
    ax.grid()

    fig.savefig(os.path.join(GRAPHS_DIR, outputFileName))
    logging.debug('Plotting line graph - end')


def plotScatter(X, yArray, outputFileName='scatter.png', xAxes=[], yAxes=[], labels=[],
                colors=[], title='', xMax=0, enableLabels=False, transpose=True, noise=None):
    xMax = np.max(yArray) + int(np.max(yArray)*0.05)
    # xMax = yMax
    logging.debug('Plotting scatter graph - begin')
    countOfSubPlots = len(yArray)
    fig, ax = plt.subplots(countOfSubPlots, 1, figsize=(12, 8))
    logging.debug('yArray = {}, yAxes = {}, labels = {}, colors = {}'.format(str(len(yArray)), str(len(yAxes)),
                                                                             str(len(labels)), str(len(colors))))
    col = 'b'
    if noise.any():
        col = np.where(noise == 0, 'b', 'g')
    if countOfSubPlots == 1:
        i = 0
        if transpose:
            if enableLabels:
                ax.scatter(yArray[i], X, color=col,
                           label=labels[i], s=SIZE, alpha=OPACITY)
            else:
                ax.scatter(yArray[i], X, color=col, s=SIZE, alpha=OPACITY)
        else:
            if enableLabels:
                ax.scatter(X, yArray[i], color=col,
                           label=labels[i], s=SIZE, alpha=OPACITY)
            else:
                ax.scatter(X, yArray[i], color=col, s=SIZE, alpha=OPACITY)
        if xMax > 0:
            ax.set_xlim(0, xMax)
        if len(xAxes) > 0:
            ax.set_xlabel(xAxes[i])
        if len(yAxes) > 0:
            ax.set_ylabel(yAxes[i])
        ax.legend(loc=2)
    else:
        for i in range(0, countOfSubPlots):
            if transpose:
                if enableLabels:
                    ax[i].scatter(yArray[i], X, color=col,
                                  label=labels[i], s=SIZE, alpha=OPACITY)
                else:
                    ax[i].scatter(yArray[i], X, color=col,
                                  s=SIZE, alpha=OPACITY)
            else:
                if enableLabels:
                    ax[i].scatter(X, yArray[i], color=col,
                                  label=labels[i], s=SIZE, alpha=OPACITY)
                else:
                    ax[i].scatter(X, yArray[i], color=col,
                                  s=SIZE, alpha=OPACITY)
            if xMax > 0:
                ax[i].set_xlim(0, xMax)
            if len(xAxes) > 0:
                ax[i].set_xlabel(xAxes[i])
            if len(yAxes) > 0:
                ax[i].set_ylabel(yAxes[i])
            ax[i].legend(loc=2)

    fig.savefig(os.path.join(GRAPHS_DIR, outputFileName))
    logging.debug('Plotting scatter graph - end')


def init():
    createDirIdenpotent(GRAPHS_DIR)


def loadBikesDataAndPlot(inputDataFilepath):
    global DATA
    data = np.loadtxt(inputDataFilepath, skiprows=1, delimiter=',')
    DATA = data

    noise = None
    try:
        noise = data[:, 13]
        print(noise)
    except Exception as e:
        print('')  # Do nothing

    # Month Vs Total Count of Cycles
    logging.info('Plotting scatter graph for total.sulfur.dioxide vs quality')
    X = data[:, 11]
    y = data[:, 6]
    yArray = []
    yArray.append(y)
    yAxes = ['quality']
    colors = ['r']
    xAxes = ['total.sulphur.dioxide']
    labels = xAxes
    plotScatter(X, yArray, outputFileName='tsdVsQuality100.png', yAxes=yAxes,
                xAxes=xAxes, title='Scatter Plot Between Total Sulfur Dioxide and Quality',
                labels=labels, colors=colors, noise=noise)


def main():
    init()
    loadBikesDataAndPlot(INPUT_DATA_FILEPATH)
    # plotLineGraph()


if __name__ == '__main__':
    main()
