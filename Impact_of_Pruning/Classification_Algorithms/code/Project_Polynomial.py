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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns

dataset = pd.read_csv('../../movie_metadata.csv')

dataset.info()

# droping the columns which is not necessary
dataset.drop(["color", "actor_1_facebook_likes", "actor_3_facebook_likes",
              "genres", "actor_1_name", "actor_2_name", "movie_title", "actor_3_name", "facenumber_in_poster",
              "plot_keywords", "title_year", "movie_imdb_link", "actor_2_facebook_likes", "aspect_ratio"], axis=1, inplace=True)

dataset.isna().sum()
dataset.info()

# Data imputation
dataset.replace({"country": np.NaN,
                 "director_name": np.NaN,
                 "language": np.NaN,
                 "content_rating": np.NaN}, value="Missing", inplace=True)

dataset['duration'] = dataset['duration'].fillna(
    value=dataset['duration'].mean())
dataset['num_user_for_reviews'] = dataset['num_user_for_reviews'].fillna(
    value=dataset['num_user_for_reviews'].mean())
dataset['budget'] = dataset['budget'].fillna(value=dataset['budget'].mean())
dataset['gross'] = dataset['gross'].fillna(value=dataset['gross'].mean())
dataset['director_facebook_likes'] = dataset['director_facebook_likes'].fillna(
    value=0)
dataset['num_critic_for_reviews'] = dataset['num_critic_for_reviews'].fillna(
    value=0)

# remove duplicates
dataset.drop_duplicates(subset=None, keep='first', inplace=True)

dataset.isna().sum()
dataset.info()

# plotting heat map to visualize correlation:
plt.figure(figsize=(18, 8), dpi=100,)
plt.subplots(figsize=(18, 8))
sns.heatmap(data=dataset.corr(), square=True, vmax=0.8, annot=True)

# Setting predictors and target variables
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
