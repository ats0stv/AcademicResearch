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
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import seaborn as sns
import os as os
from sklearn.metrics import log_loss, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

print(os.getcwd())
dataset = pd.read_csv('../../dataset/movie_metadata.csv')

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
dataset.head()
# plotting heat map to visualize correlation:
plt.figure(figsize=(18, 8), dpi=100,)
plt.subplots(figsize=(18, 8))
sns.heatmap(data=dataset.corr(), square=True, vmax=0.8, annot=True)


lb_director = LabelEncoder()
lb_country = LabelEncoder()
lb_language = LabelEncoder()
lb_content_rating = LabelEncoder()
lb_verdict = LabelEncoder()
dataset["director_code"] = lb_director.fit_transform(dataset["director_name"])
dataset["country_code"] = lb_country.fit_transform(dataset["country"])
dataset["lang_code"] = lb_language.fit_transform(dataset["language"])
dataset["content_rating_code"] = lb_content_rating.fit_transform(
    dataset["content_rating"])

dataset['verdict'] = pd.cut(dataset['imdb_score'], bins=[0, 7, 8, 8.5, 9, 10], labels=[
                            "poor", "average", "good", "very good", "excellent"], right=False)
dataset['verdict'] = lb_verdict.fit_transform(dataset['verdict'])

dataset.info()
dataset.to_csv('processed_df.csv')
# Setting predictors and target variables
X = dataset.iloc[:, np.r_[1:7, 11, 13:17]].values
X
y = dataset.iloc[:, 18].values
y
enc = OneHotEncoder(categorical_features=[0])
X = enc.fit_transform(X).toarray()
X
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
C_param_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
for i in C_param_range:
    logistic = LogisticRegression(C=i)
    logistic.fit(X_train, y_train)
    y_pred = logistic.predict(X_test)

    logistic.predict_proba(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print('The confusion matrxi: ', conf_matrix)

    print("Accuracy", accuracy_score(y_test, y_pred)*100)

logistic = LogisticRegression(C=10)
scores = cross_val_score(logistic, X, y, cv=5)
scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
