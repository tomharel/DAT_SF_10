# Classifications Models
## Logistic Regression
## Naive Bayes
## KNN
## Random Forest

# General Imports
from sys import argv
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Local Imports
from cv import cross_validate

def scoreModels(features, target, folds=10):
    "Calcs crovs-validation scores for multiple algorithms"
    import pdb
    #pdb.set_trace()
    models = []
    models.append(RandomForestClassifier(random_state=0).fit)
    models.append(LogisticRegression(C=1.0).fit)
    models.append(KNeighborsClassifier(3).fit)
    models.append(SVC(C=1.0).fit)
    models.append(GaussianNB().fit)

    for alg in models:
        print alg
        print cross_validate(features, target, alg, folds)

