"""Create simple surrogate models from data in the folder dtlzdatasets."""

import pickle
import warnings
from os import getcwd, listdir

import numpy as np
import pandas as pd
from sklearn import ensemble, svm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split as tts
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

folder = getcwd() + "/wfgdatasets/"
files = listdir(folder)
outputfolder = folder + "modellingresults/"
numfiles = len(files)
i = 0

R2results = pd.DataFrame(
    np.zeros((numfiles, 5)), columns=["file", "SVM", "NN", "Ada", "GPR"]
)
models = pd.DataFrame(
    np.full((numfiles, 5), np.nan), columns=["file", "SVM", "NN", "Ada", "GPR"]
)
for file in files:
    print("File", i + 1, "of", numfiles)
    fullfilename = folder + file
    data = pickle.load(open(fullfilename, "rb"))
    inputs = data[data.columns[0:-2]]
    f1 = data["f1"]
    f2 = data["f2"]
    R2results["file"][i] = file
    models["file"][i] = file
    inputs_train, inputs_test, f2_train, f2_test = tts(inputs, f2)
    # SVM
    max_score = 0
    best_model = None
    for j in range(3):
        clf = svm.SVR()
        clf.fit(inputs_train, f2_train)
        pred = clf.predict(inputs_test)
        score = r2_score(f2_test, pred)
        if score > max_score:
            max_score = score
            best_model = clf
    R2results["SVM"][i] = max_score
    models["SVM"][i] = best_model
    # NN
    max_score = 0
    best_model = None
    for j in range(3):
        clf = MLPRegressor()
        clf.fit(inputs_train, f2_train)
        pred = clf.predict(inputs_test)
        score = r2_score(f2_test, pred)
        if score > max_score:
            max_score = score
            best_model = clf
    R2results["NN"][i] = max_score
    models["NN"][i] = best_model
    # ADABOOST
    max_score = 0
    best_model = None
    for j in range(3):
        clf = ensemble.AdaBoostRegressor()
        clf.fit(inputs_train, f2_train)
        pred = clf.predict(inputs_test)
        score = r2_score(f2_test, pred)
        if score > max_score:
            max_score = score
            best_model = clf
    R2results["Ada"][i] = max_score
    models["Ada"][i] = best_model
    # GPR
    max_score = 0
    best_model = None
    for j in range(3):
        clf = GaussianProcessRegressor()
        clf.fit(inputs_train, f2_train)
        pred = clf.predict(inputs_test)
        score = r2_score(f2_test, pred)
        if score > max_score:
            max_score = score
            best_model = clf
    R2results["GPR"][i] = max_score
    models["GPR"][i] = best_model
    i = i + 1
R2results.to_csv("R2results.csv")
models.to_csv("models.csv")
