"""Create simple surrogate models from data in the folder dtlzdatasets."""

import pickle
import warnings
from os import getcwd, listdir

import numpy as np
import pandas as pd
from sklearn import ensemble, svm
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split as tts
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")
folderlist = ["/dtlzdatasetscsv/", "/zdtdatasetscsv/"]
for fldr in folderlist:
    folder = getcwd() + fldr
    files = listdir(folder)
    numfiles = len(files)
    i = 0
    algorithms = ["SVM", "NN", "Ada", "GPR", "SGD", "KNR", "DTR", "RFR", "ExTR", "GBR"]
    model = {
        "SVM": svm.SVR,
        "NN": MLPRegressor,
        "Ada": ensemble.AdaBoostRegressor,
        "GPR": GaussianProcessRegressor,
        "SGD": SGD,
        "KNR": KNR,
        "DTR": DTR,
        "RFR": ensemble.RandomForestRegressor,
        "ExTR": ensemble.ExtraTreesRegressor,
        "GBR": ensemble.GradientBoostingRegressor,
    }

    R2results = pd.DataFrame(
        np.zeros((numfiles, len(algorithms) + 1)), columns=["file"] + algorithms
    )
    models = pd.DataFrame(
        np.full((numfiles, len(algorithms) + 1), np.nan), columns=["file"] + algorithms
    )
    for file in files:
        print("File", i + 1, "of", numfiles)
        fullfilename = folder + file
        data = pd.read_csv(fullfilename)
        inputs = data[data.columns[0:-2]]
        f1 = data["f1"]
        f2 = data["f2"]
        R2results["file"][i] = file
        inputs_train, inputs_test, f2_train, f2_test = tts(inputs, f2)
        for algo in algorithms:
            max_score = 0
            best_model = None
            for j in range(5):
                clf = model[algo]()
                clf.fit(inputs_train, f2_train)
                pred = clf.predict(inputs_test)
                score = r2_score(f2_test, pred)
                if score > max_score:
                    max_score = score
                    best_model = clf
            R2results[algo][i] = max_score
        i = i + 1
    R2results.to_csv(folder + "R2results.csv", index=False)
