"""Conduct basic classification independent of voting."""
import pandas as pd
from sklearn.ensemble import BaggingClassifier as BC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.neighbors.nearest_centroid import NearestCentroid as NC
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import ExtraTreeClassifier as ExTC1
from sklearn.ensemble import ExtraTreesClassifier as ExTC2
from sklearn.neural_network import MLPClassifier as NNC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split as tts
import pickle as pk
from os import listdir
import matplotlib.pyplot as plt
from copy import copy


def train_classifiers():
    """
    Train classifiers
    
    DLTZ, WFG and ZDT data used. Models dumped as pickle files to ./results/bestmodels.
    
    """

    classifiers = {
        "BC": BC,
        "SVC": SVC,
        "KNC": KNC,
        "NC": NC,
        "GPC": GPC,
        "DTC": DTC,
        "NNC": NNC,
        "ExTC1": ExTC1,
        "ExTC2": ExTC2,
    }
    training_costs = pd.DataFrame(columns=classifiers.keys())
    prediction_costs = pd.DataFrame(columns=classifiers.keys())

    rootfolder = "./results/featuresR2/"
    outputfolder = "./results/bestmodels/"
    data = pd.read_csv(rootfolder + "bigdatafortraining.csv")
    inputs = data[data.keys()[1:14]]
    outputs = data[data.keys()[14:]]
    inputs_train, inputs_test, outputs_train, outputs_test = tts(inputs, outputs)
    target_train = outputs_train.idxmax(axis=1)
    target_test = outputs_test.idxmax(axis=1)

    for key, value in classifiers.items():
        model = {"model": [], "cost_train": 1, "cost_test": 1, "name": key}
        print(key, "\n")
        for i in range(50):
            clf = value()
            clf.fit(inputs_train, target_train)
            predictions = clf.predict(inputs_train)
            cost_mat = outputs_train
            cost_mat = -cost_mat.sub(cost_mat.max(axis=1), axis=0)
            num_samples, num_classes = cost_mat.shape
            cost = 0
            lenp = len(predictions)
            for index in range(num_samples):
                cost += cost_mat.iloc[index][predictions[index]]
            cost_train = cost / lenp
            predictions = clf.predict(inputs_test)
            cost_mat = outputs_test
            cost_mat = -cost_mat.sub(cost_mat.max(axis=1), axis=0)
            num_samples, num_classes = cost_mat.shape
            cost = 0
            lenp = len(predictions)
            for index in range(num_samples):
                cost += cost_mat.iloc[index][predictions[index]]
            cost_test = cost / lenp
            if cost_test < model["cost_test"]:
                model["model"] = clf
                model["cost_test"] = cost_test
                model["cost_train"] = cost_train
        pk.dump(model, open(outputfolder + key + "best.p", "wb"))


def test_classifiers():
    plt.ion()
    rootfolder = "./results/featuresR2/"
    data = pd.read_csv(rootfolder + "engineering.csv")
    root = "./results/bestmodels/"
    files = listdir(root)
    models = {}
    for file in files:
        models[file[0:-6]] = pk.load(open(root + file, "rb"))
    data_features = data[data.keys()[1:14]]
    R2all = data[data.keys()[14:]]
    for key, value in models.items():
        model = value["model"]
        algo_predicted = model.predict(data_features)
        R2predicted = [
            R2all.iloc[index][algo_predicted[index]]
            for index in range(len(algo_predicted))
        ]
        newR2 = copy(R2all)
        newR2["Predicted"] = R2predicted
        newR2['files'] = data['files']
        newR2.to_csv(
            "./results/predictionsonengineering/" + key + "prediction.csv", index=False
        )
        # newR2.plot.line()
        # plt.show()
        # plt.title(key)
    # plt.ioff()
    # plt.show()


if __name__ == "__main__":
    test_classifiers()

