import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Build a classification task using 3 informative features
rootfolder = "./results/featuresR2/"
outputfolder = "./results/classification/"
data = pd.read_csv(rootfolder + "dtlz.csv")
inputs_train = data[data.keys()[1:14]]
outputs_train = data[data.keys()[14:]]
target_train = outputs_train.idxmax(axis=1)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=2500,
                              random_state=0)

forest.fit(inputs_train, target_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(inputs_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(inputs_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(inputs_train.shape[1]), indices)
plt.xlim([-1, inputs_train.shape[1]])
plt.show()