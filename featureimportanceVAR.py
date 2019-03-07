import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2
import seaborn as sns


# Build a classification task using 3 informative features
rootfolder = "./results/featuresR2/"
outputfolder = "./results/classification/"
data = pd.read_csv(rootfolder + "dtlz.csv")
inputs_train = data[data.keys()[1:14]]
outputs_train = data[data.keys()[14:]]
target_train = outputs_train.idxmax(axis=1)
inputs_new = (inputs_train - inputs_train.min()) / (inputs_train.max()- inputs_train.min())
selector = SelectKBest(chi2, k=1)
selector.fit(inputs_new, target_train)
total = np.asarray(list(map(int, selector.get_support())))
for i in range(2, 14):
    selector = SelectKBest(chi2, k=i)
    selector.fit(inputs_new, target_train)
    total = total + np.asarray(list(map(int, selector.get_support())))
print(pd.DataFrame(total))
#plt.hist(total)
#plt.show()
"""sns.pairplot(inputs_train)
plt.rcParams['figure.figsize']=(50,50)
plt.ion()
plt.show()
fig = plt.gcf()
#fig.set_size_inches(18.5, 10.5)
fig.savefig('feature Variances.png', dpi=200)

def plot_unity(xdata, ydata, **kwargs):
    mn = min(xdata.min(), ydata.min())
    mx = max(xdata.max(), ydata.max())
    points = np.linspace(mn, mx, 100)
    plt.gca().plot(points, points, color='k', marker=None,
            linestyle='--', linewidth=1.0)"""
