import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

inputfolder = "./results/classification/"

training_costs = pd.read_csv(inputfolder + "traincost_on_bench.csv")
prediction_costs = pd.read_csv(inputfolder + "testcost_on_bench.csv")
plt.rcParams.update({"font.size": 12})

tplot = training_costs.boxplot(showfliers=False)
plt.ylim([0, 0.4])
plt.xlabel("Classification Algorithm")
plt.ylabel("Cost")
#plt.title("Average cost on training Benchmark set")
i = 1
for key, value in training_costs.items():
    y = value
    x = np.random.normal(i, 0.04, size=len(y))
    tplot.plot(x, y, "r.", alpha=0.5, markersize=12)
    i = i + 1
plt.show()

pplot = prediction_costs.boxplot(showfliers=False)
plt.ylim([0, 0.4])
plt.xlabel("Classification Algorithm")
plt.ylabel("Cost")
#plt.title("Average cost on prediction Benchmark set")
i = 1
for key, value in prediction_costs.items():
    y = value
    x = np.random.normal(i, 0.04, size=len(y))
    pplot.plot(x, y, "r.", alpha=0.5, markersize=12)
    i = i + 1
plt.show()
