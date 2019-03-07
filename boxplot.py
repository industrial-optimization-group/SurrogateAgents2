import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

inputfolder = "./results/classification/"

training_costs = pd.read_csv(inputfolder + "training costs.csv")
prediction_costs = pd.read_csv(inputfolder + "prediction costs.csv")
plt.rcParams.update({"font.size": 17})

training_costs.boxplot()
plt.ylim([0, 1])
plt.xlabel("Classificaiton Algorithm")
plt.ylabel("Cost")
plt.title("Average cost on training set (DTLZ)")
plt.show()

pplot = prediction_costs.boxplot()
plt.ylim([0, 1])
plt.xlabel("Classificaiton Algorithm")
plt.ylabel("Cost")
plt.title("Average cost on prediction set (WFG)")
i = 1
for key, value in prediction_costs.items():
    y = value
    x = np.random.normal(i, 0.04, size=len(y))
    pplot.plot(x, y, "r.", alpha=0.5)
    i = i + 1
plt.show()
