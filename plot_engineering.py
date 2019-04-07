import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# LINE PLOT CODE

problems = [
    "kursawe_f1",
    "kursawe_f2",
    "four_bar_f1",
    "four_bar_f2",
    "gear_train_f1",
    "pressure_f1",
    "pressure_f2",
    "speed_reducer_f1",
    "speed_reducer_f2",
    "speed_reducer_f3",
    "welded_beam_f1",
    'unconstrained_f_f1',
]

num_samples = ['50.', '100.', '150.', '200.', '400.', '700.', '1000.']
classifier = 'ExTC2'
rootfolder = "./results/predictionsonengineering/"
data = pd.read_csv(rootfolder + classifier + "prediction.csv")
model = [ "SVM", "NN", "Ada", "GPR", "SGD", "KNR", "DTR", "RFR", "ExTR", "GBR", 'Predicted']
colours = sns.color_palette("husl", 10) + ['Black']
testdata = data[['files']+model]
testdata= testdata.set_index('files')
testdata = testdata.unstack().reset_index()
testdata['problem'] = [0]*testdata['files']
for id in testdata.index:
    testdata['problem'][id] = [problem for problem in problems if problem in testdata['files'][id]][0]
testdata = testdata.rename(columns={'level_0':'Modelling Algorithm', 0:'R2'}) 
sns.lineplot(x='problem', y='R2', data=testdata, hue='Modelling Algorithm', palette=colours)
plt.legend(loc='lower left')
plt.title(classifier)
plt.show()
#plt.savefig(rootfolder + classifier + '.png')



"""testdata['num_samples'] = [0]*testdata['files']
for id in testdata.index:
    testdata['num_samples'][id] = [sample[0:-1] for sample in num_samples if sample in testdata['files'][id]][0]
testdata.loc[:, 'num_samples'] = testdata.num_samples.astype(float)
sns.lineplot(x='problem', y='R2', data=testdata, hue='num_samples')
plt.show()"""


# BOXPLOT CODE
"""problems = [
    "kursawe_f1",
    "kursawe_f2",
    "four_bar_f1",
    "four_bar_f2",
    "gear_train_f1",
    "pressure_f1",
    "pressure_f2",
    "speed_reducer_f1",
    "speed_reducer_f2",
    "speed_reducer_f3",
    "welded_beam_f1",
    "unconstrained_f_f1",
]

num_samples = ["50.", "100.", "150.", "200.", "400.", "700.", "1000."]
classifier = "SVC"
rootfolder = "./results/predictionsonengineering/"
data = pd.read_csv(rootfolder + classifier + "prediction.csv")
model = ["SVM", "NN", "Ada", "GPR", "SGD", "KNR", "DTR", "RFR", "ExTR", "GBR"]
maxR2 = data[model].max(axis=1)
classifiers = ["BC", "SVC", "KNC", "NC", "GPC", "DTC", "NNC", "ExTC1", "ExTC2"]
cost = pd.DataFrame(np.zeros((len(maxR2), len(classifiers))), columns=classifiers)
for classes in classifiers:
    data = pd.read_csv(rootfolder + classes + "prediction.csv")
    cost[classes] = maxR2 - data['Predicted']
print(cost)

plt.rcParams.update({"font.size": 17})
pplot = cost.boxplot()
#plt.ylim([0, 0.4])
plt.xlabel("Classificaiton Algorithm")
plt.ylabel("Cost")
plt.title("Cost on Engineering set")
i = 1
for key, value in cost.items():
    y = value
    x = np.random.normal(i, 0.04, size=len(y))
    pplot.plot(x, y, "r.", alpha=0.5)
    i = i + 1
plt.show()"""


