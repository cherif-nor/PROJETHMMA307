"""
@authors: Cherif AMGHAR
In short: Usage of linear mixed models on the dataset ESTUARIES.
"""

################################
# Packages needded
#################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import seaborn as sns

####################################
# Download dataset and delet column
####################################
Estua = pd.read_csv("C:/Users/AMGHAR MED CHERIF/Desktop/python project/PROJETHMMA307/data/Estuaries.csv")#importation des données
print(type(Estua))#Format des données
print(Estua.shape)#Taille du dataframe
print(Estua)#Affichage des données (il y a une colonne en trop)

#####################################
#Visualization of data:
#####################################
Estuaries = Estua.drop(["Unnamed: 0"] , axis = 1)#Supression de la 1ère colonne (inutile)
print(Estuaries)#données propres
print(Estuaries.shape)

#####################################
#modèle linièare mixte 1
#####################################

md=smf.mixedlm("Total ~ Modification", Estuaries, groups= Estuaries["Estuary","site"])
res=md.fit().resid
fig = sm.qqplot(res)
plt.show()

####################################
#bootstrap parametrique:
####################################
# configure bootstrap
n_iterations = 1000
n_size = int(len(Estuaries) * 0.50)
# run bootstrap
stats = list()
for i in range(n_iterations):
	# prepare train and test sets
	train = resample(values, n_samples=n_size)
	test = numpy.array([x for x in values if x.tolist() not in train.tolist()])
	# fit model
	model = DecisionTreeClassifier()
	model.fit(train[:,:-1], train[:,-1])
	# evaluate model
	predictions = model.predict(test[:,:-1])
	score = accuracy_score(test[:,-1], predictions)
	print(score)
	stats.append(score)
# plot scores
pyplot.hist(stats)
pyplot.show()
# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, numpy.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, numpy.percentile(stats, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

#####################################
#modèle linièare mixte 2
#####################################
md=smf.mixedlm("Total ~ Modification", Estuaries, groups= Estuaries["Estuary","SiteWithin"])
res=md.fit().summary()
print(res)



