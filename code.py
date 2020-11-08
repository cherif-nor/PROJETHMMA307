"""
@authors: Cherif AMGHAR
In short: Usage of linear mixed models on the dataset ESTUARIES.
"""

################################
# Packages needded
#################################
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_table('/Users/AMGHAR MED CHERIF/Desktop/python project/PROJETHMMA307/Estuaries.csv', sep='\s+', header=0)
print(data)
