# -*- coding: utf-8 -*-
"""
Compute FCI (Financial Conditions Indexes) with PCA and PLS
Romain Lafarguette, International Monetary Fund, rlafarguette@imf.org
Time-stamp: "2020-01-19 12:14:03 RLafarguette"
"""

###############################################################################
#%% Modules import
###############################################################################
import pandas as pd                                      # Dataframes
import numpy as np                                       # Numeric methods
import statsmodels as sm                                 # Statistical models
import statsmodels.formula.api as smf                    # statsmods formulas

# Functional import
from datetime import datetime as date                    # Dates management
from sklearn.preprocessing import scale                  # Zscore
from sklearn.decomposition import PCA                    # PCA
from sklearn.cross_decomposition import PLSRegression    # PLS

# Graphics
import matplotlib
matplotlib.use('TkAgg') # On IMF computers, choose this backend
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

###############################################################################
#%% Mock data loading and cleaning
###############################################################################
# US macro quarterly data dataset
# cf. https://www.statsmodels.org/0.6.1/datasets/generated/macrodata.html
df = sm.datasets.macrodata.load_pandas().data

# Time index 
df['date'] = df[['year', 'quarter']].apply(lambda row:
                                           date(int(row[0]), int(3*row[1]), 1),
                                           axis=1) + pd.offsets.QuarterEnd()

df = df.set_index(df['date']) # Index

# New variables
df['gdp'] = df['realgdp'].rolling(4).sum() # Should use nominal GDP instead...
df['gdp_yoy'] = df['gdp'].pct_change(4)
df['m1_gdp'] = df['m1']/df['gdp']
df['tbilrate_yoy'] = df['tbilrate'].diff(4) # Better to use diff here

# Forward variable (one year)
df['gdp_yoy_fwd_1y'] = df['gdp_yoy'].shift(-4)

###############################################################################
#%% Parameters
###############################################################################
# List of variables to include (only a subset)
fci_var_l = ['m1_gdp', 'tbilrate_yoy', 'infl', 'realint']

var_label_l = ['M1/GDP', 'TBills rate yoy',
               'Inflation rate', 'Real interest rate']

###############################################################################
#%% PCA approach
###############################################################################
# Remove missing values
dfn = df[fci_var_l].dropna().copy() 

# Fit the PCA on the data
pca = PCA(n_components=1) # Initialize the PCA with one component
X = scale(dfn[fci_var_l]) # Need to scale the variables before running a PCA
pca_fit = pca.fit(X) # Fit the PCA on the series

# Package the first component in a pandas dataframe
pca_transformation = pca.fit(X).transform(X).ravel()
dpca = pd.DataFrame(pca_transformation, index=dfn.index, columns=['FCI_PCA'])

# For information, store loadings, normalized loadings and variance explained
pca_loadings = pca_fit.components_.T.ravel()
norm_loadings = pca_loadings * np.sqrt(pca_fit.explained_variance_)
dpca_l = pd.DataFrame(pca_loadings, index=var_label_l,
                      columns=['PCA_loadings'])
dpca_l['normalized_PCA_loadings'] = norm_loadings
dpca_l['explained_variance'] = float(pca_fit.explained_variance_ratio_)

# Plot the FCI
dpca.plot(title='Financial Conditions Index for the United States - PCA')
plt.hlines(y=0, xmin=dpca.index.min(), xmax=dpca.index.max())
plt.xlabel('')
plt.show()

# Plot the loadings
dpca_l['PCA_loadings'].sort_values().plot.barh(title='PCA Loadings')
plt.vlines(x=0, ymin=-1, ymax=len(dpca_l))
plt.xlabel('')
plt.show()

# Plot the explained variance
dpca_l.iloc[[0]]['explained_variance'].plot.bar(
    title='Explained variance of the first component')
plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False) 
plt.show()

###############################################################################
#%% PLS approach
###############################################################################
# Remove missing values on both supervisor and regressors
dfnn = df[fci_var_l + ['gdp_yoy_fwd_1y']].dropna().copy() 

# PLS model specification 
pls = PLSRegression(n_components=1, scale=True) # Always scale 

# Model fit: Pay attention that PLS specification put first the X, then the Y
# Here, the supervisor is real GDP growth, one year ahead
pls_fit = pls.fit(dfnn[fci_var_l], dfnn['gdp_yoy_fwd_1y'])

# Return the first component of the fit
fci_pls = pls_fit.fit_transform(dfnn[fci_var_l],
                                dfnn['gdp_yoy_fwd_1y'])[0]

# Package the FCI PLS into a dataframe
dpls = pd.DataFrame(fci_pls, index=dfnn.index, columns=['FCI_PLS'])

# Package the loadings as well
dpls_l = pd.DataFrame(pls_fit.x_loadings_,
                      index=var_label_l, columns=['PLS_loadings'])

# NB: If you need variables influence in the projection, look at my
# data_reduction.py wrapper on my github repo romain_utils

# Plot the FCI
dpls.plot(title='Financial Conditions Index for the United States - PLS')
plt.hlines(y=0, xmin=dpls.index.min(), xmax=dpls.index.max())
plt.xlabel('')
plt.show()


# Plot the loadings
dpls_l['PLS_loadings'].sort_values().plot.barh(title='PLS Loadings')
plt.vlines(x=0, ymin=-1, ymax=len(dpls_l))
plt.xlabel('')
plt.show()

###############################################################################
#%% Comparison FCI : PCA and PLS
###############################################################################
dcomp = pd.merge(dpca, dpls, left_index=True, right_index=True, how='inner')

# To be interepreted the same way, need to flip the PLS, so that up means tight
dcomp['FCI_PLS_inv'] = (-1)*dcomp['FCI_PLS'].copy()

dcomp[['FCI_PCA', 'FCI_PLS_inv']].plot(
    title='FCIs comparison between unsupervised and supervised learning')
plt.hlines(y=0, xmin=dcomp.index.min(), xmax=dcomp.index.max())
plt.xlabel('')
plt.show()


