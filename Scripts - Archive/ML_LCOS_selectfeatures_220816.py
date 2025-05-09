#!/usr/bin/env python
# coding: utf-8

# ## Pre-Amble
# 
# The following code has been used to devise a machine learning algorithm to predict green steel production costs given renewable energy statistics of a specific location. This relates tothe following research article:
# 
# **Title:**
# Global green steel opportunities surrounding high quality renewable energy and iron ore deposits
# 
# **Authors:**
# Alexandra Devlin, Jannik Kossen, Haulwen Goldie-Jones, Aidong Yang
# 
# **Lead code developer:**
# Jannik Kossen
# 
# **Code support:**
# Alexandra Devlin
# 
# **Insitution:**
# University of Oxford, Parks Road, Oxford, OX1 3PJ, United Kingdom
# 
# **Corresponding author email:** aidong.yang@eng.ox.ac.uk
# 
# 
# ## Methodology
# 
# We fit a gradient-boosted regression model to directly predict (or amortize) the outcome of the compute-expensive simulation.
# While the simulation takes more than an hour to make a prediction per input location, the model can predict for hundreds of locations in less than a second.
# We rely on the gradient-boosting implementation of the popular scikit-learn toolkit (https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html).
# 
# We take special precautions to evaluate the model and fit its hyper-parameters.
# The simulation dataset is a unique set of 17 regions (plus the addition of 1 more region due to its low latitude, New Zealand).
# For each region, multiple datapoints are available.
# If we do not respect these regions when splitting into train and test regions we observe overly optimistic scores: if the model has seen a region before (even if this wase.g. for a different input year) this really helps the model predict.
# This would not be a problem, if our training set would contain _all_ possible regions.
# However, a user may query a _new_ region the model has not seen.
# In order to gain a realistic picture of our models capabilities, we want to test the model on 'new' regions that were not part of the training set.
# On such a new region, the model will then worse than on a region that was already part of the training set.
# Generalising to new regions is a big motivation for building this model in the first place, and so, when evualiting the model, we need to check if wecan deliver on the promise of saving 'simulation time'.
# 
# 
# However, some regions are rather 'extreme': they are included in model training precisely because they are unique (making sure the model will predict well on them when queried by a user).
# We do not want to test on these regions either (excluding them from training), because this will give an overly pessimistic score: it is unrealistic the user will query such an extreme region that the model has not seen during training (precisely because we take care to include all extreme regions during training).
# 
# Hence, we therefore define a set of 'safe testing regions' on which we estimate the generalisation performance of the model.
# These represent novel regions that a user might query during production: they are different to the train regions but not drastically so.
# 
# 
# We can thus generate splits of the data into a safe testing region and training regions (all other safe regions + the unsafe regions). Each safe region is a testing region once! We thus have as many splits as we have safe regions. We refer to this procedure as 'SafeKFold'.
# 
# 
# To estimate the generalization performance of the model we perform nested cross validation, e.g. as described in https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html.
# We use our custom 'SafeKFold' splitting procedure in both the outer and inner cross validation loop.
# Nested CV allows us to obtain generalisation performance estimates with uncertainties as well as fit hyperparameters in a principled manner.
# The inner cv fits the hyperparameters given a train set and the outer cv estimates the error for the found hyperparameters on a safe region; this is repeated for all safe regions.
# (The inner cv has one safe region less available to perform the SafeKFold (precisely the one that is being used in the current iteration of the outer cv loop.))
# [I think my code on the nested CV is actually fairly readable, so feel free to give it a go!]
# To obtain the final model used for production, we simply run the inner cross-validation loop on all data, to find the best hyperparameters using SafeKFold on the entire dataset.
# Given these hyparparameters, we then train a gradient-boosted regression tree on the entire dataset.
# The performance we obtain can be interpreted as the average (+- std) error when predicting on a new region (that is realistic/safe as defined above).
# 
# 
# Nested cross-validation achieves two things:
# * outer cv loop: for each outer fold (splits obtained with `SafeKFold`)
#     * obtain train and test set
#     * execute inner cv on train set --> this returns a model
#     * evaluate model on test (remember test is a new safe region)
#     * get a prediction error
# --> average prediction errors to gauge model performance 
# 
# * inner cv loop: get a training set. split this using the `SafeKFold`. (because we are now in the inner loop, there is one safe region less! (the one that is currently being held as the test set in the outer loop; remember inner_cv is executed for each iteration of the outer loop!!))
# * for each fold k
#     * obtain a train and test set
#     * for all hyper configs we want to try
#         * train gdbt(hyper config, current fold k)
#     * check which hyper performed best in this fold
# --> check which hyper setting performed best on average over all folds, train on all train data (available to the inner cv) using that hyper config --> return the best model
# 
# (note for sklearn, the GridSearchCV.predict() automatically populates this 'best model', see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV.predict)
# 

# # RE stats analysis

# In[2]:


import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict
import seaborn as sn
import logging
logging.getLogger().setLevel(logging.DEBUG)
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from IPython.display import display
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# load custom helper functions
import helper as h


# In[3]:


df_REstats = pd.read_excel('RA_data_220816_REstats_norm.xlsx', engine='openpyxl')
df_REstats.head()


# In[4]:


# Remove scrap fraction and installation year from feature list
REstats_columns = df_REstats.columns[1:]
logging.info(f'RE stats are {REstats_columns}.')

REstats_unscaled = np.array([df_REstats[i].values for i in REstats_columns]).T

REstats_scaler = StandardScaler()
REstats_scaler.fit(REstats_unscaled)
REstats_scaled = REstats_scaler.transform(REstats_unscaled)

REstats = REstats_scaled


# In[7]:


# Conduct multicollinearity analysis
# https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
    
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(REstats).correlation

corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(
    dist_linkage, labels=REstats_columns.tolist(), ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro["ivl"]))

ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]], cmap='YlGnBu_r')
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
fig.tight_layout()
plt.show()


# In[8]:


# Pick a threshold and keep a single feature from each cluster
threshold = 0.3

cluster_ids = hierarchy.fcluster(dist_linkage, threshold, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

selected_features


# # Machine Learning

# ### Customisable Options

# In[15]:


# set target
target = 'LCOS - exc ore'

# set to true if you want to enable plots
visualize = False


# ### Data Loading

# In[10]:


df = pd.read_excel('RA_data_220816.xlsx', engine='openpyxl')

# extract regions and year of weather data
extracted = h.extract_region_year(df)


# In[11]:


display(df.head().T)


# In[12]:


extracted.main_region.unique()


# In[13]:


# check selected target column
all_targets = df.columns[1:8]

# some logging
logging.info(f'Possible target columns {all_targets}.')
logging.info(f'Selected column {target}.')
assert target in all_targets, f'Given target {target} invalid.'

# define selected features from multicollinearity analysis
independent_features = df.columns[[8, 10, 14, 18, 20]]
feature_columns = independent_features
logging.info(f'Feature columns used in regression are {feature_columns}.')


# In[16]:


# data visualisation for target column

if visualize:
    logging.info(f'Plotting target column {target} against all features.')
    for col in feature_columns:
        fig, ax = plt.subplots(1, 1, dpi=200)
        df.plot.scatter(col, target, s=1, ax=ax)
        plt.show()

    plt.figure(dpi=200)
    plt.hist(y, bins=200);


# ## Regresssion

# ### Data Setup

# In[17]:


# get numpy arrays from dataframes
y_unscaled = np.array(df[target].values.reshape(-1, 1))
x_unscaled = np.array([df[i].values for i in feature_columns]).T

# standardise x and y; per column: (value - mean) / std
y_scaler = StandardScaler()
y_scaler.fit(y_unscaled)
y_scaled = y_scaler.transform(y_unscaled)

x_scaler = StandardScaler()
x_scaler.fit(x_unscaled)
x_scaled = x_scaler.transform(x_unscaled)

# shorthands
x, y = x_scaled, y_scaled


# In[18]:


# define 'safe' regions
safe_regions = [2,5]

region_years = h.extract_region_year(df)
region_years['is_safe'] = list(map(lambda x: x in safe_regions, region_years.main_region))
# pass this as group to my custom cv
main_regions = list(map(int, region_years.main_region.values))


# ## Nested CV

# This may take a decent time to run, i.e. about `12 minutes per outer fold * 6 folds = 36 minutes` on my MacBook.
# 
# I actually have not computed the 'real' solution to this, so I'd be interested to see your results!
# 
# Instead, I've only run a smaller grid for the hyperparameter search:
# All outputs below were actually computed for `DEBUG=True`
# 
# (but I've changed it to `False` to make sure you actually run the larger search for the final model! larger search should give better results!
# 
# 
# Your results should be _better_ than the ones I've created below. Let me know if this is not the case!!
# 

# In[35]:


# general model class we will use
model = GradientBoostingRegressor()

# set up parameters grid over which we will search (debug = True reduces search significantly)
DEBUG = False

if not DEBUG:
    parameters = dict(
        learning_rate=[0.01, 0.1, 0.5],
        max_depth=[3, 5, 10],
        n_estimators=[100, 5000, 10000])
else:
    parameters = dict(
        learning_rate=[0.1],
        max_depth=[3, 5],
        n_estimators=[100, 1000])


# In[36]:


outer_cv = h.CustomLeaveOneGroupOut(safe_regions)
# slightly misleading, as we will only select groups for test
# that are listed in _safe regions_
# groups not listed in safe regions will always be part of train
groups = np.array(main_regions)
n_regions = len(np.unique(main_regions))

scores = []
test_groups = []
hypers = []
grids = []
preds = []
y_trues = []
train_idxs = []
test_idxs = []


for fold, (train_index, test_index) in enumerate(outer_cv.split(x_scaled, y_scaled, groups=groups)):

    logging.info(f'Running fold {fold}')

    train_idxs.append(train_index)
    test_idxs.append(test_index)
    
    
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    current_test_group = groups[test_index[0]]
    current_train_groups = np.unique(groups[train_index])
    # only one group in test
    assert len(np.unique(groups[test_index])) == 1
    # none of the test group in train
    assert sum(groups[train_index] == current_test_group) == 0
    assert set([current_test_group]) | set(current_train_groups) == set(range(1, n_regions+1))
    # test groups + train groups == all groups
    test_groups.append(current_test_group)

    inner_safe_regions = list(set(safe_regions) - {current_test_group})
    
    inner_cv = h.CustomLeaveOneGroupOut(inner_safe_regions)
    
    grid = GridSearchCV(
        model, parameters, verbose=1, n_jobs=8, cv=inner_cv)

    grid.fit(x_train, y_train, groups=groups[train_index])
    
    grids.append(grid)
    scores.append(grid.score(x_test, y_test))
    preds.append(grid.predict(x_test))
    y_trues.append(y_test)
    hypers.append(grid.best_params_)


# no group gone through twice 
assert len(np.unique(test_groups)) == len(test_groups)
# each safe region tried once as test
assert set(test_groups) == set(safe_regions)


# ## Evaluate

# In[42]:


mean_score = np.mean(scores)

mean_score, np.std(scores)


# In[43]:


# also maybe interesting to look at the score per region
print(test_groups)
print(scores)


# In[44]:


# calculate root mean squared error (RMSE) 
rmses = []
for i in range(len(safe_regions)):
    rmses.append(np.sqrt(np.mean((preds[i]-y_trues[i][:, 0])**2)))
mean_rmse = np.mean(rmses)

mean_rmse, np.std(rmses)


# In[45]:


# can reverse the standardisation by multiplying with std of y
mean_rmse * np.sqrt(y_scaler.var_[0])


# In[46]:


# can also report the train error
train_r2 = []
for grid, train_idx in zip(grids, train_idxs):
    train_r2.append(grid.best_estimator_.score(x[train_idx], y[train_idx]))

np.mean(train_r2), np.std(train_r2)


# As expected, we almost perfectly memorise the train data.
# (Note this does _not_ indicate overfitting on the test set, but memorisation of the train data is rather typical of these types of approaches.)

# ### Visualise Results

# * we now have one best model and one test set _for each fold_
# 
# * can re-run the below plots for each of the fold by changing `SELECTED FOLD` param

# In[47]:


SELECTED_FOLD = 1
print('Selecting model that is tested on region', test_groups[SELECTED_FOLD])
selected_model = grids[SELECTED_FOLD].best_estimator_

train_idx = train_idxs[SELECTED_FOLD]
test_idx = test_idxs[SELECTED_FOLD]

train_pred = selected_model.predict(x[train_idx])
test_pred = selected_model.predict(x[test_idx])

# show scatter plot of predictions on train and test vs true values
plt.figure(dpi=200)
plt.scatter(train_pred, y[train_idx], s=1, alpha=0.4, label='train')
plt.scatter(test_pred, y[test_idx], s=1, alpha=1, label='test')

plt.legend()


# In[48]:


plt.figure(dpi=200)
plt.xlabel('|predicted-true|')
plt.hist(np.sqrt((test_pred-y[test_idx][:, 0])**2), bins=100, alpha=0.5, label='test')
plt.hist(np.sqrt((train_pred-y[train_idx][:, 0])**2), bins=100, alpha=1, label='train')
plt.legend();


# # Final Model
# 
# A side effect of the nested cv is that we are not given a single best model. Instead we get the average generalisation performance of our model (where model = gradient boosted tree + inner cv for hyper search).
# 
# However, if we want to select a 'best' model, we can just train the inner CV using all data!

# In[49]:


cv = h.CustomLeaveOneGroupOut(safe_regions)

grid = GridSearchCV(
    model, parameters, verbose=1, n_jobs=8, cv=cv)

grid.fit(x, y, groups=groups)


# In[50]:


best_model = grid.best_estimator_
grid.best_params_


# In[51]:


best_model.n_features_in_


# We plot the feature importances for this 'final model'.

# In[52]:


# get importance
importance = best_model.feature_importances_
# feature importance bar graph
plt.figure(dpi=200)
plt.title('feature importances')
spacing = np.arange(len(feature_columns))
sorting = np.argsort(importance)[::-1]

plt.barh(spacing, importance[sorting])
plt.gca().set_yticks(spacing)
plt.gca().set_yticklabels(feature_columns[sorting], rotation=0)
plt.gca().invert_yaxis();


# In[53]:


# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))


# ## Save model to load later 
# https://scikit-learn.org/dev/model_persistence.html

# In[54]:


from joblib import dump, load


# In[55]:


dump(best_model, 'LCOS_xore_model.joblib')

