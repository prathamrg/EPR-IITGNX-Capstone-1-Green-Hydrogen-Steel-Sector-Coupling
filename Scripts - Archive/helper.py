from functools import partial

import numpy as np
import pandas as pd

from sklearn.model_selection import BaseCrossValidator



def extract_region_year(df):
    # use Case IDs to extract the regions as well as the year from which the
    # wind data was taken
    # this is to later stratify by these variables in the cross-validation
    # to make sure we can generalise to new years/regions

    ids = list(df['Case ID'].values)

    def transform(name):
        # function works on a single string
        data_year = name[4:8]

        region, sim_year, scrap = name.split(' ')[2].split('-')
        main_region = ''.join([i for i in region if i.isnumeric()])

        main_region_year = f'{main_region}_{data_year}'

        return region, main_region, data_year, main_region_year

    data = np.array(list(map(transform, ids)))
    cols = 'region, main_region, data_year, main_region_year'.split(', ')

    return pd.DataFrame(data, columns=cols)


def region_year_filter(row):
    # I will try to select halve the data based on year
    # and half the data based on region
    # assuming balanced data and no correlation between the variables
    # this should leave me with about 1/4 of the data for testing!

    region = row.main_region in [str(i) for i in range(1, 9)]
    year = row.data_year in [str(i) for i in [2015, 2016, 2017]]

    return region & year


def random_region_year_filter(regions, years, row):
    # I will try to select halve the data based on year
    # and half the data based on region
    # assuming balanced data and no correlation between the variables
    # this should leave me with about 1/4 of the data for testing!

    region = row.main_region in [str(i) for i in regions]
    year = row.data_year in [str(i) for i in years]

    return region & year


def random_filter_and_eval(extracted, gen_model, x_scaled, y_scaled):

    num_regions = np.random.choice(range(2, 9), 1, replace=False)
    regions = np.random.choice(range(1, 13+1), num_regions, replace=False)
    years = np.random.choice(range(2015, 2019+1), 2)
    print(f'filter to region {regions} and years {years}')

    filtered = extracted.apply(
        partial(random_region_year_filter, regions, years), axis=1)

    fit_score(filtered, gen_model, x_scaled, y_scaled)

def fit_score(filtered, gen_model, x_scaled, y_scaled):

    fmean = filtered.mean()
    if not isinstance(filtered, np.ndarray):
        filtered = filtered.values
    # always train on larger half
    if fmean < 0.5:
        filtered = ~filtered

    x_train_filter, x_test_filter = x_scaled[filtered], x_scaled[~filtered]
    y_train_filter, y_test_filter = y_scaled[filtered], y_scaled[~filtered]

    print('train_proportion', x_train_filter.shape[0]/x_scaled.shape[0])

    gen_model.fit(x_train_filter, y_train_filter[:, 0])
    print('score', gen_model.score(x_test_filter, y_test_filter[:,0]))


def random_region_filter(regions, row):
    # I will try to select halve the data based on year
    # and half the data based on region
    # assuming balanced data and no correlation between the variables
    # this should leave me with about 1/4 of the data for testing!

    region = row.main_region in [str(i) for i in regions]

    return region


def random_region_filter_and_eval(extracted, gen_model, x_scaled, y_scaled):

    num_regions = np.random.choice(range(2, 9), 1, replace=False)
    regions = np.random.choice(range(1, 13+1), num_regions, replace=False)
    print(f'filter to region {regions}')

    filtered = extracted.apply(
        partial(random_region_filter, regions), axis=1)

    fit_score(filtered, gen_model, x_scaled, y_scaled)


def random_year_filter(years, row):
    # I will try to select halve the data based on year
    # and half the data based on region
    # assuming balanced data and no correlation between the variables
    # this should leave me with about 1/4 of the data for testing!

    year = row.data_year in [str(i) for i in years]

    return year

def random_year_filter_and_eval(extracted, gen_model, x_scaled, y_scaled):

    years = np.random.choice(range(2015, 2019+1), 3)
    print(f'filter to region years {years}')

    filtered = extracted.apply(
        partial(random_year_filter, years), axis=1)

    fit_score(filtered, gen_model, x_scaled, y_scaled)


def random_eval(extracted, gen_model, x_scaled, y_scaled):

    filtered = np.random.choice(
        2, x_scaled.shape[0], p=[0.3, 0.7]).astype(np.bool_)

    fit_score(filtered, gen_model, x_scaled, y_scaled)


class CustomLeaveOneGroupOut:
    def __init__(self, safe_regions):
        super().__init__()
        self.safe_regions = safe_regions

    def get_n_splits(self, X=None, y=None, groups=None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")

        return len(self.safe_regions)

    def split(self, X=None, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.
        groups : df for regions and years. map all the non-safe idxs to train
            do a k-Fold on the non-safe

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        main_regions = groups
        for safe_region in self.safe_regions:
            # select particular safe region
            select_safe = np.array(main_regions) == safe_region
            # safe region is test
            test = np.where(select_safe)[0]
            # everything else is train (i.e. all other safe regions and those
            # regions that are never safe)
            train = np.where(~select_safe)[0]

            yield train, test

