from __future__ import division

from collections import namedtuple
import math
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gamma
from sklearn.cross_validation import KFold, ShuffleSplit
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import f1_score, mean_squared_error, make_scorer, r2_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR


def is_data_col(c):
    front = c[0]
    try:
        int(front)
        return True
    except:
        return False


def separate_cols(full):
    data_cols = {c for c in full.columns if is_data_col(c)}
    y_cols = set(full.columns) - data_cols - {'id'}
    return data_cols, y_cols


def separate(full, target_col):
    data_cols, y_cols = separate_cols(full)

    assert target_col in y_cols

    target = full.loc[:, target_col]
    indices = pd.notnull(target)
    target = target[indices]

    data = full.loc[indices, data_cols]

    return data, target


class CvInfo:
    def __init__(self, expected_mat, predicted_mat):
        self.expected_mat = expected_mat
        self.predicted_mat = predicted_mat
        self.expected_predicted_paired = zip(expected_mat, predicted_mat)

    def _compare(self, fn):
        return [fn(expected, predicted)
                for expected, predicted in self.expected_predicted_paired]

    def root_mean_square_errors(self):
        return self._compare(lambda e, p: math.sqrt(mean_squared_error(e, p)))

    def avg_rmse(self):
        return np.average(self.root_mean_square_errors())

    def __repr__(self):
        new_line = lambda s, v: s.format(v) + "\n"

        return (new_line("root-mean-square-error scores: {}", self.root_mean_square_errors()) +
                new_line("avg root-mean-square-error: {}", self.avg_rmse()))


def verbose_cv(mat, labels, alg, n_folds=3, verbose=True, cv=None):
    if cv is None:
        cv = KFold(labels, n_folds=n_folds)

    expected_mat = []
    predicted_mat = []

    expected_mat_train = []
    predicted_mat_train = []

    def print_v(s):
        if verbose:
            print s

    for train, test in cv:
        expected = labels[test]
        print_v("train labels")
        print_v(labels[train])
        print_v("test labels")
        print_v(expected)
        print_v("about to fit")
        alg.fit(mat[train], labels[train])
        print_v("about to predict")
        predicted = alg.predict(mat[test])
        print('####')

        expected_mat.append(expected)
        predicted_mat.append(predicted)

        expected_mat_train.append(labels[train])
        predicted_mat_train.append(alg.predict(mat[train]))

    return CvInfo(expected_mat, predicted_mat), CvInfo(expected_mat_train, predicted_mat_train)


def verbose_scorer(total_runs, score_fn=f1_score):
    print('-' * total_runs)

    def verbose_score_fn(truth, predictions):
        sys.stdout.write('#')
        return score_fn(truth, predictions)

    return make_scorer(verbose_score_fn)


def run(full, target_col, random_state=1234):

    svr = LinearSVR()

    pipeline = Pipeline([('svr', svr)])

    c_range = gamma.rvs(size=100, a=1.99, random_state=random_state)

    param_dist = {"svr__C": c_range}

    data, target = separate(full, target_col)

    n_iter = 100
    cv = ShuffleSplit(len(target), n_iter=n_iter, test_size=1/6.0, random_state=random_state)

    total_runs = n_iter
    scorer = verbose_scorer(total_runs, r2_score)

    search = RandomizedSearchCV(pipeline, param_distributions=param_dist, cv=cv, scoring=scorer,
                                random_state=random_state)

    search.fit(data, target)

    return search


def calc_coeffs(cv, fit_fn, coeffs_fn, predict_fn=None, normalize=True):
    coeffs = None
    for train, test in cv:
        fit_fn(train)
        if predict_fn is not None:
            predict_fn(test)
        coeffs_step = coeffs_fn()
        if normalize:
            coeffs_step = coeffs_step/np.sum(np.abs(coeffs_step))
        coeffs = coeffs_step if coeffs is None else coeffs + coeffs_step

    return coeffs/len(cv)


def _ttest_one_tail(a, b, ttest_fn, tuple_name):
    res_t = namedtuple(tuple_name, ('statistic', 'pvalue'))
    (t, p) = ttest_fn(a, b)
    return res_t(t, p/2)


def ttest_ind_one_tail(test_arr, base_arr):
    return _ttest_one_tail(test_arr, base_arr, stats.ttest_ind, 'Ttest_indResult')


def ttest_1samp_one_tail(test_arr, mu0):
    return _ttest_one_tail(test_arr, mu0, stats.ttest_1samp, 'Ttest_1sampResult')


if __name__ == "__main__":
    import doctest

    doctest.testmod()
