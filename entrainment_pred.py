from __future__ import division

from collections import namedtuple
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gamma
from sklearn.cross_validation import permutation_test_score, ShuffleSplit
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.svm import LinearSVR, LinearSVC


# Based on:
# http://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
class LinearSVRPermuteCoef(LinearSVR):
    def __init__(self, *args, **kwargs):
        self.permute_max_coefs = []
        self.permute_min_coefs = []
        super(LinearSVR, self).__init__(*args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearSVR, self).fit(X, y, n_jobs)

        def add_coef(arr, fn):
            arr.append(fn(self.coef_))

        add_coef(self.permute_max_coefs, np.max)
        add_coef(self.permute_min_coefs, np.min)

        return self

    def resert_perm_coefs(self):
        self.permute_max_coefs = []
        self.permute_min_coefs = []


def is_data_col(c):
    front = c[0]
    try:
        int(front)
        return True
    except ValueError:
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


def verbose_scorer(total_runs, score_fn=f1_score):
    print('-' * total_runs)

    def verbose_score_fn(truth, predictions):
        sys.stdout.write('#')
        return score_fn(truth, predictions)

    return make_scorer(verbose_score_fn)


def search_all():
    import logging
    logging.basicConfig(filename='search_results.log', level=logging.DEBUG)

    data_types = {'atw': ['z', 'diff_wpm'],
                  'adw': ['z']}
    for data_type in data_types:
        full = pd.read_csv("data/step3/full_%s.csv" % data_type)
        for target_col in data_types[data_type]:
            target_col = data_type + '_' + target_col
            print("about to process: %s" % target_col)

            logging.info("results for %s" % target_col)

            search = run(full, target_col)
            search_normalize = run(full, target_col, normalize=True)

            (search, normalized) = (search, "no") if search.best_score > search_normalize.best_score \
                else (search_normalize, "yes")

            logging.info("normalized: %s" % normalized)

            logging.info("best score: %s" % search.best_score_)
            logging.info("best params: %s" % search.best_params_)

            data, target = separate(full, target_col)

            best_svr = search.best_estimator_.named_steps['svr']
            best_svr.resert_perm_coefs()

            def save_csv(desc, arr):
                np.savetxt('%s_%s.csv' % (target_col, desc), arr, delimiter=',')

            save_csv('best_coefs', best_svr.coef_)

            score, permutation_pred_scores, p_value = permutation_test_score(
                search.best_estimator_,
                data, target,
                scoring=search.scoring,
                cv=search.cv,
                n_permutations=100
            )

            logging.info("best score perms: %s" % score)

            save_csv('permute_pred_scores', permutation_pred_scores)
            save_csv('permute_max_coefs', best_svr.permute_max_coefs)
            save_csv('permute_min_coefs', best_svr.permute_min_coefs)

            logging.info("p-value: %s" % p_value)
            if p_value >= .05:
                logging.warn("p_value of %s >= .05")

            learning_curve_title = "%s Learning Curve" % target_col
            plot_learning_curve(search.best_estimator_,
                                learning_curve_title, X, y,
                                f_name=learning_curve_title.replace(' ', '_') + '.png')


def run(full, target_col, random_state=1234, c_range_alpha=.05, c_range_size=100, normalize=False):

    svr = LinearSVRPermuteCoef()
    
    pipeline_steps = [('svr', svr)]

    pipeline = Pipeline(pipeline_steps)

    c_range = gamma.rvs(size=c_range_size, a=c_range_alpha, random_state=random_state)

    param_dist = {"svr__C": c_range}

    data, target = separate(full, target_col)
    
    if normalize:
        data = scale(data)

    n_iter = 100
    cv = ShuffleSplit(len(target), n_iter=n_iter, test_size=1/6.0, random_state=random_state)

    total_runs = n_iter
    scorer = verbose_scorer(total_runs, r2_score)

    search = RandomizedSearchCV(pipeline, param_distributions=param_dist, cv=cv, scoring=scorer,
                                random_state=random_state)

    search.fit(data, target)

    return search


def run_class(full, target_col, random_state=1234, c_range_alpha=1.99, c_range_size=100):
    svc = LinearSVC()

    pipeline = Pipeline([('svc', svc)])

    c_range = gamma.rvs(size=c_range_size, a=c_range_alpha, random_state=random_state)

    param_dist = {"svc__C": c_range}

    data, target = separate(full, target_col)
    target_c = target > 0

    n_iter = 100
    cv = ShuffleSplit(len(target), n_iter=n_iter, test_size=1/6.0, random_state=random_state)

    total_runs = n_iter
    scorer = verbose_scorer(total_runs)

    search = RandomizedSearchCV(pipeline, param_distributions=param_dist, cv=cv, scoring=scorer,
                                random_state=random_state)

    search.fit(data, target_c)

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


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5),
                        f_name=None):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    f_name: string, optional
        default None
        name of file to save plot to (if set)
    """
    from sklearn.learning_curve import learning_curve

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    if f_name is not None:
        plt.savefig(f_name)

    return np.max(train_scores_mean), np.max(test_scores_mean)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
