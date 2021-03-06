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
from sklearn.learning_curve import learning_curve
from sklearn.metrics import f1_score, make_scorer, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.svm import LinearSVR, LinearSVC


def linearSVRPermuteCoefFactory():
    coeffs_state = {'min': [], 'max': []}

    class LinearSVRPermuteCoef:
        def __init__(self, **kwargs):
            self.model = LinearSVR(**kwargs)

        def fit(self, X, y):
            self.model.fit(X, y)

            self.coef_ = self.model.coef_
            self.intercept_ = self.model.intercept_

            def add_coef(arr, fn):
                arr.append(fn(self.coef_))

            add_coef(coeffs_state['max'], np.max)
            add_coef(coeffs_state['min'], np.min)

            return self

        def get_params(self, deep=True):
            return self.model.get_params(deep)

        def set_params(self, **kwargs):
            self.model.set_params(**kwargs)
            return self

        def predict(self, X):
            return self.model.predict(X)

        def score(self, X, y, sample_weight=None):
            if sample_weight is not None:
                return self.model.score(X, y, sample_weight)
            else:
                return self.model.score(X, y)

        @staticmethod
        def permute_min_coefs():
            return coeffs_state['min']

        @staticmethod
        def permute_max_coefs():
            return coeffs_state['max']

        @staticmethod
        def reset_perm_coefs():
            coeffs_state['min'] = []
            coeffs_state['max'] = []

    return LinearSVRPermuteCoef()


def is_data_col(c):
    front = c[0]
    try:
        int(front)
        return True
    except ValueError:
        return False


def jhu_edge_coordinates(left, right, jhu=pd.read_csv("data/jhu_coords.csv")):
    def loc_to_dict(loc):
        ret = dict()
        ret['x'], ret['y'], ret['z'], ret['name'] = jhu.iloc[loc, :]
        return ret

    return {'left': loc_to_dict(left),
            'right': loc_to_dict(right)}


def memoize_no_arg(fn):
    state = dict(called=False)

    def fn_wrap():
        if not state['called']:
            state['called'] = True
            state['ret'] = fn()
        return state['ret']

    return fn_wrap


@memoize_no_arg
def all_jhu_coordinates():
    jhu = pd.read_csv("data/jhu_coords.csv")
    return {"%s_%s" % (i, j): jhu_edge_coordinates(i, j, jhu)
            for i in range(189)
            for j in range(i+1, 189)}


def memoize(f):
    """
    :param f:
    :return:
    Memoization decorator for functions taking one or more arguments.
    http://code.activestate.com/recipes/578231-probably-the-fastest-memoization-decorator-in-the-/
    """
    class MemoDict(dict):
        def __init__(self, f):
            self.f = f

        def __call__(self, *args):
            return self[args]

        def __missing__(self, key):
            ret = self[key] = self.f(*key)
            return ret

    return MemoDict(f)


@memoize
def get_jhu_names(jhu_f="data/jhu_rois_left_adjusted.csv"):
    jhu = pd.read_csv(jhu_f)
    return set(jhu["name"])


def separate_cols(full):
    data_cols = {c for c in full.columns if is_data_col(c)}
    y_cols = set(full.columns) - data_cols - {'id'}
    return data_cols, y_cols


def separate(full, target_col):
    data_cols, y_cols = separate_cols(full)

    assert target_col in y_cols, "%s not in %s" % (target_col, y_cols)

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


def filter_roi_conns(df, conn_filter_fn):
    """
    :param df:
    :param conn_filter_fn:
    :return:
    """
    roi_conns = filter(is_data_col, df.columns)
    non_conns = set(df.columns) - set(roi_conns)
    filtered_roi_conns = filter(conn_filter_fn, roi_conns)
    return df[list(non_conns.union(filtered_roi_conns))]


def search_all(log_dir="data/step4/left_hemi_select_rois",
               conn_filter_fn=lambda conn: np.all(
                   [i['name'] in get_jhu_names("data/jhu_rois_left_adjusted.csv")
                    for i in all_jhu_coordinates()[conn].itervalues()])
               ):
    def _log_dir(f_name):
        import os
        return os.path.join(log_dir, f_name)

    import logging
    logging.basicConfig(filename=_log_dir('search_results.log'), 
                        level=logging.DEBUG, filemode='w+')

    data_types = {'atw': ['z'],
                  'adw': ['z']}
    for data_type in data_types:
        full = pd.read_csv("data/step3/full_%s.csv" % data_type)

        if conn_filter_fn is not None:
            full = filter_roi_conns(full, conn_filter_fn)

        for target_col in data_types[data_type]:
            target_col = data_type + '_' + target_col
            print("about to process: %s" % target_col)
            logger = logging.getLogger(target_col)

            logger.info("results for %s" % target_col)

            search = run(full, target_col)
            search_normalize = run(full, target_col, normalize=True)

            (search, normalized) = (search, "no") if search.best_score_ > search_normalize.best_score_ \
                else (search_normalize, "yes")

            logger.info("normalized: %s" % normalized)

            logger.info("best score: %s" % search.best_score_)
            logger.info("best params: %s" % search.best_params_)

            data, target = separate(full, target_col)

            best_svr = search.best_estimator_.named_steps['svr']
            best_svr.reset_perm_coefs()

            def save_csv(desc, arr):
                f_name = _log_dir('%s_%s.csv' % (target_col, desc))
                np.savetxt(f_name, arr, delimiter=',')

            save_csv('best_coefs', best_svr.coef_)

            score, permutation_pred_scores, p_value = permutation_test_score(
                search.best_estimator_,
                data.get_values(),
                target.get_values(),
                scoring=search.scoring,
                cv=search.cv,
                n_permutations=100
            )

            logger.info("best score perms: %s" % score)

            save_csv('permute_pred_scores', permutation_pred_scores)
            save_csv('permute_max_coefs', best_svr.permute_max_coefs())
            save_csv('permute_min_coefs', best_svr.permute_min_coefs())

            logger.info("p-value: %s" % p_value)
            if p_value >= .05:
                logger.warn("p_value of %s >= .05")

            train_sizes, train_scores, test_scores = learning_curve(
                search.best_estimator_,
                data.get_values(), target.get_values(),
                cv=search.cv, train_sizes=np.linspace(.1, 1.0, 5))

            save_csv("learning_curve_train_sizes", train_sizes)
            save_csv("learning_curve_train_scores", train_scores)
            save_csv("learning_curve_test_scores", test_scores)


def run(full, target_col, random_state=1234, c_range_alpha=.05, c_range_size=100, normalize=False,
        score_fn=r2_score):

    svr = linearSVRPermuteCoefFactory()
    
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
    scorer = verbose_scorer(total_runs, score_fn)

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


def plot_learning_curve_files(title,
                              train_scores,
                              test_scores,
                              train_sizes=None,
                              plot_f=None):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------

    title : string
        Title for the chart.

    train_scores : string or MxN matrix

    test_scores : string or MxN matrix

    train_sizes : string or matrix, optional

    plot_f: string, optional
        default None
        name of file to save plot to (if set)
    """

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    def load_if_filename(f):
        return np.loadtxt(f, delimiter=',') if isinstance(f, str) else f

    train_scores = load_if_filename(train_scores)
    test_scores = load_if_filename(test_scores)

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    train_sizes = range(0, len(train_scores_mean)) if \
        train_sizes is None else load_if_filename(train_sizes)

    plt.grid()

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    train_scores_std = np.std(train_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")

    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")

    plt.legend(loc="best")

    if plot_f is not None:
        plt.savefig(plot_f)

    return np.max(train_scores_mean), np.max(test_scores_mean)


def plot_learning_curve(estimator, title, X, y, cv=None,
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

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    return plot_learning_curve_files(title, train_scores, test_scores,
                                      train_sizes, f_name)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
