from __future__ import division

import os
import re

import numpy as np
import pandas as pd
from scipy import stats


def valid_id(id_str):
    """
    :param id_str: string to be tested if is valid id
    :returns: boolean indicated if id is valid
    >>> from step2_convert_excel import valid_id
    >>> valid_id("M1234")
    True
    >>> valid_id("VA")
    False
    >>> import numpy as np
    >>> valid_id(np.NaN)
    False

    """
    if isinstance(id_str, basestring):
        return re.match(r"M[0-9]{3,4}", id_str) is not None
    else:
        return False


def read_excel_data(src_file, column_mappings):
    excel = pd.read_excel(src_file, 'Data')

    data = excel[column_mappings.keys()]

    data.rename(columns=column_mappings, inplace=True)

    return data


def _date_of_test_parser(date):
    """
    :param date:
    :return:

    >>> date = "2/14/13 & 2/15/13"
    >>> import pandas as pd
    >>> expected = pd.datetime(2013, 02, 15)
    >>> from step2_convert_excel import _date_of_test_parser
    >>> actual = _date_of_test_parser(date)
    >>> assert actual == expected

    """
    if isinstance(date, pd.datetime):
        return date

    if "&" in date:
        date = date[date.find("&") + 1:]

    return pd.to_datetime(date)


def _years_diff(newer_date, older_date):
    """
    :param newer_date:
    :param older_date:
    :return:
    >>> import pandas as pd
    >>> older_date = pd.to_datetime("8/15/1952")
    >>> newer_date = pd.to_datetime("2/15/13")
    >>> import step2_convert_excel
    >>> step2_convert_excel._years_diff(newer_date, older_date)
    60.5

    """
    def yr_ratio(field, div):
        return (getattr(newer_date, field) - getattr(older_date, field))/div

    ### doesn't account for leap years, etc...
    yr = yr_ratio('year', 1)
    month = yr_ratio('month', 12)
    day = yr_ratio('day', 265)

    return yr + month + day


def convert_excel_meta(src_file, dest_dir='data/step2'):
    column_mappings = {
        'img ': 'id',
        'Gender': 'gender',
        'DOB': 'dob',
        'DateOfTest': 'dot'
    }

    data = read_excel_data(src_file, column_mappings).dropna()

    ##Neded b/c age at test is important, not current day
    dot = [_date_of_test_parser(d) for d in data['dot']]
    dob = [d if d.year < 2000 else d - pd.timeseries.offsets.relativedelta(year=100)
           for d in data['dob']]

    data['age'] = map(lambda (nd, od): _years_diff(nd, od),  zip(dot, dob))
    del data['dob']
    del data['dot']

    dest_f = os.path.join(dest_dir, 'meta.csv')
    data.to_csv(dest_f, index=False)

    return data


def _sanitize_aphasia(d):
    """
    :param d:
    :return:
    >>> from step2_convert_excel import _sanitize_aphasia
    >>> _sanitize_aphasia("Broca's")
    'broca'
    >>> _sanitize_aphasia("Anomic")
    'anomic'
    >>> _sanitize_aphasia("Wernicke's/Cond.")
    'wernicke'

    """
    return re.sub(r"[^\w].*", "", d).lower()


def convert_excel(src_file, dest_dir='data/step2/'):
    column_mappings = {
        'img ': 'id',
        'PD AVG TRW': 'pd_atw',
        'PD AVG TDW': 'pd_adw',
        'AverageTotalWords': 'se_atw',
        'AverageDifferentWords': 'se_adw',
        'Aphasia Type': 'aphasia_type'
    }

    data = read_excel_data(src_file, column_mappings)

    def float_or_nan(val):
        try:
            return float(val)
        except ValueError:
            return np.NaN

    valid_indices = [valid_id(i) for i in data['id']]
    data = data.loc[valid_indices, :]

    data['aphasia_type'] = [_sanitize_aphasia(i) for i in data['aphasia_type']]

    numeric_cols = set(column_mappings.values()) - {'id'} - {'aphasia_type'}

    for c in numeric_cols:
        data[c] = [float_or_nan(i) for i in data[c]]

    base_cols = {n[3:] for n in numeric_cols}

    def not_nan(arr):
        return np.logical_not(np.isnan(arr))

    ret = dict()

    for c in base_cols:
        def prepend(test_type): return test_type + "_" + c

        def n_nan(test_type): return not_nan(data[prepend(test_type)])

        valid_rows = np.logical_and(n_nan('se'), n_nan('pd'))
        df = data.loc[valid_rows, ['id', 'aphasia_type'] + [prepend(k) for k in ['se', 'pd']]]

        pd_z = 'pd_' + c + '_z'
        se_z = 'se_' + c + '_z'
        df[pd_z] = stats.zscore(df[pd_z[:-2]])
        df[se_z] = stats.zscore(df[se_z[:-2]])
        df[c + '_z'] = df[se_z] - df[pd_z]

        df[c + '_diff_wpm'] = df[prepend('se')] * 2 - df[prepend('pd')]

        dest_f = os.path.join(dest_dir, c+'_outcomes.csv')
        df.to_csv(dest_f, index=False)
        ret[c] = df

    return ret


def _test():
    import doctest
    doctest.testfile("step2_convert_excel.py")
