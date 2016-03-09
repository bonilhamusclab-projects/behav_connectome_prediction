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


def convert_excel(src_file, dest_dir='data/step2/'):
    excel = pd.read_excel(src_file, 'Data')

    column_mappings = {
        'img ': 'id',
        'PD AVG TRW': 'pd_atw',
        'PD AVG TDW': 'pd_adw',
        'AverageTotalWords': 'se_atw',
        'AverageDifferentWords': 'se_adw'
    }

    data = excel[column_mappings.keys()]

    data.rename(columns=column_mappings, inplace=True)

    def float_or_nan(val):
        try:
            return float(val)
        except ValueError:
            return np.NaN

    valid_indices = [valid_id(i) for i in data['id']]
    data = data.loc[valid_indices, :]

    numeric_cols = set(column_mappings.values()) - {'id'}

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
        df = data.loc[valid_rows, ['id'] + [prepend(k) for k in ['se', 'pd']]]

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
