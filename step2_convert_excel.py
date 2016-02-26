from __future__ import division

import re

import numpy as np
import pandas as pd


def valid_id(id):
    """
    >>> from step2_convert_excel import valid_id
    >>> valid_id("M1234")
    True
    >>> valid_id("VA")
    False
    >>> import numpy as np
    >>> valid_id(np.NaN)
    False

    """
    if isinstance(id, basestring):
        return re.match(r"M[0-9]{3,4}", id) is not None
    else:
        return False


def convert_excel(src_file, dest_path):
    excel = pd.read_excel(src_file, 'Data')

    column_mappings = {
        'img ': 'id',
        'PD AVG TRW': 'pd_avg_total_words',
        'PD AVG TDW': 'pd_avg_total_different_words',
        'AverageTotalWords': 'avg_total_words',
        'AverageDifferentWords': 'avg_different_words'
    }

    data = excel[column_mappings.keys()]

    data.rename(columns=column_mappings, inplace=True)

    def float_or_null(i):
        try:
            return float(i)
        except:
            return np.NaN

    numeric_cols = ['pd_avg_total_words',
                    'pd_avg_total_different_words',
                    'avg_total_words',
                    'avg_different_words']

    for c in numeric_cols:
        data[c] = [float_or_null(i) for i in data[c]]

    valid_indices = [valid_id(i) for i in data['id']]
    data = data.loc[valid_indices, :]

    data.to_csv(dest_path, index=False)
    return data


def _test():
    import doctest
    doctest.testfile("step2_convert_excel.py")