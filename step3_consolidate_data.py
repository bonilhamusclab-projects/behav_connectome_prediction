from __future__ import division

import os

import pandas as pd


def _prep_df(df):
    """
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = np.random.rand(3, 3)
    >>> from step3_consolidate_data import _prep_df
    >>> ret = _prep_df(pd.DataFrame(data))
    >>> expected_rows = ['0_1', '0_2', '1_2']
    >>> assert np.all(np.sort(ret.index) == expected_rows)
    >>> get_val = lambda c1, c2: ret[ret.index=='%s_%s' % (c1, c2)]['value'].iloc[0]
    >>> assert get_val(0, 1) == data[0][1]
    >>> assert get_val(0, 2) == data[0][2]
    >>> assert get_val(1, 2) == data[1][2]

    """

    num_rois = len(df.columns)

    tmp = df.copy()
    tmp['roi1'] = tmp.index.copy()

    ret = pd.melt(tmp, id_vars=['roi1'], var_name='roi2')
    ret = ret[ret['roi1'] < ret['roi2']]

    ret.index = ret.apply(lambda r: '%s_%s' % (r['roi1'], r['roi2']), axis=1)
    del ret['roi1']
    del ret['roi2']

    num_rows = (num_rois * (num_rois-1))/2
    assert ret.shape == (num_rows, 1)

    return ret


def consolidate(csv_dir, excel_path):

    def prep_df(csv):
        df = _prep_df(pd.read_csv(os.path.join(csv_dir, csv), header=None))
        df = df.T
        df['id'] = csv[:-4]

    dfs = pd.concat([prep_df(c) for c in os.listdir(csv_dir)], axis=1)
    assert dfs.shape[0] < dfs.shape[1]

    return dfs


def _test():
    import doctest
    doctest.testfile("step3_consolidate_data.py")
