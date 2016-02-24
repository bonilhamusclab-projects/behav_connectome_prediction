from __future__ import division

import os

import numpy as np
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


def load_x_csv(csv_path):
    df = _prep_df(pd.read_csv(csv_path, header=None))
    df = df.T
    df['id'] = os.path.basename(csv_path)[:-4]
    return df


def consolidate_x(csv_dir="data/step1/", dest_path="data/step3/X.csv"):
    fs = os.listdir(csv_dir)
    num_fs = len(fs)

    def load(csv, wait=dict(wait=0)):
        wait['wait'] += 1
        print "%s out of %s" % (wait['wait'], num_fs)
        return load_x_csv(os.path.join(csv_dir, csv))

    dfs = pd.concat([load(c) for c in fs], axis=0)
    assert dfs.shape[0] == num_fs

    if dest_path:
        dfs.to_csv(dest_path)

    return dfs


def consolidate(y_src="data/step2/outcomes.csv",
                x_src="data/step3/X.csv",
                dest_path="data/step3/full.csv"):
    x = pd.read_csv(x_src)
    y = pd.read_csv(y_src)
    full = pd.merge(x, y, on='id', how='inner')

    max_num_rows = np.min((x.shape[0], y.shape[0]))
    num_cols = x.shape[1] + y.shape[1] - 1

    assert full.shape[0] <= max_num_rows
    assert full.shape[1] == num_cols

    if dest_path:
        full.to_csv(dest_path)

    return full


def _test():
    import doctest
    doctest.testfile("step3_consolidate_data.py")
