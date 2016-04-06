from __future__ import division

import os

import numpy as np
import pandas as pd


def _prep_conn_df(df):
    """
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = np.random.rand(3, 3)
    >>> from step3_consolidate_data import _prep_conn_df
    >>> ret = _prep_conn_df(pd.DataFrame(data))
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


def load_conn_csv(csv_path):
    df = _prep_conn_df(pd.read_csv(csv_path, header=None))
    df = df.T
    df.index = [os.path.basename(csv_path)[:-4]]
    return df


def load_lesion_csv(csv_path):
    return pd.read_csv(csv_path, header=None).T


def consolidate_x(csv_dir, dest_path, load_fn):
    fs = os.listdir(csv_dir)
    num_fs = len(fs)

    def load(csv, wait=dict(wait=0)):
        wait['wait'] += 1
        print "%s out of %s" % (wait['wait'], num_fs)
        return load_fn(os.path.join(csv_dir, csv))

    dfs = pd.concat([load(c) for c in fs], axis=0)
    assert dfs.shape[0] == num_fs

    if dest_path:
        dfs.to_csv(dest_path)

    return dfs


def consolidate_conn_x(csv_dir="data/step1/conn", dest_path="data/step3/conn/X.csv"):
    print("conn")
    consolidate_x(csv_dir, dest_path, load_conn_csv)


def consolidate_lesion_x(csv_dir="data/step1/lesion", dest_path="data/step3/lesion/X.csv"):
    print("lesion")
    consolidate_x(csv_dir, dest_path, load_lesion_csv)


def consolidate_x_with_y(y_dir="data/step2/",
                x_src="data/step3/X.csv",
                dest_dir="data/step3/"):
    x = pd.read_csv(x_src, index_col=0)

    ret = dict()
    for yo in ['adw', 'atw']:
        y_src = os.path.join(y_dir, yo+'_outcomes.csv')
        y = pd.read_csv(y_src)
        full = pd.merge(x, y, right_on='id', left_index=True, how='inner')

        max_num_rows = np.min((x.shape[0], y.shape[0]))
        num_cols = x.shape[1] + y.shape[1]

        assert full.shape[0] <= max_num_rows
        assert full.shape[1] == num_cols

        if dest_dir:
            dest_path = os.path.join(dest_dir, 'full_%s.csv' % yo)
            full.to_csv(dest_path, index=False)

        ret[yo] = full

    return ret


def consolidate_lesion_x_with_y(x_src="data/step3/lesion/X.csv",
                       dest_dir="data/step3/lesion/"):
    consolidate_x_with_y(x_src=x_src, dest_dir=dest_dir)


def consolidate_conn_x_with_y(x_src="data/step3/conn/X.csv",
                       dest_dir="data/step3/conn/"):
    consolidate_x_with_y(x_src=x_src, dest_dir=dest_dir)


def run_all():
    consolidate_conn_x()
    consolidate_conn_x_with_y()

    consolidate_lesion_x()
    consolidate_lesion_x_with_y()


def _test():
    import doctest
    doctest.testfile("step3_consolidate_data.py")
