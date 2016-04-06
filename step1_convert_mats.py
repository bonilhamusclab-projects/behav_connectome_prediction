from __future__ import division

import os

import numpy as np
import scipy.io as sio


def _load_conn_mat(f):
    return sio.loadmat(f)['dti_jhu']['r'][0][0]


def _load_lesion_mat(f):
    return sio.loadmat(f)['lesion_jhu']['mean'][0][0]


def _get_data(mats_dir, load_fn):
    def load_with_msg(f):
        try:
            return load_fn(os.path.join(mats_dir, f))
        except KeyError as e:
            print "file %s not loaded" % f
            print "missing field %s" % e
            return None

    conns = [(f, load_with_msg(f)) for f in os.listdir(mats_dir)
             if (os.path.isfile(os.path.join(mats_dir, f)) and f.endswith("mat"))]

    return [fc for fc in conns if fc[1] is not None]


def _convert(mats_dir, dest_dir, load_fn):
    fs_datas = _get_data(mats_dir, load_fn)

    if dest_dir is not None:
        for (f, data) in fs_datas:
            f_name, ext = os.path.splitext(f)

            dest_f = os.path.join(dest_dir, f_name + ".csv")
            np.savetxt(dest_f, data, delimiter=',')

    return fs_datas


def convert_conns(mats_dir, dest_dir="data/step1/conn"):
    return _convert(mats_dir, dest_dir, _load_conn_mat)


def convert_lesions(mats_dir, dest_dir="data/step1/lesion"):
    return _convert(mats_dir, dest_dir, _load_lesion_mat)
