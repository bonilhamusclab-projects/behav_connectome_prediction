from __future__ import division

import os

import numpy as np
import scipy.io as sio


def _mat_file_to_conn(f):
    return sio.loadmat(f)['dti_jhu']['r'][0][0]


def _get_conns(mats_dir):

    def load_f(f):
        try:
            return _mat_file_to_conn(os.path.join(mats_dir, f))
        except KeyError as e:
            print "file %s not loaded" % f
            print "missing field %s" % e
            return None

    conns = [(f, load_f(f)) for f in os.listdir(mats_dir)
             if (os.path.isfile(os.path.join(mats_dir, f)) and f.endswith("mat"))]

    return [fc for fc in conns if fc[1] is not None]


def convert_conns(mats_dir, dest_dir=None):

    conns = _get_conns(mats_dir)

    if dest_dir is not None:
        for fc in conns:
            f, conn = fc
            f_name, ext = os.path.splitext(f)

            dest_f = os.path.join(dest_dir, f_name + ".csv")
            np.savetxt(dest_f, conn, delimiter=',')

    return conns
