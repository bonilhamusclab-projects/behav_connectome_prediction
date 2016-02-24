from __future__ import division

import pandas as pd


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

    data.to_csv(dest_path, index=False)
    return data
