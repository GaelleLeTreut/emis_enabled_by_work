# -*- coding: utf-8 -*-
# utils.py module
import numpy as np

############################
#functions for dataframe
############################
def df_to_csv_with_comment(df, output_file, comment, **kwargs):
    with open(output_file, 'w') as f:
        f.write(comment)
        f.write('\n')
        df.to_csv(f, **kwargs)

