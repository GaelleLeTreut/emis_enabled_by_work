# # -*- coding: utf-8 -*-
# """
# Created on Wed Jan 27 09:50:21 2021

# @author: GaelleLeTreut
# """
# # -*- coding: utf-8 -*-
# """
# Éditeur de Spyder

# Ceci est un script temporaire.
# """

import numpy as np
import pandas as pd
import pymrio
import os
import utils as ut
import matplotlib.pyplot as plt
import seaborn as sns

# this_file = 'Inc_Based.py'
# ##########################
# ###### Paths 
# ##########################
# # récupérer le chemin du répertoire courant
path = os.getcwd()
data_folder='data'
output_folder='outputs'
#create data_folder if not exist
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    print('Creating ' + data_folder + ' to store data')

#create output_folder if not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print('Creating ' + output_folder + ' to store outputs')

DATA_PATH = path + os.sep + data_folder + os.sep
OUTPUTS_PATH = path + os.sep + output_folder + os.sep

# # plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
# # params = {'text.usetex' : True,
# #           'font.size' : 11,
# #           'font.family' : 'lmodern',
# #         #  'text.latex.unicode': True,
# #           }
# # plt.rcParams.update(params)

# ##########################
# ###### Chargement de la base MRIO
# ##########################
# ###### Chargement de la base EXIOBASE

## Download from exiobase Zenodo
exiobase_storage = DATA_PATH+'IOT_2015_pxp.zip'
io_orig =  pymrio.parse_exiobase3(exiobase_storage)

### Repartition de la combustion des ménages MtCO2 (désagrégation suivant coefficient emissions IMACLIM2010)
F_Y_sec = pd.read_csv(DATA_PATH + 'F_Y_sec.txt', header=[0,1], index_col=0, sep="\t")

## chargement des valeurs de base
io_orig.calc_all()


## No comportement available for stressors
## https://pymrio.readthedocs.io/en/latest/notebooks/advanced_group_stressors.html
groups = io_orig.satellite.get_index(as_dict=True, grouping_pattern = {'.*- air': 'Air',})
io_orig.satellite_agg = io_orig.satellite.copy(new_name='Aggregated Air')		

for df_name, df in zip(io_orig.satellite_agg.get_DataFrame(data=False, with_unit=True, with_population=False),
                       io_orig.satellite_agg.get_DataFrame(data=True, with_unit=True, with_population=False)):
    if df_name == 'unit':
        io_orig.satellite_agg.__dict__[df_name] = df.groupby(groups).apply(lambda x: ' & '.join(x.unit.unique()))
    else:
        io_orig.satellite_agg.__dict__[df_name] = df.groupby(groups).sum()		

### Only CO2eq
io_orig.satellite_agg.D_cba.loc['Air']['FR']/1e9


