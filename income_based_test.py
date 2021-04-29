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
import scipy.linalg

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
#io_orig.calc_all()

def block_vector_to_diagonal_block_matrix(arr, blocksize):
    """Transforms an array seen as a block vector into a matrix diagonal by block

    Parameters
    ----------

    arr : numpy array
        Input array

    blocksize : int
        number of rows forming one block

    Returns
    -------
    numpy ndarray with shape (rows 'arr',
                              columns 'arr' * blocksize)

    Example
    --------

    arr:      output: (blocksize = 3)
        3 1     3 1 0 0
        4 2     4 2 0 0
        5 3     5 3 0 0
        6 9     0 0 6 9
        7 6     0 0 7 6
        8 4     0 0 8 4
    """
    nr_row = arr.shape[0]

    if np.mod(nr_row, blocksize):
        raise ValueError(
            "Number of rows of input array must be a multiple of blocksize"
        )
        
    arr_diag =  scipy.linalg.block_diag(*tuple([arr[i*blocksize:(i+1)*blocksize,:]   for i in range(nr_row//blocksize)]))
    return arr_diag


################################
#### EXTENSION FOR VALUE-ADDED
################################

list_value_added= ['Taxes less subsidies on products purchased: Total',
       'Other net taxes on production',
       'Compensation of employees; wages, salaries, & employers\' social contributions: Low-skilled',
       'Compensation of employees; wages, salaries, & employers\' social contributions: Medium-skilled',
       'Compensation of employees; wages, salaries, & employers\' social contributions: High-skilled',
       'Operating surplus: Consumption of fixed capital',
       'Operating surplus: Rents on land',
       'Operating surplus: Royalties on resources',
       'Operating surplus: Remaining net operating surplus']

df = io_orig.satellite.F.loc[list_value_added].transpose()
df.columns.name='category'

#df created only to get correct indices
kron_df = pd.concat([df] * io_orig.get_regions().size, axis=1, keys=io_orig.get_regions(),names=['region','category'])
#vector of value-added separated by countries
v = pd.DataFrame( block_vector_to_diagonal_block_matrix(df.values, io_orig.get_sectors().size), index=kron_df.index, columns=kron_df.columns)

##check if things have not been mixed up
#(v.sum(level='category',axis=1) == df).all()


io_orig.V = v

#################################
##### EXTENSION FOR CO2_emissions equivalent
#################################
##### 1/ Aggregating All CO2 and All CH4
### No comportement available for stressors
### https://pymrio.readthedocs.io/en/latest/notebooks/advanced_group_stressors.html
#groups = io_orig.satellite.get_index(as_dict=True, grouping_pattern = {'CO2.*': 'CO2','CH4.*': 'CH4'})
#io_orig.satellite_agg = io_orig.satellite.copy(new_name='Aggregated CO2 CH4')		
#
#for df_name, df in zip(io_orig.satellite_agg.get_DataFrame(data=False, with_unit=True, with_population=False),
#                       io_orig.satellite_agg.get_DataFrame(data=True, with_unit=True, with_population=False)):
#    if df_name == 'unit':
#        io_orig.satellite_agg.__dict__[df_name] = df.groupby(groups).apply(lambda x: ' & '.join(x.unit.unique()))
#    else:
#        io_orig.satellite_agg.__dict__[df_name] = df.groupby(groups).sum()		
#
#print("D_pba C02 FR in Mt:", io_orig.satellite_agg.D_pba.loc['CO2']['FR'].sum()/1e9)
#print("D_pba CH4 FR in Mt:", io_orig.satellite_agg.D_pba.loc['CH4']['FR'].sum()/1e9)
#
#
##### 2/ Converting CH4 into CO2eq - Global warming potential multiplication
#GWP_CH4= 28
### PROBLEM UNIT / changer dans unit par CO2eq =>dans le fichier unit ca marche pas
#io_orig.satellite_conv= io_orig.satellite_agg.copy(new_name='Test CO2eq')	
#for df_name, df in zip(io_orig.satellite_conv.get_DataFrame(data=False, with_unit=True, with_population=False),
#                       io_orig.satellite_conv.get_DataFrame(data=True, with_unit=True, with_population=False)):
#    if df_name == 'unit':
#        io_orig.satellite_conv.__dict__[df_name]['CH4']='kg CO2-eq'
#        io_orig.satellite_conv.unit.to_excel(OUTPUTS_PATH+'units_eq.xlsx')
#    else:
#        io_orig.satellite_conv.__dict__[df_name].loc['CH4'] = df.loc['CH4']*GWP_CH4
#        
#print("D_pba C02 FR in Mt:", io_orig.satellite_conv.D_pba.loc['CO2']['FR'].sum()/1e9)
#print("D_pba CH4 FR in Mt:", io_orig.satellite_conv.D_pba.loc['CH4']['FR'].sum()/1e9)
#
##### 3/ Grouping CH4 and CO2
#groups_CO2 = io_orig.satellite_conv.get_index(as_dict=True, grouping_pattern = {'CO2.*':'CO2eq','CH4.*':'CO2eq'})
#io_orig.satellite_eq = io_orig.satellite_conv.copy(new_name='Aggregated CO2 CH4')		
#
#for df_name, df in zip(io_orig.satellite_eq.get_DataFrame(data=False, with_unit=True, with_population=False),
#                       io_orig.satellite_eq.get_DataFrame(data=True, with_unit=True, with_population=False)):
#    if df_name == 'unit':
#        io_orig.satellite_eq.__dict__[df_name] = df.groupby(groups_CO2).apply(lambda x: ' & '.join(x.unit.unique()))
#    else:
#        io_orig.satellite_eq.__dict__[df_name] = df.groupby(groups_CO2).sum()	
#
#
#print("D_pba C02eq FR in Mt:", io_orig.satellite_eq.D_pba.loc['CO2eq']['FR'].sum()/1e9)
#
##### 4/ Extension for the CO2_emission
#io_orig.CO2_emissions = io_orig.satellite_eq.copy(new_name='CO2 emissions')		
#for df_name, df in zip(io_orig.CO2_emissions.get_DataFrame(data=False, with_unit=True, with_population=False),
#                       io_orig.CO2_emissions.get_DataFrame(data=True, with_unit=True, with_population=False)):
#    io_orig.CO2_emissions.__dict__[df_name] = df.loc['CO2eq']	
#
#print("D_pba C02eq FR in Mt:", io_orig.CO2_emissions.D_pba.loc['FR'].sum()/1e9)
#
#### Supprimer les extension intermediaire
#del io_orig.satellite_conv
#del io_orig.satellite_eq
#
#
#### New extension for Only CO2eq
#print(io_orig.satellite_agg.D_cba.loc['CO2']['FR']/1e9)

#diagonalize the GHG stressor to have the origins of impacts
io_orig.GHG_emissions = io_orig.impacts.diag_stressor('GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)')

#at this point we would like to save io_orig with keeping only GHG_emissions as a satellite account (thus removing satellite and impacts, that are no longer needed), and also v. If this saving file already exists, then upload it instead of exiobase_storage: this should speed up the process

#then we could simply a calc_all, I split here to avoid calculations in the huge extensions
io_orig.calc_system()
io_orig.GHG_emissions.calc_system(x=io_orig.x, Y=io_orig.Y, L=io_orig.L, Y_agg=None, population=io_orig.population)
io_orig.GHG_emissions.calc_income_based(x = io_orig.x, V=io_orig.V, G=io_orig.G, V_agg=None, population=io_orig.population)

#io_orig.GHG_emissions.D_iba.sum(axis=0,level='region')
#pour avoir les D_iba avec seulement le pays pour l'origine des émissions

#### Ajouter le D_iba dans pymrio.Extension? 
#io_orig2 =  pymrio.parse_exiobase3(exiobase_storage)
