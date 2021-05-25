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
# ###### Loading MRIO database
# ##########################

# ###### Chargement de la base EXIOBASE
light_exiobase_folder ='IOT_2015_pxp_GHG_emissions'
full_exiobase_storage = DATA_PATH+'IOT_2015_pxp.zip'

## Download from exiobase Zenodo
## If the light database doesn't exist already, it loads the full database
if not os.path.exists(data_folder + os.sep + light_exiobase_folder):
    print('Loading full exiobase database...')
    io_orig =  pymrio.parse_exiobase3(full_exiobase_storage)
    print('Loaded')

## chargement des valeurs de base
    #io_orig.calc_all()

### Repartition de la combustion des ménages MtCO2 (désagrégation suivant coefficient emissions IMACLIM2010)
    F_Y_sec = pd.read_csv(DATA_PATH + 'F_Y_sec.txt', header=[0,1], index_col=0, sep="\t")

#### Creating extension for FOR VALUE-ADDED
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
    
# ###### Diagonalize the GHG stressor to have the origins of impacts
    io_orig.GHG_emissions = io_orig.impacts.diag_stressor('GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)')

##deleting unecessary satellite accounts
    del io_orig.satellite
    del io_orig.impacts
    del io_orig.IOT_2015_pxp
    
## saving the pymrio database with the satellite account for GHG emissions and value-added only 
    os.makedirs(data_folder + os.sep + light_exiobase_folder)
    io_orig.save_all(data_folder + os.sep + light_exiobase_folder)
  
## If the light database does exist, it loads it instead of the full database
else: 
    print('Loading part of the exiobase database...')
    io_orig = pymrio.load_all(data_folder + os.sep + light_exiobase_folder)
    print('Loaded')
    
##########################
###### AGGREGATION PRE CALCULATION
##########################
# Correspondance au plus proche de A38 ( 35 sectors)
# corresp_table = pd.read_csv(DATA_PATH + 'exiobase_A38.csv', comment='#',header=[0,1], index_col=0, sep=';')
# corresp_table=np.transpose(corresp_table)

# sector_agg = corresp_table.values
# io_orig.aggregate(sector_agg=sector_agg,sector_names = list(corresp_table.index.get_level_values(0)))   
    

##########################
###### CALCULATION
##########################

#then we could simply a calc_all, I split here to avoid calculations in the huge extensions
io_orig.calc_system()
io_orig.GHG_emissions.calc_system(x=io_orig.x, Y=io_orig.Y, L=io_orig.L, Y_agg=None, population=io_orig.population)
io_orig.GHG_emissions.calc_income_based(x = io_orig.x, V=io_orig.V, G=io_orig.G, V_agg=None, population=io_orig.population)

#io_orig.GHG_emissions.D_iba.sum(axis=0,level='region')
#pour avoir les D_iba avec seulement le pays pour l'origine des émissions


##########################
###### AGGREGATION POST CALCULATION
##########################
# Aggregation into 35 sectors (closest correspondance with A38 nomenclature)
corresp_table = pd.read_csv(DATA_PATH + 'exiobase_A38.csv', comment='#',header=[0,1], index_col=0, sep=';')
corresp_table=np.transpose(corresp_table)
sector_agg = corresp_table.values

## Aggregation into two regions: FR and RoW
region_table = pd.read_csv(DATA_PATH + 'exiobase_FRvsRoW.csv', comment='#', index_col=0, sep=';')
region_table=np.transpose(region_table)
region_agg = region_table.values

io_orig.aggregate(region_agg=region_agg, sector_agg=sector_agg, region_names = list(region_table.index.get_level_values(0)), sector_names = list(corresp_table.index.get_level_values(0)))
## To fix : V is not aggregated

## Do we really need to put here a calc all to reassess the variables that are not aggregated? A, B L and G are not calculted...
io_orig.calc_all()


##########################
###### Emission content
##########################
inc_emis_content = pymrio.calc_N(io_orig.GHG_emissions.S, io_orig.G)*1e-3


######
### Décomposition  Emission  content (in gCO2/euro)  = S + SB + SB^2 + ..
######
inc_emis_content_direct = io_orig.GHG_emissions.S*1e-3
inc_emis_content_fo = pymrio.calc_N(io_orig.GHG_emissions.S, io_orig.B)*1e-3
inc_emis_content_so = pymrio.calc_N(io_orig.GHG_emissions.S, io_orig.B.dot(io_orig.B))*1e-3

## FRANCE only 
# emission_content_FR = np.transpose(inc_emis_content).loc[['FR']]
# income_based_emis_FR = np.transpose(io_orig.income_based_emis).loc[['FR']]

######
### Decomposition enabled emissions (in gCO2/euro)
######
# emis_enable = inc_emis_content.copy()
# for r in region_list:
#     rest_list = list(region_list)
#     rest_list.remove(r)
    
#     S_enable = io_orig.GHG_emissions.S*1e-3
#     S_enable.loc[rest_list]=0
#     emis_enable_r = pymrio.calc_iec(S_enable, io_orig.G)
#     emis_enable_r.rename(index={'emission content':'emis content from '+r},inplace=True)
#     emis_enable = emis_enable.append(emis_enable_r)
#     del rest_list
# emis_enable = emis_enable.drop(['emission content'])















