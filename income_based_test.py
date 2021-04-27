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

io_orig.value_added = io_orig.satellite.copy(new_name='Value-Added')		

for df_name, df in zip(io_orig.value_added.get_DataFrame(data=False, with_unit=True, with_population=False),
                       io_orig.value_added.get_DataFrame(data=True, with_unit=True, with_population=False)):
    io_orig.value_added.__dict__[df_name] = df.loc[list_value_added]	

# io_orig.value_added = pymrio.Extension('value_added')

################################
#### EXTENSION FOR CO2_emissions equivalent
################################
#### 1/ Aggregating All CO2 and All CH4
## No comportement available for stressors
## https://pymrio.readthedocs.io/en/latest/notebooks/advanced_group_stressors.html
groups = io_orig.satellite.get_index(as_dict=True, grouping_pattern = {'CO2.*': 'CO2','CH4.*': 'CH4'})
io_orig.satellite_agg = io_orig.satellite.copy(new_name='Aggregated CO2 CH4')		

for df_name, df in zip(io_orig.satellite_agg.get_DataFrame(data=False, with_unit=True, with_population=False),
                       io_orig.satellite_agg.get_DataFrame(data=True, with_unit=True, with_population=False)):
    if df_name == 'unit':
        io_orig.satellite_agg.__dict__[df_name] = df.groupby(groups).apply(lambda x: ' & '.join(x.unit.unique()))
    else:
        io_orig.satellite_agg.__dict__[df_name] = df.groupby(groups).sum()		

print("D_pba C02 FR in Mt:", io_orig.satellite_agg.D_pba.loc['CO2']['FR'].sum()/1e9)
print("D_pba CH4 FR in Mt:", io_orig.satellite_agg.D_pba.loc['CH4']['FR'].sum()/1e9)


#### 2/ Converting CH4 into CO2eq - Global warming potential multiplication
GWP_CH4= 28
## PROBLEM UNIT / changer dans unit par CO2eq =>dans le fichier unit ca marche pas
io_orig.satellite_conv= io_orig.satellite_agg.copy(new_name='Test CO2eq')	
for df_name, df in zip(io_orig.satellite_conv.get_DataFrame(data=False, with_unit=True, with_population=False),
                       io_orig.satellite_conv.get_DataFrame(data=True, with_unit=True, with_population=False)):
    if df_name == 'unit':
        io_orig.satellite_conv.__dict__[df_name]['CH4']='kg CO2-eq'
        io_orig.satellite_conv.unit.to_excel(OUTPUTS_PATH+'units_eq.xlsx')
    else:
        io_orig.satellite_conv.__dict__[df_name].loc['CH4'] = df.loc['CH4']*GWP_CH4
        
print("D_pba C02 FR in Mt:", io_orig.satellite_conv.D_pba.loc['CO2']['FR'].sum()/1e9)
print("D_pba CH4 FR in Mt:", io_orig.satellite_conv.D_pba.loc['CH4']['FR'].sum()/1e9)

#### 3/ Grouping CH4 and CO2
groups_CO2 = io_orig.satellite_conv.get_index(as_dict=True, grouping_pattern = {'CO2.*':'CO2eq','CH4.*':'CO2eq'})
io_orig.satellite_eq = io_orig.satellite_conv.copy(new_name='Aggregated CO2 CH4')		

for df_name, df in zip(io_orig.satellite_eq.get_DataFrame(data=False, with_unit=True, with_population=False),
                       io_orig.satellite_eq.get_DataFrame(data=True, with_unit=True, with_population=False)):
    if df_name == 'unit':
        io_orig.satellite_eq.__dict__[df_name] = df.groupby(groups_CO2).apply(lambda x: ' & '.join(x.unit.unique()))
    else:
        io_orig.satellite_eq.__dict__[df_name] = df.groupby(groups_CO2).sum()	


print("D_pba C02eq FR in Mt:", io_orig.satellite_eq.D_pba.loc['CO2eq']['FR'].sum()/1e9)

#### 4/ Extension for the CO2_emission
io_orig.CO2_emissions = io_orig.satellite_eq.copy(new_name='CO2 emissions')		
for df_name, df in zip(io_orig.CO2_emissions.get_DataFrame(data=False, with_unit=True, with_population=False),
                       io_orig.CO2_emissions.get_DataFrame(data=True, with_unit=True, with_population=False)):
    io_orig.CO2_emissions.__dict__[df_name] = df.loc['CO2eq']	

print("D_pba C02eq FR in Mt:", io_orig.CO2_emissions.D_pba.loc['FR'].sum()/1e9)

### Supprimer les extension intermediaire
del io_orig.satellite_conv
del io_orig.satellite_eq


### New extension for Only CO2eq
print(io_orig.satellite_agg.D_cba.loc['CO2']['FR']/1e9)



#### Ajouter le D_iba dans pymrio.Extension? 
#io_orig2 =  pymrio.parse_exiobase3(exiobase_storage)
