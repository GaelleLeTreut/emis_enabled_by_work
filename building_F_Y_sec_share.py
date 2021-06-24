# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 17:11:55 2021

@author: GLT
"""

import numpy as np
import pandas as pd
import os

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


# ##########################
# ###### Loading extended files 
# ##########################

F_Y_sec_extended = pd.read_csv(DATA_PATH + 'emissions_finaldemand_disag_2015.txt', header=[0], index_col=0, sep="\t")
F_Y_sec = F_Y_sec_extended[F_Y_sec_extended['SubstanceName'].isin(['CO2 - combustion', 'CH4 - combustion','N2O - combustion'])].drop(['IndustryTypeCode','AccountingYear','UnitCode','CompartmentName'],axis=1)
F_Y_sec = F_Y_sec.reset_index()
F_Y_sec = F_Y_sec.set_index(['SubstanceName', 'ProductTypeCode', 'region'])

## Conversion en kgC02eq
Convert = pd.DataFrame(index=F_Y_sec.index,columns=['Conversion to CO2eq'])
Convert.loc['CH4 - combustion'] = 25
Convert.loc['N2O - combustion'] = 298
Convert.loc['CO2 - combustion'] = 1
F_Y_sec = F_Y_sec.mul(Convert.values).groupby(axis=0,level=[2,1]).sum().unstack('region').fillna(0)
F_Y_sec.columns = F_Y_sec.columns.droplevel(0)

# ##########################
## Correspondance sector / code 
# ##########################
corresp_code_sect = pd.read_csv(DATA_PATH + 'sector_exiobase.csv', header=[0],  sep=";")
F_Y_sec_fill = pd.DataFrame(index =corresp_code_sect['ProductTypeCode'],columns=F_Y_sec.columns).fillna(0)

## Filling F_Y_sec_tofill with F_Y_sec to get a matrix with the 200products... 
## F_Y_sec_fill.combine_first(F_Y_sec) fails
## F_Y_sec_fill.update(F_Y_sec)
row_to_add = pd.DataFrame(index=F_Y_sec_fill.index.difference(F_Y_sec.index),columns=F_Y_sec.columns)
F_Y_sec = F_Y_sec.append(row_to_add).fillna(0).sort_index()

### replace code by name of sector... surement une facon mieux de faire 
F_Y_sec['sector']=corresp_code_sect['sector'].values
F_Y_sec.set_index('sector',inplace=True)

# ##########################
## Share  / repartition key to use in impacts
# ##########################
share_F_Y_sec = F_Y_sec.div(F_Y_sec.sum(axis=0))*100
share_F_Y_sec.to_pickle(DATA_PATH+'Share_F_Y_sec.pkl')