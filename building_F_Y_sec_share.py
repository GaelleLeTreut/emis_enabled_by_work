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
## Correspondance sector / code 
# ##########################
corresp_code_sect = pd.read_csv(DATA_PATH + 'sector_exiobase.csv', sep=";")
product_dict = dict(corresp_code_sect.values)

# ##########################
# ###### Loading extended files 
# ##########################

F_Y_sec_extended = pd.read_csv(DATA_PATH + 'emissions_finaldemand_disag_2015.txt', header=[0], index_col=0, sep="\t")
F_Y_sec_by_GHG = F_Y_sec_extended[F_Y_sec_extended['SubstanceName'].isin(['CO2 - combustion', 'CH4 - combustion','N2O - combustion'])] #keep only GHG stressors

if (len(F_Y_sec_by_GHG['AccountingYear'].unique()) != 1) or (len(F_Y_sec_by_GHG['UnitCode'].unique()) != 1) or (len(F_Y_sec_by_GHG['CompartmentName'].unique()) != 1):
    print('Error in indexes of F_Y_sec_by_GHG: Accounting Year or UnitCode or CompartmentName are not unique and cannot be droped')


F_Y_sec_by_GHG_no_repeat= F_Y_sec_by_GHG.groupby(['region','ProductTypeCode','SubstanceName']).apply(lambda x: x['Amount'].sum()).reset_index() #sum by IndustryTypeCode
#F_Y_sec_by_GHG_no_repeat= F_Y_sec_by_GHG.groupby(['region','IndustryTypeCode','SubstanceName']).apply(lambda x: x['Amount'].sum()).reset_index() #sum by IndustryTypeCode
F_Y_sec_by_GHG_no_repeat['sector']=F_Y_sec_by_GHG_no_repeat['ProductTypeCode'].replace(product_dict) # convert ProductTypeCode into sector
F_Y_sec_by_GHG_no_repeat['Substance'] = F_Y_sec_by_GHG_no_repeat['SubstanceName'].map(lambda x: x.split(' - ')[0]) # get rid of the " - combustion"

F_Y_sec_by_GHG_no_repeat.set_index(['region','Substance','sector'],inplace=True)

GWP_conversion = pd.DataFrame({0: [1, 25, 298]},  index=['CO2', 'CH4', 'N2O'])

F_Y_sec_by_GHG_in_CO2 = F_Y_sec_by_GHG_no_repeat.mul(GWP_conversion,level=1) #Conversion en kgC02eq

F_Y_sec_all_GHG = F_Y_sec_by_GHG_in_CO2.groupby(['region','sector']).apply(lambda x: x[0].sum()).reset_index()

#statistics in absolute number in each region
#F_Y_sec_by_GHG_in_CO2.groupby(['region','Substance']).apply(lambda x: x[0].sum()).reset_index()

#statistics by type of GHG in each region
F_Y_sec_by_GHG_in_CO2.groupby(['region']).apply(lambda x: pd.Series({'CH4': x.loc[pd.IndexSlice[:, ['CH4'],:],:][0].sum()/x[0].sum(),'N2O': x.loc[pd.IndexSlice[:, ['N2O'],:],:][0].sum()/x[0].sum(), 'CO2': x.loc[pd.IndexSlice[:, ['CO2'],:],:][0].sum()/x[0].sum()}))

#this seems to be right: testing for country SI (chosen because low share of CO2)
#tot_N2O=np.sum(F_Y_sec_by_GHG[F_Y_sec_by_GHG['SubstanceName']=='N2O - combustion'].loc['SI',:]['Amount'])*298
#tot_CO2=np.sum(F_Y_sec_by_GHG[F_Y_sec_by_GHG['SubstanceName']=='CO2 - combustion'].loc['SI',:]['Amount'])
#tot_CH4=np.sum(F_Y_sec_by_GHG[F_Y_sec_by_GHG['SubstanceName']=='CH4 - combustion'].loc['SI',:]['Amount'])*25
#tot_CH4/(tot_CO2+tot_CH4+tot_N2O)
#tot_N2O/(tot_CO2+tot_CH4+tot_N2O)



#construction of F_Y_sec: to be finished
F_Y_sec_sparsed = F_Y_sec_all_GHG.pivot(index='sector',columns='region',values=0).fillna(0) #pivoting F_Y_sec_all and fill nan
F_Y_sec_to_fill = pd.DataFrame(index =corresp_code_sect['sector'],columns=F_Y_sec_sparsed.columns)
F_Y_sec_final = F_Y_sec_to_fill.combine_first(F_Y_sec_sparsed).fillna(0)

# ##########################
## Share  / repartition key to use in impacts
# ##########################
share_F_Y_sec = F_Y_sec_final.div(F_Y_sec_final.sum(axis=0))
share_F_Y_sec.to_pickle(DATA_PATH+'Share_F_Y_sec.pkl')
