# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
import pandas as pd
import pymrio
import os


##########################
###### Paths 
##########################
# récupérer le chemin du répertoire courant
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


##########################
###### Chargement de la base MRIO
##########################
###### Chargement de la base EORA
#eora_storage = DATA_PATH+'Eora26_2015_bp'
#io_orig = pymrio.parse_eora26(year=2015, path=eora_storage)

###### Chargement de la base EXIOBASE
## Donné par Antoine
exiobase_storage = DATA_PATH + 'IOT_2015_basic.zip'
#maybe unnecessary to get the full path: if so when could keep
#exiobase_storage = data_folder + os.sep + 'IOT_2015_basic.zip'
## Download from exiobase Zenodo
##exiobase_storage = DATA_PATH+'IOT_2015_pxp.zip'

io_orig =  pymrio.parse_exiobase3(exiobase_storage)

## chargement des valeurs de base
io_orig.calc_all()


##########################
###### AGGREGATION PRE-CALCULATION
##########################
# Correspondance matrix from 200 sectos to A35 (3 sectors are not described) 
sec_agg_matrix = np.array([
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
])
#io_orig.aggregate(sector_agg=sec_agg_matrix)

# Correspondance au plus proche de A38 ( 35 sectors)
corresp_table = pd.read_csv(DATA_PATH + 'exiobase_A38.csv', comment='#',header=[0,1], index_col=0, sep=';')
sec_label = ['sector label', np.array(corresp_table.columns.get_level_values(0))]
sec_name = ['sector name', np.array(corresp_table.columns.get_level_values(1))]

io_orig.aggregate(sector_agg=np.transpose(corresp_table.values),sector_names = list(corresp_table.columns.get_level_values(0)))
#io_new = io_orig.aggregate(sector_agg=np.transpose(corresp_table.values),sector_names = list(corresp_table.columns),inplace=False)

## chargement des valeurs de base
#io_new.calc_all()

##########################
###### New calculations
##########################
##Le calcul de G et B n'est pas fait par la fonction calc_all
io_orig.B = pymrio.calc_B(io_orig.Z,io_orig.x)
io_orig.G = pymrio.calc_G(io_orig.B)

##### NOTE 
##Pour récupérer les régions et secteurs du MRIO
# io_orig.get_regions()
#io_orig.get_sectors()
## Aggrégation (prédéfini) une région pour aller plus vite dans le calcul
#io_orig = io_orig.aggregate(region_agg = 'global')


##########################
###### CHECKING - USES -SUPPLY Balance 
##########################
## -> Checking Z + Y == X 
##USES
x_recalcU = np.hstack(( io_orig.Z.values, io_orig.Y.values))
x_consistU = np.sum(x_recalcU, -1) - np.sum(io_orig.x.values,-1)
## SUPPLY
## -> verifier que sur les lignes on retrouve le X: Z + VA == X
x_recalcR = np.vstack((io_orig.value_added.F.values, io_orig.Z.values))
x_consistR = np.sum(x_recalcR, 0) - np.transpose(io_orig.x.values)
# => doit faire 0... 

### a refaire avec io_new
#x_recalcU = np.hstack(( io_orig.Z.values, io_orig.Y.values))
#x_consistU = np.sum(x_recalcU, -1) - np.sum(io_orig.x.values,-1)
## SUPPLY
## -> verifier que sur les lignes on retrouve le X: Z + VA == X
#x_recalcR = np.vstack((io_orig.value_added.F.values, io_orig.Z.values))
#x_consistR = np.sum(x_recalcR, 0) - np.transpose(io_orig.x.values)

##########################
###### CHECKING - Ghosh Account
##########################
## 1) From B:  X = BX + VA
## Checking calculation of Z from B
ZrecalcB =pymrio.calc_Z_from_B( io_orig.B ,io_orig.x)
consist_B =ZrecalcB - io_orig.Z
x_recalcGI = np.vstack(( ZrecalcB.values, io_orig.value_added.F.values))
x_consistGI = np.sum(x_recalcGI,0) - np.sum(io_orig.x.values,-1)
## 2) From G: X = G*VA
GVA = io_orig.G.values.dot(np.transpose(io_orig.value_added.F.values))
x_consistGII = np.sum(GVA,-1)-np.sum(io_orig.x.values, -1)


##########################
###### CHECKING - Emissions CO2 Energy - Magnitude
##########################
emis_comb_sec = sum(io_orig.CO2_emissions.F.loc['combustion'])
emis_comb_direct = sum(io_orig.CO2_emissions.F_Y.loc['combustion'])
# unit kg exiobase ; 1e12 for GtCO2
emis_comb_tot = (emis_comb_sec + emis_comb_direct)*1E-12
## =>  31.5 GtCO2
### Emissions France
emis_comb_sec_FR = sum( io_orig.CO2_emissions.F.loc['combustion','FR'])
emis_comb_direct_FR = sum(io_orig.CO2_emissions.F_Y.loc['combustion','FR'])
# unit kg exiobase ; 1e9 for MtCO2
emis_comb_tot_FR =  (emis_comb_sec_FR +emis_comb_direct_FR)*1E-9


##########################
###### INCOME BASED EMISSIONS AND EMISSION CONTENTS
##########################
##  from combustion only - in kgCO2
#income_based_emis_tot = (io_orig.CO2_emissions.S.loc['combustion'].dot(io_orig.G)).dot(np.transpose(io_orig.value_added.F))
#income_based_emis_tot_Gt = ((io_orig.CO2_emissions.S.loc['combustion'].dot(io_orig.G)).dot(np.transpose(io_orig.value_added.F)))*1e-12
#io_orig.inc_emis_content = pymrio.calc_iec(io_orig.CO2_emissions.S.loc['combustion'], io_orig.G)
## unit gCO2/euro for emissions content and income based emissions in tCO2
#io_orig.inc_emis_content = io_orig.inc_emis_content*1e-3
#io_orig.income_based_emis =  pymrio.calc_ibe(io_orig.inc_emis_content,io_orig.value_added.F)

##  from CO2 emissions - in kgCO2
## New line with Total 
io_orig.CO2_emissions.F.loc['Total',:]= io_orig.CO2_emissions.F.sum(axis=0)
io_orig.CO2_emissions.F_Y.loc['Total',:]= io_orig.CO2_emissions.F_Y.sum(axis=0)

### Calculate S for total CO2 emissions... 
io_orig.CO2_emissions.S = pymrio.calc_S(io_orig.CO2_emissions.F,io_orig.x)
income_based_emis_tot = (io_orig.CO2_emissions.S.loc['Total'].dot(io_orig.G)).dot(np.transpose(io_orig.value_added.F))
income_based_emis_tot_Gt = ((io_orig.CO2_emissions.S.loc['Total'].dot(io_orig.G)).dot(np.transpose(io_orig.value_added.F)))*1e-12

io_orig.inc_emis_content = pymrio.calc_iec(io_orig.CO2_emissions.S.loc['Total'], io_orig.G)
## unit gCO2/euro for emissions content and income based emissions in tCO2
io_orig.inc_emis_content = io_orig.inc_emis_content*1e-3
io_orig.income_based_emis =  pymrio.calc_ibe(io_orig.inc_emis_content,io_orig.value_added.F)

## Décomposition  Emission  content (in gCO2/euro)  = S + SB + SB^2 + ..
io_orig.inc_emis_content_direct = io_orig.CO2_emissions.S.loc['Total']*1e-3
io_orig.inc_emis_content_fo = np.transpose(pymrio.calc_iec(io_orig.CO2_emissions.S.loc['Total'],io_orig.B)*1e-3)
io_orig.inc_emis_content_so =  np.transpose(pymrio.calc_iec(io_orig.CO2_emissions.S.loc['Total'],io_orig.B.dot(io_orig.B))*1e-3)

## FRANCE only 
emission_content_FR = np.transpose(io_orig.inc_emis_content).loc[['FR']]
income_based_emis_FR = np.transpose(io_orig.income_based_emis).loc[['FR']]


##########################
###### AGGREGATION POST CALCULATION
##########################
#io_orig.aggregate(sector_agg=np.transpose(corresp_table.values))
### IEC and Income Based emissions must be aggregated (not included in the aggregate function)
## corresp_table pour aggréger Income based emissions io_orig.income_based_emis  
### io_orig.inc_emis_content = io_orig.income_based_emis / io_orig.value_added.F

##########################
###### SAVE FILES
##########################
##1) Excel files
io_orig.CO2_emissions.F_Y.to_excel(OUTPUTS_PATH+'CO2emisDirect.xlsx')
io_orig.CO2_emissions.F.to_excel(OUTPUTS_PATH+'CO2emisProd.xlsx')
io_orig.value_added.F.to_excel(OUTPUTS_PATH+'VA.xlsx')
io_orig.Z.to_excel(OUTPUTS_PATH+'IC.xlsx')
io_orig.Y.to_excel(OUTPUTS_PATH+'FD.xlsx')
io_orig.income_based_emis.to_excel(OUTPUTS_PATH+'IncBasedEmis.xlsx')
io_orig.inc_emis_content.to_excel(OUTPUTS_PATH+'inc_emis_content.xlsx')
emission_content_FR.to_excel(OUTPUTS_PATH+'inc_emis_content_FR.xlsx')
income_based_emis_FR.to_excel(OUTPUTS_PATH+'inc_based_emis_FR.xlsx')
##1) CSV files ?

##########################
###### PLOTS
##########################
import matplotlib.pyplot as plt
## Ne fonctionne pas
#io_orig.Z.plot_account('FR')
#plt.show()

# with plt.style.context('ggplot'):
#      io_orig.Z.plot_account(['Total cropland area'], figsize=(15,10))
#      plt.show()

# with plt.style.context('ggplot'):
#     eora_agg_DEU_EU_OECD.Q.plot_account(('Total cropland area', 'Total'), figsize=(8,5))
#     plt.show()



##########################
###### TO DO
##########################
## Gérer une sauvegarde? 
## voir les correspondances pour agrégation secteurs/pays


