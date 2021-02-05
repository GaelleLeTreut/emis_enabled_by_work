# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:50:21 2021

@author: GaelleLeTreut
"""
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

### Repartition de la combustion des ménages MtCO2 (désagrégation suivant coefficient emissions IMACLIM2010)
F_Y_sec = pd.read_csv(DATA_PATH + 'F_Y_sec.txt', header=[0,1], index_col=0, sep="\t")

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

# Correspondance au plus proche de A38 ( 35 sectors)
corresp_table = pd.read_csv(DATA_PATH + 'exiobase_A38.csv', comment='#',header=[0,1], index_col=0, sep=';')
sec_label = ['sector label', np.array(corresp_table.columns.get_level_values(0))]
sec_name = ['sector name', np.array(corresp_table.columns.get_level_values(1))]
corresp_table=np.transpose(corresp_table)

#sector_agg = sec_agg_matrix
#io_orig.aggregate(sector_agg=sector_agg)
sector_agg = corresp_table.values
io_orig.aggregate(sector_agg=sector_agg,sector_names = list(corresp_table.index.get_level_values(0)))
#io_new = io_orig.aggregate(sector_agg=np.transpose(corresp_table.values),sector_names = list(corresp_table.columns),inplace=False)

## Agrégation des emissions directes
F_Y_sec = np.transpose(F_Y_sec).dot(np.transpose(sector_agg))
F_Y_sec.columns = io_orig.get_sectors()

## chargement des valeurs de base
#io_new.calc_all()

### List of regions
region_list = list(io_orig.x.index.levels[0])


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

##  Emissions - CO2 emissions - in kgCO2 (F)
## New line with Total of emissions 
io_orig.CO2_emissions.F.loc['Total',:]= io_orig.CO2_emissions.F.sum(axis=0)
# Direct emissions (from combusition of HH) - in kgCO2 (orginal data in Mt-harmonisation with F data)
io_orig.CO2_emissions.F_d =F_Y_sec.sum(level=[0])
io_orig.CO2_emissions.F_d = np.transpose(io_orig.CO2_emissions.F_d.stack().to_frame())*1e9
io_orig.CO2_emissions.F_d.index.name='FD emissions'

####
## Emission intensity (S) and emission contents
####
### Calculate S for total CO2 emissions...(with or without direct emissions) 
io_orig.CO2_emissions.S = pymrio.calc_S(io_orig.CO2_emissions.F,io_orig.x)
io_orig.CO2_emissions.S_d =  pymrio.calc_S(io_orig.CO2_emissions.F_d,io_orig.x)
io_orig.CO2_emissions.S_tot = io_orig.CO2_emissions.S.loc['Total'].to_frame()+np.transpose(io_orig.CO2_emissions.S_d.values)

## Emission contents without direct emissions - unit gCO2/euro 
io_orig.inc_emis_content = pymrio.calc_iec(io_orig.CO2_emissions.S.loc['Total'], io_orig.G)*1e-3
### Emission contents included direct emissions - unit gCO2/euro 
io_orig.inc_emis_content_tot = pymrio.calc_iec(io_orig.CO2_emissions.S_tot, io_orig.G)*1e-3

####
## Income based emissions (tCO2)
####
income_based_emis_tot = (io_orig.CO2_emissions.S.loc['Total'].dot(io_orig.G)).dot(np.transpose(io_orig.value_added.F))
income_based_emis_tot_Gt = ((io_orig.CO2_emissions.S.loc['Total'].dot(io_orig.G)).dot(np.transpose(io_orig.value_added.F)))*1e-12
## without direct emissions 
io_orig.income_based_emis =  pymrio.calc_ibe(io_orig.inc_emis_content,io_orig.value_added.F)
## with direct emissions 
io_orig.income_based_emis_tot =  pymrio.calc_ibe(io_orig.inc_emis_content_tot,io_orig.value_added.F)

emis_cont = io_orig.CO2_emissions.S.loc['Total']
emis_cont_tot = io_orig.CO2_emissions.S.loc['Total']+io_orig.CO2_emissions.S_d

##########################
###### WITHOUT DIRECT EMISSIONS 
##########################

######
### Décomposition  Emission  content (in gCO2/euro)  = S + SB + SB^2 + ..
######
io_orig.inc_emis_content_direct = pd.DataFrame(io_orig.CO2_emissions.S.loc['Total']*1e-3)
io_orig.inc_emis_content_direct.rename(columns={'Total': 'Direct emis content'}, inplace=True)
io_orig.inc_emis_content_fo = np.transpose(pymrio.calc_iec(io_orig.CO2_emissions.S.loc['Total'],io_orig.B)*1e-3)
io_orig.inc_emis_content_fo.rename(columns={'emission content': 'FO emis content'}, inplace=True)
io_orig.inc_emis_content_so =  np.transpose(pymrio.calc_iec(io_orig.CO2_emissions.S.loc['Total'],io_orig.B.dot(io_orig.B))*1e-3)
io_orig.inc_emis_content_so.rename(columns={'emission content': 'SO emis content'}, inplace=True)

## FRANCE only 
emission_content_FR = np.transpose(io_orig.inc_emis_content).loc[['FR']]
income_based_emis_FR = np.transpose(io_orig.income_based_emis).loc[['FR']]

######
### Decomposition enabled emissions (in gCO2/euro)
######
emis_enable = io_orig.inc_emis_content.copy()
#emis_enable[emis_enable!=0]=0
for r in region_list:
    rest_list = list(region_list)
    rest_list.remove(r)
    
    S_enable = io_orig.CO2_emissions.S.loc['Total']*1e-3
    S_enable.loc[rest_list]=0
    emis_enable_r = pymrio.calc_iec(S_enable, io_orig.G)
    emis_enable_r.rename(index={'emission content':'emis content from '+r},inplace=True)
    emis_enable = emis_enable.append(emis_enable_r)
    del rest_list
emis_enable = emis_enable.drop(['emission content'])

##########################
###### WITH DIRECT EMISSIONS /// UPDATE ALL SECTION
##########################

######
### Décomposition  Emission  content (in gCO2/euro)  = S + SB + SB^2 + ..
######
io_orig.inc_emis_content_tot_direct = pd.DataFrame(io_orig.CO2_emissions.S_tot*1e-3)
io_orig.inc_emis_content_tot_direct.rename(columns={'Total': 'Direct emis content'}, inplace=True)
io_orig.inc_emis_content_tot_fo = np.transpose(pymrio.calc_iec(io_orig.CO2_emissions.S_tot,io_orig.B)*1e-3)
io_orig.inc_emis_content_tot_fo.rename(columns={'emission content': 'FO emis content'}, inplace=True)
io_orig.inc_emis_content_tot_so =  np.transpose(pymrio.calc_iec(io_orig.CO2_emissions.S_tot,io_orig.B.dot(io_orig.B))*1e-3)
io_orig.inc_emis_content_tot_so.rename(columns={'emission content': 'SO emis content'}, inplace=True)

## FRANCE only 
emission_content_tot_FR = np.transpose(io_orig.inc_emis_content_tot).loc[['FR']]
income_based_emis_tot_FR = np.transpose(io_orig.income_based_emis_tot).loc[['FR']]

######
### Decomposition enabled emissions (in gCO2/euro)
######
emis_enable_d = io_orig.inc_emis_content_tot.copy()
#emis_enable[emis_enable!=0]=0
for r in region_list:
    rest_list = list(region_list)
    rest_list.remove(r)
    
    S_d_enable = io_orig.CO2_emissions.S_tot*1e-3
    S_d_enable.loc[rest_list]=0
    emis_enable_r = pymrio.calc_iec(S_d_enable, io_orig.G)
    emis_enable_r.rename(index={'emission content':'emis content from '+r},inplace=True)
    emis_enable_d = emis_enable_d.append(emis_enable_r)
    del rest_list
emis_enable_d = emis_enable_d.drop(['emission content'])


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
#emission_content_FR.to_excel(OUTPUTS_PATH+'inc_emis_content_FR.xlsx')
income_based_emis_FR.to_excel(OUTPUTS_PATH+'inc_based_emis_FR.xlsx')
#io_orig.inc_emis_content_direct.to_excel(OUTPUTS_PATH+'S.xlsx')
#io_orig.G.to_excel(OUTPUTS_PATH+'G.xlsx')

##########################
###### PLOTS - WHITHOUT DIRECT EMISSIONS
##########################
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('paper', font_scale=0.9)

######
### conversion d'un dataframe multi index en dataframe 'simple' pour l'utiliser plus facilement dans seaborn
######
emis_cont = np.transpose(io_orig.inc_emis_content)
check=emis_cont.reset_index(inplace=True)

emis_cont_dir = io_orig.inc_emis_content_direct.copy()
check=emis_cont_dir.reset_index(inplace=True)

emis_cont_fo= io_orig.inc_emis_content_fo.copy()
check=emis_cont_fo.reset_index(inplace=True)

emis_cont_so= io_orig.inc_emis_content_so.copy()
check=emis_cont_so.reset_index(inplace=True)

emis_enable=np.transpose(emis_enable)
check=emis_enable.reset_index(inplace=True)

######
### Emission content - Histogramme groupé FR vs ROW
######
plt.figure(figsize=(18, 12))
sns.barplot(x="sector", hue="region", y="emission content", data=emis_cont)
plt.xlabel("Sector code", size=12)
plt.ylabel("gCO2/euro", size=12)
plt.title("Emission content - France vs Rest of World", size=12)
plt.savefig(OUTPUTS_PATH+'fig_emis_cont_FRvsRoW.jpeg', bbox_inches='tight')
plt.show()

######
### Emission content - Histogramme FRANCE
######
### Emissions content by sector
emis_cont_fr= emis_cont.loc[emis_cont['region']=='FR']
plt.figure(figsize=(18, 12))
sns.barplot(x="sector", y="emission content", data=emis_cont_fr,palette='deep')
plt.xlabel("Sector code", size=12)
plt.ylabel("gCO2/euro", size=12)
plt.title("Emission content - France", size=12)
plt.savefig(OUTPUTS_PATH+'fig_emis_cont_FR.jpeg', bbox_inches='tight')
plt.show()
plt.close()


### Emissions content = decompostion
emis_cont_decomp= emis_cont_dir.copy()
emis_cont_decomp.loc[:,'FO emis content'] = emis_cont_fo['FO emis content']
emis_cont_decomp.loc[:,'SO emis content'] = emis_cont_so['SO emis content']
emis_cont_decomp.loc[:,'Rest emis content'] =emis_cont['emission content'] - ( emis_cont_dir['Direct emis content'] + emis_cont_fo['FO emis content'] + emis_cont_so['SO emis content']) 

emis_cont_decomp_fr= emis_cont_decomp.loc[emis_cont_decomp['region']=='FR']
emis_cont_decomp_fr= emis_cont_decomp_fr.drop(['region'], axis=1)
emis_cont_decomp_fr = np.transpose(emis_cont_decomp_fr)
emis_cont_decomp_fr.columns = emis_cont_decomp_fr.loc['sector']
emis_cont_decomp_fr= emis_cont_decomp_fr.drop(['sector'], axis=0)

#sns.set()
emis_cont_decomp_fr.T.plot(kind='bar', stacked=True, figsize=(18, 12))
plt.xlabel("Sector code", size=12)
plt.xticks(rotation=0,fontsize=12)
plt.ylabel("gCO2/euro", size=12)
plt.title("Emission content decomposition- France", size=12)
plt.savefig(OUTPUTS_PATH+'fig_emis_cont_decomp_FR.jpeg', bbox_inches='tight')
plt.show()

######
### Emission content - Enable emissions
######
for r in region_list:
    emis_enable_r= emis_enable.loc[emis_cont['region']==r]
    emis_enable_r= emis_enable_r.drop(['region'], axis=1)
    emis_enable_r = np.transpose(emis_enable_r)
    emis_enable_r.columns = emis_enable_r.loc['sector']
    emis_enable_r= emis_enable_r.drop(['sector'], axis=0)

    emis_enable_r.T.plot(kind='bar', stacked=True, figsize=(18, 12))
    plt.xlabel("Sector code", size=12)
    plt.xticks(rotation=0,fontsize=12)
    plt.ylabel("gCO2/euro", size=12)
    plt.title("Enabled emission decomposition - "+r, size=12)
    plt.savefig(OUTPUTS_PATH+'fig_emis_cont_enabled_'+r+'.jpeg', bbox_inches='tight')
    plt.show()

##########################
###### PLOTS - WITH DIRECT EMISSIONS
##########################
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('paper', font_scale=0.9)

######
### conversion d'un dataframe multi index en dataframe 'simple' pour l'utiliser plus facilement dans seaborn
######
emis_cont_tot = np.transpose(io_orig.inc_emis_content_tot)
check=emis_cont_tot.reset_index(inplace=True)

emis_cont_tot_dir = io_orig.inc_emis_content_tot_direct.copy()
check=emis_cont_tot_dir.reset_index(inplace=True)

emis_cont_tot_fo= io_orig.inc_emis_content_tot_fo.copy()
check=emis_cont_tot_fo.reset_index(inplace=True)

emis_cont_tot_so= io_orig.inc_emis_content_tot_so.copy()
check=emis_cont_tot_so.reset_index(inplace=True)

emis_enable_d=np.transpose(emis_enable_d)
check=emis_enable_d.reset_index(inplace=True)

######
### Emission content - Histogramme groupé FR vs ROW
######
plt.figure(figsize=(18, 12))
sns.barplot(x="sector", hue="region", y="emission content", data=emis_cont_tot)
plt.xlabel("Sector code", size=12)
plt.ylabel("gCO2/euro", size=12)
plt.title("Total emission content - France vs Rest of World", size=12)
plt.savefig(OUTPUTS_PATH+'fig_emis_cont_FRvsRoW_tot.jpeg', bbox_inches='tight')
plt.show()

######
### Emission content - Histogramme FRANCE
######
### Emissions content by sector
emis_cont_tot_fr= emis_cont_tot.loc[emis_cont_tot['region']=='FR']
plt.figure(figsize=(18, 12))
sns.barplot(x="sector", y="emission content", data=emis_cont_tot_fr,palette='deep')
plt.xlabel("Sector code", size=12)
plt.ylabel("gCO2/euro", size=12)
plt.title("Total emission content - France", size=12)
plt.savefig(OUTPUTS_PATH+'fig_emis_cont_FR_tot.jpeg', bbox_inches='tight')
plt.show()
plt.close()


### Emissions content = decompostion
emis_cont_tot_decomp= emis_cont_tot_dir.copy()
emis_cont_tot_decomp.loc[:,'FO emis content'] = emis_cont_tot_fo['FO emis content']
emis_cont_tot_decomp.loc[:,'SO emis content'] = emis_cont_tot_so['SO emis content']
emis_cont_tot_decomp.loc[:,'Rest emis content'] =emis_cont_tot['emission content'] - ( emis_cont_tot_dir['Direct emis content'] + emis_cont_tot_fo['FO emis content'] + emis_cont_tot_so['SO emis content']) 

emis_cont_tot_decomp_fr= emis_cont_tot_decomp.loc[emis_cont_tot_decomp['region']=='FR']
emis_cont_tot_decomp_fr= emis_cont_tot_decomp_fr.drop(['region'], axis=1)
emis_cont_tot_decomp_fr = np.transpose(emis_cont_tot_decomp_fr)
emis_cont_tot_decomp_fr.columns = emis_cont_tot_decomp_fr.loc['sector']
emis_cont_tot_decomp_fr= emis_cont_tot_decomp_fr.drop(['sector'], axis=0)

#sns.set()
emis_cont_tot_decomp_fr.T.plot(kind='bar', stacked=True, figsize=(18, 12))
plt.xlabel("Sector code", size=12)
plt.xticks(rotation=0,fontsize=12)
plt.ylabel("gCO2/euro", size=12)
plt.title("Total emission content decomposition- France", size=12)
plt.savefig(OUTPUTS_PATH+'fig_emis_cont_decomp_FR_tot.jpeg', bbox_inches='tight')
plt.show()

######
### Emission content - Enable emissions
######
for r in region_list:
    emis_enable_r= emis_enable_d.loc[emis_cont['region']==r]
    emis_enable_r= emis_enable_r.drop(['region'], axis=1)
    emis_enable_r = np.transpose(emis_enable_r)
    emis_enable_r.columns = emis_enable_r.loc['sector']
    emis_enable_r= emis_enable_r.drop(['sector'], axis=0)

    emis_enable_r.T.plot(kind='bar', stacked=True, figsize=(18, 12))
    plt.xlabel("Sector code", size=12)
    plt.xticks(rotation=0,fontsize=12)
    plt.ylabel("gCO2/euro", size=12)
    plt.title("Total enabled emission decomposition - "+r, size=12)
    plt.savefig(OUTPUTS_PATH+'fig_emis_cont_enabled_'+r+'_tot.jpeg', bbox_inches='tight')
    plt.show()
    
## table to save
share_direct_emis = pd.DataFrame(emis_cont_tot_decomp_fr.loc['Direct emis content'].div(emis_cont_tot_fr['emission content'].replace(0, np.nan).values)*100)
share_direct_emis = share_direct_emis.astype(float).round(1)
share_direct_emis.columns=['% direct emissions']

share_1stdownstream_emis = pd.DataFrame(emis_cont_tot_decomp_fr.loc['FO emis content'].div(emis_cont_tot_fr['emission content'].replace(0, np.nan).values)*100)
share_1stdownstream_emis = share_1stdownstream_emis.astype(float).round(1)
share_1stdownstream_emis.columns=['% downstream at first level']

share_dom_emis = pd.DataFrame(emis_enable_d.loc[emis_enable_d['region']=='FR']['emis content from FR'].div(emis_cont_tot_fr['emission content'].replace(0, np.nan).values)*100)
share_dom_emis =share_dom_emis.astype(float).round(1)
share_dom_emis.columns=['% domestic emissions']

emis_cont_fr_to_save = emis_cont_tot_fr.copy()
emis_cont_fr_to_save.drop('region',axis=1)
emis_cont_fr_to_save['% direct emissions']=  share_direct_emis['% direct emissions'].values
emis_cont_fr_to_save['% downstream at first level']=  share_1stdownstream_emis['% downstream at first level'].values
emis_cont_fr_to_save['% domestic emissions']=  share_dom_emis['% domestic emissions'].values
emis_cont_fr_to_save = emis_cont_fr_to_save.drop(['region'], axis=1)
emis_cont_fr_to_save = emis_cont_fr_to_save.sort_values(by=['emission content'], ascending = False )

top10_table= round(emis_cont_fr_to_save[:10],1).to_latex(index=False)
least10_table= round(emis_cont_fr_to_save[-11:],1).to_latex(index=False)

print(top10_table)
print(least10_table)

VA = pd.DataFrame(np.sum(io_orig.value_added.F,axis=0))
VA.columns=['value added']
VA['share of VA']= np.nan
VA.loc[('RoW')]['share of VA'] =100*( VA.loc[('RoW')]/ VA.loc[('RoW')].sum())
VA.loc[('FR')]['share of VA'] =100*( VA.loc[('FR')]/ VA.loc[('FR')].sum())

print('Min emission content in FR (gCO2/euro of VA):',round(emis_cont_tot_fr['emission content'].min()))
print('Max emission content in FR (gCO2/euro of VA):',round(emis_cont_tot_fr['emission content'].max()))
print('mean emission content weighted by VA in FR (gCO2/euro of VA):', round(np.average(emis_cont_tot_fr['emission content'], weights=VA.loc[('FR')]['share of VA'])))
print('Standard deviation of emission content in FR (gCO2/euro of VA):',round(emis_cont_tot_fr['emission content'].std()))

print('mean emission content weighted by VA in RoW (gCO2/euro of VA):', round(np.average(emis_cont_tot.loc[emis_cont_tot['region']=='RoW']['emission content'], weights=VA.loc[('RoW')]['share of VA'])))
print('Standard deviation of emission content in RoW (gCO2/euro of VA):', round(emis_cont_tot.loc[emis_cont_tot['region']=='RoW']['emission content'].std()))
