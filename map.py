# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
import pandas as pd
import os
import geopandas as gdp
import geoviews as gv
import matplotlib.pyplot as plt
import seaborn as sns
import utils as ut
import statsmodels.api as sm
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews import opts
import utils as ut
import build_table_survey as bts

path = os.getcwd()
data_folder='data'
DATA_PATH = path + os.sep + data_folder + os.sep

### Examples
## for bokeh extension (interactive map)
## https://thedatafrog.com/fr/articles/choropleth-maps-python/
## for matplotlib extension
#https://towardsdatascience.com/lets-make-a-map-using-geopandas-pandas-and-matplotlib-to-make-a-chloropleth-map-dddc31c1983d

gv.extension('matplotlib')
#gv.extension('bokeh')

mapname = "France_regions.geojson"
sf = gdp.read_file(DATA_PATH+mapname)

## full_insee_table comes from import build_table_survey.py
# Indice de vunérabilité région
dict_codereg_to_regname= {1:'Guadeloupe',2:'Martinique',3:'Guyane',4:'Réunion',11:'Île-de-France',21:'Champagne-Ardenne',22:'Picardie',23:'Haute-Normandie',24:'Centre',25:'Basse-Normandie',26:'Bourgogne',31:'Nord-Pas-de-Calais',41:'Lorraine',42:'Alsace',43:'Franche-Comté',52:'Pays de la Loire',53:'Bretagne',54:'Poitou-Charentes',72:'Aquitaine',73:'Midi-Pyrénées',74:'Limousin',82:'Rhône-Alpes',83:'Auvergne',91:'Languedoc-Roussillon',93:'Provence-Alpes-Côte d\'Azur',94:'Corse',99:'Etranger et Tom'}

reg_emis_cont = ut.stat_data_generic(['REGT_AR'], bts.full_insee_table.dropna(subset=['REGT_AR']), ut.mean_emission_content)
reg_emis_cont['REGT_AR_NAME']=reg_emis_cont['REGT_AR'].replace(dict_codereg_to_regname)


#building table
raw_table = ut.stat_data_generic(['TRNNETO','A35','SEXE'],bts.full_insee_table, ut.mean_emission_content) 
ordered_table = raw_table.pivot_table(index=['TRNNETO','A35'],columns=['SEXE'],values='pop_mass').reset_index()
# cleaning labels
ordered_table.columns.name = 'index'
ordered_table.rename({1:'male_pop',2:'female_pop','All':'total_pop'},axis=1,inplace=True)
#add emission content
to_merge= raw_table[(raw_table['SEXE']=='All')][['TRNNETO','A35','mean emission content']]
final_table = pd.merge(ordered_table,to_merge,on=['TRNNETO','A35'],how='left')

# Tableau mean emission content by branch  
Mean_emis_branch = pd.DataFrame(to_merge.loc[to_merge['TRNNETO']=='All'][['A35','mean emission content']])
Mean_emis_branch.index =Mean_emis_branch['A35']
Mean_emis_branch.drop('A35', axis=1,inplace=True)


############################################################
####Plot Emission of France Regions
############################################################
#same plot for region (mean emission content interpreted as vulnerability index)
plt.figure(figsize=(20, 12))
sns.barplot(x=reg_emis_cont['REGT_AR_NAME'], y="mean emission content", data=reg_emis_cont,palette='deep')
plt.xlabel("Regions", size=12)
plt.xticks(rotation=90)
plt.ylabel("Vulnerability index (gCO2/euro)", size=12)
plt.title("Vulnerability index by regions", size=12)
plt.savefig(bts.OUTPUTS_PATH+'fig_mean_emis_cont_by_region.jpeg', bbox_inches='tight')

############################################################
####Plot MAP of Metropolitain France
############################################################
metropolitan_reg = reg_emis_cont.query("not(REGT_AR in [1,2,3,4,99,'All'])")
to_map = metropolitan_reg.set_index('REGT_AR_NAME')['mean emission content']
to_map_dic = to_map.to_dict()

sf.index = sf['nom']
sf['value']=sf['nom'].replace(to_map_dic)
deps = gv.Polygons(sf, vdims=['nom','value'])

## Bbokeh extension
#from geoviews import dim
# carto = deps.opts(width=600, height=600, toolbar='above', color=dim('value'),
#           colorbar=True, tools=['hover'], aspect='equal')

### Matplotib extension
variable = 'value'
# set the range for the choropleth
vmin, vmax = sf['value'].min(), sf['value'].max()
fig, ax = plt.subplots(1, figsize=(10, 6))
ax.axis('off')
sf.plot(column=variable, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')
# Create colorbar as a legend
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# empty array for the data range
sm._A = []
# add the colorbar to the figure
cbar = fig.colorbar(sm)
# add title
#plt.title("Emission content of incomes - gCO2/euro", size=12)
plt.savefig(bts.OUTPUTS_PATH+'map_emis_cont_by_reg.jpeg', bbox_inches='tight')

############################################################
####Plot Pannel of All region with relative population in each industries
############################################################
#sur chaque ligne, pour une population caractérisée par sa classe salariale et son sexe, on a la liste des proportions employées dans les différentes secteurs
relative_pop_reg = ut.stat_data_generic(['REGT_AR'],bts.full_insee_table, lambda x: ut.proportion_generic(x,'A35'))
relative_pop_reg = relative_pop_reg.fillna(0)
relative_pop_reg.set_index('REGT_AR',drop=True, inplace=True)
relative_pop_reg.columns.names=['A35']


table_relative_pop_reg  = pd.DataFrame(relative_pop_reg.stack())
table_relative_pop_reg.columns=['Relative pop by branch']
table_relative_pop_reg['mean emission content']=None
table_relative_pop_reg.index = table_relative_pop_reg.index.swaplevel(0, 1)
table_relative_pop_reg.sort_index(level=[0,1], axis=0, inplace=True)
table_relative_pop_reg.reset_index(inplace=True)
# boucle sur branch -fill table with mean emission content values
for r in Mean_emis_branch.drop('All').index.unique():
     table_relative_pop_reg.loc[table_relative_pop_reg['A35']==r,'mean emission content'] = np.repeat(Mean_emis_branch.loc[[r]].values, len(table_relative_pop_reg.loc[table_relative_pop_reg['A35']==r,'mean emission content']))
# remove Graph REGT_AR = all
table_relative_pop_reg.drop(table_relative_pop_reg.loc[table_relative_pop_reg['REGT_AR']=='All'].index, inplace=True)
# Rename regions
table_relative_pop_reg['REGT_AR']=table_relative_pop_reg['REGT_AR'].replace(dict_codereg_to_regname)
table_relative_pop_reg

region_4groups =pd.DataFrame(index=table_relative_pop_reg['REGT_AR'].unique(),columns=['top3','top','middle top','middle low', 'bottom'])
for r in list(table_relative_pop_reg['REGT_AR'].unique()):
    region_r = table_relative_pop_reg.loc[table_relative_pop_reg['REGT_AR']==r,:]
    region_4groups.loc[r,'top3'] = round(np.sum(region_r.sort_values(by=['mean emission content'], ascending = False )[:3]['Relative pop by branch']),1)
    region_4groups.loc[r,'top'] = round(np.sum(region_r.sort_values(by=['mean emission content'], ascending = False )[:10]['Relative pop by branch']),1)
    region_4groups.loc[r,'middle top'] = round(np.sum(region_r.sort_values(by=['mean emission content'], ascending = False )[10:19]['Relative pop by branch']),1)
    region_4groups.loc[r,'middle low'] = round(np.sum(region_r.sort_values(by=['mean emission content'], ascending = False )[19:27]['Relative pop by branch']),1)
    region_4groups.loc[r,'bottom'] = round(np.sum(region_r.sort_values(by=['mean emission content'], ascending = False )[-10:]['Relative pop by branch']),1)
## rearrange from region with higher emission content to lower
region_4groups = region_4groups.reindex(reg_emis_cont.sort_values('mean emission content', ascending = False)['REGT_AR_NAME'])


## Plots for all region
Tot = len(list(table_relative_pop_reg['REGT_AR'].unique()))
Cols = 3
Rows = Tot // Cols 

# sharey=True pour même scale sur y
fig, axes = plt.subplots(nrows=Rows, ncols=Cols, figsize=(40, 30),sharey=True)
#fig, axes = plt.subplots(nrows=Rows, ncols=Cols, figsize=(40, 30))
i = 0
for row in axes:
    for ax1 in row:
        r = list(reg_emis_cont.sort_values('mean emission content',ascending=False).drop(reg_emis_cont.loc[reg_emis_cont['REGT_AR_NAME']=='All'].index)['REGT_AR_NAME'].unique())[i]
        region_r = table_relative_pop_reg.loc[table_relative_pop_reg['REGT_AR']==r,:]
        region_r_raw= region_r.drop('REGT_AR',axis=1).sort_values(by='mean emission content')  
        #ax2 = ax1.twinx() # applies twinx to ax2, which is the second y axis. 
        sns.barplot(x='A35', y="Relative pop by branch", data=region_r_raw,palette='deep',ax=ax1) # plots the first set of data, and sets it to ax1. 
        #sns.scatterplot(x ='A35', y ='mean emission content', data=region_r_raw, marker='o', ax = ax2, color="firebrick", s=80) # plots the second set, and sets to ax2. 
        # these lines add the annotations for the plot. 
        #ax1.set_xlabel('branches')
        ax1.set_ylabel('pop (%)')
        #ax2.set_ylabel('emission content in gCO2/euro', size=14)
        ax1.set_title(str(r))
        i += 1
plt.tight_layout()
plt.savefig(bts.OUTPUTS_PATH+'fig_reg_panel.jpeg', bbox_inches='tight')


