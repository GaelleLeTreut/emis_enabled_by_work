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
import holoviews as hv
from holoviews import opts

exec(open("salaries_insee.py").read()) 

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

## update with emission contents / from salaries_insee.py
metropolitan_reg = reg_emis_cont.query("not(REGT_AR in [1,2,3,4,99,'All'])")
to_map = metropolitan_reg.set_index('REGT_AR_NAME')['mean emission content']
to_map_dic = to_map.to_dict()
#seems to be necessary in Windows only, maybe bypass or use try:
#to_map_dic['Île-de-France'] = to_map_dic.pop('ÃŽle-de-France')
#to_map_dic['Midi-Pyrénées'] = to_map_dic.pop('Midi-PyrÃ©nÃ©es')
#to_map_dic['Franche-Comté'] = to_map_dic.pop('Franche-ComtÃ©')
#to_map_dic['Rhône-Alpes'] = to_map_dic.pop('RhÃ´ne-Alpes')
#to_map_dic['Provence-Alpes-Côte d\'Azur'] = to_map_dic.pop('Provence-Alpes-CÃ´te d\'Azur')

sf.index = sf['nom']
sf['value']=sf['nom'].replace(to_map_dic)
deps = gv.Polygons(sf, vdims=['nom','value'])



## PLOT maps 
## Bbokeh extension
#from geoviews import dim
# carto = deps.opts(width=600, height=600, toolbar='above', color=dim('value'),
#           colorbar=True, tools=['hover'], aspect='equal')

### Matplotib extension
## Variable to plot (update)
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
plt.title("Emission content of incomes - gCO2/euro", size=12)
plt.savefig(OUTPUTS_PATH+'map_emis_cont_by_reg.jpeg', bbox_inches='tight')
