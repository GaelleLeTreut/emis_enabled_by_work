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
## Aggregation into 35 sectors (closest correspondance with A38 nomenclature)
# corresp_table = pd.read_csv(DATA_PATH + 'exiobase_A38.csv', comment='#',header=[0,1], index_col=0, sep=';')
# corresp_table=np.transpose(corresp_table)
# sector_agg = corresp_table.values

# ## Aggregation into two regions: FR and RoW
# region_table = pd.read_csv(DATA_PATH + 'exiobase_FRvsRoW.csv', comment='#', index_col=0, sep=';')
# region_table=np.transpose(region_table)
# region_agg = region_table.values

# io_orig.aggregate(region_agg=region_agg, sector_agg=sector_agg, region_names = list(region_table.index.get_level_values(0)), sector_names = list(corresp_table.index.get_level_values(0)))

# region_list = list(io_orig.get_regions()) 

##########################
###### CALCULATION
##########################

#then we could simply a calc_all, I split here to avoid calculations in the huge extensions
io_orig.calc_system()
io_orig.GHG_emissions.calc_system(x=io_orig.x, Y=io_orig.Y, L=io_orig.L, Y_agg=None, population=io_orig.population)
io_orig.GHG_emissions.calc_income_based(x = io_orig.x, V=io_orig.V, G=io_orig.G, V_agg=None, population=io_orig.population)
#compute zero
V_agg = io_orig.V.sum(level=0, axis=1, ).reindex(io_orig.get_regions(), axis=1)
io_orig.GHG_emissions.D_iba_zero_order = pymrio.tools.iomath.calc_D_iba(io_orig.GHG_emissions.S, pd.DataFrame(np.identity(np.shape(io_orig.G.values)[0])), V_agg, io_orig.get_sectors().size)
io_orig.GHG_emissions.D_iba_first_order = pymrio.tools.iomath.calc_D_iba(io_orig.GHG_emissions.S, io_orig.B, V_agg, io_orig.get_sectors().size)
#this has to be checked if correct


##########################
###### AGGREGATION POST CALCULATION
##########################
## Aggregation into 35 sectors (closest correspondance with A38 nomenclature)
corresp_table = pd.read_csv(DATA_PATH + 'exiobase_A38.csv', comment='#',header=[0,1], index_col=0, sep=';')
corresp_table=np.transpose(corresp_table)
sector_agg = corresp_table.values

## Aggregation into two regions: FR and RoW
region_table = pd.read_csv(DATA_PATH + 'exiobase_FRvsRoW.csv', comment='#', index_col=0, sep=';')
region_table=np.transpose(region_table)
region_agg = region_table.values

io_orig.aggregate(region_agg=region_agg, sector_agg=sector_agg, region_names = list(region_table.index.get_level_values(0)), sector_names = list(corresp_table.index.get_level_values(0)))

region_list = list(io_orig.get_regions())

##### GLT  TO check
## Do we really need to put here a calc all to reassess the variables that are not aggregated? A, B L and G are not calculted...
#io_orig.calc_all()


##########################
###### Emission content and decomposition
##########################
#### 10¨3gco2/10¨6euro =  X gCO2/keuro => X * 1e-3 gCO2/e 
######
### Décomposition  Emission  content (in gCO2/euro)  = S + SB + SB^2 + ..
######
# inc_emis_content = pymrio.calc_N(io_orig.GHG_emissions.S, io_orig.G)*1e-3 = io_orig.GHG_emissions.N*1e-3
inc_emis_cont = pd.DataFrame(np.transpose((io_orig.GHG_emissions.N*1e-3).sum(level=0,axis=1).sum(level=1,axis=0)).stack(),columns=['emission content'])

inc_emis_content_direct = pd.DataFrame(np.transpose((io_orig.GHG_emissions.S*1e-3).sum(level=0,axis=1).sum(level=1,axis=0)).stack(),columns=['Direct emis content'])
inc_emis_content_fo = pd.DataFrame(np.transpose((pymrio.calc_N(io_orig.GHG_emissions.S, io_orig.B)*1e-3).sum(level=0,axis=1).sum(level=1,axis=0)).stack(),columns=['FO emis content'])
inc_emis_content_so =pd.DataFrame(np.transpose((pymrio.calc_N(io_orig.GHG_emissions.S, io_orig.B.dot(io_orig.B))*1e-3).sum(level=0,axis=1).sum(level=1,axis=0)).stack(),columns=['SO emis content'])

inc_emis_cont_decomp = inc_emis_content_direct.copy()
inc_emis_cont_decomp.loc[:,'FO emis content'] = inc_emis_content_fo['FO emis content']
inc_emis_cont_decomp.loc[:,'SO emis content'] = inc_emis_content_so['SO emis content']
inc_emis_cont_decomp.loc[:,'Rest emis content'] =inc_emis_cont['emission content'] - ( inc_emis_content_direct['Direct emis content'] + inc_emis_content_fo['FO emis content'] + inc_emis_content_so['SO emis content']) 


## income based emission in France in MtCO2 
income_based_emis_FR =io_orig.GHG_emissions.D_iba.sum(axis=1).loc[['FR']]*1e-9
#io_orig.GHG_emissions.D_iba.sum(axis=0,level='region')
#pour avoir les D_iba avec seulement le pays pour l'origine des émissions

##########################
###### PLOTS 
##########################
sns.set_context('paper', font_scale=0.9)

######
### Emission content - Histogramme groupé FR vs ROW
######
check=inc_emis_cont.reset_index(inplace=True)

plt.figure(figsize=(18, 12))
sns.barplot(x="sector", hue="region", y="emission content", data=inc_emis_cont)
plt.xlabel("Sector code", size=12)
plt.ylabel("g$\mathrm{CO}_2$eq/\euro", size=12)
plt.title("Emission content - France vs Rest of World", size=12)
plt.savefig(OUTPUTS_PATH+'fig_emis_cont_FRvsRoW.jpeg', bbox_inches='tight')
plt.show()


######
### Emission content - Histogramme FRANCE
######
### Emissions content by sector
inc_emis_cont_fr= inc_emis_cont.loc[inc_emis_cont['region']=='FR']

plt.figure(figsize=(18, 12))
sns.barplot(x="sector", y="emission content", data=inc_emis_cont_fr,palette='deep')
plt.xlabel("Sector code", size=12)
plt.ylabel("g$\mathrm{CO}_2$eq/\euro", size=12)
plt.title("Total emission content - France", size=12)
plt.savefig(OUTPUTS_PATH+'fig_emis_cont_FR_tot.jpeg', bbox_inches='tight')
plt.show()
plt.close()

######
### Decomposition - Emission content FRANCE
######
inc_emis_cont_decomp_fr= np.transpose(inc_emis_cont_decomp.loc[('FR')])

#sns.set()
inc_emis_cont_decomp_fr.T.plot(kind='bar', stacked=True, figsize=(18, 12))
plt.xlabel("Sector code", size=12)
plt.xticks(rotation=0,fontsize=12)
plt.ylabel("g$\mathrm{CO}_2$eq/\euro", size=12)
plt.title("Total emission content decomposition- France", size=12)
plt.savefig(OUTPUTS_PATH+'fig_emis_cont_decomp_FR_tot.jpeg', bbox_inches='tight')
plt.show()
plt.close()

######
### Decomposition enabled emissions by country (in gCO2/euro)
######
emis_enable = (io_orig.GHG_emissions.N*1e-3).sum(level=0,axis=1)
emis_enable = emis_enable.add_prefix('Total enabled emission of ')
emis_enable.reset_index(level=1, inplace=True)
emis_enable = emis_enable.rename_axis(index=None)

for r in region_list:
 
    emis_enable_r=  emis_enable[['sector','Total enabled emission of '+str(r)]]
    emis_enable_r = np.transpose(np.transpose(emis_enable_r).add_prefix('emis cont from '))
    emis_enable_r['region']=emis_enable_r.index
    emis_enable_r= emis_enable_r.pivot(index='sector', columns='region')
    emis_enable_r = emis_enable_r.droplevel(level=0,axis=1)
    emis_enable_r= np.transpose(emis_enable_r)

    emis_enable_r.T.plot(kind='bar', stacked=True, figsize=(18, 12))
    plt.xlabel("Sector code", size=12)
    plt.xticks(rotation=0,fontsize=12)
    plt.ylabel("g$\mathrm{CO}_2eq$/\euro", size=12)
    plt.title("Total enabled emission decomposition - "+r, size=12)
    plt.savefig(OUTPUTS_PATH+'fig_emis_cont_enabled_'+r+'_tot.jpeg', bbox_inches='tight')
    plt.show()
    plt.close()


##########################
###### TABLES to save
##########################
share_direct_emis = pd.DataFrame(inc_emis_cont_decomp_fr.loc['Direct emis content'].div(inc_emis_cont_fr['emission content'].replace(0, np.nan).values)*100)
share_direct_emis = share_direct_emis.astype(float).round(1)
share_direct_emis.columns=['% direct emissions']

share_1stdownstream_emis = pd.DataFrame(inc_emis_cont_decomp_fr.loc['FO emis content'].div(inc_emis_cont_fr['emission content'].replace(0, np.nan).values)*100)
share_1stdownstream_emis = share_1stdownstream_emis.astype(float).round(1)
share_1stdownstream_emis.columns=['% downstream at first level']

##########################
### to review 
emis_enable_fr_fr = emis_enable.loc['FR'][['sector','Total enabled emission of FR']]
emis_enable_fr_fr = emis_enable_fr_fr.set_index(emis_enable_fr_fr['sector'])
emis_enable_fr_fr = emis_enable_fr_fr.drop('sector',axis=1)
emis_enable_fr_fr = emis_enable_fr_fr.squeeze()
##########################
share_dom_emis = pd.DataFrame(emis_enable_fr_fr.div(inc_emis_cont_fr['emission content'].replace(0, np.nan).values)*100)
share_dom_emis =share_dom_emis.astype(float).round(1)
share_dom_emis.columns=['% domestic emissions']

emis_cont_fr_to_save = inc_emis_cont_fr.copy()
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


VA = pd.DataFrame(io_orig.V.sum(axis=1, level=0).sum(axis=1),columns=['value added'])
VA['share of VA']= np.nan
VA['share of VA'] = VA.div(VA.sum(axis=0, level=0), level=0)*100

print('Min emission content in FR (gCO2/euro of VA):',round(inc_emis_cont_fr['emission content'].min()))
print('Max emission content in FR (gCO2/euro of VA):',round(inc_emis_cont_fr['emission content'].max()))
print('mean emission content weighted by VA in FR (gCO2/euro of VA):', round(np.average(inc_emis_cont_fr['emission content'], weights=VA.loc[('FR')]['share of VA'])))
print('Standard deviation of emission content in FR (gCO2/euro of VA):',round(inc_emis_cont_fr['emission content'].std()))

print('mean emission content weighted by VA in RoW (gCO2/euro of VA):', round(np.average(inc_emis_cont.loc[inc_emis_cont['region']=='RoW','emission content'], weights=VA.loc[('RoW'),'share of VA'])))
print('Standard deviation of emission content in RoW (gCO2/euro of VA):', round(inc_emis_cont.loc[inc_emis_cont['region']=='RoW','emission content'].std()))


##########################
###### SAVED file - to link with INSEE survey 
##########################
ut.df_to_csv_with_comment(inc_emis_cont_fr, output_folder + os.sep + 'emission_content_france.csv', '# file automatically generated by ' + os.path.basename( os.getcwd()), sep=';')
#inc_emis_cont.to_excel(OUTPUTS_PATH+'inc_emis_cont_pre.xlsx')
