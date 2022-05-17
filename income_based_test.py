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

this_file='income_based_test.py'
eol ='\n'

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

data_path = path + os.sep + data_folder + os.sep
output_path = path + os.sep + output_folder + os.sep

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
full_exiobase_storage = data_path+'IOT_2015_pxp.zip'

## Download from exiobase Zenodo
## If the light database doesn't exist already, it loads the full database
if not os.path.exists(data_folder + os.sep + light_exiobase_folder):
    print('Loading full exiobase database...')
    io_orig =  pymrio.parse_exiobase3(full_exiobase_storage)
    print('Loaded')

## chargement des valeurs de base
    #io_orig.calc_all()

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
    
# ###### GHG emissions from Final Demand
    F_Y = pd.DataFrame(io_orig.impacts.F_hh.loc['GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)'].groupby(axis=0,level=0).sum(),index=io_orig.impacts.F_hh.columns.levels[0])


# ###### GHG emissions from Final Demand by sector
    if not os.path.exists(data_folder + os.sep + "Share_F_Y_sec.pkl"):
        print('Calculating repartition key for FD emissions..')
        exec(open("building_F_Y_sec_share.py").read())
    else:
        share_F_Y_sec = pd.read_pickle(data_path + 'Share_F_Y_sec.pkl')

    #F_Y_sec (i,j): emissions from households of countries j, to final demand addressed to sector i
    F_Y_sec = share_F_Y_sec * (np.transpose(F_Y.values))
 
    Y_drop = io_orig.Y.drop(['Changes in inventories', 'Changes in valuables', 'Exports: Total (fob)','Gross fixed capital formation'], axis=1, level=1).sum(level=0,axis=1)

    #element i,j contains final consumption (except the components excluded in previous lines) of countries j addressed to sector i
    sum_Y_on_region_of_origin = Y_drop.sum(level='sector')    
        
     #spotting sector with problems: for this sector, there is emissions (according to F_Y_sec) but no final consumption (according to Y)
    df_problem = (sum_Y_on_region_of_origin == 0) & (F_Y_sec > 0)  

    #this manually reallocates emissions on F_Y_sec to other sectors where there is final consumption
    #only done for specific sectors that are quantitatively important

    for region in df_problem.columns:
        #correction emissions from Natural gas with no final demand
        sector_origin = 'Natural gas and services related to natural gas extraction, excluding surveying'
        sector_destination = 'Distribution services of gaseous fuels through mains'
        sector_destination2 = 'Biogas'
        sector_destination3 = 'Gas Works Gas'
        sector_destination4 = 'Liquefied Petroleum Gases (LPG)'
        if df_problem.loc[sector_origin, region]:#on a des émissions de gaz naturel mais pas de consommation
            if (sum_Y_on_region_of_origin.loc[sector_destination, region] !=0): #mais on a des consommations dans le secteur 'Distribution', alors on change on alloue les émissions à ce secteur-là
                F_Y_sec.loc[sector_destination, region] = F_Y_sec.loc[sector_origin, region]
                F_Y_sec.loc[sector_origin, region]=0
                print(sector_origin[0:11] +': For ' + region + ', change for '+sector_destination)
            elif (sum_Y_on_region_of_origin.loc[sector_destination2, region] !=0): #pas de consommation jusqu'à présent, mais on a des consommations dans le secteur 'Biogas', alors on change on alloue les émissions à ce secteur-là
                F_Y_sec.loc[sector_destination2, region] = F_Y_sec.loc[sector_origin, region]
                F_Y_sec.loc[sector_origin, region]=0
                print(sector_origin[0:11] +': For ' + region + ', change for '+sector_destination2)
            elif (sum_Y_on_region_of_origin.loc[sector_destination3, region] !=0): #pas de consommation jusqu'à présent, mais on a des consommations dans le secteur 'Gas Works Gas', alors on change on alloue les émissions à ce secteur-là
                F_Y_sec.loc[sector_destination3, region] = F_Y_sec.loc[sector_origin, region]
                F_Y_sec.loc[sector_origin, region]=0
                print(sector_origin[0:11] +': For ' + region + ', change for '+sector_destination3)
            elif (sum_Y_on_region_of_origin.loc[sector_destination4, region] !=0): #pas de consommation jusqu'à présent, mais on a des consommations dans le secteur 'LPG, alors on change on alloue les émissions à ce secteur-là
                F_Y_sec.loc[sector_destination4, region] = F_Y_sec.loc[sector_origin, region]
                F_Y_sec.loc[sector_origin, region]=0
                print(sector_origin[0:11] +': For ' + region + ', change for '+sector_destination4)
            else: 
                print(sector_origin[0:11]+':For ' + region + ', correction could not be made')

        #correction emissions from Gaz/Oil with no final demand
        sector_origin = 'Gas/Diesel Oil'
        sector_destination = 'Motor Gasoline'
        sector_destination2 = 'Heavy Fuel Oil'
        sector_destination3 = 'Liquefied Petroleum Gases (LPG)'
        if df_problem.loc[sector_origin, region]:#on a des émissions de gaz naturel mais pas de consommation
            if (sum_Y_on_region_of_origin.loc[sector_destination, region] !=0): #mais on a des consommations dans le secteur distribution, alors on change on alloue les émissions à ce secteur-là
                F_Y_sec.loc[sector_destination, region] = F_Y_sec.loc[sector_origin, region]
                F_Y_sec.loc[sector_origin, region]=0
                print(sector_origin +': For ' + region + ', change for '+sector_destination)
            elif (sum_Y_on_region_of_origin.loc[sector_destination2, region] !=0): #mais on a des consommations dans le secteur distribution, alors on change on alloue les émissions à ce secteur-là
                F_Y_sec.loc[sector_destination2, region] = F_Y_sec.loc[sector_origin, region]
                F_Y_sec.loc[sector_origin, region]=0
                print(sector_origin +': For ' + region + ', change for '+sector_destination2)
            elif (sum_Y_on_region_of_origin.loc[sector_destination3, region] !=0): #mais on a des consommations dans le secteur distribution, alors on change on alloue les émissions à ce secteur-là
                F_Y_sec.loc[sector_destination3, region] = F_Y_sec.loc[sector_origin, region]
                F_Y_sec.loc[sector_origin, region]=0
                print(sector_origin +': For ' + region + ', change for '+sector_destination3)
            else: 
                print(sector_origin+': For ' + region + ', correction could not be made')

        #correction emissions from Other Bituminous Coal with no final demand
        sector_origin = 'Other Bituminous Coal'
        sector_destination = 'Chemical and fertilizer minerals, salt and other mining and quarrying products n.e.c.' 
        if df_problem.loc[sector_origin, region]:#on a des émissions de gaz naturel mais pas de consommation
            if (sum_Y_on_region_of_origin.loc[sector_destination, region] !=0): #mais on a des consommations dans le secteur distribution, alors on change on alloue les émissions à ce secteur-là
                F_Y_sec.loc[sector_destination, region] = F_Y_sec.loc[sector_origin, region]
                F_Y_sec.loc[sector_origin, region]=0
                print(sector_origin +': For ' + region + ', change for '+sector_destination[0:11])
            else: 
                print(sector_origin+': For ' + region + ', correction could not be made')

 #no need to sort index here, as we work with DataFrame, indexing is recognized
    F_Y_sec_and_reg = (Y_drop / sum_Y_on_region_of_origin) * F_Y_sec
    F_Y_sec_and_reg.fillna(0, inplace =True)
    #at this stage element i, j of F_Y_sec_and_reg contains direct emissions of household of country j coming from final consumption adressed to country x sector i
    
    #reorder columns to get the same order as in rows
    F_Y_sec_and_reg = F_Y_sec_and_reg.reindex( columns = F_Y_sec_and_reg.index.get_level_values('region').drop_duplicates() ) 
    #diagonalise by sector to have same format as F
    F_Y_final = pymrio.tools.ioutil.diagonalize_blocks(F_Y_sec_and_reg.values, blocksize = io_orig.get_sectors().size).transpose()

    #transform into a dataFrame with correct indices (from F_Y_sec_and_reg)
    F_Y_final = pd.DataFrame(F_Y_final, index=F_Y_sec_and_reg.index, columns=F_Y_sec_and_reg.index)
    
    #add this as a security check in case the indices are not in the same way
    MRIO_index = io_orig.GHG_emissions.F.index
    if (F_Y_sec_and_reg.index != MRIO_index).any():
        print('reindexing F_Y_final')
        F_Y_final = F_Y_final.reindex(MRIO_index)
        F_Y_final = F_Y_final.reindex(columns = MRIO_index)

    #save in extensions
    io_orig.GHG_emissions.F_Y_final = F_Y_final

##deleting unecessary satellite accounts
    del io_orig.satellite
    del io_orig.impacts
    del io_orig.IOT_2015_pxp
    
    ##########################
    ###### AGGREGATION PRE CALCULATION
    ##########################
    ## Aggregation into 35 sectors (closest correspondance with A38 nomenclature)
    # corresp_table = pd.read_csv(data_path + 'exiobase_A38.csv', comment='#',header=[0,1], index_col=0, sep=';')
    # corresp_table=np.transpose(corresp_table)
    # sector_agg = corresp_table.values

    # ## Aggregation into two regions: FR and RoW
    # region_table = pd.read_csv(data_path + 'exiobase_FRvsRoW.csv', comment='#', index_col=0, sep=';')
    # region_table=np.transpose(region_table)
    # region_agg = region_table.values
    
    # F_Y_final = io_orig.GHG_emissions.F_Y_final
    # io_orig.GHG_emissions.F_Y_final

    # io_orig.aggregate(region_agg=region_agg, sector_agg=sector_agg, region_names =          list(region_table.index.get_level_values(0)), sector_names = list(corresp_table.index.get_level_values(0)))
    # region_list = list(io_orig.get_regions())
    ###############################
    
    io_orig.calc_system()
    io_orig.GHG_emissions.calc_system(x=io_orig.x, Y=io_orig.Y, L=io_orig.L, Y_agg=None, population=io_orig.population)
    io_orig.GHG_emissions.calc_income_based(x = io_orig.x, V=io_orig.V, G=io_orig.G, V_agg=None, population=io_orig.population)

    V_agg = io_orig.V.sum(level='region', axis=1, ).reindex(io_orig.get_regions(), axis=1)
    io_orig.GHG_emissions.D_iba_zero_order = pymrio.tools.iomath.calc_D_iba(io_orig.GHG_emissions.S, pd.DataFrame(np.identity(np.shape(io_orig.G.values)[0])), V_agg, io_orig.get_sectors().size)
    io_orig.GHG_emissions.D_iba_first_order = pymrio.tools.iomath.calc_D_iba(io_orig.GHG_emissions.S, io_orig.B, V_agg, io_orig.get_sectors().size)
    
## saving the pymrio database with the satellite account for GHG emissions and value-added only 
    os.makedirs(data_folder + os.sep + light_exiobase_folder)
    io_orig.save_all(data_folder + os.sep + light_exiobase_folder)
  
## If the light database does exist, it loads it instead of the full database
else:
   
    print('Loading part of the exiobase database...')
    io_orig = pymrio.load_all(data_folder + os.sep + light_exiobase_folder)
    print('Loaded')
    

##########################
###### AGGREGATION POST CALCULATION
##########################
## Aggregation into 35 sectors (closest correspondance with A38 nomenclature)
corresp_table = pd.read_csv(data_path + 'exiobase_A38.csv', comment='#',header=[0,1,2], index_col=0, sep=';')
corresp_table=np.transpose(corresp_table)
sector_agg = corresp_table.values

#to replace industry code by their short label
#build a dictionary, in the same time print a correspondance for Reading note
code_to_label_dic = {}
long_string = ''
for item in corresp_table.index:
    code, long_label, short_label = item
    code_to_label_dic[code] = short_label + " (" + code +")"
    long_string += code + ': '+ short_label +'; '
with open(output_path + 'caption_industry_details.tex',"w",encoding="utf8") as file:
    #trim long_string to remove UZ and last semi-colon
    file.write( long_string.rsplit(';',2)[0] )

#  Aggregation into two regions: FR and RoW
region_table = pd.read_csv(data_path + 'exiobase_FRvsRoW.csv', comment='#', index_col=0, sep=';')
region_table=np.transpose(region_table)
region_agg = region_table.values

#save F_Y_final but delete it from extensions as it cannot be aggregated
F_Y_final = io_orig.GHG_emissions.F_Y_final
del io_orig.GHG_emissions.F_Y_final

io_orig.aggregate(region_agg=region_agg, sector_agg=sector_agg, region_names = list(region_table.index.get_level_values(0)), sector_names = list(corresp_table.index.get_level_values(0)))
region_list = list(io_orig.get_regions())

##########################
##########################
###### saving data for article
##########################
##########################

V_agg = io_orig.V.sum(level='region', axis=1)

##########################
### downstream carbon intensity for aggregate sector and countries
##########################
# compute and convert in gCO2/euro
# original units of EXIOBASE: kgco2 / M euro =  X gCO2/k euro => X * 1e-3 gCO2/euro 
downstream_carbon_intensity = pymrio.tools.iomath.recalc_N(io_orig.GHG_emissions.S, io_orig.GHG_emissions.D_iba, V_agg, io_orig.get_sectors().size)*1e-3

#downstream carbon intensity with different countries (FR, RoW) in columns
carbon_intensity_by_countries = downstream_carbon_intensity.sum(axis=0).unstack(level='region')
carbon_intensity_by_countries.columns.name=None
ut.df_to_csv_with_comment( carbon_intensity_by_countries[ (carbon_intensity_by_countries['FR']!=0) | (carbon_intensity_by_countries['RoW'] !=0 )], output_path + 'carbon_intensity_by_countries.csv', '% file automatically generated by ' + this_file + eol + '% downstream carbon intensity by industries, for France and RoW')

##########################
### building table decomposing carbon intensity of sector in France
##########################
direct_carbon_intensity = pymrio.tools.iomath.recalc_N(io_orig.GHG_emissions.S, io_orig.GHG_emissions.D_iba_zero_order, V_agg, io_orig.get_sectors().size)*1e-3
first_carbon_intensity = pymrio.tools.iomath.recalc_N(io_orig.GHG_emissions.S, io_orig.GHG_emissions.D_iba_first_order, V_agg, io_orig.get_sectors().size)*1e-3

#Series of downstream carbon intensity for France
carbon_intensity_France = downstream_carbon_intensity.sum(axis=0).loc['FR',:]
share_direct = direct_carbon_intensity.sum(axis=0).loc['FR',:] / carbon_intensity_France * 100
share_first_order = first_carbon_intensity.sum(axis=0).loc['FR',:] / carbon_intensity_France * 100
share_domestic = downstream_carbon_intensity.sum(axis=0,level='region').loc['FR','FR'] / carbon_intensity_France * 100
#Series of income based emissions in MtCO2
ib_emissions_France = io_orig.GHG_emissions.D_iba.sum(axis=0).loc['FR',:]*1e-9
total_ibe_France = np.sum(ib_emissions_France)
share_emissions_France = ib_emissions_France / total_ibe_France * 100

#build and format for output in Latex
decomposition_carbon_intensity_France = pd.concat([carbon_intensity_France, share_direct, share_first_order, share_domestic, ib_emissions_France,share_emissions_France],axis=1, keys=['Downstream carbon intensity', '\% direct emissions', '\% first-level emissions', '\% domestic emissions', 'Income-based emissions (Mt\COe)', '\% of total emissions'])
decomposition_carbon_intensity_France.sort_values(by=['Downstream carbon intensity'], ascending = False, inplace = True )

decomposition_carbon_intensity_France = decomposition_carbon_intensity_France.droplevel(level='region')
decomposition_carbon_intensity_France.index.name = 'Industry'
decomposition_carbon_intensity_France.reset_index(inplace = True)

decomposition_carbon_intensity_France['Industry'].replace(code_to_label_dic, inplace=True)
decomposition_carbon_intensity_France = decomposition_carbon_intensity_France[decomposition_carbon_intensity_France['Downstream carbon intensity'] !=0]

##not to truncate long string when Industry is recoded
with pd.option_context("max_colwidth", 1000):
    decomposition_carbon_intensity_France.to_latex(output_path+"industry_carbon_intensity.tex",index=False,float_format = "{:.1f}".format,column_format='L{4cm}L{2cm}L{2cm}L{2cm}L{2cm}L{2cm}L{2cm}', escape=False)

###
###to investigate which products of Exiobase are included in aggregate industries
###
def Exiobase_product_from_industry_code(code):
    code_corresp = corresp_table.loc[code]
    for prod in code_corresp[code_corresp.ne(0)].dropna(axis=1).columns:
        print(prod)

#####################
###analyzing income-based emissions of primary inputs and industry
#####################

#aggregate value added over inputs and aggregate some inputs
# Value added in M€
V_agg_by_all_inputs = io_orig.V.sum(axis=1,level='category')#remove useless 'region' level in axis 1
#define how to aggregate primary inputs
input_groups = {'Net taxes': ['Taxes less subsidies on products purchased: Total', 'Other net taxes on production'],
        'Compensation of employees': ["Compensation of employees; wages, salaries, & employers' social contributions: Low-skilled", "Compensation of employees; wages, salaries, & employers' social contributions: Medium-skilled", "Compensation of employees; wages, salaries, & employers' social contributions: High-skilled"],
#"Compensation of employees; wages, salaries, & employers' social contributions: Low-skilled":["Compensation of employees; wages, salaries, & employers' social contributions: Low-skilled"],
#"Compensation of employees; wages, salaries, & employers' social contributions: Medium-skilled":["Compensation of employees; wages, salaries, & employers' social contributions: Medium-skilled"],
#"Compensation of employees; wages, salaries, & employers' social contributions: High-skilled":["Compensation of employees; wages, salaries, & employers' social contributions: High-skilled"],
        'Operating surplus':['Operating surplus: Consumption of fixed capital', 'Operating surplus: Royalties on resources','Operating surplus: Rents on land', 'Operating surplus: Remaining net operating surplus'],}
        
V_agg_by_inputs = pd.DataFrame({k: V_agg_by_all_inputs[v].sum(axis=1) for k, v in input_groups.items()})

#io_orig.V.sum(axis=1,level='category') #in M€
ib_emissions_by_industry_and_inputs = V_agg_by_inputs.multiply(downstream_carbon_intensity.sum(axis=0), axis = 0) * 1e-6 #multiply value added by downstream carbon intensity, resulting emissions in MtCO2
#also store total value added to perform same operations on it
ib_emissions_by_industry_and_inputs.loc[('FR','value added'),:] = V_agg_by_inputs.sum(axis=0,level='region').loc['FR',:]
ib_emissions_by_industry_and_inputs.loc[('RoW','value added'),:] = V_agg_by_inputs.sum(axis=0,level='region').loc['RoW',:]

ib_emissions_matrix_France = ib_emissions_by_industry_and_inputs.loc['FR',:].drop('value added')
ib_emissions_matrix_France['Total income-based emissions of industry'] = ib_emissions_matrix_France.sum(axis=1)
ib_emissions_matrix_France['Share of industry in total income-based emissions of France'] = ib_emissions_matrix_France['Total income-based emissions of industry'] / total_ibe_France * 100
ib_emissions_matrix_France.loc['Total income-based emissions of inputs',:]=ib_emissions_matrix_France.sum(axis=0)
ib_emissions_matrix_France.loc['Share of inputs in total income-based emissions of France',:] = ib_emissions_matrix_France.loc['Total income-based emissions of inputs',:] / total_ibe_France * 100
ib_emissions_matrix_France.loc['Share of inputs in total income-based emissions of France', 'Share of industry in total income-based emissions of France']= ''

ib_emissions_matrix_France.index.name = 'Industry'
ib_emissions_matrix_France.reset_index(inplace = True)

ib_emissions_matrix_France['Industry'].replace(code_to_label_dic, inplace=True)
ib_emissions_matrix_France = ib_emissions_matrix_France[ib_emissions_matrix_France['Total income-based emissions of industry'] !=0]

with pd.option_context("max_colwidth", 1000):
    ib_emissions_matrix_France.to_latex(output_path+"emissions_matrix.tex",index=False,float_format = "{:.1f}".format,column_format='L{4cm}L{2cm}L{2cm}L{2cm}L{2cm}L{2cm}L{2cm}L{2cm}L{2cm}')

#diagnosis of the contribution of inputs
mean_ci = io_orig.GHG_emissions.D_iba.sum(axis=0).sum(level='region') / V_agg.sum(axis=0)*1e-3
decomposition_gap_va=(V_agg_by_inputs.loc['FR',:].multiply(1/V_agg.loc['FR','FR']*100,axis=0)).subtract(V_agg_by_inputs.loc['FR',:].sum(axis=0)/V_agg.loc['FR','FR'].sum(axis=0)*100 , axis =1) #compute the difference between the share of an input in the value-added of a sector and the mean share of that input in total value-added
value_added_weight=pd.DataFrame(np.outer(V_agg.loc['FR','FR'], 1/(V_agg_by_inputs.loc['FR',:].sum(axis=0))), index= V_agg.loc['FR','FR'].index, columns = V_agg_by_inputs.loc['FR',:].sum(axis=0).index )

decomposition_gap_ci=(decomposition_gap_va * value_added_weight).multiply(carbon_intensity_France.droplevel(level='region')-mean_ci['FR'],axis=0).fillna(0)/100

decomposition_gap_ci['Downstream carbon intensity'] = carbon_intensity_France.droplevel(level='region')-mean_ci['FR']
decomposition_gap_ci.sort_values(by=['Downstream carbon intensity'], ascending = False, inplace = True )
decomposition_gap_ci.sum(axis=0) #this give the difference between ci of an input and mean ci #caution: summing carbon intensity (last columns) has no meaning



#####################
### summarizing contributions of primary inputs to income-based industry
#####################
input_contribution_table = pd.DataFrame()
input_contribution_table = pd.DataFrame( columns=ib_emissions_by_industry_and_inputs.columns)
input_contribution_table.loc['Enabled emissions (Mt\CO)',:] = ib_emissions_by_industry_and_inputs.loc['FR',:].drop('value added').sum(axis=0)
input_contribution_table.loc['Share of total emissions (\%)',:] = input_contribution_table.loc['Enabled emissions (Mt\CO)',:] / total_ibe_France * 100
input_contribution_table.loc['Value added received (G\euro)',:] = ib_emissions_by_industry_and_inputs.loc[('FR','value added'),:] *1e-3
input_contribution_table.loc['Share of value added (\%)',:] = input_contribution_table.loc['Value added received (G\euro)',:] / np.sum(input_contribution_table.loc['Value added received (G\euro)',:]) * 100
input_contribution_table.loc['Mean downstream carbon intensity (\si{g\COe\per\euro})',:] = input_contribution_table.loc['Enabled emissions (Mt\CO)',:] / input_contribution_table.loc['Value added received (G\euro)',:] * 1e3
input_contribution_table.rename(columns =
        {"Compensation of employees; wages, salaries, & employers' social contributions: Low-skilled" : 'Low-skilled labour',
         "Compensation of employees; wages, salaries, & employers' social contributions: Medium-skilled": 'Medium-skilled labour',
         "Compensation of employees; wages, salaries, & employers' social contributions: High-skilled": 'High-skilled labour',
         "Compensation of employees": 'Labour',
         'Operating surplus':'Capital',
         'Net taxes':'Administrative services'
            }, inplace= True)
input_contribution_table.index.name=''
with pd.option_context("max_colwidth", 1000):
    input_contribution_table.to_latex(output_path+"emission_per_primary_input.tex", index=True, float_format = "{:.1f}".format, column_format='L{4cm}L{2cm}L{2cm}L{2cm}', escape = False)

#####################
###save quantities used in text
#####################
with open(output_path + 'section_carbon_intensity.tex','w') as file:
    file.write('% file automatically generated by ' + this_file + eol)
    file.write('% numbers used in main text' + eol)
    file.write('\\newcommand\\minciFrance{' + "{:.1f}".format(min(carbon_intensity_France[carbon_intensity_France != 0])) + '} %min carbon intensity for France' + eol)
    file.write('\\newcommand\\maxciFrance{' + "{:.1f}".format(max(carbon_intensity_France[carbon_intensity_France != 0])) + '} %max carbon intensity for France' + eol)
    file.write('\\newcommand\\ciBZRoW{' + "{:.1f}".format(carbon_intensity_by_countries.loc['BZ','RoW']) + '} % carbon intensity for BZ in RoW' + eol)
    file.write('\\newcommand\\ciDZRoW{' + "{:.1f}".format(carbon_intensity_by_countries.loc['DZ','RoW']) + '} % carbon intensity for DZ in RoW' + eol)
    file.write('\\newcommand\\ciCDRoW{' + "{:.1f}".format(carbon_intensity_by_countries.loc['CD','RoW']) + '} % carbon intensity for CD in RoW' + eol)
    #various quantity related to AZ for caption
    file.write('\\newcommand\\ciAZFrance{' + "{:.0f}".format(carbon_intensity_France[('FR','AZ')]) + '} % carbon intensity for Agriculture in France' + eol)
    file.write('\\newcommand\\sharedirectAZFrance{' + "{:.1f}".format(share_direct[('FR','AZ')]) + '} % carbon intensity for Agriculture in France' + eol)
    file.write('\\newcommand\\sharefirstAZFrance{' + "{:.1f}".format(share_first_order[('FR','AZ')]) + '} % carbon intensity for Agriculture in France' + eol)
    file.write('\\newcommand\\remainingshareAZFrance{' + "{:.1f}".format(100-share_direct[('FR','AZ')] - share_first_order[('FR','AZ')]) + '} % carbon intensity for Agriculture in France' + eol)
    file.write('\\newcommand\\remainingshareCDFrance{' + "{:.1f}".format(100-share_direct[('FR','CD')] - share_first_order[('FR','CD')]) + '} % carbon intensity for Agriculture in France' + eol)
    file.write('\\newcommand\\domesticshareAZFrance{' + "{:.1f}".format(share_domestic[('FR','AZ')]) + '} % carbon intensity for Agriculture in France' + eol)
    file.write('\\newcommand\\abroadshareAZFrance{' + "{:.1f}".format(100-share_domestic[('FR','AZ')]) + '} % carbon intensity for Agriculture in France' + eol)
    #mean carbon intensity: obtained by dividing total IBE by total VA, converted in g/CO2
    file.write('\\newcommand\\meanciFrance{' + "{:.0f}".format(mean_ci['FR']) + '} % mean carbon intensity in France' + eol)
    file.write('\\newcommand\\meanciRoW{' + "{:.0f}".format(mean_ci['RoW']) + '} % mean carbon intensity in RoW' + eol)
    file.write('\\newcommand\\valueaddedFrance{' + "{:.0f}".format(V_agg.sum(axis=0)['FR'] * 1e-3) + '} % total value added in France' + eol)
    file.write('\\newcommand\\totaliba{' + "{:.0f}".format(total_ibe_France) + '} %total income-based emissions France' + eol)
    file.write('\\newcommand\\totalpba{' + "{:.0f}".format( io_orig.GHG_emissions.D_iba.sum(axis=1).sum(level='region')['FR']*1e-9) + '} %total production-based emissions France' + eol)
    file.write('\\newcommand\\totalcba{' + "{:.0f}".format((io_orig.GHG_emissions.D_cba.sum(axis=0).sum(level='region')['FR'] + F_Y_final.sum(axis=1).sum(level='region')['FR']) *1e-9) + '} %total consumption-based emissions France' + eol)
    file.write('\\newcommand\\shareCDtotal{' + "{:.1f}".format(share_emissions_France[('FR','CD')]) + '}%share of CD in total IB emissions France ' + eol)
    file.write('\\newcommand\\emissionscapital{' + "{:.0f}".format(input_contribution_table.loc['Enabled emissions (Mt\CO)','Capital']) + '}% emissions of capital ' + eol)
    file.write('\\newcommand\\shareemissionscapital{' + "{:.1f}".format(input_contribution_table.loc['Share of total emissions (\%)','Capital']) + '}% share of emissions of capital ' + eol)
    file.write('\\newcommand\\paymentcapital{' + "{:.0f}".format(input_contribution_table.loc['Value added received (G\euro)','Capital']) + '}% va of capital ' + eol)
    file.write('\\newcommand\\sharevacapital{' + "{:.1f}".format(input_contribution_table.loc['Share of value added (\%)','Capital']) + '}% share of value added of capital ' + eol)
    file.write('\\newcommand\\meancicapital{' + "{:.1f}".format(input_contribution_table.loc['Mean downstream carbon intensity (\si{g\COe\per\euro})','Capital']) + '}% mean ci of capital ' + eol)
    file.write('\\newcommand\\shareemissionslabour{' + "{:.1f}".format(input_contribution_table.loc['Share of total emissions (\%)','Labour']) + '}% share of emissions of capital ' + eol)
    file.write('\\newcommand\\sharevalabour{' + "{:.1f}".format(input_contribution_table.loc['Share of value added (\%)','Labour']) + '}% share of value added of capital ' + eol)
    file.write('\\newcommand\\meancilabour{' + "{:.1f}".format(input_contribution_table.loc['Mean downstream carbon intensity (\si{g\COe\per\euro})','Labour']) + '}% mean ci of capital ' + eol)
    

##########################
###### SAVED file - to link with INSEE survey 
##########################
carbon_intensity_France.name = 'carbon intensity'
ut.df_to_csv_with_comment(carbon_intensity_France, output_path + 'carbon_intensity_france.csv', '# file automatically generated by ' + this_file, sep=';')
