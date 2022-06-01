# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import os
import pandas as pd
import statistics as stat
import numpy as np
from sklearn.linear_model import LinearRegression

##########################
###### Paths 
##########################
output_folder='outputs'
#create output_folder if not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print('Creating ' + output_folder + ' to store outputs')
output_path = output_folder + os.sep

data_folder = 'data'
data_path = data_folder + os.sep

###############
# retrieving data
###############

if not os.path.isfile(output_path + 'carbon_intensity_france.csv'):
    exec(open("income_based_intensity.py").read())
carbon_intensity_france = pd.read_csv( output_path + 'carbon_intensity_france.csv', sep=';', comment='#')


#convertir le fichier dBase de l'Insee en csv s'il ne l'est pas déjà
#bien vérifier la présence du module dbf_to_csv.py dans le répertoire
if os.path.isfile(data_path + 'salaries15.csv')==True:
    print('file salaries15 is already available in csv format')
else:
    dbf_to_csv(data_path + 'salaries15.dbf')
    print('file salaries15 is now available in csv format')

#Importer base salaires15 INSEE (euro 2015)
#https://www.insee.fr/fr/statistiques/3536754#dictionnaire
full_insee_table = pd.read_csv(data_path + 'salaries15.csv',sep=',', low_memory=False)
full_insee_table = full_insee_table.dropna(subset=['A38'])

#############
# adding field of mean salary
#############

#dictionary: wage class -> mean salary value
#no value for the highest wage class at the moment
dic_TRNNETO_to_salary = {0 : stat.mean([0,200]),1 : stat.mean([200,500]),2 : stat.mean([500,1000]),3 : stat.mean([1000,1500]),4 : stat.mean([1500,2000]),5 : stat.mean([2000,3000]),6 : stat.mean([3000,4000]),7 : stat.mean([4000,6000]),8 : stat.mean([6000,8000]),9 : stat.mean([8000,10000]),10 : stat.mean([10000,12000]), 11 : stat.mean([12000,14000]),12 : stat.mean([14000,16000]), 13 : stat.mean([16000,18000]), 14 : stat.mean([18000,20000]), 15 : stat.mean([20000,22000]), 16 : stat.mean([22000,24000]), 17 : stat.mean([24000,26000]), 18 : stat.mean([26000,28000]),19 : stat.mean([28000,30000]),20 : stat.mean([30000,35000]), 21 : stat.mean([35000,40000]), 22 : stat.mean([40000,50000])}

#check whether the tail of the distribution of wages follow the Pareto law
class_effectif = full_insee_table.groupby('TRNNETO').apply(len)
#log of survival rate
y = np.log(np.cumsum(class_effectif[::-1])[::-1]/np.sum(class_effectif))
#log of minimum of class
x = np.log(np.array([100,200,500,1000,1500,2000,3000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000,26000,28000,30000,35000,40000,50000]))
#plot to check the linearity at the tail
#plt.scatter(x[-8:],y[-8:])
#plt.show()

#get the alpha using only two last classes
alpha = -  LinearRegression().fit(x[-2:].reshape(-1,1), y[-2:]).coef_

#Pareto interpolation of mean of last class
dic_TRNNETO_to_salary[23]= alpha/(alpha-1)*50000

#add field to store imputed wage
full_insee_table['salary_value']=full_insee_table['TRNNETO'].replace(dic_TRNNETO_to_salary)


#############
# adding fields of carbon intensity and income-based emissions
#############

insee_classification_to_passage_classification={"AZ":"AZ","BZ":"BZ","CA":"CA","CB":"CB","CC":"CC","CD":"CD","CE":"CE-CF","CF":"CE-CF","CE-CF":"CE-CF","CG":"CG","CH":"CH","CI":"CI","CJ":"CJ","CK":"CK","CL":"CL","CM":"CM","DZ":"DZ","EZ":"EZ","FZ":"FZ","GZ":"GZ","HZ":"HZ","IZ":"IZ","JA":"JA","JB":"JB","JC":"JC","KZ":"KZ","LZ":"LZ","MA":"MA-MC","MB":"MB","MC":"MA-MC","MA-MC":"MA-MC","NZ":"NZ","OZ":"OZ","PZ":"PZ","QA":"QA-QB","QB":"QA-QB","QA-QB":"QA-QB","RZ":"RZ","SZ":"SZ","TZ":"TZ","UZ":"UZ"}


def create_dic_from_sector_to_carbon_intensity(sector_table,carbon_intensity_table, passage_dic = insee_classification_to_passage_classification, sector_label_in_sector_table = 'A38', sector_label_in_carbon_intensity_label='sector', carbon_intensity_label='carbon intensity'):
    """
    Return a dictionary that associates each sector of sector_table to their carbon intensity given by carbon_intensity_label
    by default, assume that sector column in sector_table is labeled by 'A38', that sector column in carbon_intensity_sector is labeled by 'Code_Sector', that carbon intensity column in carbon_intensity sector is labeled by 'carbon intensity'. Finally, it assumes that the dictionary from sector_table nomenclature of sectors to carbon_intensity_table nomenclature is insee_classification_to_passage_classifcation
    """
    dic_to_carbon_intensity = {}
    for x in sector_table[sector_label_in_sector_table].unique():
        try:
            passage_sector = insee_classification_to_passage_classification[x]
            carbon_intensity = carbon_intensity_table[carbon_intensity_table[sector_label_in_carbon_intensity_label]==passage_sector][carbon_intensity_label].reset_index(drop=True)[0]
            dic_to_carbon_intensity[x] = carbon_intensity
        except KeyError:
            print('Impossible to retrieve carbon intensity of sector ' +x)
            dic_to_carbon_intensity[x]= np.nan
    return dic_to_carbon_intensity

#create dictionary of carbon intensity and add carbon intensity of branches for each observation
dic_to_carbon_intensity = create_dic_from_sector_to_carbon_intensity(full_insee_table, carbon_intensity_france) 
full_insee_table['carbon_intensity'] = full_insee_table['A38'].replace(dic_to_carbon_intensity)
full_insee_table['A35'] = full_insee_table['A38'].replace(insee_classification_to_passage_classification)
dic_to_carbon_intensity_A35 = create_dic_from_sector_to_carbon_intensity(full_insee_table, carbon_intensity_france, sector_label_in_sector_table='A35') 

#add income-based emissions for each observation
#Scaling factor to convert emissions in MtCO2 to tCO2
Scaling_factor=10**(-6)
full_insee_table['income-based_emissions'] = full_insee_table['salary_value'] * full_insee_table['carbon_intensity'] * Scaling_factor

#########
# comparison of totals (commented at this stage)
#########

##Total des emissions
##depuis la base des salaires,  MtCO2 
#Emissions_insee_tot=sum(full_insee_table['income-based_emissions']*full_insee_table['POND'])*1e-6
#print('France emissions from salaries insee: ',Emissions_insee_tot,'Mt de CO2')
#
##depuis inc_based.py, total income-based emission, MtCO2 
## income_based_emis_tot_FR vs income_based_emis_FR ( avec emission direct ou non)
#inc_based_emis= income_based_emis_tot_FR
#inc_based_emis_mrio_FR_tot=np.sum(inc_based_emis.values)*1e-6
#print('France total income-based emissions from mrio database: ',inc_based_emis_mrio_FR_tot,'Mt de CO2')
#
##depuis inc_based.py, labour factor income-based emission, MtCO2 
#inc_based_emis_mrio_FR_lab=np.sum(inc_based_emis.xs('Labour', axis=1, level=1, drop_level=False).values)*1e-6
#print('France labour factor income-based emissions from mrio database: ',inc_based_emis_mrio_FR_lab,'Mt de CO2')
#
##comparaison: moins d'un tiers des Income-Basedemissions de la France seraient attribués aux salaires
#comparaison=Emissions_insee_tot/inc_based_emis_mrio_FR_tot
#print("Insee/Mrio income-based emissions ratio:",comparaison)
#print("Insee/Mrio 'labour'based emissions ratio:",Emissions_insee_tot/inc_based_emis_mrio_FR_lab)





