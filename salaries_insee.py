# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import os
import pandas as pd
import statistics as stat
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dbf_into_csv import *
from sklearn.linear_model import LinearRegression

data_dir = 'data'

exec(open("Inc_Based.py").read())

#convertir le fichier dbf de l'insee en csv pour le traiter avec pandas
#faire attention de bien disposer du module dans le répertoire

#Ne convertir le fichier dBase en csv que s'il ne l'est pas déjà
#bien vérifier la présence du module dbf_to_csv.py dans le répertoire
if os.path.isfile(data_dir +os.sep + 'salaries15.csv')==True:
    print('le fichier salaries est déjà disponible au format csv')
else:
    dbf_to_csv(data_dir +os.sep + 'salaries15.dbf')
    print('Le fichier salaries15 est maintenant disponible au format csv')

#Importer base salaires15 INSEE (euro 2015)
#https://www.insee.fr/fr/statistiques/3536754#dictionnaire
full_insee_table = pd.read_csv(data_dir + os.sep + 'salaries15.csv',sep=',', low_memory=False)
full_insee_table = full_insee_table.dropna(subset=['A38'])

dic_TRNNETO_to_salary = {0 : stat.mean([0,200]),1 : stat.mean([200,500]),2 : stat.mean([500,1000]),3 : stat.mean([1000,1500]),4 : stat.mean([1500,2000]),5 : stat.mean([2000,3000]),6 : stat.mean([3000,4000]),7 : stat.mean([4000,6000]),8 : stat.mean([6000,8000]),9 : stat.mean([8000,10000]),10 : stat.mean([10000,12000]), 11 : stat.mean([12000,14000]),12 : stat.mean([14000,16000]), 13 : stat.mean([16000,18000]), 14 : stat.mean([18000,20000]), 15 : stat.mean([20000,22000]), 16 : stat.mean([22000,24000]), 17 : stat.mean([24000,26000]), 18 : stat.mean([26000,28000]),19 : stat.mean([28000,30000]),20 : stat.mean([30000,35000]), 21 : stat.mean([35000,40000]), 22 : stat.mean([40000,50000]),23 : 60000 }
full_insee_table['salary_value']=full_insee_table['TRNNETO'].replace(dic_TRNNETO_to_salary)

insee_classification_to_passage_classification={"AZ":"AZ","BZ":"BZ","CA":"CA","CB":"CB","CC":"CC","CD":"CD","CE":"CE_CF","CF":"CE_CF","CG":"CG","CH":"CH","CI":"CI","CJ":"CJ","CK":"CK","CL":"CL","CM":"CM","DZ":"DZ","EZ":"EZ","FZ":"FZ","GZ":"GZ","HZ":"HZ","IZ":"IZ","JA":"JA","JB":"JB","JC":"JC","KZ":"KZ","LZ":"LZ","MA":"MA_MC","MB":"MB","MC":"MA_MC","NZ":"NZ","OZ":"OZ","PZ":"PZ","QA":"QA_QB","QB":"QA_QB","RZ":"RZ","SZ":"SZ","TZ":"TZ","UZ":"UZ"}


def create_dic_from_sector_to_emission_content(sector_table,emission_content_table, passage_dic = insee_classification_to_passage_classification, sector_label_in_sector_table = 'A38', sector_label_in_emission_content_label='sector', emission_content_label='emission content'):
    """
    Return a dictionary that associates each sector of sector_table to their emission content given by emission_content_label
    by default, assume that sector column in sector_table is labeled by 'A38', that sector column in emission_content_sector is labeled by 'Code_Sector', that emission content column in emission_content sector is labeled by 'emission content'. Finally, it assumes that the dictionary from sector_table nomenclature of sectors to emission_content_table nomenclature is insee_classification_to_passage_classifcation
    """
    dic_to_emission_content = {}
    for x in sector_table[sector_label_in_sector_table].unique():
        try:
            passage_sector = insee_classification_to_passage_classification[x]
            emission_content = emission_content_table[emission_content_table[sector_label_in_emission_content_label]==passage_sector][emission_content_label].reset_index(drop=True)[0]
            dic_to_emission_content[x] = emission_content
        except KeyError:
            print('Impossible to retrieve emission content of sector ' +x)
            dic_to_emission_content[x]= np.nan
    return dic_to_emission_content

#rajoute la colonne Emissions (en t de CO2) à chaque individu référencé
# emis_cont_fr without direct emissions from inc_based.py
### emis_cont_tot_fr with direct emissions from inc_based.py
emis_cont= emis_cont_tot_fr
Scaling_factor=10**(-6)
dic_to_emission_content = create_dic_from_sector_to_emission_content(full_insee_table, emis_cont) 
full_insee_table['emission_content'] = full_insee_table['A38'].replace(dic_to_emission_content)
full_insee_table['income-based_emissions'] = full_insee_table['salary_value'] * full_insee_table['emission_content'] * Scaling_factor

#Total des emissions
#depuis la base des salaires,  MtCO2 
Emissions_insee_tot=sum(full_insee_table['income-based_emissions']*full_insee_table['POND'])*1e-6
print('France emissions from salaries insee: ',Emissions_insee_tot,'Mt de CO2')

#depuis inc_based.py, total income-based emission, MtCO2 
# income_based_emis_tot_FR vs income_based_emis_FR ( avec emission direct ou non)
inc_based_emis= income_based_emis_tot_FR
inc_based_emis_mrio_FR_tot=np.sum(inc_based_emis.values)*1e-6
print('France total income-based emissions from mrio database: ',inc_based_emis_mrio_FR_tot,'Mt de CO2')

#depuis inc_based.py, labour factor income-based emission, MtCO2 
inc_based_emis_mrio_FR_lab=np.sum(inc_based_emis.xs('Labour', axis=1, level=1, drop_level=False).values)*1e-6
print('France labour factor income-based emissions from mrio database: ',inc_based_emis_mrio_FR_lab,'Mt de CO2')

#comparaison: moins d'un tiers des Income-Basedemissions de la France seraient attribués aux salaires
comparaison=Emissions_insee_tot/inc_based_emis_mrio_FR_tot
print("Insee/Mrio income-based emissions ratio:",comparaison)
print("Insee/Mrio 'labour'based emissions ratio:",Emissions_insee_tot/inc_based_emis_mrio_FR_lab)


def ratio_of_mass(lower_bound, higher_bound, label_to_sum, label_of_bound, data):
    low_mass = np.sum( data[data[label_of_bound]<=lower_bound][label_to_sum])
    high_mass = np.sum( data[data[label_of_bound]>higher_bound][label_to_sum])
    return high_mass / low_mass
    

def mean_emission_content(table):
   """
   compute the mean emission content (weighted by value-added (here wages)
   """
   mean = np.sum(table['emission_content'] * table['salary_value'] )/ np.sum(table['salary_value'])
   return pd.Series([mean,len(table)],index=['mean emission content','pop_mass'])

def stat_data_generic(list_of_label, x, fun):
    if len(list_of_label)==0:
        a=fun(x)
        return pd.DataFrame([a.values],columns=a.index)
    else:
        label = list_of_label[0]
        x_extracted_lab = x.groupby([label]).apply(lambda y: stat_data_generic(list_of_label[1:], y, fun)).reset_index().drop('level_1',axis=1)
        x_extracted_all = stat_data_generic(list_of_label[1:],x,fun)
        x_extracted_all[label] = 'All'
        x_extracted = pd.concat([x_extracted_lab,x_extracted_all], sort=False,ignore_index=True)
        return x_extracted

# Mean emission by wage class

mean_emis_content_by_class = stat_data_generic(['TRNNETO'],full_insee_table,mean_emission_content)
# Regression 
#full_insee_table['salary_value']

# Indice de vunérabilité région
dict_codereg_to_regname= {1:'Guadeloupe',2:'Martinique',3:'Guyane',4:'Réunion',11:'Île-de-France',21:'Champagne-Ardenne',22:'Picardie',23:'Haute-Normandie',24:'Centre',25:'Basse-Normandie',26:'Bourgogne',31:'Nord-Pas-de-Calais',41:'Lorraine',42:'Alsace',43:'Franche-Comté',52:'Pays de la Loire',53:'Bretagne',54:'Poitou-Charentes',72:'Aquitaine',73:'Midi-Pyrénées',74:'Limousin',82:'Rhône-Alpes',83:'Auvergne',91:'Languedoc-Roussillon',93:'Provence-Alpes-Côte d\'Azur',94:'Corse',99:'Etranger et Tom'}

reg_emis_cont = stat_data_generic(['REGT_AR'], full_insee_table.dropna(subset=['REGT_AR']), mean_emission_content)
reg_emis_cont['REGT_AR_NAME']=reg_emis_cont['REGT_AR'].replace(dict_codereg_to_regname)

#Statistiques Descriptives
#base salaire
mean_salary=round(stat.mean(full_insee_table['salary_value']))
stdev_salary=stat.stdev(full_insee_table['salary_value'])
median_salary=stat.median(full_insee_table['salary_value'])
decile1_salary=np.percentile(full_insee_table['salary_value'],10)
decile9_salary=np.percentile(full_insee_table['salary_value'],90)
interdecile_salary=decile9_salary/decile1_salary
masses_salary = ratio_of_mass( decile1_salary, decile9_salary, 'salary_value', 'salary_value', full_insee_table)

#France Emissions
mean_emissions=stat.mean(full_insee_table['income-based_emissions'])
stdev_emissions=stat.stdev(full_insee_table['income-based_emissions'])
median_emissions=stat.median(full_insee_table['income-based_emissions'])
decile1_emissions=np.percentile(full_insee_table['income-based_emissions'],10)
decile9_emissions=np.percentile(full_insee_table['income-based_emissions'],90)
interdecile_emissions=decile9_emissions/decile1_emissions
masses_emissions_ofemitters = ratio_of_mass( decile1_emissions, decile9_emissions, 'income-based_emissions', 'income-based_emissions', full_insee_table)
masses_emissions_ofrich = ratio_of_mass( decile1_salary, decile9_salary, 'income-based_emissions', 'salary_value', full_insee_table)


#Comparaison
dispersion=interdecile_emissions>interdecile_salary
if dispersion:
    print("Il y a plus d'inégalités d'émissions que de salaires")
else:
    print("Il y a plus d'inégalités de salaires que d'émissions")
    
## PLOT
sns.set_context('paper', font_scale=0.9)

#plot for mean emission content by wage classes
plt.figure(figsize=(18, 12))
sns.barplot(x=mean_emis_content_by_class['TRNNETO'], y="mean emission content", data=mean_emis_content_by_class,palette='deep')
plt.xlabel("wage class", size=12)
plt.ylabel("gCO2/euro", size=12)
plt.title("Mean emission content by wage classes", size=12)
plt.savefig(OUTPUTS_PATH+'fig_mean_emis_cont_by_class.jpeg', bbox_inches='tight')

#same plot for region (mean emission content interpreted as vulnerability index)
plt.figure(figsize=(20, 12))
sns.barplot(x=reg_emis_cont['REGT_AR_NAME'], y="mean emission content", data=reg_emis_cont,palette='deep')
plt.xlabel("Regions", size=12)
plt.xticks(rotation=90)
plt.ylabel("Vulnerability index (gCO2/euro)", size=12)
plt.title("Vulnerability index by regions", size=12)
plt.savefig(OUTPUTS_PATH+'fig_mean_emis_cont_by_region.jpeg', bbox_inches='tight')

full_insee_table['pop_mass']=1

def make_Lorenz_and_concentration_curves(table,dic_index,pdffile):
    income_index = dic_index['income']
    emissions_index = dic_index['emissions']
    pop_index =  dic_index['pop_mass']

    #sort table by emissions
    table_sorted_by_emissions = table[:,table[emissions_index,:].argsort()]
    pop_cum_by_emissions = np.cumsum(table_sorted_by_emissions[pop_index,:])/np.sum(table_sorted_by_emissions[pop_index,:])
    emissions_cum_by_emissions = np.cumsum(table_sorted_by_emissions[emissions_index,:]*table_sorted_by_emissions[pop_index,:])/np.sum(table_sorted_by_emissions[emissions_index,:]*table_sorted_by_emissions[pop_index,:])

    #sort table by income (after having sorted by emissions, to have smoother concentration curve)
    table_sorted_by_income = table_sorted_by_emissions[:,table_sorted_by_emissions[income_index,:].argsort(kind='mergesort')]#to preserve first sorting
    pop_cum_by_income = np.cumsum(table_sorted_by_income[pop_index,:])/np.sum(table_sorted_by_income[pop_index,:])
    income_cum_by_income = np.cumsum(table_sorted_by_income[income_index, :]*table_sorted_by_income[pop_index,:])/np.sum(table_sorted_by_income[income_index,:]*table_sorted_by_income[pop_index,:])
    #reconstituted_cum = np.cumsum(table_sorted_by_income[0,:]**0.34)/np.sum(table_sorted_by_income[0,:]**0.34)
    emissions_cum_by_income = np.cumsum(table_sorted_by_income[emissions_index,:]*table_sorted_by_income[pop_index,:])/np.sum(table_sorted_by_income[emissions_index,:]*table_sorted_by_income[pop_index,:])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot([0,1],[0,1],'k:',linewidth=1)
    plt.plot(pop_cum_by_income, emissions_cum_by_income,label='concentration curve of income-based emissions')
    plt.plot(pop_cum_by_emissions, emissions_cum_by_emissions,label='Lorenz curve for income-based emissions')
    plt.plot(pop_cum_by_income, income_cum_by_income,label='Lorenz curve for income')
    plt.xlabel('Cumulative share of French wage earners')
    plt.ylabel('Cumulative share of income or income-based emissions')
    plt.axis([0,1,0,1])
    plt.legend(loc='upper left')
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(pdffile,bbox_inches='tight')
    plt.close()



pop_mass_per_sector_x_salary=full_insee_table.groupby(['TRNNETO','A38']).size().reset_index(name='pop_mass')
pop_mass_per_sector_x_salary['emission_content'] = pop_mass_per_sector_x_salary['A38'].replace(dic_to_emission_content)
pop_mass_per_sector_x_salary['salary_value'] = pop_mass_per_sector_x_salary['TRNNETO'].replace(dic_TRNNETO_to_salary)
pop_mass_per_sector_x_salary['salary_mass'] = pop_mass_per_sector_x_salary['salary_value'] * pop_mass_per_sector_x_salary['pop_mass']
pop_mass_per_sector_x_salary['emissions_mass'] = pop_mass_per_sector_x_salary['salary_mass'] * pop_mass_per_sector_x_salary['emission_content']
pop_mass_per_sector_x_salary['emissions_capita'] = pop_mass_per_sector_x_salary['salary_value'] * pop_mass_per_sector_x_salary['emission_content']

fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlabel('Wages')
plt.ylabel('Emission content')
plt.scatter(pop_mass_per_sector_x_salary['salary_value'], pop_mass_per_sector_x_salary['emission_content'], s=pop_mass_per_sector_x_salary['pop_mass']/100,marker='o')
plt.savefig(OUTPUTS_PATH + 'wages_per_sector.pdf',bbox_inches='tight')
plt.close()

make_Lorenz_and_concentration_curves(np.transpose(np.array(pop_mass_per_sector_x_salary[['pop_mass','salary_value', 'emissions_capita']])),{'pop_mass':0,'income':1,'emissions':2},OUTPUTS_PATH + 'Lorenz_curve_from_aggregate.pdf')


#regression 
def compute_and_print_elasticity(x,y,independent_variable,study,weight=None,print_rsq=False):
    x_log = np.log(x).reshape((-1,1))
    y_log = np.log(y)
    model = LinearRegression().fit(x_log, y_log,sample_weight = weight)
    elasticity = model.coef_[0]
    print(independent_variable + " elasticity in " +  study + " is " +"{:.2f}".format(elasticity))
    if print_rsq:
        print('and coefficient of determination is '+"{:.2f}".format(model.score(x_log,y_log, sample_weight = weight)))

#the computation of the two next elasticities will be the same if one uses full_insee_table without weighting instead of pop_mass_per_sector_x_salary 
compute_and_print_elasticity(np.array(pop_mass_per_sector_x_salary['salary_value']),np.array(pop_mass_per_sector_x_salary['emissions_capita']), 'salary', 'income-based emissions', weight = pop_mass_per_sector_x_salary['pop_mass'],print_rsq=True)

compute_and_print_elasticity(np.array(pop_mass_per_sector_x_salary['salary_value']),np.array(pop_mass_per_sector_x_salary['emission_content']), 'salary', 'emission_content', weight = pop_mass_per_sector_x_salary['pop_mass'],print_rsq=True)

mean_emis_content_class_only = mean_emis_content_by_class[ mean_emis_content_by_class['TRNNETO'] != 'All' ]
mean_emis_content_class_only['salary_value'] = mean_emis_content_class_only['TRNNETO'].replace(dic_TRNNETO_to_salary)
compute_and_print_elasticity(np.array(mean_emis_content_class_only['salary_value']),np.array(mean_emis_content_class_only['mean emission content']),'salary','mean emission content per class', weight = mean_emis_content_class_only['pop_mass'], print_rsq=True)


#statistiques by sex and class
sex_class = stat_data_generic(['TRNNETO','SEXE'],full_insee_table,mean_emission_content)
#les femmes ont un contenu en émissions beaucoup plus faibles que les hommes

mean_emission_content_by_age = stat_data_generic(['AGE'],full_insee_table,mean_emission_content)
mean_emission_content_by_age[mean_emission_content_by_age['pop_mass']>=1000]
