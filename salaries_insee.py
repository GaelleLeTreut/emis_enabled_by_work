# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import os
import pandas as pd
import statistics as stat
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dbf_into_csv import *
from sklearn.linear_model import LinearRegression

data_dir = 'data'
output_dir = 'outputs'

###############
# retrieving data
###############


#computed by Inc_Based.py
emis_cont = pd.read_csv( output_dir + os.sep + 'emission_content_france.csv', sep=';', index_col=0, comment='#')


#convertir le fichier dBase de l'Insee en csv s'il ne l'est pas déjà
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
alpha = LinearRegression().fit(x[-2:].reshape(-1,1), y[-2:]).coef_

#Pareto interpolation of mean of last class
dic_TRNNETO_to_salary[23]= alpha/(alpha-1)*50000

#add field to store imputed wage
full_insee_table['salary_value']=full_insee_table['TRNNETO'].replace(dic_TRNNETO_to_salary)


#############
# adding fields of emission content and income-based emissions
#############

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

#create dictionary of emission content and add emission content of branches for each observation
dic_to_emission_content = create_dic_from_sector_to_emission_content(full_insee_table, emis_cont) 
full_insee_table['emission_content'] = full_insee_table['A38'].replace(dic_to_emission_content)

#add income-based emissions for each observation
#Scaling factor to convert emissions in MtCO2 to tCO2
Scaling_factor=10**(-6)
full_insee_table['income-based_emissions'] = full_insee_table['salary_value'] * full_insee_table['emission_content'] * Scaling_factor


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

#######################
# statistical test of independence between branches and wage classes
#######################

#build the frequency matrix if independent
X='A38'
Y='TRNNETO'
cont = full_insee_table[[X,Y]].pivot_table(index=X,columns=Y,aggfunc=len,margins=True,margins_name="Total")
c = cont.fillna(0)
#chi-squared test of independence
# do not include margins in contingency matrix, otherwise the dof are not accurate
st_chi2, st_p, st_dof, st_exp = st.chi2_contingency(c.iloc[:-1,:-1])

##chi2 by hand
##compute the contingency matrix in case of independence
#tx = cont.loc[:,["Total"]]
#ty = cont.loc[["Total"],:]
#n = len(full_insee_table)
#indep = tx.dot(ty) / n
##compute xi2 by hand, ok to include margins, as they are zero here
#measure = (c-indep)**2/indep
#xi_n = measure.sum().sum() #this is the same as st_chi2
##picture the matrix of gap
#table = measure/xi_n
#sns.heatmap(table.iloc[:-1,:-1])#,annot=c.iloc[:-1,:-1])
#plt.show()



# Mean emission by wage class

mean_emis_content_by_class = ut.stat_data_generic(['TRNNETO'],full_insee_table, ut.mean_emission_content)
# Regression 
#full_insee_table['salary_value']

# Indice de vunérabilité région
dict_codereg_to_regname= {1:'Guadeloupe',2:'Martinique',3:'Guyane',4:'Réunion',11:'Île-de-France',21:'Champagne-Ardenne',22:'Picardie',23:'Haute-Normandie',24:'Centre',25:'Basse-Normandie',26:'Bourgogne',31:'Nord-Pas-de-Calais',41:'Lorraine',42:'Alsace',43:'Franche-Comté',52:'Pays de la Loire',53:'Bretagne',54:'Poitou-Charentes',72:'Aquitaine',73:'Midi-Pyrénées',74:'Limousin',82:'Rhône-Alpes',83:'Auvergne',91:'Languedoc-Roussillon',93:'Provence-Alpes-Côte d\'Azur',94:'Corse',99:'Etranger et Tom'}

reg_emis_cont = stat_data_generic(['REGT_AR'], full_insee_table.dropna(subset=['REGT_AR']), mean_emission_content)
reg_emis_cont['REGT_AR_NAME']=reg_emis_cont['REGT_AR'].replace(dict_codereg_to_regname)

#Statistiques Descriptives
#base salaire
mean_salary=round(stat.mean(full_insee_table['salary_value']))
variance_salary=stat.variance(full_insee_table['salary_value'])
median_salary=stat.median(full_insee_table['salary_value'])
decile1_salary=np.percentile(full_insee_table['salary_value'],10)
decile9_salary=np.percentile(full_insee_table['salary_value'],90)
interdecile_salary=decile9_salary/decile1_salary
masses_salary = ratio_of_mass( decile1_salary, decile9_salary, 'salary_value', 'salary_value', full_insee_table)

#France Emissions
mean_emissions=stat.mean(full_insee_table['income-based_emissions'])
variance_emissions=stat.variance(full_insee_table['income-based_emissions'])
median_emissions=stat.median(full_insee_table['income-based_emissions'])
decile1_emissions=np.percentile(full_insee_table['income-based_emissions'],10)
decile9_emissions=np.percentile(full_insee_table['income-based_emissions'],90)
interdecile_emissions=decile9_emissions/decile1_emissions
masses_emissions_ofemitters = ratio_of_mass( decile1_emissions, decile9_emissions, 'income-based_emissions', 'income-based_emissions', full_insee_table)
masses_emissions_ofrich = ratio_of_mass( decile1_salary, decile9_salary, 'income-based_emissions', 'salary_value', full_insee_table)

#comparison of product
mean_content = stat.mean(full_insee_table['emission_content'])
variance_content = stat.variance(full_insee_table['emission_content'])
test= (variance_emissions - (Scaling_factor**2) *(variance_content * variance_salary + variance_salary*(mean_content**2)+variance_content*(mean_salary**2)))/variance_emissions

#comparison of variance with sums of log
full_insee_table['log_income-based_emissions'] = np.log(full_insee_table['income-based_emissions'])
full_insee_table['log_emission_content'] = np.log(full_insee_table['emission_content'])
full_insee_table['log_salary_value'] = np.log(full_insee_table['salary_value'])

variance_log_emissions = stat.variance(full_insee_table['log_income-based_emissions'])
variance_log_content = stat.variance(full_insee_table['log_emission_content'])
variance_log_salary_value = stat.variance(full_insee_table['log_salary_value'])
test2=(variance_log_emissions - variance_log_content - variance_log_salary_value)/variance_log_emissions

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
sex_class['SEXE'] = sex_class['SEXE'].replace(2, 'Female')
sex_class['SEXE'] = sex_class['SEXE'].replace(1, 'Male')
#les femmes ont un contenu en émissions beaucoup plus faibles que les hommes

plt.figure(figsize=(18, 12))
sns.barplot(x="TRNNETO", hue="SEXE", y="mean emission content", data=sex_class)
plt.xlabel("wage class", size=12)
plt.ylabel("gCO2/euro", size=12)
plt.title("Mean emission content by sex and class", size=12)
plt.savefig(OUTPUTS_PATH+'fig_emis_cont_sex_and_class.jpeg', bbox_inches='tight')
plt.show()

sex_class.drop(sex_class.loc[sex_class['SEXE']=='All'].index, inplace=True)
sex_class.drop(sex_class.loc[sex_class['TRNNETO']=='All'].index, inplace=True)

# plt.figure(figsize=(18, 12))
# sns.kdeplot(data=sex_class, x="mean emission content", hue="SEXE", multiple="stack")
# plt.show()


#Table for Gaelle
#building table
raw_table = stat_data_generic(['TRNNETO','A38','SEXE'],full_insee_table,mean_emission_content) 
ordered_table = raw_table.pivot_table(index=['TRNNETO','A38'],columns=['SEXE'],values='pop_mass').reset_index()
# cleaning labels
ordered_table.columns.name = 'index'
ordered_table.rename({1:'male_pop',2:'female_pop','All':'total_pop'},axis=1,inplace=True)
#add emission content
to_merge= raw_table[(raw_table['SEXE']=='All')][['TRNNETO','A38','mean emission content']]
final_table = pd.merge(ordered_table,to_merge,on=['TRNNETO','A38'],how='left')

# Tableau mean emission content by branch  
Mean_emis_branch = pd.DataFrame(to_merge.loc[to_merge['TRNNETO']=='All'][['A38','mean emission content']])
Mean_emis_branch.index =Mean_emis_branch['A38']
Mean_emis_branch.drop('A38', axis=1,inplace=True)

#sur chaque ligne, pour une population caractérisée par sa classe salariale et son sexe, on a la liste des proportions employées dans les différentes secteurs
#full_insee_table.drop(full_insee_table.loc[full_insee_table'A38']='CD')
relative_pop = stat_data_generic(['TRNNETO','SEXE'],full_insee_table, lambda x: proportion_generic(x,'A38'))
relative_pop = relative_pop.fillna(0)
relative_pop['SEXE'] = relative_pop['SEXE'].replace(2, 'Female')
relative_pop['SEXE'] = relative_pop['SEXE'].replace(1, 'Male')

relative_pop.set_index(['TRNNETO','SEXE'], inplace=True)
relative_pop.columns.name= 'A38'
relative_pop= relative_pop.sort_index(axis=1)

## table with diff between male and female pop share 
diff_share_pop = relative_pop.xs('Male', axis=0, level=1, drop_level=True) - relative_pop.xs('Female', axis=0, level=1, drop_level=True)
a, b = relative_pop.index.levels
table_diff_pop = relative_pop.reindex(pd.MultiIndex.from_product([a, [*b, 'diff pop share']]))
table_diff_pop.index.names =['TRNNETO', 'SEXE']
table_diff_pop.loc[pd.IndexSlice[:,('diff pop share')], :] = diff_share_pop.values
table_diff_pop.drop(['Male','Female','All'], level='SEXE',inplace=True)
table_diff_pop  = pd.DataFrame(table_diff_pop.stack())
table_diff_pop = table_diff_pop.droplevel('SEXE')
table_diff_pop.columns=['diff pop by branch']
table_diff_pop.index = table_diff_pop.index.swaplevel(0, 1)
table_diff_pop['mean emission content']=None
table_diff_pop.sort_index(level=0, axis=0, inplace=True)
table_diff_pop.reset_index(inplace=True)
# boucle sur branch -fill table with mean emission content values
for r in Mean_emis_branch.drop('All').index.unique():
     table_diff_pop.loc[table_diff_pop['A38']==r,'mean emission content'] = np.repeat(Mean_emis_branch.loc[[r]].values, len(table_diff_pop.loc[table_diff_pop['A38']==r,'mean emission content']))

## module pour aligner les axes... install via pip
#from mpl_axes_aligner import align
#import math

## Plot histo diff gender in share accross branch - Graph ALL mean
fig, ax1 = plt.subplots(figsize=(18, 12)) # initializes figure and plots
#ax2 = ax1.twinx() # applies twinx to ax2, which is the second y axis. 
ax2 = ax1.twinx()
low1 = min(table_diff_pop.loc[table_diff_pop["TRNNETO"] =='All', "diff pop by branch"])
high1 = max(table_diff_pop.loc[table_diff_pop["TRNNETO"] =='All', "diff pop by branch"])
#plt.ylim([math.ceil(low1-0.5*(high1-low1)), math.ceil(high1+0.5*(high1-low1))])
f1 =sns.barplot(x='A38', y="diff pop by branch", data=table_diff_pop[table_diff_pop['TRNNETO']=='All'].sort_values(by='mean emission content').drop('TRNNETO',axis=1), ax = ax1) # plots the first set of data, and sets it to ax1. 
f2= sns.scatterplot(x ='A38', y ='mean emission content', data=table_diff_pop[table_diff_pop['TRNNETO']=='All'].drop('TRNNETO',axis=1).sort_values(by='mean emission content'), marker='o', ax = ax2, color="firebrick", s=80) # plots the second set, and sets to ax2. 
# these lines add the annotations for the plot. 
#mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.2)
#f1.set_ylim(math.ceil(low1-0.5*(high1-low1)),math.ceil(high1+0.5*(high1-low1)))
ax1.set_xlabel('branches', size=14)
ax1.set_ylabel('Gap between male and female population share (%)', size=14)
ax2.set_ylabel('emission content in gCO2/euro', size=14)
plt.title("wage group", size=14)
plt.show()

##Test Plots - remove CD 
test= table_diff_pop.drop(table_diff_pop.loc[table_diff_pop['A38']=='CD'].index).loc[table_diff_pop.drop(table_diff_pop.loc[table_diff_pop['A38']=='CD'].index)["TRNNETO"] =='All']
fig, ax1 = plt.subplots(figsize=(18, 12)) # initializes figure and plots
#ax2 = ax1.twinx() # applies twinx to ax2, which is the second y axis. 
ax2 = ax1.twinx()
low1 = min(table_diff_pop.loc[table_diff_pop["TRNNETO"] =='All', "diff pop by branch"])
high1 = max(table_diff_pop.loc[table_diff_pop["TRNNETO"] =='All', "diff pop by branch"])
#plt.ylim([math.ceil(low1-0.5*(high1-low1)), math.ceil(high1+0.5*(high1-low1))])
f1 =sns.barplot(x='A38', y="diff pop by branch", data=test.sort_values(by='mean emission content').drop('TRNNETO',axis=1), ax = ax1) # plots the first set of data, and sets it to ax1. 
f2= sns.scatterplot(x ='A38', y ='mean emission content', data=test.sort_values(by='mean emission content'), marker='o', ax = ax2, color="firebrick", s=80) # plots the second set, and sets to ax2. 
# these lines add the annotations for the plot. 
#mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.2)
#f1.set_ylim(math.ceil(low1-0.5*(high1-low1)),math.ceil(high1+0.5*(high1-low1)))
ax1.set_xlabel('branches without CD', size=14)
ax1.set_ylabel('Gap between male and female population share (%)', size=14)
ax2.set_ylabel('emission content in gCO2/euro', size=14)
plt.title("wage group", size=14)
plt.show()


# remove Graph TRNNETO = all
table_diff_pop.drop(table_diff_pop.loc[table_diff_pop['TRNNETO']=='All'].index, inplace=True)

## Plots for all groups boucle sur les classes  
for r in list(table_diff_pop['TRNNETO'].unique()):
    class_r = table_diff_pop.loc[table_diff_pop['TRNNETO']==r,:]
    class_r_raw= class_r.drop('TRNNETO',axis=1).sort_values(by='mean emission content')

    fig, ax1 = plt.subplots(figsize=(18, 12)) # initializes figure and plots
    ax2 = ax1.twinx() # applies twinx to ax2, which is the second y axis. 
    f =sns.barplot(x='A38', y="diff pop by branch", data=class_r_raw, ax = ax1) # plots the first set of data, and sets it to ax1. 
    sns.scatterplot(x ='A38', y ='mean emission content', data=class_r_raw, marker='o', ax = ax2, color="firebrick", s=80) # plots the second set, and sets to ax2. 
    #mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.2)
    # these lines add the annotations for the plot. 
    ax1.set_xlabel('branches', size=14)
    ax1.set_ylabel(' Gap between male and female population share (%)', size=14)
    ax2.set_ylabel('emission content in gCO2/euro', size=14)
    plt.title("wage group:"+str(r), size=14)
    plt.show()

####
## table with absolute male and female pop share
####
# table_relative_pop  = pd.DataFrame(relative_pop.stack())
# table_relative_pop.columns=['Relative pop by branch']
# table_relative_pop.index = table_relative_pop.index.swaplevel(1, 2)
# table_relative_pop['mean emission content']=None
# table_relative_pop.index = table_relative_pop.index.swaplevel(0, 1)
# table_relative_pop.sort_index(level=0, axis=0, inplace=True)
# table_relative_pop.reset_index(inplace=True)
# table_relative_pop.drop(table_relative_pop.loc[table_relative_pop['SEXE']=='All'].index, inplace=True)
# #### boucle sur branch -fill table with mean emission content values
# for r in Mean_emis_branch.drop('All').index.unique():
#     table_relative_pop.loc[table_relative_pop['A38']==r,'mean emission content'] = np.repeat(Mean_emis_branch.loc[[r]].values, len(table_relative_pop.loc[table_relative_pop['A38']==r,'mean emission content']))

## Plot Graph ALL mean
# fig, ax1 = plt.subplots(figsize=(18, 12)) # initializes figure and plots
# ax2 = ax1.twinx() # applies twinx to ax2, which is the second y axis. 
# f =sns.barplot(x='A38', hue="SEXE", y="Relative pop by branch", data=table_relative_pop[table_relative_pop['TRNNETO']=='All'].sort_values(by='mean emission content').drop('TRNNETO',axis=1), ax = ax1) # plots the first set of data, and sets it to ax1. 
# plt.setp(f.get_legend().get_texts(), fontsize=14) # for legend text
# plt.setp(f.get_legend().get_title(), fontsize=14)
# sns.scatterplot(x ='A38', y ='mean emission content', data=table_relative_pop[table_relative_pop['TRNNETO']=='All'].drop('TRNNETO',axis=1).sort_values(by='mean emission content'), marker='o', ax = ax2, color="firebrick", s=80) # plots the second set, and sets to ax2. 
# mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.01)
# ##### these lines add the annotations for the plot. 
# ax1.set_xlabel('branches', size=14)
# ax1.set_ylabel('Population share (%)', size=14)
# ax2.set_ylabel('emission content in gCO2/euro', size=14)
# plt.title("wage group", size=14)
# plt.show()
# #### remove Graph TRNNETO = all
# table_relative_pop.drop(table_relative_pop.loc[table_relative_pop['TRNNETO']=='All'].index, inplace=True)

# #### Plot for all groups - boucle sur les classes  
# for r in list(table_relative_pop['TRNNETO'].unique()):
#     class_r = table_relative_pop.loc[table_relative_pop['TRNNETO']==r,:]
#     class_r_raw= class_r.drop('TRNNETO',axis=1).sort_values(by='mean emission content')

#     fig, ax1 = plt.subplots(figsize=(18, 12)) # initializes figure and plots
#     ax2 = ax1.twinx() # applies twinx to ax2, which is the second y axis. 
#     f =sns.barplot(x='A38', hue="SEXE", y="Relative pop by branch", data=class_r_raw, ax = ax1) # plots the first set of data, and sets it to ax1. 
#     plt.setp(f.get_legend().get_texts(), fontsize=14) # for legend text
#     plt.setp(f.get_legend().get_title(), fontsize=14)
#     sns.scatterplot(x ='A38', y ='mean emission content', data=class_r_raw, marker='o', ax = ax2, color="firebrick", s=80) # plots the second set, and sets to ax2. 
#     # these lines add the annotations for the plot. 
#     ax1.set_xlabel('branches', size=14)
#     ax1.set_ylabel('Population share (%)', size=14)
#     ax2.set_ylabel('emission content in gCO2/euro', size=14)
#     plt.title("wage group:"+str(r), size=14)
#     plt.show()

#statistiques by sex and branch
sex_branch = stat_data_generic(['A38','SEXE'],full_insee_table,mean_emission_content)
sex_branch['SEXE'] = sex_branch['SEXE'].replace(2, 'Female')
sex_branch['SEXE'] = sex_branch['SEXE'].replace(1, 'Male')

#statistiques by age
mean_emission_content_by_age = stat_data_generic(['AGE'],full_insee_table,mean_emission_content)
mean_emission_content_by_age[mean_emission_content_by_age['pop_mass']>=1000]


######
###  Region
###### 
#sur chaque ligne, pour une population caractérisée par sa classe salariale et son sexe, on a la liste des proportions employées dans les différentes secteurs
relative_pop_reg = stat_data_generic(['REGT_AR'],full_insee_table, lambda x: proportion_generic(x,'A38'))
relative_pop_reg = relative_pop_reg.fillna(0)
relative_pop_reg.set_index('REGT_AR',drop=True, inplace=True)
relative_pop_reg.columns.names=['A38']

table_relative_pop_reg  = pd.DataFrame(relative_pop_reg.stack())
table_relative_pop_reg.columns=['Relative pop by branch']
table_relative_pop_reg['mean emission content']=None
table_relative_pop_reg.index = table_relative_pop_reg.index.swaplevel(0, 1)
table_relative_pop_reg.sort_index(level=[0,1], axis=0, inplace=True)
table_relative_pop_reg.reset_index(inplace=True)
# boucle sur branch -fill table with mean emission content values
for r in Mean_emis_branch.drop('All').index.unique():
     table_relative_pop_reg.loc[table_relative_pop_reg['A38']==r,'mean emission content'] = np.repeat(Mean_emis_branch.loc[[r]].values, len(table_relative_pop_reg.loc[table_relative_pop_reg['A38']==r,'mean emission content']))
# remove Graph REGT_AR = all
table_relative_pop_reg.drop(table_relative_pop_reg.loc[table_relative_pop_reg['REGT_AR']=='All'].index, inplace=True)
# Rename regions
table_relative_pop_reg['REGT_AR']=table_relative_pop_reg['REGT_AR'].replace(dict_codereg_to_regname)

## one plot all regions
# fig, ax1 = plt.subplots(figsize=(18, 12)) # initializes figure and plots
# f1 =sns.barplot(x="A38", hue="REGT_AR", y="Relative pop by branch", data=table_relative_pop_reg.sort_values(by='mean emission content'), ax = ax1) # plots the 
# plt.xlabel("branches", size=12)
# plt.ylabel("Relative pop by branch", size=12)
# plt.title("repartition of pop by region", size=12)
# plt.show()

## Plots for all region
for r in list(table_relative_pop_reg['REGT_AR'].unique()):
    region_r = table_relative_pop_reg.loc[table_relative_pop_reg['REGT_AR']==r,:]
    region_r_raw= region_r.drop('REGT_AR',axis=1).sort_values(by='mean emission content')

    fig, ax1 = plt.subplots(figsize=(18, 12)) # initializes figure and plots
    ax2 = ax1.twinx() # applies twinx to ax2, which is the second y axis. 
    f =sns.barplot(x='A38', y="Relative pop by branch", data=region_r_raw, ax = ax1) # plots the first set of data, and sets it to ax1. 
    sns.scatterplot(x ='A38', y ='mean emission content', data=region_r_raw, marker='o', ax = ax2, color="firebrick", s=80) # plots the second set, and sets to ax2. 
    # these lines add the annotations for the plot. 
    ax1.set_xlabel('branches', size=14)
    ax1.set_ylabel('population share (%)', size=14)
    ax2.set_ylabel('emission content in gCO2/euro', size=14)
    plt.title("Region:"+str(r), size=14)
    plt.show()
    
# reg_emis_cont = reg_emis_cont.sort_values(by=['mean emission content'], ascending = False )