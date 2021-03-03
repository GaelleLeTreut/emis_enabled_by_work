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
import utils as ut
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from dbf_into_csv import *
from sklearn.linear_model import LinearRegression
import build_table_survey as bts

##########################
###### Paths 
##########################
output_folder='outputs'
#create output_folder if not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print('Creating ' + output_folder + ' to store outputs')
OUTPUTS_PATH = output_folder + os.sep

data_folder = 'data'
DATA_PATH = data_folder + os.sep

this_file = 'salaries_insee.py'


#######################
# statistical test of independence between branches and wage classes
#######################

#build the frequency matrix if independent
X='A38'
Y='TRNNETO'
cont = bts.full_insee_table[[X,Y]].pivot_table(index=X,columns=Y,aggfunc=len,margins=True,margins_name="Total")
c = cont.fillna(0)
#chi-squared test of independence
# do not include margins in contingency matrix, otherwise the dof are not accurate
st_chi2, st_p, st_dof, st_exp = st.chi2_contingency(c.iloc[:-1,:-1])

##chi2 by hand
##compute the contingency matrix in case of independence
#tx = cont.loc[:,["Total"]]
#ty = cont.loc[["Total"],:]
#n = len(bts.full_insee_table)
#indep = tx.dot(ty) / n
##compute xi2 by hand, ok to include margins, as they are zero here
#measure = (c-indep)**2/indep
#xi_n = measure.sum().sum() #this is the same as st_chi2
##picture the matrix of gap
#table = measure/xi_n
#sns.heatmap(table.iloc[:-1,:-1])#,annot=c.iloc[:-1,:-1])
#plt.show()

###################
# Descriptive statistics
###################

#not used at the moment

##of wages
#mean_salary=round(stat.mean(bts.full_insee_table['salary_value']))
#variance_salary=stat.variance(bts.full_insee_table['salary_value'])
#median_salary=stat.median(bts.full_insee_table['salary_value'])
#decile1_salary=np.percentile(bts.full_insee_table['salary_value'],10)
#decile9_salary=np.percentile(bts.full_insee_table['salary_value'],90)
#interdecile_salary=decile9_salary/decile1_salary
#masses_salary = ratio_of_mass( decile1_salary, decile9_salary, 'salary_value', 'salary_value', bts.full_insee_table)
#
##of emissions
#mean_emissions=stat.mean(bts.full_insee_table['income-based_emissions'])
#variance_emissions=stat.variance(bts.full_insee_table['income-based_emissions'])
#median_emissions=stat.median(bts.full_insee_table['income-based_emissions'])
#decile1_emissions=np.percentile(bts.full_insee_table['income-based_emissions'],10)
#decile9_emissions=np.percentile(bts.full_insee_table['income-based_emissions'],90)
#interdecile_emissions=decile9_emissions/decile1_emissions
#masses_emissions_ofemitters = ratio_of_mass( decile1_emissions, decile9_emissions, 'income-based_emissions', 'income-based_emissions', bts.full_insee_table)
#masses_emissions_ofrich = ratio_of_mass( decile1_salary, decile9_salary, 'income-based_emissions', 'salary_value', bts.full_insee_table)

##variance of income-based emissions compared with product of emission content and wages in case of independence
#mean_content = stat.mean(bts.full_insee_table['emission_content'])
#variance_content = stat.variance(bts.full_insee_table['emission_content'])
#test= (variance_emissions - (Scaling_factor**2) *(variance_content * variance_salary + variance_salary*(mean_content**2)+variance_content*(mean_salary**2)))/variance_emissions

#comparison of variance of log of income-based emissions with sums of variance of log of emission content and of log of wages
bts.full_insee_table['log_income-based_emissions'] = np.log(bts.full_insee_table['income-based_emissions'])
bts.full_insee_table['log_emission_content'] = np.log(bts.full_insee_table['emission_content'])
bts.full_insee_table['log_salary_value'] = np.log(bts.full_insee_table['salary_value'])

matrix = np.cov(np.transpose(np.array(bts.full_insee_table[['log_emission_content','log_salary_value']])))
print('decomposition of variance of log of emission-based')
print("{:.2f}".format(np.sum(matrix)) + " = " + "{:.2f}".format(matrix[0,0]) +" + " + "{:.2f}".format(matrix[1,1]) + " + 2*" +"{:.2f}".format(matrix[0,1]))
relative_matrix = matrix /np.sum(matrix)
print('decomposition of variance of log of emission-based in relative terms')
print("{:.2f}".format(np.sum(relative_matrix)) + " = " + "{:.2f}".format(relative_matrix[0,0]) +" + " + "{:.2f}".format(relative_matrix[1,1]) + " + 2*" +"{:.2f}".format(relative_matrix[0,1]))
print('variability of emissions content is ' + "{:.2f}".format(matrix[0,0]/matrix[1,1]) + ' times that of wages')

#################
# Lorenz and concentration curves
#################

#construct the data for Lorenz curves from grouping by wage classes and branches (because that is all what matters)
bts.full_insee_table['pop_mass']=1
pop_mass_per_sector_x_salary=bts.full_insee_table.groupby(['TRNNETO','A35']).size().reset_index(name='pop_mass')
pop_mass_per_sector_x_salary['emission_content'] = pop_mass_per_sector_x_salary['A35'].replace(bts.dic_to_emission_content_A35)
pop_mass_per_sector_x_salary['salary_value'] = pop_mass_per_sector_x_salary['TRNNETO'].replace(bts.dic_TRNNETO_to_salary)
pop_mass_per_sector_x_salary['salary_mass'] = pop_mass_per_sector_x_salary['salary_value'] * pop_mass_per_sector_x_salary['pop_mass']
pop_mass_per_sector_x_salary['emissions_mass'] = pop_mass_per_sector_x_salary['salary_mass'] * pop_mass_per_sector_x_salary['emission_content']
pop_mass_per_sector_x_salary['emissions_capita'] = pop_mass_per_sector_x_salary['salary_value'] * pop_mass_per_sector_x_salary['emission_content']

ut.make_Lorenz_and_concentration_curves(np.transpose(np.array(pop_mass_per_sector_x_salary[['pop_mass','salary_value', 'emissions_capita']])),{'pop_mass':0,'income':1,'emissions':2},OUTPUTS_PATH + 'Lorenz_curve_French_employee','% data for Lorenz and concentration curves for French employees \n% file automatically created from ' + this_file )

#lowest emitting sector is QA and there is a person of the highest income classes there
#highest emitting sector is CD and there is a person of the lowest income classes there
if ((23 in bts.full_insee_table[bts.full_insee_table['A35']=='QA-QB']['TRNNETO'].unique()) and (0 in bts.full_insee_table[bts.full_insee_table['A35']=='CD']['TRNNETO'].unique() )):
    print('A person of the highest income classes of the lowest emitting sectors emits' + "{:.2f}".format(((bts.dic_TRNNETO_to_salary[23]*bts.dic_to_emission_content_A35['QA-QB'])/(bts.dic_TRNNETO_to_salary[0]*bts.dic_to_emission_content['CD']))[0]) + ' as a person in the lowest income classes of the highest emitting sector, although the reatio of the wages is '+"{:.2f}".format((bts.dic_TRNNETO_to_salary[23]/bts.dic_TRNNETO_to_salary[0])[0])+'.')


##compute share of wages captured by each class
wage_captured = ut.share_generic('salary_value',bts.full_insee_table,['TRNNETO'])
np.sum(wage_captured['share_of_salary_value'][:5])

emis_captured = ut.share_generic('income-based_emissions',bts.full_insee_table,['TRNNETO'])
np.sum(emis_captured['share_of_income-based_emissions'][:5])

#for some reasons this does not work


##############
#regression 
##############

def estimate_OLS(X):
    X2=sm.add_constant(X)
    est=sm.OLS(np.array(bts.full_insee_table['log_emission_content']),X2)
    return est.fit()


#regression of emission content against salary
est_wages_alone = estimate_OLS( np.array(bts.full_insee_table['log_salary_value']).reshape((-1,1)))
print('Regressing mean emission content against log of wages')
print(est_wages_alone.summary())
#get several parameters
#est_salary_alone.params #coefficient of regression
#est_salary_alone.pvalues #p-values
#est_salary_alone.conf_int #confidence interval of coefficient

#regression of emission content against sexe
est_sex_alone = estimate_OLS( np.array(bts.full_insee_table['SEXE']).reshape((-1,1)))
print('Regressing mean emission content against sex')
print(est_sex_alone.summary())

#regression of emission content against wages and sexe
est_wages_and_sex = estimate_OLS( bts.full_insee_table[['log_salary_value','SEXE']])
print('Regressing mean emission content against wages and sex')
print(est_wages_and_sex.summary())

df = summary_col([est_wages_and_sex], stars=True,float_format='%0.3f',info_dict={'$R^2$':lambda x: "{:.3f}".format(x.rsquared)})
latex_str = df.as_latex()
eof='\n'
list_of_line = latex_str.split(eof)

##to test output
#with open('econometric_results.tex','w') as file:
#    file.write(df.as_latex())
#    file.close()

#then tweak the string to format as wanted
with open(OUTPUTS_PATH+'econometric_results.tex','w') as file:
    file.write('\\begin{tabular}{N{3cm}N{2cm}}'+eof)
    file.write('\\toprule'+eof)
    file.write('dependent variable & log carbon intensity\\\\'+eof)
    file.write('\\midrule'+eof)
    file.write('log wage' + ''.join(list_of_line[9].partition('&')[1:]) + eof)
    file.write(list_of_line[10] + eof)
    file.write('female employee' + ''.join(list_of_line[11].partition('&')[1:]) + eof)
    file.write(list_of_line[12] + eof)
    file.write('intercept' + ''.join(list_of_line[7].partition('&')[1:]) + eof)
    file.write(list_of_line[8] + eof)
    file.write('\\midrule'+eof)
    file.write(list_of_line[15].replace('\\$','$') + eof)
    file.write('\\bottomrule'+eof)
    file.write('\\end{tabular}'+eof)
    file.close()


#regression of wages against sex to have the difference in log point
X=np.array(bts.full_insee_table['SEXE']).reshape((-1,1))
X2=sm.add_constant(X)
est=sm.OLS(np.array(bts.full_insee_table['log_salary_value']),X2)
est_wages_against_sex = est.fit()
print(est_wages_against_sex.summary())


#build the frequency matrix if independent
X='A35'
Y='SEXE'
cont = bts.full_insee_table[[X,Y]].pivot_table(index=X,columns=Y,aggfunc=len,margins=True,margins_name="Total")
c = cont.fillna(0)
##compute the contingency matrix in case of independence
tx = cont.loc[:,["Total"]]
ty = cont.loc[["Total"],:]
n = len(bts.full_insee_table)
indep = tx.dot(ty) / n
#compute xi2 by hand, ok to include margins, as they are zero here
measure = (c-indep)**2/indep
xi_n = measure.sum().sum() #this is the same as st_chi2
#picture the matrix of gap
table = measure/xi_n
sns.heatmap(table.iloc[:-1,:-1])#,annot=c.iloc[:-1,:-1])
#plt.show()
plt.close()
#three striking sector

#share captured by women
pop_share = ut.share_generic('POND',bts.full_insee_table,['SEXE'])
wage_share = ut.share_generic('salary_value',bts.full_insee_table,['SEXE'])
emis_share = ut.share_generic('income-based_emissions',bts.full_insee_table,['SEXE'])

with open(OUTPUTS_PATH+'share-captured-by-gender.tex','w') as file:
    file.write('\\begin{tabular}{L{1.5cm}L{2cm}L{2cm}L{3cm}}'+eof)
    file.write('\\toprule'+eof)
    file.write('& share of population & share of wages earned & share of income-based emissions \\\\'+eof)
    file.write('\\midrule'+eof)
    file.write('men &' + "{:.1f}".format(pop_share.iloc[0,1]) + '&'+"{:.1f}".format(wage_share.iloc[0,1]) +'&'+"{:.1f}".format(emis_share.iloc[0,1]) +'\\\\' +eof)
    file.write('women &' + "{:.1f}".format(pop_share.iloc[1,1]) + '&'+"{:.1f}".format(wage_share.iloc[1,1]) +'&'+"{:.1f}".format(emis_share.iloc[1,1]) +'\\\\' +eof)
    file.write('\\bottomrule'+eof)
    file.write('\\end{tabular}'+eof)
#
#emis_captured = ut.share_generic('income-based_emissions',bts.full_insee_table,['TRNNETO'])
#np.sum(emis_captured['share_of_income-based_emissions'][:5])


#employment of women by working condition
working_condition_by_sex=bts.full_insee_table.groupby(['CPFD','SEXE']).apply(len).reset_index()
working_condition_by_sex['proportion_by_sex']= pd.concat((working_condition_by_sex[working_condition_by_sex['SEXE']==1][0]/np.sum(working_condition_by_sex[working_condition_by_sex['SEXE']==1][0]), working_condition_by_sex[working_condition_by_sex['SEXE']==2][0]/np.sum(working_condition_by_sex[working_condition_by_sex['SEXE']==2][0])))

#difference of wages at full time jobs
full_time_table=bts.full_insee_table[bts.full_insee_table['CPFD']=='C']
X=np.array(full_time_table['SEXE']).reshape((-1,1))
X2=sm.add_constant(X)
est=sm.OLS(np.array(full_time_table['log_salary_value']),X2)
est_wages_against_sex = est.fit()
print(est_wages_against_sex.summary())

#mean_emis_content_class_only = mean_emis_content_by_class[ mean_emis_content_by_class['TRNNETO'] != 'All' ]
#mean_emis_content_class_only['salary_value'] = mean_emis_content_class_only['TRNNETO'].replace(bts.dic_TRNNETO_to_salary)
#compute_and_print_elasticity(np.array(mean_emis_content_class_only['salary_value']),np.array(mean_emis_content_class_only['mean emission content']),'salary','mean emission content per class', weight = mean_emis_content_class_only['pop_mass'], print_rsq=True)

## PLOT
sns.set_context('paper', font_scale=0.9)

mean_emis_content_by_class = ut.stat_data_generic(['TRNNETO'],bts.full_insee_table, ut.mean_emission_content)

#lowest and highest value per income class
mean_emis_content_by_class.sort_values('mean emission content')


#plot for mean emission content by wage classes
plt.figure(figsize=(18, 12))
sns.barplot(x=mean_emis_content_by_class['TRNNETO'], y="mean emission content", data=mean_emis_content_by_class,palette='deep')
plt.xlabel("wage class", size=12)
plt.ylabel("gCO2/euro", size=12)
#plt.title("Mean carbon intensity by wage classes", size=12)
plt.savefig(OUTPUTS_PATH+'fig_mean_emis_cont_by_class.jpeg', bbox_inches='tight')


fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlabel('Wages')
plt.ylabel('Emission content')
plt.scatter(pop_mass_per_sector_x_salary['salary_value'], pop_mass_per_sector_x_salary['emission_content'], s=pop_mass_per_sector_x_salary['pop_mass']/100,marker='o')
plt.savefig(OUTPUTS_PATH + 'wages_per_sector.pdf',bbox_inches='tight')
plt.close()

#statistiques by sex and class
sex_class = ut.stat_data_generic(['TRNNETO','SEXE'],bts.full_insee_table, ut.mean_emission_content)
#here look for mean emission content by sex

sex_class['SEXE'].replace({1:'Male',2:'Female'},inplace=True)
#les femmes ont un contenu en émissions beaucoup plus faibles que les hommes

plt.figure(figsize=(18, 12))
sns.barplot(x="TRNNETO", hue="SEXE", y="mean emission content", data=sex_class)
plt.xlabel("wage class", size=12)
plt.ylabel("gCO2/euro", size=12)
plt.title("Mean emission content by sex and class", size=12)
plt.savefig(OUTPUTS_PATH+'fig_emis_cont_sex_and_class.jpeg', bbox_inches='tight')
#plt.show()
plt.close()

sex_class.drop(sex_class.loc[sex_class['SEXE']=='All'].index, inplace=True)
sex_class.drop(sex_class.loc[sex_class['TRNNETO']=='All'].index, inplace=True)

# plt.figure(figsize=(18, 12))
# sns.kdeplot(data=sex_class, x="mean emission content", hue="SEXE", multiple="stack")
# plt.show()


#Table for Gaelle
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

#sur chaque ligne, pour une population caractérisée par sa classe salariale et son sexe, on a la liste des proportions employées dans les différentes secteurs
#bts.full_insee_table.drop(bts.full_insee_table.loc[bts.full_insee_table'A35']='CD')
relative_pop = ut.stat_data_generic(['TRNNETO','SEXE'],bts.full_insee_table, lambda x: ut.proportion_generic_weighted(x,'A35'))
relative_pop = relative_pop.fillna(0)
relative_pop['SEXE'].replace({1:'Male',2:'Female'},inplace=True)

relative_pop.set_index(['TRNNETO','SEXE'], inplace=True)
relative_pop.columns.name= 'A35'
relative_pop= relative_pop.sort_index(axis=1)


table_group_pop = relative_pop.drop(['Male','Female'], level='SEXE')
table_group_pop= pd.DataFrame(table_group_pop.stack())
table_group_pop = table_group_pop.droplevel('SEXE')
table_group_pop.columns=['pop by branch']
table_group_pop.index = table_group_pop.index.swaplevel(0, 1)
table_group_pop['mean emission content']=None
table_group_pop.sort_index(level=0, axis=0, inplace=True)
table_group_pop.reset_index(inplace=True)
# boucle sur branch -fill table with mean emission content values
for r in Mean_emis_branch.drop('All').index.unique():
     table_group_pop.loc[table_group_pop['A35']==r,'mean emission content'] = np.repeat(Mean_emis_branch.loc[[r]].values, len(table_group_pop.loc[table_group_pop['A35']==r,'mean emission content']))
table_group_pop.drop(table_group_pop.loc[table_group_pop['TRNNETO']=='All'].index, inplace=True)

## Plots for all groups boucle sur les classes  into one graph
Tot = len(list(table_group_pop['TRNNETO'].unique()))
Cols = 4
Rows = Tot // Cols 
# sharey=True pour même scale sur y
fig, axes = plt.subplots(nrows=Rows, ncols=Cols, figsize=(40, 30),sharey=True)
#fig, axes = plt.subplots(nrows=Rows, ncols=Cols, figsize=(40, 30))
i = 0
for row in axes:
    for ax1 in row:
        r = list(table_group_pop['TRNNETO'].unique())[i]
        class_r = table_group_pop.loc[table_group_pop['TRNNETO']==r,:]
        class_r_raw= class_r.drop('TRNNETO',axis=1).sort_values(by='mean emission content')
        #ax2 = ax1.twinx() # applies twinx to ax2, which is the second y axis. 
        sns.barplot(x='A35', y="pop by branch", data=class_r_raw, ax = ax1, palette='deep') # plots the first set of data, and sets it to ax1. 
        #sns.scatterplot(x ='A35', y ='mean emission content', data=region_r_raw, marker='o', ax = ax2, color="firebrick", s=80) # plots the second set, and sets to ax2. 
        # these lines add the annotations for the plot. 
        #ax1.set_xlabel('branches')
        ax1.set_ylabel('pop (%)')
        #ax2.set_ylabel('emission content in gCO2/euro', size=14)
        ax1.set_title("wage group:"+str(r))
        i += 1
plt.tight_layout()
plt.savefig(OUTPUTS_PATH+'fig_group_pop_panel.jpeg', bbox_inches='tight')


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
     table_diff_pop.loc[table_diff_pop['A35']==r,'mean emission content'] = np.repeat(Mean_emis_branch.loc[[r]].values, len(table_diff_pop.loc[table_diff_pop['A35']==r,'mean emission content']))

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
f1 =sns.barplot(x='A35', y="diff pop by branch", data=table_diff_pop[table_diff_pop['TRNNETO']=='All'].sort_values(by='mean emission content').drop('TRNNETO',axis=1), ax = ax1, palette='deep') # plots the first set of data, and sets it to ax1. 
f2= sns.scatterplot(x ='A35', y ='mean emission content', data=table_diff_pop[table_diff_pop['TRNNETO']=='All'].drop('TRNNETO',axis=1).sort_values(by='mean emission content'), marker='o', ax = ax2, color="firebrick", s=80) # plots the second set, and sets to ax2. 
# these lines add the annotations for the plot. 
#mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.2)
#f1.set_ylim(math.ceil(low1-0.5*(high1-low1)),math.ceil(high1+0.5*(high1-low1)))
ax1.set_xlabel('branches', size=14)
ax1.set_ylabel('Gap between male and female population share (%)', size=14)
ax2.set_ylabel('carbont intensity in gCO2/euro', size=14)
plt.title("wage group", size=14)
plt.savefig(OUTPUTS_PATH+'fig_relative_pop_and_intensity_by_industry.jpeg', bbox_inches='tight')
#plt.show()
plt.close()

##Test Plots - remove CD 
test= table_diff_pop.drop(table_diff_pop.loc[table_diff_pop['A35']=='CD'].index).loc[table_diff_pop.drop(table_diff_pop.loc[table_diff_pop['A35']=='CD'].index)["TRNNETO"] =='All']
fig, ax1 = plt.subplots(figsize=(18, 12)) # initializes figure and plots
#ax2 = ax1.twinx() # applies twinx to ax2, which is the second y axis. 
ax2 = ax1.twinx()
low1 = min(table_diff_pop.loc[table_diff_pop["TRNNETO"] =='All', "diff pop by branch"])
high1 = max(table_diff_pop.loc[table_diff_pop["TRNNETO"] =='All', "diff pop by branch"])
#plt.ylim([math.ceil(low1-0.5*(high1-low1)), math.ceil(high1+0.5*(high1-low1))])
f1 =sns.barplot(x='A35', y="diff pop by branch", data=test.sort_values(by='mean emission content').drop('TRNNETO',axis=1), ax = ax1, palette='deep') # plots the first set of data, and sets it to ax1. 
f2= sns.scatterplot(x ='A35', y ='mean emission content', data=test.sort_values(by='mean emission content'), marker='o', ax = ax2, color="firebrick", s=80) # plots the second set, and sets to ax2. 
# these lines add the annotations for the plot. 
#mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.2)
#f1.set_ylim(math.ceil(low1-0.5*(high1-low1)),math.ceil(high1+0.5*(high1-low1)))
ax1.set_xlabel('branches without CD', size=14)
ax1.set_ylabel('Gap between male and female population share (%)', size=14)
ax2.set_ylabel('emission content in gCO2/euro', size=14)
plt.title("wage group", size=14)
#plt.show()
plt.close()


# remove Graph TRNNETO = all
table_diff_pop.drop(table_diff_pop.loc[table_diff_pop['TRNNETO']=='All'].index, inplace=True)

## Plots for all groups boucle sur les classes  
#for r in list(table_diff_pop['TRNNETO'].unique()):
#    class_r = table_diff_pop.loc[table_diff_pop['TRNNETO']==r,:]
#    class_r_raw= class_r.drop('TRNNETO',axis=1).sort_values(by='mean emission content')
#
#    fig, ax1 = plt.subplots(figsize=(18, 12)) # initializes figure and plots
#    ax2 = ax1.twinx() # applies twinx to ax2, which is the second y axis. 
#    f =sns.barplot(x='A35', y="diff pop by branch", data=class_r_raw, ax = ax1, palette='deep') # plots the first set of data, and sets it to ax1. 
#    sns.scatterplot(x ='A35', y ='mean emission content', data=class_r_raw, marker='o', ax = ax2, color="firebrick", s=80) # plots the second set, and sets to ax2. 
#    #mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.2)
#    # these lines add the annotations for the plot. 
#    ax1.set_xlabel('branches', size=14)
#    ax1.set_ylabel(' Gap between male and female population share (%)', size=14)
#    ax2.set_ylabel('emission content in gCO2/euro', size=14)
#    plt.title("wage group:"+str(r), size=14)
#    #plt.show()
#    plt.close()




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
#     table_relative_pop.loc[table_relative_pop['A35']==r,'mean emission content'] = np.repeat(Mean_emis_branch.loc[[r]].values, len(table_relative_pop.loc[table_relative_pop['A35']==r,'mean emission content']))

## Plot Graph ALL mean
# fig, ax1 = plt.subplots(figsize=(18, 12)) # initializes figure and plots
# ax2 = ax1.twinx() # applies twinx to ax2, which is the second y axis. 
# f =sns.barplot(x='A35', hue="SEXE", y="Relative pop by branch", data=table_relative_pop[table_relative_pop['TRNNETO']=='All'].sort_values(by='mean emission content').drop('TRNNETO',axis=1), ax = ax1) # plots the first set of data, and sets it to ax1. 
# plt.setp(f.get_legend().get_texts(), fontsize=14) # for legend text
# plt.setp(f.get_legend().get_title(), fontsize=14)
# sns.scatterplot(x ='A35', y ='mean emission content', data=table_relative_pop[table_relative_pop['TRNNETO']=='All'].drop('TRNNETO',axis=1).sort_values(by='mean emission content'), marker='o', ax = ax2, color="firebrick", s=80) # plots the second set, and sets to ax2. 
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
#     f =sns.barplot(x='A35', hue="SEXE", y="Relative pop by branch", data=class_r_raw, ax = ax1,palette='deep') # plots the first set of data, and sets it to ax1. 
#     plt.setp(f.get_legend().get_texts(), fontsize=14) # for legend text
#     plt.setp(f.get_legend().get_title(), fontsize=14)
#     sns.scatterplot(x ='A35', y ='mean emission content', data=class_r_raw, marker='o', ax = ax2, color="firebrick", s=80) # plots the second set, and sets to ax2. 
#     # these lines add the annotations for the plot. 
#     ax1.set_xlabel('branches', size=14)
#     ax1.set_ylabel('Population share (%)', size=14)
#     ax2.set_ylabel('emission content in gCO2/euro', size=14)
#     plt.title("wage group:"+str(r), size=14)
#     plt.show()

#statistiques by sex and branch
sex_branch = ut.stat_data_generic(['A35','SEXE'],bts.full_insee_table,ut.mean_emission_content)
sex_branch['SEXE'].replace({1:'Male',2:'Female'},inplace=True)

#statistiques by age
mean_emission_content_by_age = ut.stat_data_generic(['AGE'],bts.full_insee_table,ut.mean_emission_content)
mean_emission_content_by_age[mean_emission_content_by_age['pop_mass']>=1000]

