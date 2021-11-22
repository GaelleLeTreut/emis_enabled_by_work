# -*- coding: utf-8 -*-
# utils.py module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############################
#functions for dataframe
############################
def df_to_csv_with_comment(df, output_file, comment, **kwargs):
    with open(output_file, 'w') as f:
        f.write(comment)
        f.write('\n')
        df.to_csv(f, **kwargs)

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

def mean_salary(table):
   """
   compute the mean emission content (weighted by value-added (here wages)
   """
   mean = np.sum(table['salary_value'] )/ len(table)
   return pd.Series([mean],index=['mean wages'])

def new_proportion(x, category):
    """
    Return proportion of each modality of a category

    DataFrame, string -> Series
    compute the proportion (in percentage) of each modality of the category (category) of DataFrame x (with weigth given by weight_label)
    """
    return (x.groupby(category).apply(lambda y: pd.Series({'proportion':len(y)}))/len(x)).reset_index()

def proportion_weighted_by_wages(x, category):
    """
    Return proportion of each modality of a category

    DataFrame, string -> Series
    compute the proportion (in percentage) of each modality of the category (category) of DataFrame x (with weigth given by weight_label)
    """
    return (x.groupby(category).apply(lambda y: pd.Series({'proportion':np.sum(y['salary_value'])}))/np.sum(x['salary_value'])).reset_index()

def proportion_generic(x, category):
    """
    Return proportion of each modality of a category

    DataFrame, string -> Series
    compute the proportion (in percentage) of each modality of the category (category) of DataFrame x (with weigth given by weight_label)
    """
    def fun(modality, x):
        #return pd.Series({'proportion_of_'+str(modality)+'_in_'+category: 100 * len(x[x[category] == modality]) /len(x)})
        return pd.Series({str(modality): 100 * len(x[x[category] == modality]) /len(x)})
    return apply_over_labels( fun, sorted(x[category].unique()), x )

def proportion_generic_weighted(x, category):
    """
    Return proportion of each modality of a category

    DataFrame, string -> Series
    compute the proportion (in percentage) of each modality of the category (category) of DataFrame x (with weigth given by weight_label)
    """
    def fun(modality, x):
        #return pd.Series({'proportion_of_'+str(modality)+'_in_'+category: 100 * len(x[x[category] == modality]) /len(x)})
        return pd.Series({str(modality): 100 * np.sum(x[x[category] == modality]['salary_value']) /np.sum(x['salary_value'])})
    return apply_over_labels( fun, sorted(x[category].unique()), x )

def apply_over_labels(fun,list_of_label, *args):
    """apply function for each label"""
    d=pd.Series([])
    for l in list_of_label:
        d = d.append( fun(l, *args) )
    return d

def stat_data_generic_df(list_of_label, x, fun):
    if len(list_of_label)==0:
        return fun(x)
    else:
        label = list_of_label[0]
        x_extracted_lab = x.groupby([label]).apply(lambda y: stat_data_generic_df(list_of_label[1:], y, fun)).reset_index().drop('level_1',axis=1)
        x_extracted_all = stat_data_generic_df(list_of_label[1:],x,fun)
        x_extracted_all[label] = 'All'
        x_extracted = pd.concat([x_extracted_lab,x_extracted_all], sort=False,ignore_index=True)
        return x_extracted

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

def share_generic(label, x, list_of_category):
    """
    Return the share of variable for each combination of  category

    string, DataFrame, list of string -> DataFrame
    compute the share of total mass of variable label of DataFrame x that belongs to each combination of modalities of list_of_category (with weigth given by weight_label)
    Caution variable label must have an extensive meaning, otherwise computing the share over a DataFrame is meaningless
    """
    share_label='share_of_'+label
    fun = lambda y: pd.Series({share_label:np.sum(y[label])})
    df = stat_data_generic(list_of_category, x, fun)
    df[share_label] = df[share_label]/(fun(x)[share_label])*100
    return df

def make_Lorenz_and_concentration_curves(table,dic_index,file_name, comment):
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
    Lorenz_curve_data=pd.DataFrame({'pop_cum_by_income':pop_cum_by_income, 'pop_cum_by_emissions':pop_cum_by_emissions, 'emissions_cum_by_emissions':emissions_cum_by_emissions, 'emissions_cum_by_income':emissions_cum_by_income, ' income_cum_by_income': income_cum_by_income})
    df_to_csv_with_comment(Lorenz_curve_data, file_name+'.csv', comment, index=False)
   
   #figure no longer drawn
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #plt.plot([0,1],[0,1],'k:',linewidth=1)
    #plt.plot(pop_cum_by_income, emissions_cum_by_income,label='concentration curve of income-based emissions')
    #plt.plot(pop_cum_by_emissions, emissions_cum_by_emissions,label='Lorenz curve for income-based emissions')
    #plt.plot(pop_cum_by_income, income_cum_by_income,label='Lorenz curve for income')
    #plt.xlabel('Cumulative share of French wage earners')
    #plt.ylabel('Cumulative share of income or income-based emissions')
    #plt.axis([0,1,0,1])
    #plt.legend(loc='upper left')
    #ax.set_aspect('equal', adjustable='box')
    #plt.savefig(file_name + '.pdf',bbox_inches='tight')
    #plt.close()

