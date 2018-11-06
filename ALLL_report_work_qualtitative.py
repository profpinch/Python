# -*- coding: utf-8 -*-
"""
Draft Code for Tony Frame, Audit Support Work.

"""

from fredapi import Fred
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm #for regression estimate
import seaborn as sns; sns.set(color_codes=True)
from functools import reduce
import scipy as sp

def Univariate_Regression(X,Y,ticker=False,const=1,regression_output=False):
    '''
    Input:
    X = Explanatory variable for OLS regression
    Y = Dependent variable for OLS regression
    ticker = variable used for the qualitative analysis, if ticker = true then generate graphs for all FRED tickers (no Overlays in graphs)
    const = 1 indicates to add constant. if user specified anything else then no constant
    regression_output = False indicates that summary of regression is supressed. True indicates that it is not suppressed
    
    Output:
    1. Graphical result of regression (with confidence bands)
    2. Regression summary 
    '''
    
    #regression estimate with constant
    if const == 1:
        X1 = sm.add_constant(X)
    else:
        X1 = X
    
    model1 = sm.OLS(Y, X1).fit()
    #Get the slope, parameter and p-value
    intercept = np.round(model1.params[0],3)
    parameter = np.round(model1.params[1],3)
    p_val_param = np.round(model1.pvalues[1],3)
    
    if ticker == False:
        if p_val_param > 0.1:
            ax1 = sns.regplot(x=X,y=Y, line_kws={'label':"y={0} + {1}X \n(p-value parameter: {2} (not significant at 10% level))".format(intercept,parameter,p_val_param)})
            ax1.legend(loc='best')
            ax1.set_title("{0} vs. {1}".format(Y.name,X.name),fontsize=18)
            ax1.figure.set_size_inches(12,8)
        else:
            ax1 = sns.regplot(x=X,y=Y, line_kws={'label':"y={0} + {1}X \n(p-value parameter: {2} (significant at 10% level))".format(intercept,parameter,p_val_param)})
            ax1.legend(loc='best')
            ax1.set_title("{0} vs. {1}".format(Y.name,X.name),fontsize=18)
            ax1.figure.set_size_inches(12,8)
        

    if ticker == True:
        if p_val_param > 0.1:
            plt.figure()
            ax1 = sns.regplot(x=X,y=Y, line_kws={'label':"y={0} + {1}X \n(p-value parameter: {2} (not significant at 10% level))".format(intercept,parameter,p_val_param)})
            ax1.legend(loc='best')
            ax1.set_title("{0} vs. {1}".format(Y.name,X.name),fontsize=18)
            ax1.figure.set_size_inches(12,8)
        else:
            plt.figure()
            ax1 = sns.regplot(x=X,y=Y, line_kws={'label':"y={0} + {1}X \n(p-value parameter: {2} (significant at 10% level))".format(intercept,parameter,p_val_param)})
            ax1.legend(loc='best')
            ax1.set_title("{0} vs. {1}".format(Y.name,X.name),fontsize=18)
            ax1.figure.set_size_inches(12,8)
            
    if regression_output == False:
        pass
    else:
        return model1.summary()

    
'''def Multivariate_Regression(X,Y,Date,const=1):
    
    Input:
    X = Explanatory variable for OLS regression (more than one)
    Y = Dependent variable for OLS regression
    const = 1 indicates to add constant. if user specified anything else then no constant
    Dates = Dates from data
    
    Output:
    1. Regression Summary
    
    #regression estimate with constant
    if const == 1:
        X1 = sm.add_constant(X)
    else:
        X1 = X1
    
    model = sm.OLS(Y, X1).fit()
    
    #Graph fitted values
    fit = model.fittedvalues
    plt.figure(2,figsize=(12,8))
    plt.plot(Date,fit,color='blue',linewidth=2,label='Fitted Values/Predicted Values')
    plt.plot(Date,Y,color = 'green', linewidth=2, label='{0}'.format(Y.name))
    plt.grid(True)
    plt.legend(loc='best',fancybox=True)
    plt.suptitle("Predicted Values vs. {0}".format(Y.name),fontsize=18)
    plt.setp(plt.gca().xaxis.get_majorticklabels(),rotation=90)

    return model.summary() '''
    
def Add_FRED_Data(ticker,indicator,data, merge=True):
    '''
    Input:
    Ticker = list of tickers that correspond to FRED database
    indicator = list of integers that indicate for each ticker if percentage change is used to transform data or not
    data = original data that has been tranformed to pct change. THis is used to add FRED data to original data set
    merge = indicator if data set is to be merged on original data
    
    Output:
    df_final =  combined dataset of all tickers and original data set
    
    '''
    #the fred API key so we can pull data from FRED
    fred = Fred(api_key='f8da225a41713e19fd1f1929d99a2e05')
    #function to create YearQ1/Q2/Q3/Q4 format (used at end)
    def convert_to_quarter(d):
        ts = pd.Timestamp(pd.to_datetime(d,format='%Y%m'))
        return "{}Q{}".format(ts.year, ts.quarter)
    
    FRED_Data = []
    
    for j in range(len(ticker)):
        FRED = fred.get_series(ticker[j])
        
        #Step 1: investigate if monthly or quarterly. If monthly data transform to quarterly by mean aggregation rule
        date1,date2 = FRED.index[0],FRED.index[1]        
        #convert to quarterly if freq is monthly
        if (date2-date1).days >= 28:
            FRED = FRED.resample('QS').mean()
        else:
            FRED=FRED
            
        #Step 2: investigate what indicator is assigned for each variable. 2= transform to pct change, 1=difference, 0=no transformation
        if indicator[j] == 2:
            FRED = FRED.pct_change(periods=4)*100 #year over year percentage change
        elif indicator[j] == 1:
            FRED = FRED.diff() #difference
        elif indicator[j]== 0:
            FRED=FRED #no change to variable
        
        #Convert to Quarterly String to match the original dataset
        Date = []
        for i in range(len(FRED)):
            Date.append(convert_to_quarter(FRED.index[i]))
        FRED_Data.append(pd.DataFrame({'Date':Date,'{0}'.format(ticker[j]):FRED}))
        
    #Now Merge the datasets into the original Dataset
    df_combo = reduce(lambda left,right: pd.merge(left,right,on='Date'),FRED_Data)
    if merge == True:
        df_final = pd.merge(data,df_combo,on='Date')
        df_final.index = df_final['Date']
        df_final = df_final.drop('Date',axis=1)
    else:
        df_final=df_combo
        df_final.index = df_final['Date']
        df_final = df_final.drop('Date',axis=1)
            
    return df_final    

def Lagged_DataFrame(df,lags=1):
    '''
    creates a lag dataframe for all variables in a dataset. Needed for optimzation of the lagged variables
    '''
    
    def suffixed_columns(df, suffix):
        return ["{}{}".format(column, suffix) for column in df.columns]
    
    
    def lag(df, n):
        new_df = df.shift(n)
        new_df.columns = suffixed_columns(df, "_Lag{:02d}".format(n))
        return new_df

    return pd.concat([df] + [lag(df, i) for i in range(1, lags + 1)], axis=1)


def Selecting_Lag_Regression_Factor(data_lag,data_orig,ticker,supress_graph=False):
    '''
    Input:
    data_lag = is data preprocessed with lags for each ticker
    data_orig = original data, which is needed to get Quarterly Loss %
    ticker = the tickers from FRED data, used to filter out regression analysis
    
    Output:
    append_final_reg = a list with the selected regressions for each ticker based on AIC optimization
    append_final_res = a list of results based on AIC analysis. shows AIC for each lag variables for a given ticker
    Selections = final selections for each ticker
    '''
    #now for each unique ticker we want to investigate the lag-length and which one is the best.
    append_final_reg = []
    append_final_res = []
    Selections = []
    for tick in ticker:
        filter_col = [col for col in data_lag if col.startswith(tick)]
        ticker1 = data_lag[filter_col]
        ticker1['Date'] = ticker1.index
        #now match column 'Quarterly Loss %' on this dataframe
        intermediate_df = data_orig[['Date','Quarterly Loss %']]
        ticker1 = pd.merge(ticker1,intermediate_df,on='Date',how='left')
        ticker1 = ticker1.set_index("Date")
        ticker1 = ticker1.dropna()
        
        Save_AIC = []
        for col in filter_col:
            X = sm.add_constant(ticker1[col])
            model1 = sm.OLS(ticker1['Quarterly Loss %'], X).fit()
            Save_AIC.append(model1.aic)
        #get together DF
        Results = pd.DataFrame({'Col Name':filter_col,'AIC':Save_AIC})
        Selection = Results.loc[Results['AIC'].idxmin()]
        X = sm.add_constant(ticker1[Selection['Col Name']])
        reg_res = sm.OLS(ticker1['Quarterly Loss %'], X).fit()
        append_final_res.append(Results)
        append_final_reg.append(reg_res.summary())
        Selections.append(Selection)
        
    #Last step. Generate the graphs for the selected regressions
        if supress_graph == False:
            Univariate_Regression(ticker1[Selection['Col Name']],ticker1['Quarterly Loss %'],ticker=True,const=1,regression_output=False)
        else:
            pass
    
    return append_final_reg, append_final_res, Selections


def Qualitative_Assessments(data,ticker):
    #Step 1 obtain graphs and regression estimates for all factors vs. Quarterly Loss % (contemp. effect. for lagged effect look at lag regression selection function above)
    regressions = []
    for i in range(len(ticker)):
        regressions.append(Univariate_Regression(data[ticker[i]],data['Quarterly Loss %'],ticker=True,regression_output=True))

def Heatmap_Factors(df,data_lags,Selections, revert):
    '''
    Input:
    df = original data set, used to match dates with lagged dataset
    data_lags = dataset that has lagged factor variables
    Selections = list that contains the selected lagged variable regression based on optimization
    revert = list of True/False statements. if True then the colorbar is reverted for the factor. i.e. GDP increases, then losses should decrease, i.e. need a reverted colorbar
    
    Output:
    Heatmap that shows the quarterly loss % (ranked) and a specific factor
    '''
    
    
    for j in range(len(Selections)):
        df['Date'] = df.index
        data_lags['Date'] = data_lags.index
        merged_data = pd.merge(df,data_lags,on='Date')
        merged_data = merged_data.set_index("Date")
        #now select the data which is needed to test
        
        #now do a heat map analysis first split the data
        df1 = np.round(pd.DataFrame(merged_data['Quarterly Loss %']),3)
        df2 = np.round(pd.DataFrame(merged_data[Selections[j][1]]),3)
        
        #replicate Excels PERCENTRANK.INC by using scipy
        percentile_array = df1.drop(df1.idxmax())
        percentile_array2 = df2.drop(df2.idxmax()) #IMPORTANT as Excel excludes
        rank_array_df1 = []
        rank_array_df2 = []
        for i in range(len(df1)):
            rank_array_df1.append(sp.stats.percentileofscore(percentile_array,df1.ix[i][0],kind='strict'))
            rank_array_df2.append(sp.stats.percentileofscore(percentile_array2,df2.ix[i][0],kind='strict'))
        
        df1_final = pd.DataFrame({'{0} Ranked(in %)'.format(df1.dtypes.index[0]):np.round(rank_array_df1,2)}, index=df1.index)
        df2_final = pd.DataFrame({'{0} Ranked(in %)'.format(df2.dtypes.index[0]):np.round(rank_array_df2,2)}, index=df2.index)
        
        
        #now do a heat map analysis  
        fig, (ax,ax2) = plt.subplots(ncols=2, figsize=(10,14))
        fig.subplots_adjust(wspace=0.03)
        
        sns.heatmap(df1_final, cmap='RdYlGn_r',fmt="g", ax=ax,linewidths=0.5, annot=True, cbar=False)
        fig.colorbar(ax.collections[0], ax=ax, location = "left", use_gridspec=False, pad=0.2)
        if revert[j] == True:
            sns.heatmap(df2_final, cmap='RdYlGn',fmt="g", ax=ax2 ,linewidths=0.5, annot=True, cbar=False)
        else:
            sns.heatmap(df2_final, cmap='RdYlGn_r',fmt="g", ax=ax2 ,linewidths=0.5, annot=True, cbar=False)
            
        fig.colorbar(ax2.collections[0], ax=ax2, location = "right", use_gridspec=False, pad=0.2)
        ax2.yaxis.tick_right()
        ax2.tick_params(rotation=0)
        ax.set_ylabel('')
        ax2.set_ylabel('')
        plt.suptitle("Heatmap of {0} and {1}".format(df1_final.dtypes.index[0],df2_final.dtypes.index[0]), fontsize = 14, y=0.92)
        plt.show()
    






'''
This section is used to test the functions and run a couple of examples
'''

#Import the dataset
df = pd.read_excel(open('Loss Rates.xlsx','rb'),sheet_name='Sheet1')
df['Quarterly Loss %'] = df['Quarterly Loss %'].apply(lambda x: x*100)

''' #generate univariate graph
uni_reg = Univariate_Regression(df['LN (Qual Reserve)'],df['LN (Problem Loans)'],1,True)
#Generate multivariate regression
X = pd.DataFrame({'Problem Loans':df['Problem Loans'], 'Non-Accruals':df['Non-Accruals'], 'Total Delinq':df['Total Delinquencies \n']})
multi_reg = Multivariate_Regression(X,df['Net Charge-offs'],df['Date'],1)'''

df = df.set_index(df['Date'],drop=True)

"""Obtain FRED Data based on tickers""" 
ticker = ['CARQGSP','CAUR','RSXFS','CANA','CASTHPI']
#five factors based on Provident memorandum: 
#'CARQGSP' = Real GDP for CA, 
#'CAUR' = unemp rate CA, 
#RSXFS =  Advanced Retail Sales, 
#'CANA' = All Employees (nonfarm) in CA, 
#All-Transactions House Price Index for California = 'CASTHPI'
indicator = [2,0,2,2,2]
New_Data =  Add_FRED_Data(ticker,indicator,df)


"""Qualitative_Assessments: optimize lag-length and then run a regression on that."""

#Step 1 obtain Fred data that has not been matched to original data set as we do not want to lose observations when doing lag-lengths
Fred_data = Add_FRED_Data(ticker,indicator,df,merge=False)
#for each of the variables create lag lengths max = 6 
Fred_data_lags = Lagged_DataFrame(Fred_data,lags=6) #remember Fred_data needs NO merging with orig, i.e. set merge=False

#Step 2 find the optimal lag based on the AIC and Step 3 create the regression graphs for the selected variables
regressions,results_aic,Selections = Selecting_Lag_Regression_Factor(Fred_data_lags,df,ticker,supress_graph=False) #lagged effect
QA = Qualitative_Assessments(New_Data,ticker) #contemp effect

#step 4 generate the heatmaps based on the selection (output from "Selecting_Lag_Regression_Factort() function)
revert = [True,False,True,True,True]
Heatmap_Factors(df,Fred_data_lags,Selections,revert) #heatmaps
