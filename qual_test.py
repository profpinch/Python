# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sb

""" Start and end dates for FRED data retrieval """
start = dt.datetime(2013, 1, 31)
end = dt.datetime(2016, 9, 30)

""" Get macroeconomic variables for qualitative data. We can get more variables but these were just the start """
fedfunds = web.DataReader("FEDFUNDS", "fred", start, end)
unemployment = web.DataReader("UNRATE", "fred", start, end)
case_shiller = web.DataReader("CSUSHPINSA", "fred", start, end)
ninetydpd  = web.DataReader("NPTLTL", "fred", start, end)
charge_offs = web.DataReader("CORALACBN", "fred", start, end)

""" Read bank data as a CSV file

NOTE: The data is an aggregation of various portfolio segments with three columns, 
Date, Qual_Reserve, and Total_Reserve, that have repeating data since it just include both
qualitative and total loan loss reserves in aggregate. I'm just working with one segment here (C&I).
Would probably be best to take a file like this and break out the segments into their own data frames
and analyze independently. """
UBOC = pd.read_csv("C:/Users/aframe/Documents/UBOCQualitative.csv", index_col = 0, encoding='latin-1')
uboc = pd.DataFrame(UBOC)
uboc.index = pd.to_datetime(uboc.index)
uboc = uboc.sort_index()

""" Reduce monthly data to quarterly data """
unempqtrly = unemployment.iloc[2::3, :]
ffqtrly = fedfunds.iloc[2::3, :]
cashqtrly = case_shiller.iloc[2::3,:]

""" Merge all FRED data """
freddata = pd.merge(ffqtrly, cashqtrly, left_index = True, right_index = True)
freddata = pd.merge(freddata, unempqtrly, left_index = True, right_index = True)

""" Merge client Qualitative Factor data with FRED data """
combined = pd.merge_asof(freddata, uboc, left_index = True, right_index = True, direction = 'nearest')

""" Some EDA (Exploratory Data Analysis) of scores in a client's Qualitative Framework, assuming they use a qualitative factor scorecard.
Other banks may just weight their reserves based on %s allocated for each factor. Regressions would tell us if their thinking was 
reasonable/appropriate.

NOTE: Goal is to have one plot for each factor but I haven't made this work yet """
econ_dist = sb.distplot(uboc['EconomicConditions'])
loanvol_dist = sb.distplot(uboc['LoanVolume_NewLoans'])
credpol_dist = sb.distplot(uboc['CreditPolicies'])
lendmgmt_dist = sb.distplot(uboc['LendingManagement'])
porttrend_dist = sb.distplot(uboc['PortfolioTrends'])
riskid_dist = sb.distplot(uboc['RiskIdentification'])
extfact_dist = sb.distplot(uboc['ExternalFactors'])
credcon_dist = sb.distplot(uboc['CreditConcentrations'])

""" Scatter plots using matplotlib
NOTE: Like the distplots above, these are supposed to be separate scatter plots but I haven't made this work either. """
scatter_plot = plt.figure()
axes1 = scatter_plot.add_subplot(1,1,1)
axes1.scatter(combined['UNRATE'], combined['Qual_Reserve'])
axes1.set_title('Scatterplot of Unemployment Rate vs. Qualitative Reserve')
axes1.set_xlabel('Unemployment (%)')
axes1.set_ylabel('Qualitative Reserve')
scatter_plot

scatter_plot1 = plt.figure()
axes1 = scatter_plot.add_subplot(1,1,1)
axes1.scatter(combined['UNRATE'], combined['EconomicConditions'])
axes1.set_title('Scatterplot of Unemployment Rate vs. Economic Conditions')
axes1.set_xlabel('Unemployment (%)')
axes1.set_ylabel('Economic Conditions Score')
scatter_plot1

scatter_plot2 = plt.figure()
axes1 = scatter_plot.add_subplot(1,1,1)
axes1.scatter(combined['FEDFUNDS'], combined['Qual_Reserve'])
axes1.set_title('Scatterplot of Fed Funds vs. Qualitative Reserve')
axes1.set_xlabel('Fed Funds (%)')
axes1.set_ylabel('Qualitative Reserve')
scatter_plot2

scatter_plot3 = plt.figure()
axes1 = scatter_plot.add_subplot(1,1,1)
axes1.scatter(combined['FEDFUNDS'], combined['EconomicConditions'])
axes1.set_title('Scatterplot of Fed Funds vs. Economic Conditions')
axes1.set_xlabel('Fed Funds (%)')
axes1.set_ylabel('Economic Conditions Score')
scatter_plot3

""" There should be two scatter plots for each macro variable from FRED: 1 vs. the Qualitative Reserve, 1 vs. the Economic Conditions Score in the data frame.


Additional code to be written would incorporate regression analyses and possibly some ML technique(s) that recommends a preferred model. 
Need other/different data to include (i.e. losses, risk ratings to build a transition matrix model, more granularity), determine metrics/success criteria (AUROC/AUC, potentially).
Other *potential* features: report generation """

