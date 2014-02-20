#!/usr/bin/env python
# Python script to test connection to external data sources to write JSON data to a csv file. Yahoo Finance is a test case.

import json
import csv
import urllib


def main(*args): 
	for ticker in args:
		url = urllib.urlopen('http://query.yahooapis.com/v1/public/yql?q=SELECT%20*%20FROM%20yahoo.finance.keystats%20WHERE%20symbol%3D%27'
		+ ticker + '%27&format=json&env=store%3A%2F%2Fdatatables.org%2Falltableswithkeys&callback=')
		content = url.read()
		values = json.loads(content, 'utf-8')
		data = values['query']['results']['stats']
		with open(ticker+'.csv', 'w') as f:
			writer = csv.writer(f, lineterminator = '\n')
			for k, v in data.iteritems():
				if k == 'TotalDebt':
					try:
						value = data['TotalDebt']['content']
						timepd = data['TotalDebt']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd)) 
				elif k == 'ReturnonEquity':
					try:
						value = data['ReturnonEquity']['content']
						timepd = data['ReturnonEquity']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'TrailingPE':
					try:
						value = data['TrailingPE']['content']
						timepd = data['TrailingPE']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'RevenuePerShare':
					try:
						value = data['RevenuePerShare']['content']
						timepd = data['RevenuePerShare']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'EnterpriseValue':
					try:
						value = data['EnterpriseValue']['content']
						timepd = data['EnterpriseValue']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'p_52_WeekLow':
					try:
						value = data['p_52_WeekLow']['content']
						timepd = data['p_52_WeekLow']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))    
				elif k == 'MarketCap':
					try:
						value = data['MarketCap']['content']
						timepd = data['MarketCap']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'TrailingAnnualDividendYield':
					try:
						value = data['TrailingAnnualDividendYield']['content']
						timepd = data['TrailingAnnualDividendYield']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'PriceBook':
					try:
						value = data['PriceBook']['content']
						timepd = data['PriceBook']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'EBITDA':
					try:
						value = data['EBITDA']['content']
						timepd = data['EBITDA']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'ShortRatio':
					try:
						value = data['ShortRatio']['content']
						timepd = data['ShortRatio']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'EnterpriseValueEBITDA':
					try:
						value = data['EnterpriseValueEBITDA']['content']
						timepd = data['EnterpriseValueEBITDA']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'ShortPercentageofFloat':
					try:
						value = data['ShortPercentageofFloat']['content']
						timepd = data['ShortPercentageofFloat']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'EnterpriseValueRevenue':
					try:
						value = data['EnterpriseValueRevenue']['content']
						timepd = data['EnterpriseValueRevenue']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'OperatingCashFlow':
					try:
						value =  data['OperatingCashFlow']['content']
						timepd = data['OperatingCashFlow']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'LastSplitFactor':
					try:
						value = data['LastSplitFactor']['content']
						timepd = data['LastSplitFactor']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'PriceSales':
					try:
						value = data['PriceSales']['content']
						timepd = data['PriceSales']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'MostRecentQuarter':
					try:
						value = data['MostRecentQuarter']['content']
						timepd = data['MostRecentQuarter']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'ReturnonAssets':
					try:
						value = data['ReturnonAssets']['content']
						timepd = data['ReturnonAssets']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'p_52_WeekHigh':
					try:
						value = data['p_52_WeekHigh']['content']
						timepd = data['p_52_WeekHigh']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'SharesShort':
					try:
						value = data['SharesShort']['content']
						timepd = data['SharesShort']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'CurrentRatio':
					try:
						value = data['CurrentRatio']['content']
						timepd = data['CurrentRatio']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'BookValuePerShare':
					try:
						value = data['BookValuePerShare']['content']
						timepd = data['BookValuePerShare']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'ProfitMargin':
					try:
						value = data['ProfitMargin']['content']
						timepd = data['ProfitMargin']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'TotalCashPerShare':
					try:
						value = data['TotalCashPerShare']['content']
						timepd = data['TotalCashPerShare']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'QtrlyRevenueGrowth':
					try:
						value = data['QtrlyRevenueGrowth']['content']
						timepd = data['QtrlyRevenueGrowth']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'GrossProfit':
					try:
						value = data['GrossProfit']['content']
						timepd = data['GrossProfit']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'AvgVol':
					try:
						value = data['AvgVol']['content']
						timepd = data['AvgVol']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'LeveredFreeCashFlow':
					try:
						value = data['LeveredFreeCashFlow']['content']
						timepd = data['LeveredFreeCashFlow']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'PEGRatio':
					try:
						value = data['PEGRatio']['content']
						timepd = data['PEGRatio']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'QtrlyEarningsGrowth':
					try:
						value = data['QtrlyEarningsGrowth']['content']
						timepd = data['QtrlyEarningsGrowth']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'TotalCash':
					try:
						value = data['TotalCash']['content']
						timepd = data['TotalCash']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'Revenue':
					try:
						value = data['Revenue']['content']
						timepd = data['Revenue']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'ForwardPE':
					try:
						value = data['ForwardPE']['content']
						timepd = data['ForwardPE']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'DilutedEPS':
					try:
						value = data['DilutedEPS']['content']
						timepd = data['DilutedEPS']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'OperatingMargin':
					try:
						value = data['OperatingMargin']['content']
						timepd = data['OperatingMargin']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'TotalDebtEquity':
					try:
						value = data['TotalDebtEquity']['content']
						timepd = data['TotalDebtEquity']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				elif k == 'NetIncomeAvltoCommon':
					try:
						value = data['NetIncomeAvltoCommon']['content']
						timepd = data['NetIncomeAvltoCommon']['term']
					except TypeError, ex:
						value = 0
						timepd = 0
					writer.writerow((k, value, timepd))
				else:
					writer.writerow((k, v))

if __name__ == 'main':
    main()