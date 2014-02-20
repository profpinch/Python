#!/usr/bin/env python
# Python script to test connection to external data sources to write JSON data to a csv file. FRED is a test case.

import json
import csv
import urllib

APIKey = '################&file_type=json' # FRED API Key and JSON response spec


def main(*args):
    for i in args:
        serieslink = 'http://api.stlouisfed.org/fred/series/observations?series_id='+i+'&api_key='+APIKey
        temp0 = urllib.urlopen(serieslink)
        temp00 = temp0.read()
        temp000 = json.loads(temp00, 'utf-8')
        timeseries = temp000['observations']
        with open(i+'.csv', 'w') as f:
            writer = csv.writer(f, lineterminator = '\n')
            for x in timeseries:
                date = x['date']
                value = x['value']
                writer.writerow((date, value, i))

if __name__ == 'main':
    main()