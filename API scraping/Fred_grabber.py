#Getting Time-Series Data From Fred API
import numpy as np
import pandas as pd
from fredapi import Fred

country_list = ['Australia', 'Austria', 'Belgium', 'Canada', 'Chile', 'Czech Republic', 'Denmark', 'Estonia', 'Finland',
                'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Israel', 'Italy', 'Japan', 'Korea',
                'Latvia', 'Lithuania', 'Luxembourg','Mexico', 'Netherlands', 'New Zealand', 'Norway', 'Poland',
                'Portugal', 'Slovak Republic', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom'
                , 'United States']

# Enviornment variable for Fred API key (Account: gechang1122@gmail)
API_key = 'f883bba8409588288a20b1febca0b3dc'
def get_Fred_Master():
    seriesID_dict = {'RGDPg':{},'INDg':{},'Spread':{}, 'Stock':{}}
    table_dict = {'RGDPg':{},'INDg':{},'Spread':{}, 'Stock':{}}

    country_code_df = pd.read_excel('ISSO_CODE.xlsx')
    country_Alpha2_L = []
    country_Alpha3_L = []
    for country in country_list:
        country_Alpha2_L.append(country_code_df.loc[country_code_df['Country_Name'] == country]['Alpha_2'].values[0])
        country_Alpha3_L.append(country_code_df.loc[country_code_df['Country_Name'] == country]['Alpha_3'].values[0])

    #Real GDP for OECD Countries

    # get Real Gross Domestic Product series, if not found for certain countries -> use alternatives
    fred = Fred(api_key = API_key)
    root = 'CLVMNACSCAB1GQ'
    list_alt = []
    for i, country in enumerate(country_list):
        try:
            data = fred.get_series(root + country_Alpha2_L[i])
            seriesID_dict['RGDPg'][country_list[i]] = root + country_Alpha2_L[i]
            table_dict['RGDPg'][country] = data
        except ValueError:
            print('no such series for ' + country)
            list_alt.append(country)
    # handling the rest countries:
    list_alt_seriesID = ['AUSGDPRQDSMEI', 'NAEXKP01CAQ189S', 'NAEXKP01CLQ652S', 'CLVMNACSCAB1GQEL','CLVMNACSAB1GQIS',
                       'CLVMNACSAB1GQIE', '', 'JPNRGDPEXP', 'NAEXKP01KRQ189S','', 'NAEXKP01NZQ189S',
                       'CLVMNACSAB1GQSK', 'NAEXKP01TRQ652S', 'GDPC1']
    for i, country in enumerate(list_alt):
        try:
            data = fred.get_series(list_alt_seriesID[i])
            seriesID_dict['RGDPg'][country_list[i]] = list_alt_seriesID[i]
            table_dict['RGDPg'][country] = data
            print('alternative series added for ' + country)
        except ValueError:
            print('no alternative series found for ' + country)

    # get production of total industry series, if not found for certain countries -> use alternatives
    fred = Fred(api_key = API_key)
    root = 'PROINDQISMEI'
    list_alt = []
    for i, country in enumerate(country_list):
        try:
            data = fred.get_series(country_Alpha3_L[i] + root)
            seriesID_dict['INDg'][country_list[i]] = country_Alpha3_L[i] + root
            table_dict['INDg'][country] = data
        except ValueError:
            print('no such series for ' + country)
            list_alt.append(country)

    # get production of total industry series, if not found for certain countries -> use alternatives
    fred = Fred(api_key = API_key)
    root_1 = 'IRLTLT01'
    freq = 'M'
    root_2 = '156N'
    list_alt = []
    for i, country in enumerate(country_list):
        try:
            data = fred.get_series(root_1 + country_Alpha2_L[i] + freq + root_2)
            seriesID_dict['Spread'][country_list[i]] = root_1 + country_Alpha2_L[i] + freq + root_2
            table_dict['Spread'][country] = data
        except ValueError:
            print('no such series for ' + country)
            list_alt.append(country)

    # get production of total industry series, if not found for certain countries -> use alternatives
    fred = Fred(api_key = API_key)
    root_1 = 'SPASTT01'
    freq = 'M'
    root_2 = '661N'
    list_alt = []
    for i, country in enumerate(country_list):
        try:
            data = fred.get_series(root_1 + country_Alpha2_L[i] + freq + root_2)
            seriesID_dict['Stock'][country_list[i]] = root_1 + country_Alpha2_L[i] + freq + root_2
            table_dict['Stock'][country] = data
        except ValueError:
            print('no such series for ' + country)
            list_alt.append(country)

    #Combining results into one single table
    # Step 1: Merge the indicators
    country_table_dict = {}
    for country in country_list:
        country_table_dict[country] = pd.DataFrame(columns = ['date'])
        for indicator in table_dict.keys():
            if country in table_dict[indicator].keys():
                temp = table_dict[indicator][country].reset_index()
                temp.columns = ['date', indicator]
                # merge indicators into country_table
                country_table_dict[country] = country_table_dict[country].merge(temp, how = 'outer', on = 'date')
                country_table_dict[country]['country'] = country
    fred_table = pd.concat(country_table_dict.values()).reset_index(drop = True)

    return fred_table
