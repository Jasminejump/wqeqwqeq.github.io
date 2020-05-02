import numpy as np
import pandas as pd
import requests
# Environment Var
server_address = 'https://www.climatewatchdata.org'
history_emission_addon = '/api/v1/data/historical_emissions'
linkages_NDC_SDG_addon = '/api/v1/data/ndc_sdg'
content_NDC_addon = '/api/v1/data/ndc_content'

#Getter modules for ClimateWatch
# Retrieves time series data for historical emissions
'''
Query Parameters
    data_sources - source_ids[] - emission data source id (CAIT, PIK, UNFCCC)
    gwps - gwp_ids[] - emission gwps id
    gases - gas_ids[] - emission data source id (CAIT, PIK, UNFCCC)
    sectors - sector_ids[] - sector id
    regions - regions[] - region ISO code 3
    start_year - start_year - Show results from this year onwards
    end_year - end_year - Show results up to this year
    sort_col - sort_col - column to sort the table by
    sort_dir - sort_dir - sort direction (ASC or DESC)
'''
def get_emission():
    # initiating empty DataFrame for emission history data
    emission_df = pd.DataFrame()
    i = 1 # increment var
    while(True):
        url = server_address + history_emission_addon + '?page=' + str(i)
        decode_success = False
        #handling json decoding error exception by sending request repeatly
        while not decode_success:
            try:
                r = requests.get(url)
#                 while r.json().get('status')!='OK':               #Only for webiste with request rate limit
#                     from time import sleep
#                     from random import random
#                     sleep(random())
#                     r = requests.get(url)
                if len(r.json()['data']) == 0:
                    print('end')
                    return emission_df
                decode_success = True
            except:
                pass
        print('processing page:' + str(i))
        response_dict = r.json()
        emission_df = emission_df.append(pd.DataFrame(response_dict['data']))
        i +=1
    print('Dimension:' + emission_df.shape)
    return emission_df

# Retrieves time series data for NDC SDG linkages
'''Query Parameters
goals - goal_ids[] - goal id
targets - target_ids[] - target id
sectors - sector_ids[] - sector id
countries - countries[] - country ISO code
sort_col - sort_col - column to sort the table by
sort_dir - sort_dir - sort direction (ASC or DESC)
'''
def get_linkages():
    # initiating empty DataFrame for NDC SDG linkages
    linkages_df = pd.DataFrame()
    i = 1 # increment var
    while(True):
        url = server_address + linkages_NDC_SDG_addon + '?page=' + str(i)
        decode_success = False
        #handling json decoding error exception by sending request repeatly
        while not decode_success:
            try:
                r = requests.get(url)
                if len(r.json()['data']) == 0:
                    print('end')
                    return linkages_df
                decode_success = True
            except:
                pass
        print('processing page:' + str(i))
        response_dict = r.json()
        linkages_df = linkages_df.append(pd.DataFrame(response_dict['data']))
        i +=1
    print('Dimension:' + linkages_df.shape)
    return linkages_df

# Retrieves time series data for NDC content
'''
Query Parameters
    countries - countries[] - country ISO code
    sources - source_ids[] - source id
    indicators - indicator_ids[] - indicator id
    categories - category_ids[] - category id
    labels - label_ids[] - label id
    sectors - sector_ids[] - sector id
    sort_col - sort_col - column to sort the table by
    sort_dir - sort_dir - sort direction (ASC or DESC)
'''
def get_content_NDC():
    # initiating empty DataFrame for NDC content
    content_df = pd.DataFrame()
    i = 1 # increment var
    while(True):
        url = server_address + content_NDC_addon + '?page=' + str(i)
        decode_success = False
        #handling json decoding error exception by sending request repeatly
        while not decode_success:
            try:
                r = requests.get(url)
                if len(r.json()['data']) == 0:
                    print('end')
                    return content_df
                decode_success = True
            except:
                pass
        print('processing page:' + str(i))
        response_dict = r.json()
        content_df = content_df.append(pd.DataFrame(response_dict['data']))
        i +=1
    print(content_df.shape)
    return content_df
#############################################################################
def handle_emission_data(emission_df):
    emission_df.reset_index(drop = True, inplace = True)
    emission_df['indicator_name'] = emission_df['data_source'] + '//' + emission_df['gas'] + '//' + emission_df['sector']
    emission_df.drop(columns = ['unit','data_source','gas','sector','id','iso_code3'], inplace = True)
    country_table_dict = {}
    # iterrows will iterate through rows and ['column name'] will return the element, rather than a dataSeries
    for index, row in emission_df.iterrows():
        if index%1000 == 0:
            print('row: ', index)
        if str(row['country']) not in country_table_dict.keys():
            country_table_dict[str(row['country'])] = pd.DataFrame(row['emissions']).rename(columns = {'value': str(row['indicator_name'])})
        else:
            if str(row['indicator_name']) in country_table_dict[str(row['country'])].columns:
                continue
            temp = pd.DataFrame(row['emissions']).rename(columns = {'value': str(row['indicator_name'])})
            country_table_dict[str(row['country'])] = country_table_dict[str(row['country'])].merge(temp, on = 'year', how = 'outer')
    # Final handling of individual country tables
    for country_name in country_table_dict.keys():
        country_table_dict[country_name]['country'] = country_name
    master_table = pd.concat(list(country_table_dict.values()), ignore_index=True)
    master_table.sort_values(by = ['country', 'year'], inplace = True)
    return master_table
