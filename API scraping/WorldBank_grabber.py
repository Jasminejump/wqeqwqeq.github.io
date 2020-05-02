#!/usr/bin/env python3

import pandas as pd
import wbdata
import pickle
from threading import Thread
        
def get_data(i,table_list,id_value_dict,error_list):
    '''
    i: identifier
    table_list: a list to store data from WorldBank
    id_value_dict: a dictionary to store id,value pair. For some identifiers with the same id, the id,value pair retrieved from get_data()
                   is different from the id,value pair retrieved from get_indicator(). Store the id,value pair from get_data() in case needed.
    error: a list to store identifiers failed to fetch data
    '''
    global n
    try:
        temp = wbdata.get_data(i)
        id_value_dict[temp[0]['indicator']['id']] = temp[0]['indicator']['value']
        for i in range(len(temp)):
            temp[i]['indicator']=temp[i]['indicator']['value']
            temp[i]['country']=temp[i]['country']['value']            
        table = pd.DataFrame.from_dict(temp).dropna()
        table = table.set_index(['country','date']).sort_index()
        table = table.rename(columns={'value':table.indicator[0]})
        table = pd.DataFrame(table[table.indicator[0]])
        table_list.append(table)
        n+=1
        print(n,'done',end='...',flush=True)
    except:
        error_list.append(i)
        n+=1
        print(n,'done',end='...',flush=True)

# read local pickle file storing dictionary of id,value pair
# you can also retreive it through get_indicator() as the document says
with open('keys.pkl','rb') as f:
    d = pickle.load(f)

ind = list(range(0,17201,100))+[len(d)]

for j in range(len(ind)-1):
    id_value_dict = {}
    error_list = []
    table_list = []
    threadlist = []
    print('get data for ',ind[j],'-',ind[j+1],'...')
    n = 0
    for i in list(d.keys())[ind[j]:ind[j+1]]:
        t = Thread(target=get_data,args=(i,table_list,id_value_dict,error_list))
        t.start()
        threadlist.append(t)
    for i in threadlist:
        i.join()
    with open(''.join(['data/data_',str(ind[j]),'.pkl']),'wb') as f:
        pickle.dump(table_list,f)
    with open(''.join(['data/dict_',str(ind[j]),'.pkl']),'wb') as f:
        pickle.dump(id_value_dict,f)
    with open(''.join(['data/error_',str(ind[j]),'.pkl']),'wb') as f:
        pickle.dump(error_list,f)
    print('\n',ind[j],'-',ind[j+1],'done')

        
    
