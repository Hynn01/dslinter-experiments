#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import pickle

def get_cpc_info(cpc_code,summary=0):
    
    if summary ==  1 :
        txt = cpc_titles[cpc_code[0:1]] 
    else:
        if summary == 0:
            txt = cpc_titles[cpc_code]
        else:
            txt = ""
            for k,v in cpc_titles.items():
                if k.startswith(cpc_code):
                    #print(k)
                    txt += cpc_titles[k] + "."
            
    return txt.lower()


with open('/kaggle/input/cpc-titles-pickle/cpc_titles.pickle','rb') as handle:
    cpc_titles = pickle.load(handle)

print("Exact....")
print(get_cpc_info('A47',0))
print('Entire Category...')
print(get_cpc_info('A47',1))
print("Starts with....")
print(get_cpc_info('A47',2))

