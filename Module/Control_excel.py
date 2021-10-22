#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pytz
import datetime

def update_excel(excel_path, sheet_name, update_data):
    tz = pytz.timezone('Asia/Seoul')
    now = datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
    time_df = pd.DataFrame(columns=['Time'])
    time_df.loc[0] = now
    
    excel_df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    if type(update_data) == type(pd.Series()):
        update_data_df = pd.DataFrame(columns=update_data.index)
        update_data_df.loc[0] = update_data.values
        update_data = pd.concat([update_data_df, time_df], axis=1)
        excel_df = pd.concat([excel_df, update_data], axis=0, join='outer')
        
    elif type(update_data) == type(pd.DataFrame()):
        update_data = update_data.reset_index(drop=True)
        time_df_list = [time_df for _ in range(len(update_data))]
        time_df = pd.concat(time_df_list, axis=0).reset_index(drop=True)
        update_data = pd.concat([update_data, time_df], axis=1) 
        excel_df = pd.concat([excel_df, update_data], axis=0, join='outer')
        
    else:
        assert False, 'update_data를 Series나 DataFrame 형태로 입력해주세요.'
    
    excel_df = excel_df.reset_index(drop=True)
    excel_df_columns = excel_df.columns.tolist()
    excel_df_columns.remove('Time')
    excel_df = excel_df.drop_duplicates(subset=excel_df_columns)
        
    writer = pd.ExcelWriter(excel_path)
    excel_df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()
    return excel_df
