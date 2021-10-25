#!/usr/bin/env python
# coding: utf-8

from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns

import warnings
warnings.filterwarnings(action='ignore')


# VIF 확인
def show_vif(df):
    vif = []
    for idx in range(len(df.columns)):
        vif.append(variance_inflation_factor(df.values, idx))

    vif_dataframe = pd.DataFrame()
    vif_dataframe['columns'] = df.columns
    vif_dataframe['VIF'] = vif
    return vif_dataframe

# 다중공선성 제거(vif > 10 이상인거 제거)
def remove_multicollinearity(df):
    while True:
        vif_dataframe = show_vif(df)
        
        print(len(vif_dataframe[vif_dataframe['VIF'] >= 10]))
        if len(vif_dataframe[vif_dataframe['VIF'] >= 10]) == 0:
            break
        
        remove_column = vif_dataframe[vif_dataframe['VIF'] >= 10].sort_values(by='VIF', ascending=False)['columns'].reset_index(drop=True)[0]
        print(f"remove_column: {remove_column}")
        df = df.drop(remove_column, axis=1)
    return df


# ttest의 p-value값
def ttest_pvalue(df, label_column):
    label_0_data = np.array(df.loc[df[label_column] == 0, :].drop(label_column, axis=1))
    label_1_data = np.array(df.loc[df[label_column] == 1, :].drop(label_column, axis=1))
    
    # t-test
    statistic, p_values = stats.ttest_ind(label_0_data, label_1_data, axis=0, equal_var=True)
    
    # p_value dataframe
    ttest_pvalue_dataframe = pd.DataFrame({'columns':df.drop(label_column, axis=1).columns, 'p-value':p_values})
    
    return ttest_pvalue_dataframe