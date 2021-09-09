#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def data_distribution(y_data, data_purpose='Total'):
    
    unique, counts = np.unique(y_data, return_counts=True)
    fig ,ax = plt.subplots(figsize=(10, 5))
    ax.set_xticks(unique)
    ax.set_xticklabels(unique)
    ax.set_xlabel('ECS',fontsize=15)
    ax.set_ylabel('Number',fontsize=15)
    ax.set_title('Number of '+data_purpose + ' Data', fontsize=25)
    ax.bar(unique,counts, width=0.3)
    plt.show()

