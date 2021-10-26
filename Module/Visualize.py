#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
import seaborn as sns

def data_distribution(y_data, data_purpose='Total', column_name='ECS'):
    
    unique, counts = np.unique(y_data, return_counts=True)
    fig ,ax = plt.subplots(figsize=(10, 5))
    ax.set_xticks(unique)
    ax.set_xticklabels(unique)
    ax.set_xlabel(column_name,fontsize=15)
    ax.set_ylabel('Number',fontsize=15)
    ax.set_title('Number of '+data_purpose + ' Data', fontsize=25)
    ax.bar(unique,counts, width=0.3)
    
    for x_idx in range(len(unique)):
        ax.text(x_idx, counts[x_idx],
            s = str(np.round(counts[x_idx], 2)),
                    color='red',
                    ha='center', va='bottom')
    plt.show()
    

def roc_graph(y_test, prediction_proba):

    fpr, tpr, thresholds = roc_curve(y_test, prediction_proba)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(fpr, tpr, color='red')
    ax.plot([0, 1], [0, 1], color='green', linestyle='--')
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title('ROC Curve', fontsize=15)
    plt.show()

def confusion_matrix_heatmap(y_data, prediction):
    cm = confusion_matrix(y_data, prediction)

    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax = sns.heatmap(cm, 
                        linewidths = 0.1,
                       square=True, cmap=plt.cm.PuBu,
                       linecolor='white', annot=True, annot_kws={'size':20}, fmt='d')
    
    ax.set_ylabel('Ground truth', fontsize=20)
    ax.set_xlabel('Prediction', fontsize=20)
    
    plt.show()

