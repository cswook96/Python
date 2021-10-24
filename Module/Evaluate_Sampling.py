#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from imblearn.under_sampling import EditedNearestNeighbours, OneSidedSelection, CondensedNearestNeighbour, NeighbourhoodCleaningRule, RandomUnderSampler, TomekLinks
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score




def evaluate_sampling(x_train, y_train, x_test, y_test, eval_type='test'):
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    enn = EditedNearestNeighbours()
    oss = OneSidedSelection(random_state=0)
    cnn = CondensedNearestNeighbour(random_state=0)
    ncr = NeighbourhoodCleaningRule()
    rus = RandomUnderSampler(random_state=0)
    tl = TomekLinks()
    adasyn = ADASYN(random_state=0)
    blsmote = BorderlineSMOTE(random_state=0)
    ros = RandomOverSampler(random_state=0)
    smote = SMOTE(random_state=0)
    smoteenn = SMOTEENN(random_state=0)
    smotetomek = SMOTETomek(random_state=0)
    
    samplers = [enn, oss, cnn, ncr, rus, tl, adasyn, blsmote, ros, smote, smoteenn, smotetomek]
    
    no_sample_model = LogisticRegression(C=1.5, max_iter=1000, random_state=0)
    no_sample_model.fit(x_train, y_train)
    no_sample_prediction = (no_sample_model.predict_proba(x_test)[:, 1] > 0.5).astype('int32')
    
    no_sample_acc = accuracy_score(y_test, no_sample_prediction) 
    no_sample_precision = precision_score(y_test, no_sample_prediction)
    no_sample_recall = recall_score(y_test, no_sample_prediction)
    no_sample_f1 = f1_score(y_test, no_sample_prediction)  
    no_sample_auc = roc_auc_score(y_test, no_sample_model.predict_proba(x_test)[:, 1])
    
    sampling_datas = {}
    sampling_name = ['No Sampling', 'ENN', 'OSS', 'CNN', 'NCR', 'RUS', 'TL', 'ADASYN', 'BLSMOTE', 'ROS', 'SMOTE', 'SMOTE+ENN', 'SMOTE+Tomek']
    
    accuracys = [no_sample_acc]
    precisions = [no_sample_precision]
    recalls = [no_sample_recall]
    f1_scores = [no_sample_f1]
    aucs = [no_sample_auc]
    
    
    for idx, sampler in enumerate(samplers):
        x_sampling, y_sampling = sampler.fit_resample(x_train, y_train)

        lr_model = LogisticRegression(C=1.5, max_iter=1000, random_state=0)
        lr_model.fit(x_sampling, y_sampling)
        
        if eval_type == 'test':
            prediction = (lr_model.predict_proba(x_test)[:, 1] > 0.5).astype('int32')
            acc = accuracy_score(y_test, prediction) 
            precision = precision_score(y_test, prediction)
            recall = recall_score(y_test, prediction)
            f1 = f1_score(y_test, prediction)  
            auc = roc_auc_score(y_test, lr_model.predict_proba(x_test)[:, 1])

        
        elif eval_type == 'train':
            prediction = (lr_model.predict_proba(x_sampling)[:, 1] > 0.5).astype('int32')
            acc = accuracy_score(y_sampling, prediction) 
            precision = precision_score(y_sampling, prediction)
            recall = recall_score(y_sampling, prediction)
            f1 = f1_score(y_sampling, prediction)  
            auc = roc_auc_score(y_sampling, lr_model.predict_proba(x_sampling)[:, 1])

        sampling_datas[sampling_name[idx+1]] = (x_sampling, y_sampling)
        accuracys.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        aucs.append(auc)
    
    
        
        
        
    metrics = [accuracys, precisions, recalls, f1_scores, aucs]
    
    
    xticks = [i for i in range(len(sampling_name))]
    yticks = [i for i in np.arange(0, 1, 0.1)]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1_score', 'AUROC']
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 20), sharex=True, sharey=True)
    
    axes[0].set_title('Model performance based on sampling', fontsize=30)
    axes[len(metrics)-1].set_xlabel('Sampling Method', fontsize=15)
    axes[len(metrics)-1].set_xticks(xticks)
    axes[len(metrics)-1].set_yticks(yticks)
    axes[len(metrics)-1].set_xticklabels(sampling_name, ha='right')
    axes[len(metrics)-1].tick_params(axis='x', labelsize=10, rotation=30)
    
    for idx in range(len(metrics)):
        axes[idx].set_ylabel(metric_names[idx], fontsize=15)
        axes[idx].bar(xticks, metrics[idx])
        
        for x_idx, x_value in enumerate(xticks):
            axes[idx].text(x_value, metrics[idx][x_idx], 
                          s=str(np.round(metrics[idx][x_idx], 2)),
                          color='red',
                          ha='center', va='bottom')
    
    fig.subplots_adjust(hspace=0.08)
    
    plt.show()
    
    return sampling_datas, np.round(accuracys, 2), np.round(precisions, 2), np.round(recalls, 2), np.round(f1_scores, 2) ,np.round(aucs, 2)