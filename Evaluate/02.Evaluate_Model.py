import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

import tensorflow as tf

from Evaluate_Sampling import evaluate_sampling

import warnings
warnings.filterwarnings(action='ignore')




def evaluate_model(x_train, y_train, x_test, y_test, EPOCH):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    num_colums = len(x_train[0])
    
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_split=15, min_samples_leaf=4, random_state=0)
    svm_model = SVC(C=0.7, kernel='rbf', gamma=1, random_state=0, probability=True)
    lr_model = LogisticRegression(C=1.5, max_iter=1000, random_state=0)
    knn_model = KNeighborsClassifier(weights='distance')
    gb_model = GradientBoostingClassifier(n_estimators=30, max_depth=10, min_samples_split=15,min_samples_leaf=5, random_state=0)
    xgb_model = XGBClassifier(min_child_weight=20, max_depth=10, gamma=0.3, random_state=0)
    lgbm_model = LGBMClassifier(min_child_samples=35, max_depth=11, random_state=0)
    
    vote_models = [
    ('rf', rf_model),
    ('svm', svm_model),
    ('lr', lr_model),
    ('knn', knn_model),
    ('gb', gb_model),
    ('xgb', xgb_model),
    ('lgbm', lgbm_model),
    ]

    vote_soft = VotingClassifier(vote_models, voting='soft')    

    stack_models = [
        ('rf', rf_model),
        ('svm', svm_model),
        ('lr', lr_model),
        ('knn', knn_model),
        ('gb', gb_model),
        ('lgbm', lgbm_model),
    ]

    stack_model = StackingClassifier(stack_models, final_estimator=xgb_model)


    def denseunit(vector, growth_rate, regularizer):
        x = tf.keras.layers.BatchNormalization()(vector)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dense(growth_rate, kernel_regularizer=regularizer)(x)
        x = tf.keras.layers.Concatenate()([vector, x])
        return x

    def get_ann(num_columns):
        leakyrelu = tf.keras.layers.LeakyReLU()
        regularizer = tf.keras.regularizers.l2(l2=0.05)

        input_ = tf.keras.layers.Input(shape=(num_columns, ))
        x = tf.keras.layers.Dense(512, activation=leakyrelu, kernel_regularizer=regularizer)(input_)

        for _ in range(128):
            x = denseunit(x, 4, regularizer)

        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Dense(256, activation=leakyrelu, kernel_regularizer=regularizer)(x)
        x = tf.keras.layers.Dense(32, activation=leakyrelu, kernel_regularizer=regularizer)(x)

        x = tf.keras.layers.Dropout(0.3)(x)

        output_ = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.models.Model(inputs=input_, outputs=output_)
        return model

    ann_model = get_ann(num_colums)
    
    models = [rf_model, svm_model, lr_model, knn_model, gb_model, xgb_model, lgbm_model,vote_soft, stack_model, ann_model]
    
    accs = []
    precisions = []
    recalls = []
    f1_scores = []
    aucs = []
    
    for model in models:
        if model == ann_model:
            model.compile(optimizer='adam',
                              loss='binary_crossentropy',
                              metrics=['acc'])


            checkpoint_path = './ANN.ckpt'
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                save_weights_only=True,
                                                                save_best_only=True,
                                                                monitor = 'val_loss',
                                                                verbose=0)


            reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=0)

            model.fit(x_train, y_train,
                      validation_data=(x_test, y_test),
                      epochs = EPOCH,
                      batch_size=16, validation_batch_size=16,
                      callbacks = [checkpoint, reduceLR],
                      verbose=0
                     )
            model.load_weights(checkpoint_path)
            prediction = (model.predict(x_test) > 0.5).astype('int32')
            prediction_proba = model.predict(x_test)
            
        else:
            model.fit(x_train, y_train)
            prediction = (model.predict_proba(x_test)[:, 1] > 0.5).astype('int32')
            prediction_proba = model.predict_proba(x_test)[:, 1]
        
        acc = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction)
        recall = recall_score(y_test, prediction)
        f1 = f1_score(y_test, prediction)
        auc = roc_auc_score(y_test, prediction_proba)
        
        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        aucs.append(auc)
        
    metrics = [accs, precisions, recalls, f1_scores, aucs]
        
    ml_names = ['Random Forest', 'SVM', 'Logistic Regression', 'K-Nearest Neighbors', 'Gradient Boosting', 'XGBoost', 'LightGBM','Voting(soft)', 'Stacking', 'ANN']
    xticks = [i for i in range(len(ml_names))]
    yticks = [i for i in np.arange(0, 1, 0.1)]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1_score', 'AUROC']
        
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 20), sharex=True, sharey=True)
    
    axes[0].set_title('Model performance based on Machine Learning', fontsize=30)
    axes[len(metrics)-1].set_xlabel('Machine Learning Method', fontsize=15)
    axes[len(metrics)-1].set_xticks(xticks)
    axes[len(metrics)-1].set_yticks(yticks)
    axes[len(metrics)-1].set_xticklabels(ml_names, ha='right')
    axes[len(metrics)-1].tick_params(axis='x', labelsize=10, rotation=30)
    
    for idx, ax in enumerate(axes.flat):
        ax.set_ylabel(metric_names[idx], fontsize=15)
        ax.bar(xticks, metrics[idx])
        
        for x_idx in range(len(ml_names)):
            ax.text(x_idx, metrics[idx][x_idx],
                   s = str(np.round(metrics[idx][x_idx], 2)),
                          color='red',
                          ha='center', va='bottom')
    
    fig.subplots_adjust(hspace=0.08)

    return models, np.round(accs, 2), np.round(precisions, 2), np.round(recalls, 2), np.round(f1_scores, 2), np.round(aucs, 2)