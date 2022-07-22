# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:00:00 2022

@author: aaron
"""

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import Sequential, Input
from tensorflow.keras.utils import plot_model

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler,StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os

class EDA:
    def allcolumns(self):
        pd.set_option('display.max_columns',None)
    
    def countplot(self,cat,df):
        for i in cat:
            plt.figure()
            sns.countplot(df[i])
            plt.show()

    def distplot(self,con,df):
        for i in con:
            plt.figure()
            sns.distplot(df[i])
            plt.show()

    def label_encoder(self,cat,df):
        for i in cat:
            le = LabelEncoder()
            temp = df[i]
            temp[temp.notnull()] = le.fit_transform(temp[temp.notnull()])
            df[i] = pd.to_numeric(temp,errors='coerce')
            ENCODER_PATH = os.path.join(os.getcwd(),'models',i + '_encoder.pkl')
            pickle.dump(le,open(ENCODER_PATH,'wb'))
        return df
    
    def one_hot_encoder(self,y):
        ohe = OneHotEncoder(sparse=False)
        if y.ndim == 1:
            y = ohe.fit_transform(np.expand_dims(y,axis=-1))
        else:
            y = ohe.fit_transform(y)
        OHE_PATH = os.path.join(os.getcwd(),'models','ohe.pkl')

        with open(OHE_PATH,'wb') as file:
            pickle.dump(OHE_PATH,file)
        return y
    
    def simple_imputer(self,df,cat,con):
        
        df_simple = df
        
        for i in con:
            df_simple[i] = df[i].fillna(df[i].median())

        for i in cat:
            df_simple[i] = df[i].fillna(df[i].mode()[0])

        print(df_simple.describe().T)
        return df_simple

    def knn_imputer(self,df,cat):
        knn = KNNImputer() 
        df_knn = df
        df_knn = knn.fit_transform(df)
        df_knn = pd.DataFrame(df_knn)
        df_knn.columns = df.columns
        for i in cat:
            df_knn[i] = np.floor(df_knn[i]).astype(int)
        print(df_knn.describe().T)
        return df_knn
    
    def iterative_imputer(self,df,cat):
        ii = IterativeImputer() 
        df_ii = df
        df_ii = ii.fit_transform(df)
        df_ii = pd.DataFrame(df_ii)
        df_ii.columns = df.columns
        for i in cat:
            df_ii[i] = np.floor(df_ii[i]).astype(int)
        print(df_ii.describe().T)
        return df_ii
    
    def con_vs_con_features_selection(self,df,con,target,selected_features,
                                      corr_target=0.6,figsize=(20,12)):
        
        cor = df.loc[:,con].corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(cor,cmap=plt.cm.Reds,annot=True)
        plt.show()
                
        for i in con:
            print(i)
            print(cor[i].loc[target])
            if (cor[i].loc[target]) >= corr_target:
                selected_features.append(i)
                print(i,' ',cor[i].loc[target])
        return selected_features
    
    def cat_vs_con_features_selection(self,df,con,target,selected_features,
                                      target_score=0.6,
                                      solver='lbfgs',max_iter=100):
        
        for i in con:
            lr = LogisticRegression(solver=solver,max_iter=max_iter)   
            lr.fit(np.expand_dims(df[target],axis=-1),df[i])
            lr_score = lr.score(np.expand_dims(df[target],axis=-1),df[i])
            print(i)
            print(lr_score)
            if lr_score >= target_score:
                selected_features.append(i)
        return selected_features
    
    def cat_vs_cat_features_selection(self,df,cat,target,selected_features,
                                      target_score=0.6):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher, 
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        for i in cat:
            cramers_confusion_matrix = pd.crosstab(df[i],df[target]).to_numpy()
            chi2 = ss.chi2_contingency(cramers_confusion_matrix)[0]
            n = cramers_confusion_matrix.sum()
            phi2 = chi2/n
            r,k = cramers_confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            cramers_score = np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
            print(i)
            print(cramers_score)
            if cramers_score >= target_score:
                selected_features.append(i)
                # print(i,' ',cramers_score)
        return selected_features

    def min_max_scaler(self,X):
        mms = MinMaxScaler()    
        
        if X.ndim == 1:
            X = mms.fit_transform(np.expand_dims(X,axis=-1))
        else:    
            X = mms.fit_transform(X)

        MMS_PATH = os.path.join(os.getcwd(),'models','mms.pkl')

        with open(MMS_PATH,'wb') as file:
            pickle.dump(mms,file)
        
        return X
    
    def standard_scaler(self,X):
        ss = StandardScaler()    
        
        if X.ndim == 1:
            X = ss.fit_transform(np.expand_dims(X,axis=-1))
        else:    
            X = ss.fit_transform(X)

        SS_PATH = os.path.join(os.getcwd(),'models','ss.pkl')

        with open(SS_PATH,'wb') as file:
            pickle.dump(ss,file)
        
        return X

class ModelDevelopment:
    def simple_dl_model(self,X_train,y_train,activation_dense='relu',
                        activation_output='softmax',
                        nb_node=128,dropout_rate=0.3):
        """
        
        This is a simple 3 layers Deep Learning model using Sequential approach

        Parameters
        ----------
        X_train : TYPE
            DESCRIPTION.
        no_class : TYPE
            DESCRIPTION.
        nb_node : TYPE, optional
            DESCRIPTION. The default is 128.
        dropout_rate : TYPE, optional
            DESCRIPTION. The default is 0.3.

        Returns
        -------
        model : TYPE
            DESCRIPTION.

        """
        
        model = Sequential()
        model.add(Input(shape=np.shape(X_train)[1:]))
        model.add(Dense(nb_node,activation=activation_dense))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_node,activation=activation_dense))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_node,activation=activation_dense))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(len(np.unique(y_train,axis=0)),
                            activation=activation_output))
        model.summary()
        
        plot = plot_model(model,show_shapes=True)
        PLOT_PATH = os.path.join(os.getcwd(),'statics','model.png')
        
        return model

    def save_model(self,model):
        MODEL_PATH = os.path.join(os.getcwd(),'models','model.h5')
        model.save(MODEL_PATH)
        
# class ModelEvaluation:
#     def plot_hist(self,hist):
        
#         keys = list(hist.history.keys())
        
#         plt.figure()
#         plt.plot(hist.history['loss'])
#         plt.plot(hist.history['val_loss])
#         plt.xlabel('Epoch')
#         plt.legend(['Training Loss','Validation Loss'])
#         plt.show()

#         plt.figure()
#         plt.plot(hist.history['acc'])
#         plt.plot(hist.history['val_acc'])
#         plt.xlabel('Epoch')
#         plt.legend(['Training Accuracy','Validation Accuracy'])
#         plt.show()