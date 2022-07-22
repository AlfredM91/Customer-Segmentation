# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:36:16 2022

@author: aaron
"""

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import plot_model

from cs_module import EDA, ModelDevelopment
import matplotlib.pyplot as plt
from datetime import datetime
import missingno as msno
import pandas as pd
import numpy as np
import os

#%% Constants

CSV_PATH = os.path.join(os.getcwd(),'Train.csv')
LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.now().
                         strftime('%Y%m%d-%H%M%S'))
PLOT_PATH = os.path.join(os.getcwd(),'statics','model.png')

#%% Step 1) Data Loading

df = pd.read_csv(CSV_PATH)

#%% Step 2) Data Inspection

eda = EDA()
eda.allcolumns()

df.info()
df.describe().T
df.describe(include='all').T
df.isnull().sum()
df.duplicated().sum()

cat = list(df.columns[df.dtypes=='object'])
cat.remove('id')
cat.extend(['day_of_month','num_contacts_in_campaign',
            'num_contacts_prev_campaign','term_deposit_subscribed'])
con = list(df.drop(labels=cat,axis=1))
con.remove('id')

df[cat].describe(include='all').T

eda.countplot(cat, df)
eda.distplot(con, df)

df[cat].boxplot()
df[con].boxplot()

msno.matrix(df)
msno.bar(df)

#%% Step 3) Data Cleaning

# Filling nulls

df = df.replace('unknown',None)
df.isnull().sum()
df = df.drop(labels=['id','days_since_prev_campaign_contact'],axis=1) # too many nulls

con.remove('days_since_prev_campaign_contact')

# 1. Simple imputer

df_simple = eda.simple_imputer(df, cat, con)
df_simple.describe().T

# 2. KNN imputer

df = eda.label_encoder(cat, df)
df_knn = eda.knn_imputer(df,cat)
df_knn[con].describe().T

# 3. Iterative imputer

df_ii = eda.iterative_imputer(df,cat)
df_ii[con].describe().T


# We will be using Iterative Imputer

# Removing outliers - nothing to remove as all values are within acceptable 
# range despite some outliers present in balance and last_contact_duration features

# Removing duplicates

df_ii.duplicated().sum()

# No duplicates to remove

#%% Step 4) Features Selection

selected_features = eda.cat_vs_cat_features_selection(df, cat,
                                                      'prev_campaign_outcome',
                                                      [],target_score=0.01)


selected_features = eda.cat_vs_con_features_selection(df, con,
                                                      'prev_campaign_outcome',
                                                      selected_features,
                                                      target_score=0.04,
                                                      solver='saga')

print(selected_features)

# selected_features = ['job_type',
#  'marital',
#  'education',
#  'default',
#  'housing_loan',
#  'personal_loan',
#  'communication_type',
#  'month',
#  'prev_campaign_outcome',
#  'day_of_month',
#  'num_contacts_in_campaign',
#  'num_contacts_prev_campaign',
#  'term_deposit_subscribed',
#  'customer_age']

# Selected features are as listed in the variable selected_features

#%% Step 5) Data Preprocessing

X = df_ii[selected_features]
y = df_ii['prev_campaign_outcome']

X = eda.min_max_scaler(X)
y = eda.min_max_scaler(np.expand_dims(y,axis=-1))
y = eda.one_hot_encoder(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,
                                                 random_state=123)

#%% Model Development

md = ModelDevelopment() 
model = md.simple_dl_model(X_train, y_train,nb_node=512)
plot_model(model,show_shapes=True, to_file=PLOT_PATH)

#%% Model Compilation

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')

#%% Model Training


tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)

early_callback = EarlyStopping(monitor='val_loss',patience=10)

hist = model.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test),
                 callbacks=[tensorboard_callback,early_callback])

#%%% Model Evaluation

print(hist.history.keys())

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.xlabel('epoch')
plt.legend(['Training Loss','Validation Loss'])
plt.show()

plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.xlabel('epoch')
plt.legend(['Training Accuracy','Validation Accuracy'])
plt.show()

print(model.evaluate(X_test,y_test))

# go to Anaconda prompt and type tensorboard --logdir "actual LOGS_PATH"
# then open the localhost url

#%% Model Analysis

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)
y_true = np.argmax(y_test,axis=1)

cm = confusion_matrix(y_true,y_pred)
cr = classification_report(y_true,y_pred)

labels = ['success','failure','other']
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

print(cr)

# We don't have enough data, only 201 data, and too many outliers in target
# We need at least 1k data for DL, but 10k is better

#%% Model Saving

md.save_model(model)















