#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
import category_encoders as ce
from sklearn.model_selection import train_test_split
from zipfile import ZipFile


# In[53]:


#read zip files, print directories, extract all files
with ZipFile('data.zip', 'r') as zip:
    zip.printdir()
    zip.extractall() 


# In[77]:


#Loaded train, test, and tube data into pandas dataframes
test_set= pd.read_csv('competition_data/test_set.csv')
train_set= pd.read_csv('competition_data/train_set.csv')
tube_data= pd.read_csv('competition_data/tube.csv')
bill_data= pd.read_csv('competition_data/bill_of_materials.csv')


# In[91]:


bill_data.columns
comp_columns= ['component_id_1', 'component_id_2',
       'component_id_3', 'component_id_4',
       'component_id_5', 'component_id_6',
       'component_id_7', 'component_id_8']
bill_data[comp_columns]= bill_data[comp_columns].fillna('None')
quantity_columns= ['quantity_1',
       'quantity_2', 'quantity_3', 
       'quantity_4', 'quantity_5',
       'quantity_6', 'quantity_7',
       'quantity_8']
bill_data[quantity_columns]= bill_data[quantity_columns].fillna(0)


# In[93]:


columns= ['tube_assembly_id', 'material_id', 'diameter', 'wall', 'length', 'bend_radius']
tube_merge= tube_data[columns]


# In[94]:


train_set= pd.merge(train_set, tube_merge)
test_set= pd.merge(test_set, tube_merge)


# In[95]:


train_set= pd.merge(train_set, bill_data)
test_set= pd.merge(test_set, bill_data)


# In[96]:


train_set.head()


# In[97]:


#grabbed the unique tube assembly ids for train and test data
train_assembly_unique= train_set['tube_assembly_id'].unique()
test_assembly_unique= test_set['tube_assembly_id']. unique()


# In[98]:


#checked if there are any unique assemblies between train and test
set(train_assembly_unique) & set(test_assembly_unique)


# In[99]:


#split the unique assembies into train and validate groups
train_assemblies, val_assemblies= train_test_split(train_assembly_unique, train_size= 0.80, test_size=0.2, random_state= 42)


# In[100]:


#looked at shape of both train and val assemblies
train_assemblies.shape, val_assemblies.shape


# In[101]:


#created train an val datasets by grabbing only data associated with the unique assemblies for that dataset
train= train_set[train_set['tube_assembly_id'].isin(train_assemblies)]
val= train_set[train_set['tube_assembly_id'].isin(val_assemblies)]


# In[102]:


#checked the shape of train and validate
train.shape, val.shape


# In[103]:


#created a wrangle function
def wrangle(df):
    df= df.copy()
    df['quote_date']= pd.to_datetime(df['quote_date'], infer_datetime_format= True)
    df['quote_date_year'] = df['quote_date'].dt.year
    df['quote_date_month'] = df['quote_date'].dt.month
    df= df.drop(columns=['quote_date'])
    return df


# In[104]:


train= wrangle(train)
val= wrangle(val)


# In[112]:


train_target= np.log1p(train['cost'])
train_features= train.drop(columns=['cost', 'tube_assembly_id'])
val_target= np.log1p(val['cost'])
val_features= val.drop(columns=['cost', 'tube_assembly_id'])


# In[113]:


#created a pipeline with an ordinal encoder and randomforestregressor
pipeline= make_pipeline(ce.OrdinalEncoder(),
                       RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))


# In[114]:


#fitted training features and target to pipeline
pipeline.fit(train_features, train_target)


# In[115]:


#made predictions on validations features
y_pred= pipeline.predict(val_features)


# In[118]:


#root mean squared error with log target
np.sqrt(mean_squared_error(val_target, y_pred))


# In[119]:


#R squared score
r2_score(val_target, y_pred)

