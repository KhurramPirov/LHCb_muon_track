#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# In[ ]:





# In[3]:


metrics = np.zeros((1, 4))

XGB = xgb.XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=1000, random_state=404, verbose = 0, n_jobs=-1)
#XGB=joblib.load('XGB0.joblib')
#for i in range(11):
name = 'data_undersampled.hdf'## путь к файлу
features = pd.read_csv('RF_top20.csv')
data = pd.read_hdf(name)
y = data['label']
X = data[features['name']]
Lextra = data[['Lextra_X0', 'Lextra_X1', 'Lextra_X2', 'Lextra_X3', 'Lextra_Y0', 
                    'Lextra_Y1', 'Lextra_Y2', 'Lextra_Y3']].values
Matched_hit = data[['MatchedHit_X0','MatchedHit_X1', 'MatchedHit_X2', 'MatchedHit_X3', 'MatchedHit_Y0', 
                         'MatchedHit_Y1','MatchedHit_Y2','MatchedHit_Y3']].values
d = np.sum((Lextra-Matched_hit)**2, axis=1)
X.join(pd.DataFrame(d))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
print(1)

XGB.fit(X_train.values, y_train.values)  
metrics[0, 0] = roc_auc_score(y_train.values, XGB.predict_proba(X_train.values)[:,1])
metrics[0, 1] = roc_auc_score(y_test.values, XGB.predict_proba(X_test.values)[:,1])
metrics[0, 2] = f1_score(y_train.values, XGB.predict(X_train.values))
metrics[0, 3] = f1_score(y_test.values, XGB.predict(X_test.values))


joblib.dump(XGB,'XGB_top20+d'+'.joblib')## сохраняется модель


# In[ ]:


df_score = pd.DataFrame(data=metrics, columns=['roc_auc_train', 'roc_auc_test', 'f1_train', 'f1_test'])
df_score.head()
df_score.to_csv('XGB_top20+d.csv')## сохраняются скоры


# In[ ]:




