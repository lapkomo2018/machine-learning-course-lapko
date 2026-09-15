# Practical 3 — complexity and regularization on Bike Sharing
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

df=pd.read_csv(Path(__file__).resolve().parents[1] / 'data' / 'hour.csv')
y=df['cnt']
cols=['hr','temp','atemp','hum','windspeed','yr','mnth','season','workingday','weathersit']
X=df[cols]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
cv=KFold(5,shuffle=True,random_state=42)
for degree in [1,2,3]:
    for name,reg in [('OLS',LinearRegression()),('Ridge',Ridge(alpha=10)),('Lasso',Lasso(alpha=.1,max_iter=20000)),('ElasticNet',ElasticNet(alpha=.1,l1_ratio=.5,max_iter=20000))]:
        pipe=Pipeline([('poly',PolynomialFeatures(degree=degree,include_bias=False)),('scale',StandardScaler()),('reg',reg)])
        s=cross_validate(pipe,Xtr,ytr,cv=cv,scoring='neg_root_mean_squared_error',return_train_score=True,n_jobs=-1)
        print(degree,name,'train',-s['train_score'].mean(),'cv',-s['test_score'].mean(),'std',s['test_score'].std())
# Tune degree/alpha only on CV, then fit selected pipeline and evaluate test once.
