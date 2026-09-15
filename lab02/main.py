# Lab 2 — Titanic classification
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import *

data_path=Path(__file__).resolve().parents[1] / 'data' / 'titanic.csv'
source=data_path if data_path.exists() else 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df=pd.read_csv(source)
df['AtRisk']=1-df['Survived']
features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X=df[features]; y=df['AtRisk']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
num=['Age','SibSp','Parch','Fare']; cat=['Pclass','Sex','Embarked']
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),cat)])
models={
    'Baseline': DummyClassifier(strategy='most_frequent'),
    'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42),
    'GaussianNB': GaussianNB(),
}
cv=StratifiedKFold(5,shuffle=True,random_state=42)
results=[]
for name, estimator in models.items():
    pipe=Pipeline([('pre',pre),('clf',estimator)])
    scores=cross_validate(pipe,Xtr,ytr,cv=cv,scoring=['accuracy','precision','recall','f1','roc_auc'])
    row={metric: scores[f'test_{metric}'].mean() for metric in ['accuracy','precision','recall','f1','roc_auc']}
    row['f1_std']=scores['test_f1'].std()
    results.append((name,row))
    print(name,row)

best_name=max(results,key=lambda item:item[1]['f1'])[0]
model=Pipeline([('pre',pre),('clf',models[best_name])]).fit(Xtr,ytr)
p=model.predict_proba(Xte)[:,1] if hasattr(model,'predict_proba') else model.predict(Xte).astype(float)
print('Selected',best_name)
for t in [.2,.3,.4,.5,.6,.7]:
    pred=(p>=t).astype(int)
    print(t, precision_score(yte,pred), recall_score(yte,pred), f1_score(yte,pred), confusion_matrix(yte,pred).ravel())
