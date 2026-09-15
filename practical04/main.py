# Practical 4 — threshold, error costs and decision policy
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

data_path=Path(__file__).resolve().parents[1] / 'data' / 'titanic.csv'
source=data_path if data_path.exists() else 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df=pd.read_csv(source); df['AtRisk']=1-df['Survived']
features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X=df[features]; y=df['AtRisk']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
num=['Age','SibSp','Parch','Fare']; cat=['Pclass','Sex','Embarked']
pre=ColumnTransformer([
 ('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),num),
 ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)
])
def make_model(): return Pipeline([('pre',pre),('model',LogisticRegression(max_iter=2000,random_state=42))])
cv=StratifiedKFold(5,shuffle=True,random_state=42)
oof=np.empty(len(Xtr)); fold=np.empty(len(Xtr),dtype=int)
for i,(a,b) in enumerate(cv.split(Xtr,ytr),1):
    m=make_model().fit(Xtr.iloc[a],ytr.iloc[a])
    oof[b]=m.predict_proba(Xtr.iloc[b])[:,1]; fold[b]=i

rows=[]
for t in np.linspace(.05,.95,91):
    p=(oof>=t).astype(int); tn,fp,fn,tp=confusion_matrix(ytr,p).ravel()
    rows.append([t,precision_score(ytr,p,zero_division=0),recall_score(ytr,p),f1_score(ytr,p),p.mean(),fp,fn,tp,tn,fp+5*fn])
res=pd.DataFrame(rows,columns=['threshold','precision','recall','f1','action_rate','fp','fn','tp','tn','cost'])
feasible=res[res.action_rate<=.35]
best=feasible.loc[feasible.cost.idxmin()]
print('Selected OOF policy\n',best)

final=make_model().fit(Xtr,ytr); score=final.predict_proba(Xte)[:,1]; pred=(score>=best.threshold).astype(int)
tn,fp,fn,tp=confusion_matrix(yte,pred).ravel()
print('TEST',{'precision':precision_score(yte,pred),'recall':recall_score(yte,pred),'f1':f1_score(yte,pred),'action_rate':pred.mean(),'fp':fp,'fn':fn,'cost':fp+5*fn})

# Optional three-zone policy around the chosen threshold
low=max(.05,float(best.threshold)-.10); high=min(.95,float(best.threshold)+.15)
def action(score):
    if score<low: return 'standard'
    if score<high: return 'manual_review'
    return 'priority_assistance'
print(pd.Series([action(s) for s in oof]).value_counts(normalize=True))
