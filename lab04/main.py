# Lab 4 — Decision Trees and Ensembles on Titanic
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score

data_path=Path(__file__).resolve().parents[1] / 'data' / 'titanic.csv'
source=data_path if data_path.exists() else 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df=pd.read_csv(source); df['AtRisk']=1-df['Survived']
features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X=df[features]; y=df['AtRisk']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
num=['Age','SibSp','Parch','Fare']; cat=['Pclass','Sex','Embarked']
pre=ColumnTransformer([
 ('num',SimpleImputer(strategy='median'),num),
 ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),cat)
])
cv=StratifiedKFold(5,shuffle=True,random_state=42)

for depth in [2,3,4,5,7,None]:
    m=Pipeline([('pre',pre),('model',DecisionTreeClassifier(max_depth=depth,min_samples_leaf=5,random_state=42))])
    s=cross_validate(m,Xtr,ytr,cv=cv,scoring='f1',return_train_score=True)
    print('depth',depth,'train',s['train_score'].mean(),'cv',s['test_score'].mean(),'std',s['test_score'].std())

models={
 'Baseline':DummyClassifier(strategy='most_frequent'),
 'Tree':Pipeline([('pre',pre),('model',DecisionTreeClassifier(max_depth=2,min_samples_leaf=5,random_state=42))]),
 'RandomForest':Pipeline([('pre',pre),('model',RandomForestClassifier(n_estimators=300,max_depth=7,min_samples_leaf=3,random_state=42,n_jobs=-1))]),
 'HistGB':Pipeline([('pre',pre),('model',HistGradientBoostingClassifier(max_leaf_nodes=15,learning_rate=.08,random_state=42))])
}
rows=[]
for name,m in models.items():
    s=cross_validate(m,Xtr,ytr,cv=cv,scoring='f1',return_train_score=True)
    rows.append((name,s['train_score'].mean(),s['test_score'].mean(),s['test_score'].std()))
    print(rows[-1])

best_name=max(rows,key=lambda r:r[2])[0]; final=models[best_name].fit(Xtr,ytr); pred=final.predict(Xte)
print('Selected',best_name,'TEST accuracy',accuracy_score(yte,pred),'F1',f1_score(yte,pred))

rf=models['RandomForest'].fit(Xtr,ytr)
pi=permutation_importance(rf,Xte,yte,scoring='f1',n_repeats=10,random_state=42,n_jobs=-1)
print(pd.DataFrame({'feature':Xte.columns,'importance':pi.importances_mean,'std':pi.importances_std}).sort_values('importance',ascending=False))
