# Lab 3 — kNN / NearestCentroid / SVM on Titanic
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

data_path=Path(__file__).resolve().parents[1] / 'data' / 'titanic.csv'
source=data_path if data_path.exists() else 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df=pd.read_csv(source)
df['AtRisk']=1-df['Survived']
features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X=df[features]; y=df['AtRisk']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
num=['Age','SibSp','Parch','Fare']; cat=['Pclass','Sex','Embarked']

def prep(scale):
    steps=[('imp',SimpleImputer(strategy='median'))]
    if scale=='standard': steps.append(('sc',StandardScaler()))
    elif scale=='minmax': steps.append(('sc',MinMaxScaler()))
    return ColumnTransformer([
        ('num',Pipeline(steps),num),
        ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)
    ])

cv=StratifiedKFold(5,shuffle=True,random_state=42)
for scale in ['none','standard','minmax']:
    for name,est in [
        ('kNN',KNeighborsClassifier(n_neighbors=7,weights='distance')),
        ('LinearSVM',SVC(kernel='linear',C=1)),
        ('RBFSVM',SVC(kernel='rbf',C=1))]:
        pipe=Pipeline([('pre',prep(scale)),('model',est)])
        sc=cross_validate(pipe,Xtr,ytr,cv=cv,scoring='f1')
        print(name,scale,sc['test_score'].mean(),sc['test_score'].std())

knn=Pipeline([('pre',prep('standard')),('model',KNeighborsClassifier())])
knn_grid={'model__n_neighbors':[3,5,7,11,15],'model__weights':['uniform','distance'],'model__p':[1,2]}
knn_search=GridSearchCV(knn,knn_grid,cv=cv,scoring='f1',n_jobs=-1).fit(Xtr,ytr)

svm=Pipeline([('pre',prep('standard')),('model',SVC(probability=True,random_state=42))])
svm_grid={'model__kernel':['linear','rbf'],'model__C':[.1,1,5,10],'model__gamma':['scale','auto']}
svm_search=GridSearchCV(svm,svm_grid,cv=cv,scoring='f1',n_jobs=-1).fit(Xtr,ytr)

cent=Pipeline([('pre',prep('standard')),('model',NearestCentroid())])
cent_cv=cross_validate(cent,Xtr,ytr,cv=cv,scoring='f1')['test_score']
print('NearestCentroid',cent_cv.mean())
print('Best kNN',knn_search.best_score_,knn_search.best_params_)
print('Best SVM',svm_search.best_score_,svm_search.best_params_)

final=svm_search.best_estimator_ if svm_search.best_score_>=knn_search.best_score_ else knn_search.best_estimator_
final.fit(Xtr,ytr); pred=final.predict(Xte)
print('TEST accuracy',accuracy_score(yte,pred),'F1',f1_score(yte,pred))
if hasattr(final,'predict_proba'):
    print('ROC-AUC',roc_auc_score(yte,final.predict_proba(Xte)[:,1]))
