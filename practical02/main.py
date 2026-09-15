# Practical 2 — Bike Sharing correct experiment
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# In the course repository the file is data/hour.csv
df=pd.read_csv(Path(__file__).resolve().parents[1] / 'data' / 'hour.csv')
y=df['cnt']
X=df.drop(columns=['cnt','casual','registered','instant','dteday'])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
cv=KFold(5,shuffle=True,random_state=42)
models={
 'baseline':DummyRegressor(strategy='mean'),
 'ridge':Ridge(alpha=10.0),
 'rf':RandomForestRegressor(n_estimators=300,min_samples_leaf=2,n_jobs=-1,random_state=42),
 'hgb':HistGradientBoostingRegressor(max_iter=300,learning_rate=.08,random_state=42)
}
rows=[]
for name,m in models.items():
 s=cross_validate(m,Xtr,ytr,cv=cv,scoring={'rmse':'neg_root_mean_squared_error','mae':'neg_mean_absolute_error','r2':'r2'},n_jobs=-1)
 rows.append([name,-s['test_rmse'].mean(),s['test_rmse'].std(),-s['test_mae'].mean(),s['test_r2'].mean()])
print(pd.DataFrame(rows,columns=['model','cv_rmse','rmse_std','cv_mae','cv_r2']).sort_values('cv_rmse'))
best=models[min(rows,key=lambda x:x[1])[0]]
best.fit(Xtr,ytr); pred=best.predict(Xte)
print('TEST RMSE',mean_squared_error(yte,pred)**.5,'MAE',mean_absolute_error(yte,pred),'R2',r2_score(yte,pred))
