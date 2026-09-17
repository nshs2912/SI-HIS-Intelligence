"""SI-HIS ML Engine: supervised ML + robust forecasting."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing

RANDOM_STATE=42

def _split(df, target, features):
    d=df[features+[target]].copy().dropna(subset=[target])
    if len(d)<30 or d[target].nunique()<2:return None
    # Temporal split when a date is available; otherwise deterministic holdout.
    if 'Tanggal Sakit' in d.columns:
        d['Tanggal Sakit']=pd.to_datetime(d['Tanggal Sakit'],errors='coerce')
        d=d.sort_values('Tanggal Sakit'); cut=max(int(len(d)*.8),1); tr=d.iloc[:cut]; te=d.iloc[cut:]
    else:
        d=d.sample(frac=1,random_state=RANDOM_STATE); cut=max(int(len(d)*.8),1); tr=d.iloc[:cut]; te=d.iloc[cut:]
    if len(te)==0 or te[target].nunique()<2: return None
    Xtr=tr[features]; Xte=te[features]; ytr=tr[target].astype(int); yte=te[target].astype(int)
    return Xtr,Xte,ytr,yte

def _classifier(df,target,features):
    s=_split(df,target,features)
    if s is None:return {'status':'error','message':'Data/target tidak cukup untuk supervised learning.'}
    Xtr,Xte,ytr,yte=s; cats=[c for c in features if df[c].dtype=='object']; nums=[c for c in features if c not in cats]
    prep=ColumnTransformer([('num',SimpleImputer(strategy='median'),nums),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cats)])
    model=Pipeline([('prep',prep),('model',RandomForestClassifier(n_estimators=300,min_samples_leaf=3,class_weight='balanced',random_state=RANDOM_STATE,n_jobs=-1))]); model.fit(Xtr,ytr); p=model.predict_proba(Xte)[:,1]; yhat=(p>=.5).astype(int)
    out={'status':'ok','model':model,'roc_auc':roc_auc_score(yte,p),'pr_auc':average_precision_score(yte,p),'accuracy':accuracy_score(yte,yhat),'precision':precision_score(yte,yhat,zero_division=0),'recall':recall_score(yte,yhat,zero_division=0),'f1':f1_score(yte,yhat,zero_division=0),'feature_importance':pd.DataFrame({'Feature':features,'Importance':np.nan})}
    return out

def train_case_severity(df):
    d=df.copy(); d['Severity_Target']=((d.get('Is_Meninggal',0).astype(int)==1)|d.get('Status Penderita','').eq('Rawat Inap')).astype(int)
    features=['Umur','Jenis Kelamin','Pekerjaan','Status Imunisasi','Status Komorbid','Riwayat Perjalanan','Faktor Risiko Lain']; return _classifier(d,'Severity_Target',features)
def predict_case_severity(df,model): return _predict(df,model,'Severity_Risk')

def train_klb_prediction(df):
    d=df.copy(); d['Tanggal Sakit']=pd.to_datetime(d['Tanggal Sakit'],errors='coerce'); g=d.groupby(['Desa/Kelurahan','Tanggal Sakit']).size().reset_index(name='Daily_Cases'); g['Rolling7']=g.groupby('Desa/Kelurahan')['Daily_Cases'].transform(lambda x:x.rolling(7,min_periods=1).sum()); g['Next7_Max']=g.groupby('Desa/Kelurahan')['Daily_Cases'].transform(lambda x:x.shift(-1).rolling(7,min_periods=1).sum()); g['KLB_Target']=(g['Next7_Max']>=5).astype(int); g['Lag1']=g.groupby('Desa/Kelurahan')['Daily_Cases'].shift(1); g['Lag7']=g.groupby('Desa/Kelurahan')['Daily_Cases'].shift(7); g['Growth7']=g['Rolling7']/g.groupby('Desa/Kelurahan')['Rolling7'].shift(7).replace(0,np.nan); return _classifier(g.fillna(0),'KLB_Target',['Daily_Cases','Rolling7','Lag1','Lag7','Growth7'])
def predict_klb(df,model): return _predict(df,model,'KLB_Risk')

def train_spatial_outbreak(df):
    d=df.copy(); d['Tanggal Sakit']=pd.to_datetime(d['Tanggal Sakit'],errors='coerce'); g=d.groupby(['Desa/Kelurahan','Tanggal Sakit']).agg(Daily_Cases=('Nama','count'),Lat=('Latitude','mean'),Lon=('Longitude','mean')).reset_index(); g['Rolling7']=g.groupby('Desa/Kelurahan')['Daily_Cases'].transform(lambda x:x.rolling(7,min_periods=1).sum()); g['Next7']=g.groupby('Desa/Kelurahan')['Daily_Cases'].transform(lambda x:x.shift(-1).rolling(7,min_periods=1).sum()); g['Spatial_Target']=(g['Next7']>=max(5,g['Daily_Cases'].median()*2)).astype(int); g=g.replace([np.inf,-np.inf],np.nan).fillna(0); return _classifier(g,'Spatial_Target',['Daily_Cases','Rolling7','Lat','Lon'])
def predict_spatial_outbreak(df,model): return _predict(df,model,'Spatial_Risk')

def train_vulnerable_population(df):
    d=df.copy(); d['Vulnerable_Target']=((d.get('Is_Meninggal',0).astype(int)==1)|d.get('Status Penderita','').eq('Rawat Inap')).astype(int); features=['Umur','Jenis Kelamin','Pekerjaan','Status Imunisasi','Status Komorbid','Riwayat Perjalanan','Faktor Risiko Lain']; return _classifier(d,'Vulnerable_Target',features)
def predict_vulnerable_population(df,model): return _predict(df,model,'Vulnerable_Risk')

def _predict(df,model,name):
    d=df.copy(); features=[c for c in ['Umur','Jenis Kelamin','Pekerjaan','Status Imunisasi','Status Komorbid','Riwayat Perjalanan','Faktor Risiko Lain','Daily_Cases','Rolling7','Lag1','Lag7','Growth7','Lat','Lon'] if c in d.columns and c in getattr(model.named_steps['prep'],'feature_names_in_',d.columns)]
    if not features:
        try: features=list(model.named_steps['prep'].feature_names_in_)
        except Exception: return pd.DataFrame()
    try:p=model.predict_proba(d[features])[:,1]
    except Exception:return pd.DataFrame()
    out=d.copy(); out[name]=p; out[name+'_Level']=pd.cut(p,[-.01,.33,.66,1.01],labels=['LOW','MEDIUM','HIGH']); return out

def robust_forecast(df,forecast_days=14):
    if len(df)<30:return {'status':'error','message':'Minimal 30 observasi harian untuk forecasting robust.'}
    x=df.copy().sort_values('Tanggal Sakit'); x['Jumlah Kasus']=pd.to_numeric(x['Jumlah Kasus'],errors='coerce').fillna(0); y=x['Jumlah Kasus'].values.astype(float)
    try: ets=ExponentialSmoothing(y,trend='add',damped_trend=True,seasonal=None).fit(optimized=True); f=np.maximum(ets.forecast(forecast_days),0)
    except Exception: return {'status':'error','message':'ETS gagal di-fit.'}
    # Simple backtest on last 20% as an objective check.
    cut=max(int(len(y)*.8),14); train=y[:cut]; test=y[cut:]
    try: bt=ExponentialSmoothing(train,trend='add',damped_trend=True,seasonal=None).fit(optimized=True).forecast(len(test)); mae=mean_absolute_error(test,np.maximum(bt,0)); rmse=np.sqrt(mean_squared_error(test,np.maximum(bt,0)))
    except Exception: mae=rmse=np.nan
    dates=pd.date_range(x['Tanggal Sakit'].max()+pd.Timedelta(days=1),periods=forecast_days); s=np.std(y-ets.fittedvalues); peak=int(np.argmax(f)); return {'status':'ok','forecast':f,'dates':dates,'lower':np.maximum(f-1.96*s,0),'upper':f+1.96*s,'peak_date':dates[peak],'peak_value':f[peak],'mae':mae,'rmse':rmse}
