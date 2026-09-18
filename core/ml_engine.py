"""SI-HIS Machine Learning Engine v2.
Supervised prediction, temporal validation, explainability and robust forecasting.
Derived targets are explicitly demo labels unless a validated outcome is supplied.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score, mean_absolute_error,
    mean_squared_error, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import average_precision_score
from .epidemiology_ml import detect_temporal_anomalies, detect_change_points, build_spatial_neighbor_features, train_growth_risk_model, predict_growth_risk, train_spatiotemporal_risk_model, predict_spatiotemporal_risk, train_disease_specific_growth_models, cluster_population_vulnerability, rank_areas_by_continuous_signal, build_continuous_epidemiology_signals
import joblib

RANDOM_STATE = 42
SEVERITY_FEATURES = ['Umur','Jenis Kelamin','Pekerjaan','Status Imunisasi','Status Komorbid','Riwayat Perjalanan','Faktor Risiko Lain']


def _ensure_date(df):
    d=df.copy()
    if 'Tanggal Sakit' in d.columns: d['Tanggal Sakit']=pd.to_datetime(d['Tanggal Sakit'],errors='coerce')
    return d


def _temporal_split(df,target,features,date_col='Tanggal Sakit',test_size=.2):
    cols=list(dict.fromkeys(features+[target]+([date_col] if date_col in df.columns else [])))
    d=df[cols].copy().dropna(subset=[target])
    if len(d)<40 or d[target].nunique()<2:return None
    if date_col in d.columns and d[date_col].notna().sum()>=10: d=d.dropna(subset=[date_col]).sort_values(date_col)
    else: d=d.sort_index()
    cut=max(int(len(d)*(1-test_size)),1)
    if cut>=len(d):return None
    tr,te=d.iloc[:cut],d.iloc[cut:]
    if tr[target].nunique()<2 or te[target].nunique()<2:return None
    return tr[features],te[features],tr[target].astype(int),te[target].astype(int),tr,te


def _build_classifier(features,model_type='rf'):
    numeric=[c for c in features if c in ['Umur','Daily_Cases','Rolling7','Lag1','Lag7','Growth7','Lat','Lon','Neighbor_Cases','Density']]
    categorical=[c for c in features if c not in numeric]; transformers=[]
    if numeric: transformers.append(('num',SimpleImputer(strategy='median'),numeric))
    if categorical: transformers.append(('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),categorical))
    prep=ColumnTransformer(transformers)
    clf=LogisticRegression(max_iter=1200,class_weight='balanced',random_state=RANDOM_STATE) if model_type=='logistic' else RandomForestClassifier(n_estimators=400,min_samples_leaf=3,class_weight='balanced',random_state=RANDOM_STATE,n_jobs=-1)
    return Pipeline([('prep',prep),('model',clf)])


def _feature_importance(model,X,y,features):
    try:
        r=permutation_importance(model,X,y,n_repeats=8,random_state=RANDOM_STATE,scoring='average_precision')
        return pd.DataFrame({'Feature':list(X.columns),'Importance':r.importances_mean,'Std':r.importances_std}).sort_values('Importance',ascending=False).reset_index(drop=True)
    except Exception:return pd.DataFrame({'Feature':features,'Importance':np.nan,'Std':np.nan})


def _metrics_narrative(metrics,label,positive_rate):
    auc=metrics.get('roc_auc',np.nan); pr=metrics.get('pr_auc',np.nan); rec=metrics.get('recall',np.nan); spec=metrics.get('specificity',np.nan); brier=metrics.get('brier',np.nan)
    if np.isfinite(auc):
        discrimination='kemampuan membedakan kasus positif dan negatif berada pada tingkat yang perlu dibaca bersama PR-AUC dan konteks data.'
        if auc >= .80: discrimination='kemampuan diskriminasi model pada holdout terlihat kuat.'
        elif auc >= .70: discrimination='kemampuan diskriminasi model pada holdout terlihat moderat.'
        else: discrimination='kemampuan diskriminasi model pada holdout masih terbatas; hasil sebaiknya tidak dipakai sebagai keputusan tunggal.'
    else: discrimination='AUC tidak dapat dihitung secara andal pada holdout ini.'
    return (f"Model {label} diuji pada holdout temporal. {discrimination} "
            f"Recall={rec:.2f} menunjukkan proporsi target positif yang berhasil terdeteksi, sedangkan specificity={spec:.2f} menunjukkan proporsi non-target yang berhasil disaring. "
            f"PR-AUC={pr:.2f} perlu dibaca terutama ketika kejadian positif relatif jarang (positive rate={positive_rate:.1%}). "
            f"Brier={brier:.3f} menggambarkan kualitas probabilitas prediksi; semakin kecil umumnya semakin baik. "
            "Feature importance adalah kontribusi prediktif pada holdout, bukan bukti hubungan sebab-akibat. Output ini merupakan decision-support dan memerlukan validasi epidemiologi/klinis.")


def _train_classifier(df,target,features,label):
    split=_temporal_split(df,target,features)
    if split is None:return {'status':'error','message':f'Data/target {label} tidak cukup atau holdout temporal hanya memiliki satu kelas.'}
    Xtr,Xte,ytr,yte,tr,te=split; model=_build_classifier(features); model.fit(Xtr,ytr); p=model.predict_proba(Xte)[:,1]; pred=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(yte,pred,labels=[0,1]).ravel()
    metrics={'roc_auc':roc_auc_score(yte,p),'pr_auc':average_precision_score(yte,p),'accuracy':accuracy_score(yte,pred),'precision':precision_score(yte,pred,zero_division=0),'recall':recall_score(yte,pred,zero_division=0),'specificity':tn/(tn+fp) if tn+fp else np.nan,'f1':f1_score(yte,pred,zero_division=0),'brier':brier_score_loss(yte,p)}
    positive_rate=float(yte.mean())
    metrics['Narrative']=_metrics_narrative(metrics,label,positive_rate)
    return {'status':'ok','model':model,'features':features,'label':label,'metrics':metrics,**{k:v for k,v in metrics.items() if k != 'Narrative'},'feature_importance':_feature_importance(model,Xte,yte,features),'holdout_start':te['Tanggal Sakit'].min() if 'Tanggal Sakit' in te else None,'train_rows':len(tr),'test_rows':len(te),'positive_rate':positive_rate}


def _severity_target(df):
    death=pd.to_numeric(df.get('Is_Meninggal',0),errors='coerce').fillna(0).astype(int); inpatient=df.get('Status Penderita',pd.Series('',index=df.index)).astype(str).str.lower().eq('rawat inap'); return ((death==1)|inpatient).astype(int)


def train_case_severity(df):
    d=df.copy(); d['Severity_Target']=_severity_target(d); return _train_classifier(d,'Severity_Target',SEVERITY_FEATURES,'case severity')

def predict_case_severity(df,model):return _predict_generic(df,model,'Severity_Risk',SEVERITY_FEATURES)

def train_vulnerable_population(df):
    d=df.copy(); d['Vulnerable_Target']=_severity_target(d); return _train_classifier(d,'Vulnerable_Target',SEVERITY_FEATURES,'vulnerable population')

def predict_vulnerable_population(df,model):return _predict_generic(df,model,'Vulnerable_Risk',SEVERITY_FEATURES)


def _daily_panel(df):
    d=_ensure_date(df).dropna(subset=['Tanggal Sakit','Desa/Kelurahan']).copy()
    if d.empty:return pd.DataFrame()
    rows=[]
    for village,g in d.groupby('Desa/Kelurahan'):
        dates=pd.date_range(g['Tanggal Sakit'].min().normalize(),g['Tanggal Sakit'].max().normalize(),freq='D'); s=g.set_index('Tanggal Sakit').resample('D').size().reindex(dates,fill_value=0)
        lat=pd.to_numeric(g['Latitude'],errors='coerce').mean() if 'Latitude' in g else np.nan; lon=pd.to_numeric(g['Longitude'],errors='coerce').mean() if 'Longitude' in g else np.nan
        x=pd.DataFrame({'Desa/Kelurahan':village,'Tanggal Sakit':dates,'Daily_Cases':s.values,'Lat':lat,'Lon':lon}); x['Rolling7']=x['Daily_Cases'].rolling(7,min_periods=7).sum(); x['Lag1']=x['Daily_Cases'].shift(1); x['Lag7']=x['Daily_Cases'].shift(7); x['Prev7']=x['Daily_Cases'].shift(1).rolling(7,min_periods=7).sum(); x['Growth7']=np.where(x['Prev7']>0,x['Rolling7']/x['Prev7']-1,np.nan); rows.append(x)
    return pd.concat(rows,ignore_index=True)


def _future_target(panel,threshold=None):
    p=panel.copy();
    if threshold is None:
        raw=[]
        for _,g in p.groupby('Desa/Kelurahan',sort=False):
            v=g['Daily_Cases'].to_numpy(float); raw.extend([v[i+1:i+8].sum() for i in range(max(0,len(v)-7))])
        threshold=max(5.0,float(np.nanquantile(raw,.75))) if raw else 5.0
    future=[]
    for _,g in p.groupby('Desa/Kelurahan',sort=False):
        v=g['Daily_Cases'].to_numpy(float); future.extend([v[i+1:i+8].sum() if i<len(g)-7 else np.nan for i in range(len(g))])
    p['Next7_Total']=future; p['Outbreak_Target']=(p['Next7_Total']>=threshold).astype('float'); p.attrs['derived_threshold']=threshold
    return p.dropna(subset=['Next7_Total'])


def train_klb_prediction(df):
    p=_future_target(_daily_panel(df),None)
    if p.empty:return {'status':'error','message':'Riwayat harian per desa belum cukup untuk label KLB 7 hari.'}
    r=_train_classifier(p,'Outbreak_Target',['Daily_Cases','Rolling7','Lag1','Lag7','Growth7'],'KLB/outbreak 7-day'); r['derived_threshold']=p.attrs.get('derived_threshold',5); return r


def _latest_village_features(df,spatial=False):
    p=_daily_panel(df)
    if p.empty:return pd.DataFrame()
    latest=p.sort_values('Tanggal Sakit').groupby('Desa/Kelurahan',as_index=False).tail(1).copy()
    if spatial:
        coords=latest[['Lat','Lon']].to_numpy(float); dens=[]
        for i,(lat,lon) in enumerate(coords):
            if not np.isfinite(lat) or not np.isfinite(lon):dens.append(0);continue
            dist=111.2*np.sqrt(((coords[:,0]-lat)*np.cos(np.radians(lat)))**2+(coords[:,1]-lon)**2); dens.append(float(np.sum((dist<=20)&np.isfinite(dist))-1))
        latest['Density']=dens
    return latest


def train_spatial_outbreak(df):
    p=_future_target(_daily_panel(df),None)
    if p.empty:return {'status':'error','message':'Riwayat harian-spasial belum cukup untuk training.'}
    p['Density']=0.0
    r=_train_classifier(p,'Outbreak_Target',['Daily_Cases','Rolling7','Lag1','Lag7','Growth7','Lat','Lon','Density'],'spatial outbreak 7-day'); r['derived_threshold']=p.attrs.get('derived_threshold',5); return r

def predict_klb(df,model):return _predict_generic(_latest_village_features(df),model,'KLB_Risk',['Daily_Cases','Rolling7','Lag1','Lag7','Growth7'])
def predict_spatial_outbreak(df,model):return _predict_generic(_latest_village_features(df,True),model,'Spatial_Risk',['Daily_Cases','Rolling7','Lag1','Lag7','Growth7','Lat','Lon','Density'])


def _predict_generic(df,model,name,features):
    if df is None or df.empty:return pd.DataFrame()
    use=[c for c in features if c in df.columns]
    try:p=model.predict_proba(df[use])[:,1]
    except Exception:return pd.DataFrame()
    out=df.copy(); out[name]=p; out[name+'_Level']=pd.cut(p,[-.01,.33,.66,1.01],labels=['LOW','MEDIUM','HIGH']).astype(str); return out.sort_values(name,ascending=False)


def _fit_ets(y):return ExponentialSmoothing(y,trend='add',damped_trend=True,seasonal=None).fit(optimized=True)
def _seasonal_naive(y,h,period=7):return np.repeat(y[-1],h) if len(y)<period else np.resize(y[-period:],h)


def robust_forecast(df,forecast_days=14):
    if df is None or len(df)<30:return {'status':'error','message':'Minimal 30 observasi harian untuk forecasting robust.'}
    x=_ensure_date(df).sort_values('Tanggal Sakit').copy(); x['Jumlah Kasus']=pd.to_numeric(x['Jumlah Kasus'],errors='coerce').fillna(0); x=x.set_index('Tanggal Sakit').resample('D')['Jumlah Kasus'].sum().asfreq('D',fill_value=0); y=x.to_numpy(float); cut=max(int(len(y)*.8),14); train,test=y[:cut],y[cut:]; candidates={}
    try:m=_fit_ets(train); candidates['ETS']=np.maximum(m.forecast(len(test)),0) if len(test) else np.array([])
    except Exception:pass
    candidates['SeasonalNaive7']=_seasonal_naive(train,len(test),7); candidates['Naive']=np.repeat(train[-1],len(test)); scores={k:mean_absolute_error(test,v) for k,v in candidates.items()} if len(test) else {}; valid={k:v for k,v in scores.items() if np.isfinite(v)}
    if not valid:return {'status':'error','message':'Backtest forecasting gagal.'}
    inv={k:1/max(v,1e-6) for k,v in valid.items()}; z=sum(inv.values()); weights={k:v/z for k,v in inv.items()}
    try:ets_full=_fit_ets(y); f_ets=np.maximum(ets_full.forecast(forecast_days),0)
    except Exception:ets_full=None; f_ets=np.repeat(y[-1],forecast_days)
    preds={'ETS':f_ets,'SeasonalNaive7':_seasonal_naive(y,forecast_days,7),'Naive':np.repeat(y[-1],forecast_days)}; f=sum(weights.get(k,0)*preds[k] for k in preds); dates=pd.date_range(x.index.max()+pd.Timedelta(days=1),periods=forecast_days); fitted=ets_full.fittedvalues if ets_full is not None else np.repeat(y.mean(),len(y)); s=float(np.std(y-fitted)) if len(y)>1 else 0; best=min(valid,key=valid.get)
    narrative=(f"Forecast 14 hari menggunakan ensemble berbobot dari {', '.join(preds.keys())}. "
               f"Pada backtest temporal, model dengan MAE terendah adalah {best} (MAE {valid[best]:.2f}). "
               f"Bobot ensemble mengikuti performa backtest; semakin baik MAE, semakin besar kontribusinya. "
               f"Prediksi puncak sekitar {float(np.max(f)):.1f} kasus pada {dates[int(np.argmax(f))].strftime('%d %b %Y')}. "
               "Interval yang ditampilkan menggambarkan ketidakpastian berbasis residual model dan bukan interval prediksi terkalibrasi epidemiologis. "
               "Gunakan bersama kurva epidemik, Rₜ, EWS, dan konteks intervensi sebelum keputusan operasional.")
    return {'status':'ok','forecast':np.maximum(f,0),'dates':dates,'lower':np.maximum(f-1.96*s,0),'upper':f+1.96*s,'peak_date':dates[int(np.argmax(f))],'peak_value':float(np.max(f)),'mae':float(valid[best]),'rmse':float(np.sqrt(np.mean((test-candidates[best])**2))) if len(test) else np.nan,'models':list(preds),'weights':weights,'backtest_mae':valid,'narrative':narrative}


def save_model(result,path):
    if result and result.get('status')=='ok':joblib.dump(result['model'],path);return path
    return None

def load_model(path):return joblib.load(path)


def train_epidemiological_intelligence(df):
    """Run ML-oriented epidemiological screening bundle."""
    result = build_continuous_epidemiology_signals(df)
    growth = train_growth_risk_model(df)
    ranking = rank_areas_by_continuous_signal(df)
    vulnerability = cluster_population_vulnerability(df)
    spatiotemporal = train_spatiotemporal_risk_model(df)
    disease_models = train_disease_specific_growth_models(df)
    result.update({"growth_risk": growth, "spatiotemporal_risk": spatiotemporal, "disease_specific_growth": disease_models, "continuous_ranking": ranking, "vulnerability_clustering": vulnerability})
    return result


def predict_epidemiological_growth(df, model_result):
    return predict_growth_risk(df, model_result)


def predict_epidemiological_spatiotemporal(df, model_result):
    return predict_spatiotemporal_risk(df, model_result)
