"""SI-HIS Machine Learning Engine v2.
Supervised prediction, temporal validation, explainability and robust forecasting.
Derived targets are explicitly demo labels unless a validated outcome is supplied.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score, mean_absolute_error,
    mean_squared_error, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib

RANDOM_STATE = 42

# Small-stratum reliability is treated as a first-class ML output.
# The exact thresholds are configurable and are screening safeguards, not
# universal epidemiological rules. Small counts can produce unstable rates.
STABILITY_THRESHOLDS = {"low_max_n": 15, "medium_max_n": 29}

# Operational screening thresholds for observation-count stability.
# These are not universal epidemiological/statistical cut-offs and do not
# establish representativeness, statistical validity, or population risk.
STABILITY_GUIDE = [
    {
        "range": "n ≤ 15",
        "label": "RENDAH",
        "meaning": "Sampel sangat kecil; estimasi mudah berubah; tidak layak untuk generalisasi",
    },
    {
        "range": "16–29",
        "label": "SEDANG",
        "meaning": "Sampel masih terbatas; interpretasi perlu kehati-hatian",
    },
    {
        "range": "≥ 30",
        "label": "CUKUP STABIL",
        "meaning": "Jumlah observasi relatif lebih memadai untuk analisis deskriptif, tetapi bukan jaminan representatif",
    },
]

def _stability_label(n, events=None):
    n = int(n or 0)
    if n <= STABILITY_THRESHOLDS["low_max_n"]:
        return "RENDAH"
    if n <= STABILITY_THRESHOLDS["medium_max_n"]:
        return "SEDANG"
    return "CUKUP STABIL"

def _stability_note(n, events=None):
    label = _stability_label(n, events)
    if label == "RENDAH":
        return "Sampel sangat kecil; estimasi mudah berubah; tidak layak untuk generalisasi."
    if label == "SEDANG":
        return "Sampel masih terbatas; interpretasi perlu kehati-hatian."
    return "Jumlah observasi relatif lebih memadai untuk analisis deskriptif, tetapi bukan jaminan representatif."

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
    return {'status':'ok','model':model,'features':features,'label':label,'metrics':metrics,**{k:v for k,v in metrics.items() if k != 'Narrative'},'feature_importance':_feature_importance(model,Xte,yte,features),'holdout_start':te['Tanggal Sakit'].min() if 'Tanggal Sakit' in te else None,'train_rows':len(tr),'test_rows':len(te),'positive_rate':positive_rate,'stability':_stability_label(len(te),int(yte.sum())),'stability_note':_stability_note(len(te),int(yte.sum()))}


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



# ---------------------------------------------------------------------------
# Continuous epidemiological intelligence signals
# These are supporting ML/AI modules around the five primary SI-HIS engines.
# They do not create a sixth clinical decision engine; they enrich the
# continuous DATA -> ANALYSIS -> PREDICTION -> PRESCRIPTION -> NEW DATA loop.
# ---------------------------------------------------------------------------

def temporal_anomaly_detection(df, contamination=0.05):
    """Detect unusual daily village burden using Isolation Forest."""
    p=_daily_panel(df)
    if p.empty:return {'status':'error','message':'Panel temporal per desa belum tersedia.','data':pd.DataFrame()}
    x=p[['Daily_Cases','Rolling7','Growth7']].replace([np.inf,-np.inf],np.nan).dropna()
    if len(x)<30 or x.nunique().sum()<2:return {'status':'error','message':'Observasi temporal belum cukup untuk anomaly detection.','data':pd.DataFrame()}
    model=IsolationForest(n_estimators=250,contamination=contamination,random_state=RANDOM_STATE)
    model.fit(x)
    p2=p.loc[x.index].copy(); p2['Anomaly_Score']=(-model.score_samples(x)).astype(float); p2['Anomaly_Flag']=(model.predict(x)==-1)
    return {'status':'ok','data':p2.sort_values('Anomaly_Score',ascending=False),'n_anomalies':int(p2['Anomaly_Flag'].sum()),'method':'Isolation Forest'}


def temporal_change_points(df, window=7, z_threshold=2.0):
    """Transparent change-point signal from rolling mean shift."""
    p=_daily_panel(df)
    if p.empty:return {'status':'error','message':'Panel temporal belum tersedia.','data':pd.DataFrame()}
    rows=[]
    for village,g in p.groupby('Desa/Kelurahan',sort=False):
        g=g.sort_values('Tanggal Sakit').copy(); s=g['Daily_Cases'].astype(float)
        before=s.shift(1).rolling(window,min_periods=window).mean(); after=s.rolling(window,min_periods=window).mean()
        sd=s.shift(1).rolling(window,min_periods=window).std().replace(0,np.nan)
        g['Change_Score']=((after-before).abs()/sd).replace([np.inf,-np.inf],np.nan).fillna(0)
        g['Change_Point_Flag']=g['Change_Score']>=z_threshold; rows.append(g)
    out=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    return {'status':'ok','data':out.sort_values('Change_Score',ascending=False),'n_change_points':int(out['Change_Point_Flag'].sum()),'method':'Rolling mean shift'}


def growth_risk_prediction(df):
    """Rank current growth/acceleration signals for each village."""
    p=_latest_village_features(df)
    if p.empty:return {'status':'error','message':'Fitur pertumbuhan per desa belum tersedia.','data':pd.DataFrame()}
    q=p.copy(); q['Growth_Risk_Score']=(q['Growth7'].clip(lower=-1,upper=3).fillna(0).add(1).clip(lower=0)/4*100).round(2)
    q['Acceleration_Signal']=(q['Rolling7'].fillna(0)-q['Lag7'].fillna(0)).round(2)
    q['Growth_Risk_Level']=pd.cut(q['Growth_Risk_Score'],[-.01,33,66,100.01],labels=['LOW','MEDIUM','HIGH']).astype(str)
    return {'status':'ok','data':q.sort_values(['Growth_Risk_Score','Acceleration_Signal'],ascending=False),'method':'Growth + acceleration signal'}


def spatial_neighbor_intelligence(df, radius_km=20):
    """Calculate geographically proximate case burden without claiming causality."""
    p=_latest_village_features(df,True)
    if p.empty or not {'Lat','Lon'}.issubset(p.columns):return {'status':'error','message':'Koordinat desa belum tersedia.','data':pd.DataFrame()}
    coords=p[['Lat','Lon']].to_numpy(float); daily=p['Daily_Cases'].to_numpy(float); neigh=[]
    for i,(lat,lon) in enumerate(coords):
        if not np.isfinite(lat) or not np.isfinite(lon):neigh.append(0.0);continue
        dist=111.2*np.sqrt(((coords[:,0]-lat)*np.cos(np.radians(lat)))**2+(coords[:,1]-lon)**2)
        mask=(dist<=radius_km)&np.isfinite(dist); neigh.append(float(daily[mask].sum()-daily[i]))
    p['Neighbor_Cases']=neigh; p['Neighbor_Risk_Score']=(100*p['Neighbor_Cases']/max(float(max(p['Neighbor_Cases'].max(),1)),1)).round(2)
    return {'status':'ok','data':p.sort_values('Neighbor_Risk_Score',ascending=False),'radius_km':radius_km,'method':'Geo-distance neighbor burden'}


def vulnerability_clustering(df):
    """Cluster person-level vulnerability and produce explicit interpretable strata.

    The KMeans cluster remains exploratory. The companion stratification table
    is deterministic and reports age, occupation, comorbidity, burden and CFR.
    Small strata are flagged so high observed CFR cannot be mistaken for stable
    population risk.
    """
    d=df.copy()
    if len(d)<30 or 'Umur' not in d.columns:
        return {'status':'error','message':'Data person belum cukup untuk vulnerability clustering/stratification.','data':pd.DataFrame()}

    age=pd.to_numeric(d['Umur'],errors='coerce')
    d['Age_Group']=pd.cut(
        age,
        bins=[-np.inf,5,19,46,65,np.inf],
        labels=['<5 tahun','5–18 tahun','19–45 tahun','46–64 tahun','≥65 tahun'],
        right=False
    ).astype(str)

    def yes(s):
        return s.astype(str).str.strip().str.lower().isin(
            ['ya','yes','positif','ada','ada komorbid','berisiko','tinggi','1','true']
        )

    d['_Death']=_safe_binary(d['Is_Meninggal']).fillna(0).astype(int) if 'Is_Meninggal' in d.columns else 0
    d['_Comorbidity']=yes(d['Status Komorbid']).astype(int) if 'Status Komorbid' in d.columns else 0

    # Explainable KMeans representation.
    x=pd.DataFrame(index=d.index)
    x['Umur']=age.fillna(age.median())
    for col in ['Status Komorbid','Status Imunisasi','Pekerjaan','Merokok','Aktivitas Fisik']:
        if col in d.columns:
            x[col+'_Yes']=yes(d[col]).astype(int)
    model=KMeans(n_clusters=2,n_init=10,random_state=RANDOM_STATE)
    d['Vulnerability_Cluster']=model.fit_predict(x)

    occupation=d['Pekerjaan'].fillna('Tidak diketahui').astype(str) if 'Pekerjaan' in d.columns else pd.Series('Semua pekerjaan',index=d.index)
    strata=d.assign(_Occupation=occupation).groupby(
        ['Age_Group','_Occupation','_Comorbidity'],dropna=False
    ).agg(
        Jumlah_Observasi=('_Death','size'),
        Meninggal=('_Death','sum')
    ).reset_index()
    strata['CFR_Persen']=np.where(
        strata['Jumlah_Observasi']>0,
        strata['Meninggal']/strata['Jumlah_Observasi']*100,
        np.nan
    ).round(2)
    strata['Stabilitas']=strata['Jumlah_Observasi'].map(_stability_label)
    strata['Interpretasi']=np.where(
        strata['Stabilitas'].eq('RENDAH'),
        'Sampel sangat kecil; estimasi mudah berubah; tidak layak untuk generalisasi.',
        np.where(
            strata['Stabilitas'].eq('SEDANG'),
            'Sampel masih terbatas; interpretasi perlu kehati-hatian.',
            'Jumlah observasi relatif lebih memadai untuk analisis deskriptif, tetapi bukan jaminan representatif.'
        )
    )
    strata['Usia']=strata['Age_Group']
    strata['Pekerjaan']=strata['_Occupation']
    strata['Komorbid']=np.where(strata['_Comorbidity'].eq(1),'Ada','Tidak')
    strata['Stability_Note']=strata['Jumlah_Observasi'].map(_stability_note)

    # Ranking is deliberately based on burden + observed CFR with a small-cell
    # penalty, rather than raw CFR alone.
    n_score=(strata['Jumlah_Observasi']/max(float(strata['Jumlah_Observasi'].max()),1)*100)
    cfr_score=(strata['CFR_Persen']/max(float(strata['CFR_Persen'].max()),1)*100)
    stability_factor=np.where(strata['Jumlah_Observasi']<16,0.35,
                              np.where(strata['Jumlah_Observasi']<30,0.70,1.0))
    strata['Prioritas_Sinyal_Score']=(0.50*n_score+0.50*cfr_score)*stability_factor
    strata['Prioritas_Sinyal_Score']=strata['Prioritas_Sinyal_Score'].round(2)
    strata=strata.sort_values(
        ['Prioritas_Sinyal_Score','Jumlah_Observasi','CFR_Persen'],
        ascending=[False,False,False]
    ).reset_index(drop=True)
    strata.insert(0,'Peringkat',np.arange(1,len(strata)+1))
    strata['Strata_Label']=(
        'Usia '+strata['Usia'].astype(str)
        +' × '+strata['Pekerjaan'].astype(str)
        +' × Komorbid '+strata['Komorbid'].astype(str)
    )
    display_cols=['Peringkat','Strata_Label','Usia','Pekerjaan','Komorbid',
                  'Jumlah_Observasi','Meninggal','CFR_Persen','Stabilitas',
                  'Prioritas_Sinyal_Score','Interpretasi','Stability_Note']
    return {
        'status':'ok',
        'data':d,
        'cluster_profile':d.groupby('Vulnerability_Cluster').size().rename('Cases').reset_index(),
        'vulnerability_strata':strata[display_cols],
        'top_stratum':strata.iloc[0].to_dict() if not strata.empty else None,
        'method':'KMeans exploratory clustering + explicit vulnerability stratification',
        'stability_guide':STABILITY_GUIDE,
        'stability_definition':'Stabilitas observasi adalah screening berbasis jumlah observasi untuk membantu membaca kestabilan deskriptif; bukan ukuran representativitas populasi, validasi model, atau risiko individu.',
        'guardrail':'Observed CFR in small strata is an unstable descriptive signal, not population risk.'
    }

def prioritize_continuous_signals(anomaly=None,change_points=None,growth=None,neighbor=None):
    """Fuse transparent surveillance signals into an operational queue."""
    frames=[]
    if isinstance(anomaly,dict) and isinstance(anomaly.get('data'),pd.DataFrame) and not anomaly['data'].empty:
        a=anomaly['data'][['Desa/Kelurahan','Tanggal Sakit','Anomaly_Score']].copy();a['Anomaly']=a['Anomaly_Score'];frames.append(a)
    if isinstance(change_points,dict) and isinstance(change_points.get('data'),pd.DataFrame) and not change_points['data'].empty:
        x=change_points['data'][['Desa/Kelurahan','Tanggal Sakit','Change_Score']].copy();frames.append(x)
    if isinstance(growth,dict) and isinstance(growth.get('data'),pd.DataFrame) and not growth['data'].empty:
        x=growth['data'][['Desa/Kelurahan','Tanggal Sakit','Growth_Risk_Score']].copy();frames.append(x)
    if not frames:return {'status':'error','message':'Belum ada sinyal yang dapat digabungkan.','data':pd.DataFrame()}
    base=frames[0].copy()
    for x in frames[1:]:base=base.merge(x,on=['Desa/Kelurahan','Tanggal Sakit'],how='outer')
    for c in ['Anomaly','Change_Score','Growth_Risk_Score']:
        if c not in base:base[c]=0.0
        mx=float(pd.to_numeric(base[c],errors='coerce').replace([np.inf,-np.inf],np.nan).max() or 0);base[c+'_Norm']=np.where(mx>0,pd.to_numeric(base[c],errors='coerce').fillna(0)/mx*100,0)
    base['Priority_Score']=(0.35*base['Anomaly_Norm']+0.30*base['Change_Score_Norm']+0.35*base['Growth_Risk_Score_Norm']).round(2)
    base['Priority_Level']=pd.cut(base['Priority_Score'],[-.01,33,66,100.01],labels=['LOW','MEDIUM','HIGH']).astype(str)
    return {'status':'ok','data':base.sort_values('Priority_Score',ascending=False),'method':'Transparent multi-signal prioritization'}


def calibration_report(y_true,probabilities,bins=10):
    y=np.asarray(y_true); p=np.asarray(probabilities,float)
    if len(y)==0 or len(y)!=len(p):return pd.DataFrame()
    q=pd.DataFrame({'y':y,'p':p}); q['bin']=pd.qcut(q['p'].rank(method='first'),q=min(bins,len(q)),labels=False,duplicates='drop')
    return q.groupby('bin',as_index=False).agg(Mean_Predicted=('p','mean'),Observed_Rate=('y','mean'),N=('y','size'))


def population_stability_index(reference,current,bins=10):
    """PSI-style drift indicator for one numeric feature."""
    r=pd.to_numeric(pd.Series(reference),errors='coerce').dropna(); c=pd.to_numeric(pd.Series(current),errors='coerce').dropna()
    if len(r)<20 or len(c)<20:return np.nan
    cuts=np.unique(np.quantile(r,np.linspace(0,1,bins+1)))
    if len(cuts)<3:return 0.0
    cuts[0],cuts[-1]=-np.inf,np.inf
    rp=pd.cut(r,cuts,include_lowest=True).value_counts(normalize=True,sort=False).to_numpy(); cp=pd.cut(c,cuts,include_lowest=True).value_counts(normalize=True,sort=False).to_numpy()
    rp=np.clip(rp,1e-6,None);cp=np.clip(cp,1e-6,None);return float(np.sum((cp-rp)*np.log(cp/rp)))


def spatiotemporal_risk(df, radius_km=20):
    """Fuse recent temporal burden with nearby burden into a transparent risk signal."""
    p=_latest_village_features(df,True)
    if p.empty:return {'status':'error','message':'Fitur spatio-temporal belum tersedia.','data':pd.DataFrame()}
    n=spatial_neighbor_intelligence(df,radius_km)
    if n.get('status')!='ok':return n
    q=n['data'].copy()
    q['Recent_Burden']=q['Rolling7'].fillna(q['Daily_Cases']).astype(float)
    rb=q['Recent_Burden']; nb=q['Neighbor_Cases'].fillna(0).astype(float)
    def norm(s):
        m=float(s.max()) if len(s) else 0
        return s/m*100 if m>0 else s*0
    q['SpatioTemporal_Risk_Score']=(0.60*norm(rb)+0.40*norm(nb)).round(2)
    q['SpatioTemporal_Risk_Level']=pd.cut(q['SpatioTemporal_Risk_Score'],[-.01,33,66,100.01],labels=['LOW','MEDIUM','HIGH']).astype(str)
    return {'status':'ok','data':q.sort_values('SpatioTemporal_Risk_Score',ascending=False),'radius_km':radius_km,'method':'Recent burden + spatial neighbour burden'}

def time_person_place_risk(df):
    """Transparent composite surveillance score across TIME, PERSON and PLACE."""
    d=_ensure_date(df).copy()
    if d.empty:return {'status':'error','message':'Data TIME + PERSON + PLACE belum tersedia.','data':pd.DataFrame()}
    d['Tanggal Sakit']=pd.to_datetime(d.get('Tanggal Sakit'),errors='coerce')
    time_score=d.groupby(d['Tanggal Sakit'].dt.date).size().rename('Time_Cases')
    d['Time_Cases']=d['Tanggal Sakit'].dt.date.map(time_score).fillna(0)
    place_col='Desa/Kelurahan' if 'Desa/Kelurahan' in d.columns else ('Kecamatan' if 'Kecamatan' in d.columns else None)
    if place_col:
        ps=d.groupby(place_col).size().rename('Place_Cases'); d['Place_Cases']=d[place_col].map(ps).fillna(0)
    else:d['Place_Cases']=0
    person_cols=[c for c in ['Umur','Status Komorbid','Status Imunisasi'] if c in d.columns]
    if 'Umur' in d.columns:
        age=pd.to_numeric(d['Umur'],errors='coerce'); d['Person_Score']=np.select([age>=65,age<5],[100,80],default=40)
    else:d['Person_Score']=40
    def norm(s):
        m=float(pd.to_numeric(s,errors='coerce').max())
        return pd.to_numeric(s,errors='coerce').fillna(0)/m*100 if m>0 else s*0
    d['TIME_Score']=norm(d['Time_Cases']); d['PLACE_Score']=norm(d['Place_Cases'])
    d['TIME_PERSON_PLACE_Risk']=(0.30*d['TIME_Score']+0.35*d['Person_Score']+0.35*d['PLACE_Score']).round(2)
    d['TPP_Risk_Level']=pd.cut(d['TIME_PERSON_PLACE_Risk'],[-.01,33,66,100.01],labels=['LOW','MEDIUM','HIGH']).astype(str)
    return {'status':'ok','data':d.sort_values('TIME_PERSON_PLACE_Risk',ascending=False),'method':'TIME + PERSON + PLACE composite surveillance score','person_features':person_cols}

def disease_specific_growth_model(df, min_days=14):
    """Estimate disease-specific recent growth using log-linear regression by disease."""
    d=_ensure_date(df).copy()
    if 'Diagnosis Konfirm' not in d.columns or d.empty:return {'status':'error','message':'Diagnosis Konfirm belum tersedia.','data':pd.DataFrame()}
    d['Disease']=d['Diagnosis Konfirm'].astype(str).str.strip()
    d=d[(d['Disease']!='')&(d['Disease'].str.lower()!='nan')].dropna(subset=['Tanggal Sakit'])
    if d.empty:return {'status':'error','message':'Data penyakit bertanggal belum tersedia.','data':pd.DataFrame()}
    rows=[]
    for disease,g in d.groupby('Disease'):
        s=g.groupby(g['Tanggal Sakit'].dt.normalize()).size()
        if len(s)<min_days:continue
        idx=np.arange(len(s)); y=np.log1p(s.to_numpy(float))
        slope=np.polyfit(idx,y,1)[0]; growth_rate=float(np.expm1(slope))
        rows.append({'Disease':disease,'Days':len(s),'Latest_Daily_Cases':int(s.iloc[-1]),'Growth_Rate_Per_Day':growth_rate,'Doubling_Time_Days':float(np.log(2)/slope) if slope>0 else np.inf})
    out=pd.DataFrame(rows)
    if out.empty:return {'status':'error','message':f'Belum ada penyakit dengan minimal {min_days} hari observasi.','data':out}
    out['Growth_Risk_Score']=(100*out['Growth_Rate_Per_Day'].clip(lower=0).rank(pct=True)).round(2)
    out['Growth_Risk_Level']=pd.cut(out['Growth_Risk_Score'],[-.01,33,66,100.01],labels=['LOW','MEDIUM','HIGH']).astype(str)
    return {'status':'ok','data':out.sort_values('Growth_Rate_Per_Day',ascending=False),'method':'Disease-specific log-linear growth model','min_days':min_days}

def save_model(result,path):
    if result and result.get('status')=='ok':joblib.dump(result['model'],path);return path
    return None

def load_model(path):return joblib.load(path)
