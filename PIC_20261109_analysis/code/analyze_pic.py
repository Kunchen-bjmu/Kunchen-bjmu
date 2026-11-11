"""PIC derivative analysis; no row-level data in public outputs.
Run from any cwd: python /path/to/code/analyze_pic.py --input /path/to/PIC
Dependencies: numpy pandas scipy scikit-learn statsmodels matplotlib markdown threadpoolctl.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import argparse, json, hashlib, platform, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold, GridSearchCV, cross_val_predict, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, roc_curve, precision_recall_curve, confusion_matrix
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

SEED=20261109
ROOT=Path(__file__).resolve().parents[1]
for folder in ['tables','figures','private']:(ROOT/folder).mkdir(exist_ok=True,parents=True)
NUM=['age_month','chart_1001_min','chart_1001_max','chart_1003_max','chart_1016_min','lab_5129_min','lab_5033_max','lab_5041_max','lab_5223_max']
VITAL=NUM[1:5]; LAB=NUM[5:]; ALL=NUM+['gender_is_male']
LABEL={'age_month':'近似月龄','gender_is_male':'男性标记','chart_1001_min':'体温最低值*','chart_1001_max':'体温最高值*','chart_1003_max':'心率最高值*','chart_1016_min':'收缩压最低值*','lab_5129_min':'血小板最低值*','lab_5033_max':'尿素相关最高值*','lab_5041_max':'肌酐最高值*','lab_5223_max':'血糖最高值*'}
MODELS={'M0':'人口学','M1':'基础临床','M2':'增加检验缺失标记','M3':'增加检验数值','M4':'梯度提升补充'}
COLORS={'M0':'#8c969f','M1':'#1f668d','M2':'#d18b32','M3':'#25826f','M4':'#9a5578'}
plt.rcParams.update({'font.family':['Times New Roman','Songti SC','DejaVu Sans'],'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','figure.dpi':130,'savefig.bbox':'tight'})

def save_table(obj,name):
    obj.to_csv(ROOT/'tables'/name,index=False,encoding='utf-8-sig')
def save_fig(fig,name):
    for ext in ['png','pdf','svg']: fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=190)
    plt.close(fig)
def announce(msg): print(time.strftime('%H:%M:%S'),msg,flush=True)

class Features(BaseEstimator,TransformerMixin):
    def __init__(self,mode='M3',winsor=False,spline=False):
        self.mode=mode;self.winsor=winsor;self.spline=spline
    def fit(self,X,y=None):
        self.numeric_=['age_month']+(VITAL if self.mode!='M0' else [])+(LAB if self.mode in ['M3','M4'] else [])
        self.flags_=(VITAL if self.mode!='M0' else [])+(LAB if self.mode in ['M2','M3','M4'] else [])
        self.lo_=X[self.numeric_].quantile(.005); self.hi_=X[self.numeric_].quantile(.995)
        self.knots_=np.quantile(X.age_month,[.1,.5,.9])
        a,names,isnum=self._raw(X)
        self.names_full_=names;self.isnum_full_=np.array(isnum)
        self.medians_=np.nanmedian(a,axis=0)
        if np.isnan(self.medians_).any(): raise ValueError('Training fold has an all-missing selected feature')
        a=np.where(np.isnan(a),self.medians_,a)
        self.scale_=np.where(self.isnum_full_,a.std(axis=0),1.)
        self.scale_=np.where(self.scale_>1e-12,self.scale_,1.)
        self.mean_=np.where(self.isnum_full_,a.mean(axis=0),0.)
        self.keep_=np.ptp(a,axis=0)>0
        self.names_=np.array(names)[self.keep_].tolist()
        return self
    def _raw(self,X):
        vals=[];names=[];kind=[]
        for c in self.numeric_:
            x=X[c].to_numpy(dtype=float).copy()
            if self.winsor and c!='age_month':x=np.clip(x,self.lo_[c],self.hi_[c])
            if c=='age_month' and self.spline:
                k1,k2,k3=self.knots_
                if k1==k2 or k2==k3:raise ValueError('Duplicate spline knots')
                h=(np.maximum(x-k1,0)**3-(k3-k1)/(k3-k2)*np.maximum(x-k2,0)**3+(k2-k1)/(k3-k2)*np.maximum(x-k3,0)**3)/(k3-k1)**2
                vals.extend([x,h]);names.extend(['age_linear','age_rcs']);kind.extend([True,True])
            else:
                vals.append(np.log1p(x) if c=='age_month' else x);names.append('log1p_age' if c=='age_month' else c);kind.append(True)
        vals.append(X.gender_is_male.to_numpy(float));names.append('gender_is_male');kind.append(False)
        for c in self.flags_:
            vals.append(X[c].isna().to_numpy(float));names.append(c+'__missing');kind.append(False)
        return np.column_stack(vals),names,kind
    def transform(self,X):
        a,_,_=self._raw(X);a=np.where(np.isnan(a),self.medians_,a)
        return ((a-self.mean_)/self.scale_)[:,self.keep_]

def pipeline(mode,C=1.,winsor=False,spline=False):
    return Pipeline([('features',Features(mode,winsor,spline)),('model',LogisticRegression(C=C,max_iter=3000,tol=1e-7,class_weight=None,solver='lbfgs'))])
def metrics(y,p):return {'Brier':brier_score_loss(y,p),'AUROC':roc_auc_score(y,p),'AP':average_precision_score(y,p),'LogLoss':log_loss(y,p)}
def calibration(y,p):
    z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a=sm.GLM(y,np.ones((len(y),1)),family=sm.families.Binomial(),offset=z).fit()
        b=sm.GLM(y,sm.add_constant(z),family=sm.families.Binomial()).fit()
    return float(a.params[0]),float(b.params[0]),float(b.params[1])
def wilson(k,n):
    if not n:return np.nan,np.nan
    q=1.95996398454;ph=k/n;den=1+q*q/n;ctr=(ph+q*q/2/n)/den
    half=q*np.sqrt(ph*(1-ph)/n+q*q/(4*n*n))/den
    return ctr-half,ctr+half

def main(input_path):
    start=time.time(); source=Path(input_path)/'icu_first24hours.csv'
    cols=['SUBJECT_ID','HADM_ID','ADMITTIME','HOSPITAL_EXPIRE_FLAG']+ALL
    raw=pd.read_csv(source,usecols=cols)
    raw['_date']=pd.to_datetime(raw.ADMITTIME,errors='coerce')
    first=raw.sort_values(['SUBJECT_ID','_date','HADM_ID'],kind='stable').drop_duplicates('SUBJECT_ID')
    good=lambda d:d.age_month.ge(0)&d.gender_is_male.isin([0,1])&d.HOSPITAL_EXPIRE_FLAG.isin([0,1])&d.SUBJECT_ID.notna()&d.HADM_ID.notna()&d._date.notna()
    d=first.loc[good(first)].copy().reset_index(drop=True)
    assert len(d)==12622 and d.HOSPITAL_EXPIRE_FLAG.sum()==759
    tr,te=train_test_split(np.arange(len(d)),test_size=.2,random_state=SEED,stratify=d.HOSPITAL_EXPIRE_FLAG)
    assert set(d.iloc[tr].SUBJECT_ID).isdisjoint(set(d.iloc[te].SUBJECT_ID))
    X=d[ALL];y=d.HOSPITAL_EXPIRE_FLAG.to_numpy(int)
    dev=X.iloc[tr];test=X.iloc[te];yd=y[tr];yt=y[te]
    manifest=d[['SUBJECT_ID','HADM_ID']].copy();manifest['split']='development';manifest.loc[te,'split']='test';manifest.to_csv(ROOT/'private/split_manifest.csv',index=False)
    flow=pd.DataFrame([['原始住院记录',len(raw),int(raw.HOSPITAL_EXPIRE_FLAG.sum())],['每人首条可见住院',len(first),int(first.HOSPITAL_EXPIRE_FLAG.sum())],['主队列',len(d),int(y.sum())],['开发集',len(tr),int(yd.sum())],['测试集',len(te),int(yt.sum())]],columns=['stage','n','deaths'])
    save_table(flow,'01_cohort_flow.csv')
    # Table 1 tests apply to observed values only; no feature selection follows.
    tab=[]
    for c in ALL:
        s0=X.loc[y==0,c].dropna();s1=X.loc[y==1,c].dropna()
        fmt=lambda s:f'{s.median():.2f} [{s.quantile(.25):.2f}, {s.quantile(.75):.2f}]'
        if c=='gender_is_male':
            contingency=np.array([[(s0==1).sum(),(s0==0).sum()],[(s1==1).sum(),(s1==0).sum()]])
            pv=stats.chi2_contingency(contingency,correction=False)[1]
            a=f'{int(s0.sum())}/{len(s0)} ({100*s0.mean():.1f}%)';b=f'{int(s1.sum())}/{len(s1)} ({100*s1.mean():.1f}%)'
        else:pv=stats.mannwhitneyu(s0,s1,alternative='two-sided').pvalue;a=fmt(s0);b=fmt(s1)
        tab.append({'field':c,'provisional_label':LABEL[c],'survivors':a,'deaths':b,'missing_survivors':int(X.loc[y==0,c].isna().sum()),'missing_deaths':int(X.loc[y==1,c].isna().sum()),'p':pv})
    tab=pd.DataFrame(tab);tab['p_BH']=multipletests(tab.p,method='fdr_bh')[1];save_table(tab,'02_table1.csv')
    missing=pd.DataFrame({'field':VITAL+LAB,'provisional_label':[LABEL[c] for c in VITAL+LAB],
       'missing_survivors_pct':[100*X.loc[y==0,c].isna().mean() for c in VITAL+LAB],
       'missing_deaths_pct':[100*X.loc[y==1,c].isna().mean() for c in VITAL+LAB]})
    save_table(missing,'03_missingness.csv')
    cv=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(dev,yd))
    fitted={};pred={};params={};cvrows=[];thresholds=[]
    for mode in ['M0','M1','M2','M3','M4']:
        announce('训练 '+mode)
        base=pipeline(mode)
        if mode=='M4':
            base.set_params(model=GradientBoostingClassifier(n_estimators=100,min_samples_leaf=30,random_state=SEED))
            grid={'model__max_depth':[1,2],'model__learning_rate':[.03,.1]}
        else:grid={'model__C':[.01,.1,1.,10.,100.]}
        search=GridSearchCV(base,grid,scoring='neg_log_loss',cv=cv,n_jobs=1,refit=True,error_score='raise')
        search.fit(dev,yd);m=search.best_estimator_;fitted[mode]=m;params[mode]=search.best_params_
        for par,score,std in zip(search.cv_results_['params'],search.cv_results_['mean_test_score'],search.cv_results_['std_test_score']):
            cvrows.append({'model':mode,'params':json.dumps(par),'cv_logloss':-score,'cv_sd':std})
        # All held-out predictions are generated only after CV selects params.
        pred[mode]=m.predict_proba(test)[:,1]
        oof=cross_val_predict(clone(m),dev,yd,cv=cv,method='predict_proba',n_jobs=1)[:,1]
        fpr,tpr,th=roc_curve(yd,oof,drop_intermediate=False)
        eligible=th[(tpr>=.8)&np.isfinite(th)];threshold=float(eligible.max())
        tn,fp,fn,tp=confusion_matrix(yt,pred[mode]>=threshold).ravel()
        thresholds.append({'model':mode,'threshold_from_development_OOF':threshold,'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'PPV':tp/(tp+fp) if tp+fp else np.nan,'NPV':tn/(tn+fn),'TP':tp,'FP':fp,'TN':tn,'FN':fn})
        announce(mode+' 已锁定 '+str(params[mode]))
    save_table(pd.DataFrame(cvrows),'04_cross_validation.csv');save_table(pd.DataFrame(thresholds),'05_threshold_metrics.csv')
    base_p=np.full(len(yt),yd.mean());base_b=brier_score_loss(yt,base_p)
    results=[]
    for mode,p in pred.items():
        a,b,slope=calibration(yt,p)
        results.append(dict(model=mode,label=MODELS[mode],**metrics(yt,p),mean_predicted=float(p.mean()),observed_rate=float(yt.mean()),Brier_skill=1-brier_score_loss(yt,p)/base_b,calibration_intercept=a,calibration_joint_intercept=b,calibration_slope=slope))
    res=pd.DataFrame(results)
    predictions=d.iloc[te][['SUBJECT_ID','HADM_ID','HOSPITAL_EXPIRE_FLAG']].copy()
    for m,p in pred.items():predictions[m]=p
    predictions.to_csv(ROOT/'private/test_predictions.csv',index=False)
    announce('测试患者配对bootstrap 2000次')
    rng=np.random.default_rng(SEED+1);boot=np.empty((2000,5,3))
    for i in range(2000):
        ix=rng.integers(0,len(yt),len(yt));yy=yt[ix]
        if np.unique(yy).size<2:
            boot[i,:,:]=np.nan
            continue
        for j,m in enumerate(pred):
            pp=pred[m][ix]
            boot[i,j]=[brier_score_loss(yy,pp),roc_auc_score(yy,pp),average_precision_score(yy,pp)]
        if (i+1)%500==0:announce(f'测试bootstrap {i+1}/2000')
    for k,col in enumerate(['Brier','AUROC','AP']):
        res[col+'_lo']=np.nanpercentile(boot[:,:,k],2.5,axis=0);res[col+'_hi']=np.nanpercentile(boot[:,:,k],97.5,axis=0)
    save_table(res,'06_test_performance.csv')
    diffs=[]
    for a,b in [('M3','M1'),('M2','M1'),('M3','M2'),('M4','M3')]:
        ia=list(pred).index(a);ib=list(pred).index(b)
        for k,metric in enumerate(['Brier_improvement','AUROC_gain','AP_gain']):
            sign=-1 if k==0 else 1
            samples=sign*(boot[:,ia,k]-boot[:,ib,k])
            col=['Brier','AUROC','AP'][k];point=sign*(res.set_index('model').loc[a,col]-res.set_index('model').loc[b,col])
            diffs.append({'comparison':f'{a} vs {b}','metric':metric,'estimate':point,'lo':np.percentile(samples,2.5),'hi':np.percentile(samples,97.5),'valid_bootstrap':int(np.isfinite(samples).sum())})
    diffs=pd.DataFrame(diffs);save_table(diffs,'07_incremental_value.csv')
    # Model-parameter bootstrap; standardized to the same main developer SD.
    announce('开发集M3系数稳定性bootstrap 1000次')
    m=fitted['M3'];ft=m.named_steps['features'];ref_names=ft.names_;ref_scale=dict(zip(ft.names_full_,ft.scale_));ref_kind=dict(zip(ft.names_full_,ft.isnum_full_))
    coef_samples=[];rng=np.random.default_rng(SEED+2)
    for i in range(1000):
        ix=rng.integers(0,len(yd),len(yd));bm=clone(m).fit(dev.iloc[ix],yd[ix]);bf=bm.named_steps['features']
        scales=dict(zip(bf.names_full_,bf.scale_));bc=dict(zip(bf.names_,bm.named_steps['model'].coef_[0]))
        coef_samples.append([bc.get(c,np.nan)/scales[c]*ref_scale[c] if ref_kind[c] else bc.get(c,np.nan) for c in ref_names])
        if (i+1)%250==0:announce(f'系数bootstrap {i+1}/1000')
    samples=np.array(coef_samples);point=m.named_steps['model'].coef_[0]
    coefs=pd.DataFrame({'feature':ref_names,'beta_standardized':point,'lo':np.nanpercentile(samples,2.5,axis=0),'hi':np.nanpercentile(samples,97.5,axis=0),'valid_replicates':np.isfinite(samples).sum(axis=0)})
    save_table(coefs,'08_coefficient_stability.csv')
    # Sensitivities use the already-selected main C, never select by test result.
    announce('预设敏感性分析')
    sens=[]
    def run_sens(name,frame,winsor=False,spline=False,group=False):
        xx=frame[ALL];yy=frame.HOSPITAL_EXPIRE_FLAG.to_numpy(int)
        if group:
            a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(xx,yy,frame.SUBJECT_ID))
            assert set(frame.iloc[a].SUBJECT_ID).isdisjoint(set(frame.iloc[b].SUBJECT_ID))
        else:a,b=train_test_split(np.arange(len(frame)),test_size=.2,random_state=SEED,stratify=yy)
        if yy[a].sum()<30 or yy[b].sum()<10:
            sens.append({'scenario':name,'model':'not_fit','n':len(frame),'events':int(yy.sum()),'note':'事件过少，跳过'});return
        for mode in ['M1','M3']:
            pm=pipeline(mode,C=params[mode]['model__C'],winsor=winsor,spline=spline).fit(xx.iloc[a],yy[a])
            pp=pm.predict_proba(xx.iloc[b])[:,1]
            sens.append(dict(scenario=name,model=mode,n=len(frame),events=int(yy.sum()),test_n=len(b),test_events=int(yy[b].sum()),**metrics(yy[b],pp)))
    allad=raw.loc[good(raw)].reset_index(drop=True)
    run_sens('全部住院 按患者分组',allad,group=True)
    run_sens('排除0近似月龄',d.loc[d.age_month>=1].reset_index(drop=True))
    run_sens('完整候选病例',d.dropna(subset=ALL).reset_index(drop=True))
    run_sens('训练集0.5至99.5百分位缩尾',d,winsor=True)
    run_sens('年龄限制性立方样条',d,spline=True)
    run_sens('排除216及以上近似月龄',d.loc[d.age_month<216].reset_index(drop=True))
    screened=d.copy();screen_counts=[]
    rules={'chart_1001_min':(25,45),'chart_1001_max':(25,45),'chart_1003_max':(20,350),'chart_1016_min':(10,300),'lab_5129_min':(0,2000),'lab_5033_max':(0,None),'lab_5041_max':(0,None),'lab_5223_max':(0,None)}
    for c,(lo,hi) in rules.items():
        mask=(screened[c].le(lo) if lo==0 else screened[c].lt(lo))
        if hi is not None:mask|=screened[c].gt(hi)
        screen_counts.append({'field':c,'marked_n':int(mask.sum()),'rule':f'lower={lo}, upper={hi}; conditional on inferred units'})
        screened.loc[mask,c]=np.nan
    run_sens('推定单位下宽界限置缺失',screened)
    save_table(pd.DataFrame(sens),'09_sensitivity.csv');save_table(pd.DataFrame(screen_counts),'10_screening_sensitivity_rules.csv')
    announce('绘制主要图表')
    fig,ax=plt.subplots(figsize=(9,6.5));ax.axis('off')
    texts=[f'原始衍生数据\n13,258次住院 / 780次死亡',f'每人首条可见住院\n12,690人 / 761次死亡',f'排除68条负月龄记录\n主队列12,622人 / 759次死亡',f'开发集 {len(tr):,}人 / {yd.sum()}次死亡',f'测试集 {len(te):,}人 / {yt.sum()}次死亡']
    pos=[(.5,.88),(.5,.65),(.5,.42),(.25,.14),(.75,.14)]
    for i,(t,(xx,yy)) in enumerate(zip(texts,pos)):
        ax.text(xx,yy,t,ha='center',va='center',fontsize=12,bbox={'boxstyle':'round,pad=.8','facecolor':'#edf3f7','edgecolor':'#47738e'},transform=ax.transAxes)
    for a,b in [(0,1),(1,2),(2,3),(2,4)]:
        ax.annotate('',xy=(pos[b][0],pos[b][1]+.08),xytext=(pos[a][0],pos[a][1]-.08),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#47738e','lw':1.4})
    ax.set_title('研究样本与内部验证划分',pad=15);save_fig(fig,'01_cohort_flow')
    fig,ax=plt.subplots(figsize=(10,6));z=np.arange(len(missing));ax.barh(z-.17,missing.missing_survivors_pct,.32,label='存活',color='#33779b');ax.barh(z+.17,missing.missing_deaths_pct,.32,label='死亡',color='#ba654d')
    ax.set_yticks(z,[f'{LABEL[c]}\n{c}' for c in missing.field]);ax.invert_yaxis();ax.set_xlabel('缺失比例 (%)');ax.set_xlim(0,100);ax.legend();ax.set_title('主队列中的缺失模式（*名称未获官方字典核验）');fig.tight_layout();save_fig(fig,'02_missingness')
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for mode,p in pred.items():
        f,t,_=roc_curve(yt,p);pr,re,_=precision_recall_curve(yt,p)
        rr=res.set_index('model').loc[mode]
        axs[0].plot(f,t,color=COLORS[mode],label=f'{mode}  AUC {rr.AUROC:.3f}')
        axs[1].step(re,pr,where='post',color=COLORS[mode],label=f'{mode}  AP {rr.AP:.3f}')
    axs[0].plot([0,1],[0,1],'--',color='#aaa');axs[1].axhline(yt.mean(),ls='--',color='#aaa',label=f'死亡比例 {yt.mean():.3f}')
    axs[0].set(xlabel='1－特异度',ylabel='敏感度',title='测试集ROC曲线');axs[1].set(xlabel='召回率',ylabel='精确率',title='测试集PR曲线')
    for ax in axs:ax.legend(fontsize=9);ax.set_xlim(0,1);ax.set_ylim(0,1)
    fig.tight_layout();save_fig(fig,'03_ROC_PR')
    calrows=[];fig,axs=plt.subplots(1,3,figsize=(14,4.5))
    for ax,mode in zip(axs,['M1','M2','M3']):
        p=pred[mode];groups=pd.qcut(p,q=10,duplicates='drop');cal=pd.DataFrame({'p':p,'y':yt,'bin':groups}).groupby('bin',observed=True).agg(mean_predicted=('p','mean'),observed=('y','mean'),n=('y','size'),events=('y','sum')).reset_index(drop=True)
        ci=np.array([wilson(r.events,r.n) for r in cal.itertuples()]);cal['lo']=ci[:,0];cal['hi']=ci[:,1];cal['model']=mode;calrows.extend(cal.to_dict('records'))
        ax.errorbar(cal.mean_predicted,cal.observed,yerr=np.array([cal.observed-cal.lo,cal.hi-cal.observed]),fmt='o-',color=COLORS[mode],capsize=3)
        lim=max(cal.mean_predicted.max(),cal.hi.max())*1.1;ax.plot([0,lim],[0,lim],'--',color='#aaa');ax.set(xlim=(0,lim),ylim=(0,lim),xlabel='平均预测概率',ylabel='观察死亡比例',title=f'{mode} 风险校准')
        ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle('按预测十分位校准（误差条为事件率Wilson 95%区间）');fig.tight_layout();save_fig(fig,'04_calibration');save_table(pd.DataFrame(calrows),'11_calibration_bins.csv')
    fig,axs=plt.subplots(1,3,figsize=(13,4.5))
    for ax,metric in zip(axs,['Brier','AUROC','AP']):
        vals=res[metric].to_numpy();ax.errorbar(vals,np.arange(5),xerr=[vals-res[metric+'_lo'],res[metric+'_hi']-vals],fmt='o',capsize=4,color='#275c7b');ax.set_yticks(np.arange(5),res.model);ax.invert_yaxis();ax.set_title(metric+('（越小越好）' if metric=='Brier' else '（越大越好）'));ax.grid(axis='x',alpha=.15)
    fig.suptitle('测试集表现与2000次配对bootstrap区间');fig.tight_layout();save_fig(fig,'05_performance')
    fig,axs=plt.subplots(1,3,figsize=(14,4.5))
    for ax,metric in zip(axs,['Brier_improvement','AUROC_gain','AP_gain']):
        z=diffs.loc[diffs.metric==metric];ax.errorbar(z.estimate,np.arange(len(z)),xerr=[z.estimate-z.lo,z.hi-z.estimate],fmt='o',capsize=4,color='#25826f');ax.axvline(0,color='#999',ls='--');ax.set_yticks(np.arange(len(z)),z.comparison);ax.invert_yaxis();ax.set_title(metric);ax.grid(axis='x',alpha=.15)
    fig.suptitle('模型增量（正值表示改善）');fig.tight_layout();save_fig(fig,'06_incremental_value')
    fig,ax=plt.subplots(figsize=(10,8));show=coefs.sort_values('beta_standardized');lab=[]
    for x in show.feature:
        if x=='log1p_age':lab.append('log(1＋近似月龄) / 1 SD')
        elif x.endswith('__missing'):lab.append(x.replace('__missing','')+' 缺失 / 有记录')
        else:lab.append(x+(' / 0→1' if x=='gender_is_male' else ' / 1 SD'))
    ax.errorbar(show.beta_standardized,np.arange(len(show)),xerr=[show.beta_standardized-show.lo,show.hi-show.beta_standardized],fmt='o',capsize=3,color='#285e7d');ax.axvline(0,ls='--',color='#999');ax.set_yticks(np.arange(len(show)),lab);ax.set_xlabel('标准化log-odds系数（非因果效应）');ax.set_title('M3开发集系数稳定性：固定C、1000次重拟合');fig.tight_layout();save_fig(fig,'07_coefficient_stability')
    ss=pd.DataFrame(sens);ss=ss[ss.model!='not_fit'];fig,ax=plt.subplots(figsize=(10,5.5))
    scenarios=ss.scenario.unique()
    for j,mode in enumerate(['M1','M3']):
        z=ss[ss.model==mode].set_index('scenario').reindex(scenarios);ax.scatter(z.Brier,np.arange(len(z))+(j-.5)*.15,label=mode,color=COLORS[mode])
    ax.set_yticks(np.arange(len(scenarios)),scenarios);ax.invert_yaxis();ax.set_xlabel('Brier分数（越小越好）');ax.set_title('敏感性分析（不同队列之间不直接排名）');ax.legend();fig.tight_layout();save_fig(fig,'08_sensitivity')
    meta=dict(seed=SEED,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cohort_n=len(d),events=int(y.sum()),development_n=len(tr),development_events=int(yd.sum()),test_n=len(te),test_events=int(yt.sum()),chosen_params=params,null_Brier=base_b,development_event_rate=float(yd.mean()),test_event_rate=float(yt.mean()),runtime_seconds=time.time()-start,primary_uses_raw_numeric_values=True,unit_verification=False,test_bootstrap_n=2000,coefficient_bootstrap_n=1000,software={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':__import__('sklearn').__version__,'statsmodels':sm.__version__})
    (ROOT/'run_summary.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    announce('完成分析 '+json.dumps(meta,ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);args=ap.parse_args()
    with threadpool_limits(limits=1):main(args.input)
