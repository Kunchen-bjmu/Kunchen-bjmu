"""Redraw display-only figures from locked predictions and aggregate results.
No model refitting or parameter changes.
"""
from analyze_pic import ROOT,COLORS,save_fig,plt,PercentFormatter
import pandas as pd
from sklearn.metrics import roc_curve,precision_recall_curve
P=pd.read_csv(ROOT/'private/test_predictions.csv')
R=pd.read_csv(ROOT/'tables/06_test_performance.csv').set_index('model')
y=P.HOSPITAL_EXPIRE_FLAG.to_numpy()
fig,axs=plt.subplots(1,2,figsize=(12,5))
for m in R.index:
    p=P[m].to_numpy();f,t,_=roc_curve(y,p);pr,re,_=precision_recall_curve(y,p)
    axs[0].plot(f,t,color=COLORS[m],label=f'{m}  AUC {R.loc[m,"AUROC"]:.3f}')
    axs[1].step(re,pr,where='post',color=COLORS[m],label=f'{m}  AP {R.loc[m,"AP"]:.3f}')
axs[0].plot([0,1],[0,1],'--',color='#aaa')
axs[1].axhline(y.mean(),ls='--',color='#aaa',label=f'死亡比例 {y.mean():.3f}')
axs[0].set(xlabel='1－特异度',ylabel='敏感度',title='测试集ROC曲线')
axs[1].set(xlabel='召回率',ylabel='精确率',title='测试集PR曲线')
for ax in axs:ax.legend(fontsize=9);ax.set_xlim(0,1);ax.set_ylim(0,1)
fig.tight_layout();save_fig(fig,'03_ROC_PR')
C=pd.read_csv(ROOT/'tables/11_calibration_bins.csv')
fig,axs=plt.subplots(1,3,figsize=(14,4.5))
for ax,m in zip(axs,['M1','M2','M3']):
    z=C[C.model==m];ax.errorbar(z.mean_predicted,z.observed,yerr=[z.observed-z.lo,z.hi-z.observed],fmt='o-',color=COLORS[m],capsize=3)
    lim=max(z.mean_predicted.max(),z.hi.max())*1.1
    ax.plot([0,lim],[0,lim],'--',color='#aaa');ax.set(xlim=(0,lim),ylim=(0,lim),xlabel='平均预测概率',ylabel='观察死亡比例',title=f'{m} 风险校准')
    ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
fig.suptitle('按预测十分位校准（误差条为事件率Wilson 95%区间）');fig.tight_layout();save_fig(fig,'04_calibration')
print('Redrawn ROC/PR and calibration from locked predictions; no refit.')
