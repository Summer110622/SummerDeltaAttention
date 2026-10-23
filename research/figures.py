from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
P=Path(__file__).parent
S=pd.read_csv(P/'summary_results.csv')
BLUE='#1678a2'; TEAL='#119b81'; ORANGE='#c17a30'; GREY='#626f80'; LIGHT='#e6eef2'; RED='#ba4951'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})

def sub(n,noise,ret,method):
 v=S[(S.n==n)&(S.noise==noise)&(S.retention==ret)&(S.method==method)]
 return float(v.mean_mse.iloc[0]),float(v['sem'].iloc[0])

# diagram
fig,ax=plt.subplots(figsize=(10.4,3.3));ax.set_xlim(0,10.4);ax.set_ylim(0,3.3);ax.axis('off')
def node(x,y,w,h,label,subline,edge):
 rect=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.11,rounding_size=.15',edgecolor=edge,facecolor='white',linewidth=1.4)
 ax.add_patch(rect);ax.text(x+w/2,y+h*.63,label,ha='center',va='center',weight='bold',fontsize=10,color=edge)
 ax.text(x+w/2,y+h*.27,subline,ha='center',va='center',fontsize=7.8,color=GREY)
def arrow(x1,y1,x2,y2):
 ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,color=GREY,linewidth=1.25))
node(.12,1.36,1.55,.82,'Input token','q, k, v',GREY)
node(2.24,2.2,2.12,.82,'KDA state','channel-wise decay',BLUE)
node(4.92,2.2,2.1,.82,'KDA readout','q^T S',BLUE)
node(2.24,.62,2.12,.82,'SDA fast state','no-decay delta rule',TEAL)
node(2.24,-.02,2.12,.44,'KV archive','retained subset',TEAL)
node(5.02,.53,2.0,.82,'Indexed residual','Top-1 + confidence',TEAL)
node(8,.58,2.1,.82,'SDA readout','fast + correction',TEAL)
arrow(1.78,1.8,2.2,2.58);arrow(4.48,2.61,4.85,2.61)
arrow(1.78,1.69,2.17,1.0);arrow(4.47,1.03,4.94,1.03);arrow(4.45,.19,5.0,.78);arrow(7.15,.94,7.9,.94)
ax.text(8.1,2.47,'Comparison principle:',fontsize=8.6,weight='bold',color=GREY)
ax.text(8.1,2.15,'same keys / values',fontsize=8.4,color=GREY)
ax.text(8.1,1.89,'separate calibration',fontsize=8.4,color=GREY)
ax.text(8.1,1.63,'different memory budgets',fontsize=8.4,color=RED)
fig.savefig(P/'fig_architecture.pdf');fig.savefig(P/'fig_architecture.png',dpi=180);plt.close(fig)

methods=[('Full KV softmax (sharp)','Full KV',GREY),('KDA channel-gated proxy','KDA proxy',BLUE),('No-decay DeltaNet (calibrated beta)','DeltaNet',ORANGE),('Indexed KV (zero fallback)','KV only',RED),('SDA v5-Gated (original)','SDA orig.',GREY),('SDA v5-Gated (calibrated)','SDA calibrated',TEAL)]
fig,ax=plt.subplots(figsize=(9,3.6))
vals=np.array([sub(512,0,.5,k)[0] for k,_,_ in methods]); err=np.array([sub(512,0,.5,k)[1] for k,_,_ in methods]);
y=np.arange(len(methods))
ax.barh(y,vals,xerr=err,color=[m[2] for m in methods],alpha=.88,height=.64,error_kw={'capsize':2})
ax.set_yticks(y,[m[1] for m in methods]);ax.invert_yaxis();ax.set_xlabel('Value reconstruction MSE (lower is better)');ax.set_xlim(0,1.12)
for i,(k,_,_) in enumerate(methods):
 mem=S[(S.n==512)&(S.noise==0)&(S.retention==.5)&(S.method==k)].memory_kib.iloc[0]
 ax.text(vals[i]+.028,i,f'{vals[i]:.4f}  |  {mem:.0f} KiB',va='center',fontsize=8)
ax.grid(axis='x',alpha=.13);fig.savefig(P/'fig_main_comparison.pdf');fig.savefig(P/'fig_main_comparison.png',dpi=170);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(9.4,3.45),gridspec_kw={'width_ratios':[1.1,1]})
for method,label,col in [('SDA v5-Gated (calibrated)','SDA calibrated',TEAL),('SDA v5-Gated (original)','SDA original',GREY),('Indexed KV (zero fallback)','KV only',RED)]:
 s=S[(S.n==512)&(S.noise==0)&(S.method==method)].sort_values('retention')
 axs[0].errorbar(s.retention*100,s.mean_mse,yerr=s['sem'],marker='o',capsize=2,label=label,color=col,linewidth=2)
for method,label,col in [('KDA channel-gated proxy','KDA proxy',BLUE),('No-decay DeltaNet (calibrated beta)','DeltaNet',ORANGE)]:
 val=sub(512,0,.5,method)[0];axs[0].axhline(val,label=label,color=col,linestyle='--',linewidth=1.5)
axs[0].set(xlabel='KV retention (%)',ylabel='MSE',title='Retention / reconstruction error');axs[0].set_xticks([10,25,50,100]);axs[0].grid(alpha=.16);axs[0].legend(fontsize=7,loc='upper right')
for method,label,col in [('SDA v5-Gated (calibrated)','SDA calibrated',TEAL),('Indexed KV (zero fallback)','KV only',RED)]:
 s=S[(S.n==512)&(S.noise==0)&(S.method==method)].sort_values('retention')
 axs[1].plot(s.memory_kib,s.mean_mse,'-o',color=col,label=label,linewidth=2)
for method,label,col in [('KDA channel-gated proxy','KDA proxy',BLUE),('No-decay DeltaNet (calibrated beta)','DeltaNet',ORANGE)]:
 val=sub(512,0,.5,method)[0];axs[1].scatter(4,val,c=col,s=60,label=label)
axs[1].scatter(192,sub(512,0,.5,'Full KV softmax (sharp)')[0],marker='s',c=GREY,s=55,label='Full KV')
axs[1].set(xlabel='Stored state + KV (KiB)',ylabel='MSE',title='Accuracy / storage trade-off');axs[1].grid(alpha=.16);axs[1].legend(fontsize=7,loc='upper right');fig.tight_layout(pad=1.3)
fig.savefig(P/'fig_budget_pareto.pdf');fig.savefig(P/'fig_budget_pareto.png',dpi=170);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(9.4,3.3))
for method,label,col in [('KDA channel-gated proxy','KDA proxy',BLUE),('SDA v5-Gated (calibrated)','SDA calibrated',TEAL),('SDA v5-Gated (original)','SDA original',GREY),('Indexed KV (zero fallback)','KV only',RED),('Full KV softmax (sharp)','Full KV (sharp)',ORANGE)]:
 s=S[(S.n==512)&(S.retention==.5)&(S.method==method)].sort_values('noise')
 axs[0].errorbar(s.noise,s.mean_mse,yerr=s['sem'],label=label,color=col,marker='o',capsize=2,linewidth=1.8)
 s=S[(S.noise==0)&(S.retention==.5)&(S.method==method)].sort_values('n')
 axs[1].plot(s.n,s.mean_mse,'-o',label=label,color=col,linewidth=1.8)
axs[0].set(xlabel='Query noise sigma',ylabel='MSE',title='Robustness at N = 512');axs[1].set(xlabel='Number of written KV pairs',ylabel='MSE',title='Sequence-length generalization (sigma = 0)');axs[1].set_xscale('log',base=2);axs[1].set_xticks([128,512,2048],['128','512','2048'])
for ax in axs:ax.grid(alpha=.16)
axs[0].legend(fontsize=7,loc='upper left');fig.tight_layout(pad=1.3)
fig.savefig(P/'fig_noise_scaling.pdf');fig.savefig(P/'fig_noise_scaling.png',dpi=170);plt.close(fig)
print('done')
