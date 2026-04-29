import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
N = 1200

# ── 1. Realistic Indian loan applicant dataset ────────────────────────────────
age        = np.random.randint(22, 62, N)
income_lpa = np.round(np.random.lognormal(mean=np.log(6), sigma=0.6, size=N), 2)
loan_amt   = np.round(income_lpa * np.random.uniform(0.5, 4.0, N), 2)
loan_term  = np.random.choice([12,24,36,48,60], N, p=[0.1,0.2,0.35,0.25,0.1])
cibil_score= np.clip(np.random.normal(720, 80, N).astype(int), 300, 900)
emp_years  = np.random.randint(0, 20, N)
existing_loans = np.random.choice([0,1,2,3], N, p=[0.4,0.35,0.18,0.07])
education  = np.random.choice(['Graduate','Post-Graduate','Under-Graduate','Diploma'],
                               N, p=[0.45,0.25,0.2,0.1])
employment = np.random.choice(['Salaried','Self-Employed','Business','Freelance'],
                               N, p=[0.55,0.2,0.18,0.07])
property_area = np.random.choice(['Urban','Semi-Urban','Rural'], N, p=[0.45,0.35,0.2])
gender     = np.random.choice(['Male','Female'], N, p=[0.62,0.38])
dependents = np.random.choice([0,1,2,3], N, p=[0.35,0.3,0.25,0.1])

# Default probability — influenced by realistic factors
default_prob = (
    0.12
    - 0.0003*(cibil_score - 600)
    + 0.04*(loan_amt / (income_lpa + 0.1) - 2)
    + 0.02*existing_loans
    - 0.005*emp_years
    + 0.03*(employment == 'Freelance').astype(int)
    + 0.02*(property_area == 'Rural').astype(int)
    - 0.01*(education == 'Post-Graduate').astype(int)
    + 0.02*dependents
)
default_prob = np.clip(default_prob, 0.04, 0.88)
default      = (np.random.random(N) < default_prob).astype(int)

df = pd.DataFrame({
    'Age': age, 'Annual_Income_LPA': income_lpa, 'Loan_Amount_LPA': loan_amt,
    'Loan_Term_Months': loan_term, 'CIBIL_Score': cibil_score,
    'Employment_Years': emp_years, 'Existing_Loans': existing_loans,
    'Education': education, 'Employment_Type': employment,
    'Property_Area': property_area, 'Gender': gender,
    'Dependents': dependents, 'Default': default
})
df['Loan_to_Income_Ratio'] = (df['Loan_Amount_LPA'] / df['Annual_Income_LPA']).round(3)

print(f"✅ Dataset: {len(df)} applicants | Default rate: {df['Default'].mean()*100:.1f}%")

# ── 2. Feature Engineering & Encoding ────────────────────────────────────────
df_model = df.copy()
df_model['Is_Graduate']    = (df['Education'].isin(['Graduate','Post-Graduate'])).astype(int)
df_model['Is_Salaried']    = (df['Employment_Type'] == 'Salaried').astype(int)
df_model['Is_Urban']       = (df['Property_Area'] == 'Urban').astype(int)
df_model['Is_Female']      = (df['Gender'] == 'Female').astype(int)
df_model['CIBIL_Category'] = pd.cut(df['CIBIL_Score'],
                                     bins=[0,579,669,739,799,900],
                                     labels=[0,1,2,3,4]).astype(int)

features = ['Age','Annual_Income_LPA','Loan_Amount_LPA','Loan_Term_Months',
            'CIBIL_Score','Employment_Years','Existing_Loans','Dependents',
            'Loan_to_Income_Ratio','Is_Graduate','Is_Salaried','Is_Urban',
            'Is_Female','CIBIL_Category']

X = df_model[features]
y = df_model['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc = scaler.transform(X_test)

# ── 3. Train 3 Models ─────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42),
}

results = {}
for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_tr_sc, y_train)
        y_pred  = model.predict(X_te_sc)
        y_proba = model.predict_proba(X_te_sc)[:,1]
    else:
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
    results[name] = {
        'model': model, 'pred': y_pred, 'proba': y_proba,
        'acc':   round(accuracy_score(y_test, y_pred)*100, 2),
        'auc':   round(roc_auc_score(y_test, y_proba), 4),
        'cm':    confusion_matrix(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True),
    }
    print(f"   {name}: Acc={results[name]['acc']}% | AUC={results[name]['auc']}")

best_name = max(results, key=lambda k: results[k]['auc'])
best      = results[best_name]
print(f"\n🏆 Best Model: {best_name} (AUC={best['auc']})")

# ── 4. DASHBOARD ──────────────────────────────────────────────────────────────
BG    = '#0A0E1A'; PANEL = '#111827'; PANEL2= '#1A2235'
TEXT  = '#F0F4FF'; SUB   = '#8892A4'; ACC   = '#6C63FF'
GREEN = '#00D97E'; RED   = '#FF4560'; GOLD  = '#FFB400'; TEAL  = '#00C9C9'

fig = plt.figure(figsize=(20, 24), facecolor=BG)
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.48, wspace=0.32,
                        top=0.94, bottom=0.04, left=0.07, right=0.97)

fig.text(0.5,0.965,'CREDIT RISK SCORING MODEL  —  INDIAN LOAN APPLICANTS', ha='center',
         fontsize=20, fontweight='bold', color=TEXT, fontfamily='monospace')
fig.text(0.5,0.950,f'1,200 Applicants  |  14 Features  |  3 Models Compared  |  Best: {best_name}  (AUC={best["auc"]})',
         ha='center', fontsize=10, color=SUB)

# KPI cards
ax_k = fig.add_subplot(gs[0,:])
ax_k.set_facecolor(BG); ax_k.axis('off')
kpis = [
    ("1,200","Loan Applicants\nAnalysed",ACC),
    (f"{df['Default'].mean()*100:.1f}%","Dataset Default\nRate",RED),
    (f"{best['acc']}%",f"Best Model\nAccuracy",GREEN),
    (f"{best['auc']}","Best Model\nAUC-ROC Score",TEAL),
    ("14","Features\nEngineered",GOLD),
    ("3","ML Models\nCompared",ACC),
]
cw = 1/6
for i,(val,lbl,col) in enumerate(kpis):
    x = i*cw+0.01
    ax_k.add_patch(FancyBboxPatch((x,0.05),cw-0.02,0.88, boxstyle="round,pad=0.02",
                  facecolor=PANEL2, transform=ax_k.transAxes, zorder=0, edgecolor=col, linewidth=1.5))
    ax_k.text(x+(cw-0.02)/2, 0.62, val, transform=ax_k.transAxes,
              ha='center', va='center', fontsize=15, fontweight='bold', color=col)
    ax_k.text(x+(cw-0.02)/2, 0.25, lbl, transform=ax_k.transAxes,
              ha='center', va='center', fontsize=8.5, color=SUB, linespacing=1.4)

# Chart 1: ROC Curves
ax1 = fig.add_subplot(gs[1,0])
ax1.set_facecolor(PANEL)
colors_roc = [ACC, GREEN, GOLD]
for (name, res), col in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res['proba'])
    ax1.plot(fpr, tpr, color=col, linewidth=2, label=f"{name} (AUC={res['auc']})")
ax1.plot([0,1],[0,1], color=SUB, linewidth=1, linestyle='--', alpha=0.5)
ax1.fill_between(*roc_curve(y_test, best['proba'])[:2], alpha=0.08, color=GREEN)
ax1.set_xlabel('False Positive Rate', color=SUB, fontsize=9)
ax1.set_ylabel('True Positive Rate', color=SUB, fontsize=9)
ax1.set_title('ROC Curves — Model Comparison', color=TEXT, fontsize=11, fontweight='bold', pad=8)
ax1.legend(facecolor=PANEL2, edgecolor='#2D3748', labelcolor=TEXT, fontsize=8.5)
ax1.tick_params(colors=SUB, labelsize=8)
for sp in ax1.spines.values(): sp.set_color('#2D3748')
ax1.grid(alpha=0.08, color=SUB)

# Chart 2: Confusion Matrix (best model)
ax2 = fig.add_subplot(gs[1,1])
ax2.set_facecolor(PANEL)
cm = best['cm']
im = ax2.imshow(cm, cmap='Blues', aspect='auto')
for i in range(2):
    for j in range(2):
        ax2.text(j,i,str(cm[i,j]), ha='center', va='center',
                 fontsize=18, fontweight='bold',
                 color='white' if cm[i,j]>cm.max()*0.5 else TEXT)
ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
ax2.set_xticklabels(['Predicted\nNo Default','Predicted\nDefault'], color=TEXT, fontsize=9)
ax2.set_yticklabels(['Actual\nNo Default','Actual\nDefault'], color=TEXT, fontsize=9)
ax2.set_title(f'Confusion Matrix — {best_name}', color=TEXT, fontsize=11, fontweight='bold', pad=8)
for sp in ax2.spines.values(): sp.set_color('#2D3748')
rep = best['report']
ax2.text(0.5,-0.18, f"Precision: {rep['1']['precision']:.2f}  |  Recall: {rep['1']['recall']:.2f}  |  F1: {rep['1']['f1-score']:.2f}",
         transform=ax2.transAxes, ha='center', color=SUB, fontsize=8.5)

# Chart 3: Feature Importance
ax3 = fig.add_subplot(gs[2,0])
ax3.set_facecolor(PANEL)
rf_model = results['Random Forest']['model']
fi = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)
colors_fi = [GREEN if v > fi.median() else TEAL for v in fi.values]
bars = ax3.barh(fi.index, fi.values, color=colors_fi, height=0.65, edgecolor='none', zorder=3)
for bar, val in zip(bars, fi.values):
    ax3.text(val+0.001, bar.get_y()+bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=7.5, color=TEXT)
ax3.set_title('Feature Importance — Random Forest', color=TEXT, fontsize=11, fontweight='bold', pad=8)
ax3.tick_params(colors=SUB, labelsize=8)
for sp in ax3.spines.values(): sp.set_color('#2D3748')
ax3.grid(axis='x', alpha=0.1, color=SUB, zorder=0)

# Chart 4: CIBIL Score vs Default Rate
ax4 = fig.add_subplot(gs[2,1])
ax4.set_facecolor(PANEL)
bins_cibil = [300,500,580,670,740,800,900]
labels_cibil = ['300-500\n(Poor)','500-580\n(Fair)','580-670\n(Good)',
                 '670-740\n(Very Good)','740-800\n(Excellent)','800+\n(Exceptional)']
df['CIBIL_Band'] = pd.cut(df['CIBIL_Score'], bins=bins_cibil, labels=labels_cibil)
cibil_def = df.groupby('CIBIL_Band', observed=True)['Default'].mean() * 100
bar_c4 = [RED if v>30 else GOLD if v>15 else GREEN for v in cibil_def.values]
bars4 = ax4.bar(range(len(cibil_def)), cibil_def.values, color=bar_c4, width=0.6, edgecolor='none', zorder=3)
ax4.set_xticks(range(len(cibil_def)))
ax4.set_xticklabels(cibil_def.index, color=SUB, fontsize=7.5)
for bar,val in zip(bars4, cibil_def.values):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f'{val:.1f}%', ha='center', fontsize=8.5, color=TEXT, fontweight='bold')
ax4.set_ylabel('Default Rate (%)', color=SUB, fontsize=9)
ax4.set_title('Default Rate by CIBIL Score Band', color=TEXT, fontsize=11, fontweight='bold', pad=8)
ax4.tick_params(colors=SUB, labelsize=8)
for sp in ax4.spines.values(): sp.set_color('#2D3748')
ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax4.grid(axis='y', alpha=0.1, color=SUB, zorder=0)

# Chart 5: Model comparison bars + Income vs Default
ax5 = fig.add_subplot(gs[3,0])
ax5.set_facecolor(PANEL)
model_names = list(results.keys())
accs = [results[m]['acc'] for m in model_names]
aucs = [results[m]['auc']*100 for m in model_names]
x_pos = np.arange(len(model_names))
w = 0.35
b1 = ax5.bar(x_pos-w/2, accs, width=w, color=ACC,  label='Accuracy (%)', edgecolor='none', zorder=3)
b2 = ax5.bar(x_pos+w/2, aucs, width=w, color=GOLD, label='AUC × 100',    edgecolor='none', zorder=3)
for bar,val in zip(list(b1)+list(b2), accs+aucs):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
             f'{val:.1f}', ha='center', fontsize=8, color=TEXT, fontweight='bold')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(['Logistic\nRegression','Random\nForest','Gradient\nBoosting'], color=TEXT, fontsize=8)
ax5.set_ylabel('Score', color=SUB, fontsize=9)
ax5.set_title('Model Performance Comparison', color=TEXT, fontsize=11, fontweight='bold', pad=8)
ax5.legend(facecolor=PANEL2, edgecolor='#2D3748', labelcolor=TEXT, fontsize=8)
ax5.tick_params(colors=SUB, labelsize=8)
for sp in ax5.spines.values(): sp.set_color('#2D3748')
ax5.set_ylim(0, 105)
ax5.grid(axis='y', alpha=0.1, color=SUB, zorder=0)

# Chart 6: Loan-to-Income Ratio vs Default
ax6 = fig.add_subplot(gs[3,1])
ax6.set_facecolor(PANEL)
lti_bins = [0,1,1.5,2,2.5,3,10]
lti_lbls = ['<1x','1-1.5x','1.5-2x','2-2.5x','2.5-3x','>3x']
df['LTI_Band'] = pd.cut(df['Loan_to_Income_Ratio'], bins=lti_bins, labels=lti_lbls)
lti_def = df.groupby('LTI_Band', observed=True)['Default'].mean()*100
bar_c6  = [RED if v>35 else GOLD if v>20 else GREEN for v in lti_def.values]
bars6   = ax6.bar(range(len(lti_def)), lti_def.values, color=bar_c6, width=0.6, edgecolor='none', zorder=3)
ax6.set_xticks(range(len(lti_def)))
ax6.set_xticklabels(lti_def.index, color=SUB, fontsize=8.5)
for bar,val in zip(bars6, lti_def.values):
    ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f'{val:.1f}%', ha='center', fontsize=8.5, color=TEXT, fontweight='bold')
ax6.set_ylabel('Default Rate (%)', color=SUB, fontsize=9)
ax6.set_title('Default Rate by Loan-to-Income Ratio', color=TEXT, fontsize=11, fontweight='bold', pad=8)
ax6.tick_params(colors=SUB, labelsize=8)
for sp in ax6.spines.values(): sp.set_color('#2D3748')
ax6.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax6.grid(axis='y', alpha=0.1, color=SUB, zorder=0)

fig.text(0.5,0.008,'Data: Simulated Indian Loan Applicants  |  Analysis: Hruthvik HS  |  Tools: Python (Pandas, Scikit-learn, Matplotlib)',
         ha='center', fontsize=8, color=SUB)

plt.savefig('/mnt/user-data/outputs/Project3_Credit_Risk/credit_risk_dashboard.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✅ Dashboard saved!")

df.to_csv('/mnt/user-data/outputs/Project3_Credit_Risk/loan_applicants.csv', index=False)

metrics_df = pd.DataFrame([{
    'Model': name, 'Accuracy_Pct': res['acc'], 'AUC_ROC': res['auc'],
    'Precision': round(res['report']['1']['precision'],4),
    'Recall':    round(res['report']['1']['recall'],4),
    'F1_Score':  round(res['report']['1']['f1-score'],4),
} for name, res in results.items()])
metrics_df.to_csv('/mnt/user-data/outputs/Project3_Credit_Risk/model_metrics.csv', index=False)
print("✅ CSVs saved!")
print(f"\n🎉 Credit Risk Model COMPLETE!")
print(f"   Best Model: {best_name} | Accuracy: {best['acc']}% | AUC: {best['auc']}")
