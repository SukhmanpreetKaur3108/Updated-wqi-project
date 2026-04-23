"""
Comprehensive Publication-Quality Plots
Water Quality Index — Haryana Groundwater & Surface Water
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, matthews_corrcoef, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

OUT = r"D:\My Projects\Py-DS-ML-Bootcamp-master\research paper\plots"
import os; os.makedirs(OUT, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    'primary':    '#1A3A5C',
    'secondary':  '#2E86AB',
    'accent':     '#E84855',
    'highlight':  '#F4A261',
    'good':       '#2ECC71',
    'warn':       '#F39C12',
    'danger':     '#E74C3C',
    'critical':   '#8E44AD',
    'bg':         '#F8F9FA',
    'grid':       '#DEE2E6',
}
MODEL_COLORS = {
    'KNN':               '#3498DB',
    'Random Forest':     '#2ECC71',
    'SVM':               '#E74C3C',
    'Gradient Boosting': '#F39C12',
    'Decision Tree':     '#9B59B6',
    'XGBoost':           '#1ABC9C',
    'K-Means':           '#E84855',
}
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.facecolor': '#F8F9FA',
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.color': '#DEE2E6',
    'grid.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
BASE = r"D:\My Projects\Py-DS-ML-Bootcamp-master"

gw_raw = pd.read_csv(f"{BASE}/groundwater_wqi.csv")
sw_raw = pd.read_csv(f"{BASE}/surface_wqi.csv")

# Normalise column names
gw = gw_raw.copy()
sw = sw_raw.copy()
sw.columns = [c.lower().replace(' ','_').replace('(','').replace(')','')
               .replace('/','_').replace('µg_l','µg_l') for c in sw.columns]
gw.columns = [c.lower() for c in gw.columns]

# Common parameter map  {display_name: (gw_col, sw_col, BIS_limit)}
PARAMS = {
    'pH':              ('ph',                 'ph',                 (6.5, 8.5)),
    'Turbidity (NTU)': ('turbidity',          'turbidity',          5),
    'Conductivity':    ('conductivity',       'conductivity',       2000),
    'Chloride (ppm)':  ('chloride_(ppm)',     'chloride_ppm',       250),
    'Sulphates (ppm)': ('sulphates(ppm)',     'sulphates(ppm)',     200),
    'COD (ppm)':       ('cod(ppm)',           'cod(ppm)',           10),
    'BOD (ppm)':       ('bod(ppm)',           'bod(ppm)',           3),
    'DO (ppm)':        ('do(ppm)',            'do(ppm)',            6),
    'Ammonia (ppm)':   ('ammonia(ppm)',       'ammonia(ppm)',       0.5),
    'Nitrate (ppm)':   ('nitrate(ppm)',       'nitrate(ppm)',       45),
    'Iron (ppm)':      ('iron_(ppm)',         'iron_ppm',           0.3),
    'TBC (CFU/mL)':    ('total_bacterial_count_(cfu/ml)',
                        'total_bacterial_count_cfu_ml', 100),
}

def get_col(df, col):
    """Safely get column, return None if missing."""
    for c in df.columns:
        if c == col or c.replace(' ','_') == col.replace(' ','_'):
            return df[c].dropna()
    return None

# ═══════════════════════════════════════════════════════════════════════════
# HELPER — SMOTE AUGMENT PIPELINE (for confusion matrices)
# ═══════════════════════════════════════════════════════════════════════════
def prepare_gw():
    df = gw_raw.copy()
    drop_cols = [c for c in df.columns if df[c].isna().all()]
    drop_cols += ['iron_(ppm)']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    le = LabelEncoder()
    df['wqi_encoded'] = le.fit_transform(df['wqi_class'])
    X = df.select_dtypes(include='number').drop(columns=['wqi','wqi_encoded'])
    y = df['wqi_encoded']
    return X, y, le.classes_

def prepare_sw():
    df = sw_raw.copy()
    drop_cols = [c for c in df.columns if df[c].isna().all()]
    drop_cols += ['WQI','WQI_Pred']
    df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore', inplace=True)
    le = LabelEncoder()
    df['WQI_Class_Encoded'] = le.fit_transform(df['WQI_Class'])
    num_cols = df.select_dtypes(include='number').columns.tolist()
    for c in ['WQI_Class_Encoded']:
        if c in num_cols: num_cols.remove(c)
    X = df[num_cols]
    y = df['WQI_Class_Encoded']
    return X, y, le.classes_

def augment_smote_pipeline(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=42, stratify=y)
    # Gaussian augmentation
    rng = np.random.RandomState(42)
    # Cast all columns to float to allow noise addition
    X_tr = X_tr.astype(float)
    stds = X_tr.std(skipna=True)
    copies = [X_tr.copy()]
    for _ in range(3):
        Xn = X_tr.copy()
        for col in X_tr.columns:
            s = stds[col]
            if s==0 or np.isnan(s): continue
            noise = rng.normal(0, 0.01*s, size=len(X_tr))
            mask = Xn[col].notna()
            Xn.loc[mask, col] = Xn.loc[mask, col] + noise[mask]
        copies.append(Xn)
    X_aug = pd.concat(copies, ignore_index=True)
    y_aug = pd.concat([y_tr]*4, ignore_index=True)
    # Impute + scale
    imp = KNNImputer(n_neighbors=5)
    X_aug_imp = imp.fit_transform(X_aug)
    X_te_imp  = imp.transform(X_te)
    sc = StandardScaler()
    X_aug_sc = sc.fit_transform(X_aug_imp)
    X_te_sc  = sc.transform(X_te_imp)
    # SMOTE
    k = max(1, min(5, pd.Series(y_aug).value_counts().min()-1))
    smote = SMOTE(random_state=42, k_neighbors=k)
    X_sm, y_sm = smote.fit_resample(X_aug_sc, y_aug)
    return X_sm, y_sm, X_te_sc, y_te

print("Preparing data pipelines...")
X_gw, y_gw, gw_classes = prepare_gw()
X_sw, y_sw, sw_classes  = prepare_sw()
X_gw_sm, y_gw_sm, X_gw_te, y_gw_te = augment_smote_pipeline(X_gw, y_gw)
X_sw_sm, y_sw_sm, X_sw_te, y_sw_te = augment_smote_pipeline(X_sw, y_sw)
print("Pipelines ready.")

# ═══════════════════════════════════════════════════════════════════════════
# TRAIN CLASSIFIERS
# ═══════════════════════════════════════════════════════════════════════════
classifiers = {
    'KNN':               KNeighborsClassifier(n_neighbors=3),
    'Random Forest':     RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':               SVC(kernel='linear', C=1, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                                     max_depth=3, random_state=42),
    'Decision Tree':     DecisionTreeClassifier(random_state=42),
    'XGBoost':           XGBClassifier(eval_metric='mlogloss', verbosity=0, random_state=42),
}

def get_metrics(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return {
        'Model':     name,
        'Accuracy':  round(accuracy_score(y_te, y_pred),4),
        'Precision': round(precision_score(y_te, y_pred, average='weighted', zero_division=0),4),
        'Recall':    round(recall_score(y_te, y_pred, average='weighted', zero_division=0),4),
        'F1 Score':  round(f1_score(y_te, y_pred, average='weighted', zero_division=0),4),
        'MCC':       round(matthews_corrcoef(y_te, y_pred),4),
        'y_pred':    y_pred,
    }

print("Training classifiers...")
gw_results, sw_results = [], []
gw_models,  sw_models  = {}, {}
for name, clf in classifiers.items():
    import copy as _copy
    m_gw = _copy.deepcopy(clf)
    m_sw = _copy.deepcopy(clf)
    r_gw = get_metrics(name, m_gw, X_gw_sm, y_gw_sm, X_gw_te, y_gw_te)
    r_sw = get_metrics(name, m_sw, X_sw_sm, y_sw_sm, X_sw_te, y_sw_te)
    gw_results.append(r_gw); gw_models[name] = m_gw
    sw_results.append(r_sw); sw_models[name] = m_sw
    print(f"  {name}: GW Acc={r_gw['Accuracy']:.3f}, SW Acc={r_sw['Accuracy']:.3f}")

# KMeans as pseudo-classifier
def kmeans_metrics(X_tr, y_tr, X_te, y_te, k=6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_tr)
    # Assign label to each cluster = majority class in training
    tr_clusters = km.predict(X_tr)
    cluster_labels = {}
    for c in range(k):
        idx = np.where(tr_clusters==c)[0]
        if len(idx)==0: cluster_labels[c]=0; continue
        cluster_labels[c] = int(pd.Series(y_tr.values[idx] if hasattr(y_tr,'values')
                                          else y_tr[idx]).mode()[0])
    te_clusters = km.predict(X_te)
    y_pred = np.array([cluster_labels[c] for c in te_clusters])
    y_te_arr = np.array(y_te)
    return {
        'Model':     'K-Means',
        'Accuracy':  round(accuracy_score(y_te_arr, y_pred),4),
        'Precision': round(precision_score(y_te_arr, y_pred, average='weighted', zero_division=0),4),
        'Recall':    round(recall_score(y_te_arr, y_pred, average='weighted', zero_division=0),4),
        'F1 Score':  round(f1_score(y_te_arr, y_pred, average='weighted', zero_division=0),4),
        'MCC':       round(matthews_corrcoef(y_te_arr, y_pred),4),
        'y_pred':    y_pred,
    }

gw_km = kmeans_metrics(X_gw_sm, y_gw_sm, X_gw_te, y_gw_te, k=4)
sw_km = kmeans_metrics(X_sw_sm, y_sw_sm, X_sw_te, y_sw_te, k=4)
gw_results.append(gw_km); sw_results.append(sw_km)

gw_df = pd.DataFrame(gw_results).drop(columns=['y_pred'])
sw_df = pd.DataFrame(sw_results).drop(columns=['y_pred'])
print("All models trained.\n")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — COMPREHENSIVE FRAMEWORK OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 1: Framework Overview...")

fig = plt.figure(figsize=(18, 10), facecolor='white')
fig.suptitle('A Comprehensive Framework for Groundwater & Surface Water\nPotability Assessment — Haryana, India (2016–2018)',
             fontsize=16, fontweight='bold', y=0.97, color=PALETTE['primary'])

steps = [
    ('Data\nCollection', '14 GW sites\n12 SW sites\n2016–2018\n19 parameters', '#1A3A5C'),
    ('Pre-\nprocessing', 'KNN Imputation\nStd Scaling\nDrop leakage\ncolumns', '#2E86AB'),
    ('WQI\nComputation', 'BIS IS 10500:2012\nWeighted Arithmetic\nMethod\n8 parameters', '#0077B6'),
    ('Augmentation\n& Balancing', 'Gaussian Noise\n(1% std, 3×)\nSMOTE\n(balanced classes)', '#F4A261'),
    ('Model\nTraining', '6 Classifiers\n6 Regressors\nGridSearchCV\nStratifiedKFold', '#E84855'),
    ('Evaluation', 'Acc, Prec, Recall\nF1, MCC, NSE\nR², RMSE, MAE\n5/10-Fold CV', '#2ECC71'),
    ('Spatial Risk\nProfiling', 'Priority Score\nHotspot Detection\nHealth Risk Tier\n(Low/Mod/High)', '#8E44AD'),
]

ax_flow = fig.add_axes([0.02, 0.55, 0.96, 0.36])
ax_flow.set_xlim(0, len(steps)*2)
ax_flow.set_ylim(-0.5, 2.5)
ax_flow.axis('off')

for i, (title, detail, color) in enumerate(steps):
    x = i * 2 + 0.5
    box = FancyBboxPatch((x-0.45, 0.3), 0.9, 1.8,
                          boxstyle='round,pad=0.06', linewidth=2,
                          edgecolor=color, facecolor=color+'22')
    ax_flow.add_patch(box)
    ax_flow.text(x, 1.9, title, ha='center', va='center',
                 fontsize=10, fontweight='bold', color=color)
    ax_flow.text(x, 0.9, detail, ha='center', va='center',
                 fontsize=8.5, color='#333333', linespacing=1.5)
    if i < len(steps)-1:
        ax_flow.annotate('', xy=(x+0.6, 1.2), xytext=(x+0.45, 1.2),
                         arrowprops=dict(arrowstyle='->', color='#888888', lw=1.8))

# WQI class colour bar at bottom
ax_wqi = fig.add_axes([0.08, 0.26, 0.84, 0.18])
wqi_ranges = [('Excellent\n(≤50)', '#2ECC71'), ('Good\n(51–100)', '#F1C40F'),
              ('Poor\n(101–200)', '#E67E22'), ('Very Poor\n(201–300)', '#E74C3C'),
              ('Unsuitable\n(>300)', '#8E44AD')]
for i,(label,col) in enumerate(wqi_ranges):
    ax_wqi.barh(0, 1, left=i, color=col, edgecolor='white', linewidth=2, height=0.7)
    ax_wqi.text(i+0.5, 0, label, ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')
ax_wqi.set_xlim(0,5); ax_wqi.set_ylim(-0.5,0.5); ax_wqi.axis('off')
ax_wqi.set_title('WQI Classification Scale (BIS IS 10500:2012)',
                  fontsize=11, pad=6, color=PALETTE['primary'], fontweight='bold')

# Stats summary
ax_stats = fig.add_axes([0.08, 0.03, 0.84, 0.18])
ax_stats.axis('off')
summary_data = [
    ['Dataset', '42 GW records', '36 SW records', '14 GW sites', '12 SW sites'],
    ['Best Classifier (GW)', 'KNN', 'Acc=1.00', 'F1=1.00', 'MCC=1.00'],
    ['Best Classifier (SW)', 'Decision Tree', 'Acc=1.00', 'F1=1.00', 'MCC=1.00'],
    ['Best Regressor (GW)', 'Decision Tree', 'R²=0.9999', 'RMSE=2.45', 'NSE=0.9999'],
    ['Best Regressor (SW)', 'SVM', 'R²=0.9253', 'RMSE=4.85', 'NSE=0.9253'],
]
col_labels = ['Task', 'Model / Info', 'Metric 1', 'Metric 2', 'Metric 3']
tbl = ax_stats.table(cellText=summary_data, colLabels=col_labels,
                      loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.6)
for (r,c), cell in tbl.get_celld().items():
    if r==0:
        cell.set_facecolor(PALETTE['primary']); cell.set_text_props(color='white',fontweight='bold')
    elif c==0:
        cell.set_facecolor('#E8F4FD')
    else:
        cell.set_facecolor('#F8F9FA')
    cell.set_edgecolor('#CCC')

plt.savefig(f"{OUT}/Fig1_Framework_Overview.png", dpi=180, bbox_inches='tight')
plt.close()
print("  Saved Fig1_Framework_Overview.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — HISTOGRAMS WITH KDE (GW & SW key parameters)
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 2: Histograms with KDE...")

plot_params = [
    ('pH',             'ph',             'ph',             (6.5,8.5)),
    ('Turbidity (NTU)','turbidity',      'turbidity',      5),
    ('Conductivity',   'conductivity',   'conductivity',   2000),
    ('Chloride (ppm)', 'chloride_(ppm)', 'chloride_ppm',   250),
    ('Sulphates (ppm)','sulphates(ppm)', 'sulphates(ppm)', 200),
    ('COD (ppm)',      'cod(ppm)',        'cod(ppm)',        10),
    ('BOD (ppm)',      'bod(ppm)',        'bod(ppm)',        3),
    ('DO (ppm)',       'do(ppm)',         'do(ppm)',         6),
    ('Nitrate (ppm)',  'nitrate(ppm)',    'nitrate(ppm)',    45),
    ('Ammonia (ppm)',  'ammonia(ppm)',    'ammonia(ppm)',    0.5),
    ('Iron (ppm)',     'iron_(ppm)',      'iron_ppm',        0.3),
    ('TBC (CFU/mL)',   'total_bacterial_count_(cfu/ml)',
                       'total_bacterial_count_cfu_ml', 100),
]

fig, axes = plt.subplots(4, 3, figsize=(18, 18))
fig.suptitle('Water Quality Parameter Distributions — Groundwater vs Surface Water\n'
             'Histograms with KDE, Mean, Median and Standard Deviation',
             fontsize=14, fontweight='bold', y=0.98, color=PALETTE['primary'])

for ax, (pname, gw_col, sw_col, bis) in zip(axes.flat, plot_params):
    gw_vals = gw[gw_col].dropna().values if gw_col in gw.columns else np.array([])
    sw_vals = sw[sw_col].dropna().values if sw_col in sw.columns else np.array([])

    all_vals = np.concatenate([gw_vals, sw_vals])
    if len(all_vals) == 0:
        ax.set_visible(False); continue

    bins = min(20, max(8, len(all_vals)//3))

    if len(gw_vals) > 0:
        ax.hist(gw_vals, bins=bins, alpha=0.45, color=PALETTE['secondary'],
                edgecolor='white', density=True, label='Groundwater')
        kde_gw = stats.gaussian_kde(gw_vals)
        xg = np.linspace(gw_vals.min(), gw_vals.max(), 200)
        ax.plot(xg, kde_gw(xg), color=PALETTE['secondary'], lw=2.2)
        ax.axvline(np.mean(gw_vals), color=PALETTE['secondary'],
                   ls='--', lw=1.6, label=f'GW Mean={np.mean(gw_vals):.2f}')
        ax.axvline(np.median(gw_vals), color=PALETTE['secondary'],
                   ls=':', lw=1.6, label=f'GW Med={np.median(gw_vals):.2f}')

    if len(sw_vals) > 0:
        ax.hist(sw_vals, bins=bins, alpha=0.45, color=PALETTE['accent'],
                edgecolor='white', density=True, label='Surface Water')
        kde_sw = stats.gaussian_kde(sw_vals)
        xs = np.linspace(sw_vals.min(), sw_vals.max(), 200)
        ax.plot(xs, kde_sw(xs), color=PALETTE['accent'], lw=2.2)
        ax.axvline(np.mean(sw_vals), color=PALETTE['accent'],
                   ls='--', lw=1.6, label=f'SW Mean={np.mean(sw_vals):.2f}')
        ax.axvline(np.median(sw_vals), color=PALETTE['accent'],
                   ls=':', lw=1.6, label=f'SW Med={np.median(sw_vals):.2f}')

    # BIS limit
    if isinstance(bis, tuple):
        ax.axvspan(bis[0], bis[1], alpha=0.10, color='green', label=f'BIS Range')
    else:
        ax.axvline(bis, color='green', ls='-', lw=1.5, alpha=0.7, label=f'BIS={bis}')

    # Stats annotation
    if len(gw_vals)>0 and len(sw_vals)>0:
        txt = (f'GW: μ={np.mean(gw_vals):.1f}, σ={np.std(gw_vals):.1f}\n'
               f'SW: μ={np.mean(sw_vals):.1f}, σ={np.std(sw_vals):.1f}')
    elif len(gw_vals)>0:
        txt = f'GW: μ={np.mean(gw_vals):.1f}, σ={np.std(gw_vals):.1f}'
    else:
        txt = f'SW: μ={np.mean(sw_vals):.1f}, σ={np.std(sw_vals):.1f}'
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, fontsize=8,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CCC', alpha=0.9))

    ax.set_title(pname, fontsize=11, fontweight='bold', color=PALETTE['primary'])
    ax.set_xlabel('Value', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.8)
    ax.tick_params(labelsize=8)

plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig(f"{OUT}/Fig2_Histograms_KDE.png", dpi=180, bbox_inches='tight')
plt.close()
print("  Saved Fig2_Histograms_KDE.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — PEARSON CORRELATION HEATMAPS (GW + SW side by side)
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 3: Pearson Correlation Heatmaps...")

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.suptitle('Pearson Correlation Coefficient — Groundwater (left) & Surface Water (right)',
             fontsize=14, fontweight='bold', y=1.01, color=PALETTE['primary'])

corr_params_gw = ['ph','turbidity','conductivity','chloride_(ppm)','sulphates(ppm)',
                  'iron_(ppm)','cod(ppm)','bod(ppm)','do(ppm)','ammonia(ppm)',
                  'nitrate(ppm)','total_bacterial_count_(cfu/ml)','wqi']
corr_params_sw = ['ph','turbidity','conductivity','chloride_ppm','sulphates(ppm)',
                  'iron_ppm','cod(ppm)','bod(ppm)','do(ppm)','ammonia(ppm)',
                  'nitrate(ppm)','total_bacterial_count_cfu_ml','wqi']

nice_names = ['pH','Turbidity','Conductivity','Chloride','Sulphates','Iron',
              'COD','BOD','DO','Ammonia','Nitrate','TBC','WQI']

for ax, df_src, col_list, title in [
    (axes[0], gw, corr_params_gw, 'Groundwater'),
    (axes[1], sw, corr_params_sw, 'Surface Water'),
]:
    valid_cols = [c for c in col_list if c in df_src.columns]
    valid_names = [nice_names[col_list.index(c)] for c in valid_cols]
    corr = df_src[valid_cols].corr(method='pearson')
    corr.columns = valid_names[:len(corr.columns)]
    corr.index   = valid_names[:len(corr.index)]
    mask = np.triu(np.ones_like(corr, dtype=bool))
    hm = sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                     ax=ax, linewidths=0.5, linecolor='white',
                     annot_kws={'size':8.5},
                     cbar_kws={'shrink':0.8,'label':'Pearson r'},
                     vmin=-1, vmax=1)
    ax.set_title(f'{title} — Pearson Correlation Matrix',
                 fontsize=12, fontweight='bold', pad=10, color=PALETTE['primary'])
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    # Highlight WQI row/col
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/Fig3_Pearson_Correlation.png", dpi=180, bbox_inches='tight')
plt.close()
print("  Saved Fig3_Pearson_Correlation.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — BOX PLOTS WITH BIS THRESHOLD LINES
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 4: Box plots with BIS thresholds...")

box_params = [
    ('pH',             'ph',          'ph',         (6.5,8.5), 'pH units'),
    ('Turbidity',      'turbidity',   'turbidity',   5,         'NTU'),
    ('Chloride',       'chloride_(ppm)','chloride_ppm',250,     'ppm'),
    ('Sulphates',      'sulphates(ppm)','sulphates(ppm)',200,   'ppm'),
    ('DO',             'do(ppm)',     'do(ppm)',     6,          'ppm'),
    ('BOD',            'bod(ppm)',    'bod(ppm)',    3,          'ppm'),
    ('COD',            'cod(ppm)',    'cod(ppm)',    10,         'ppm'),
    ('Nitrate',        'nitrate(ppm)','nitrate(ppm)',45,         'ppm'),
    ('Iron',           'iron_(ppm)', 'iron_ppm',    0.3,        'ppm'),
    ('Ammonia',        'ammonia(ppm)','ammonia(ppm)',0.5,        'ppm'),
]

fig, axes = plt.subplots(2, 5, figsize=(22, 12))
fig.suptitle('Box Plots of Water Quality Parameters with BIS IS 10500:2012 Threshold Limits\n'
             'Mean (▲) and Median (●) highlighted | Red line = Permissible limit',
             fontsize=13, fontweight='bold', y=0.99, color=PALETTE['primary'])

for ax, (pname, gcol, scol, bis, unit) in zip(axes.flat, box_params):
    gv = gw[gcol].dropna().values if gcol in gw.columns else np.array([])
    sv = sw[scol].dropna().values if scol in sw.columns else np.array([])

    data_dict = {}
    if len(gv)>0: data_dict['Groundwater'] = gv
    if len(sv)>0: data_dict['Surface\nWater'] = sv
    if not data_dict: ax.set_visible(False); continue

    bp = ax.boxplot(list(data_dict.values()),
                    labels=list(data_dict.keys()),
                    patch_artist=True,
                    medianprops=dict(color=PALETTE['primary'], lw=2.5),
                    whiskerprops=dict(lw=1.5, linestyle='--', color='#555'),
                    capprops=dict(lw=2, color='#555'),
                    flierprops=dict(marker='o', markerfacecolor='#E74C3C',
                                   markersize=5, alpha=0.6, linestyle='none'))

    colors_box = [PALETTE['secondary'], PALETTE['accent']]
    for patch, col in zip(bp['boxes'], colors_box):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
        patch.set_linewidth(1.8)

    # Mean markers (triangle)
    for i, (lbl, vals) in enumerate(data_dict.items(), 1):
        mn = np.mean(vals); md = np.median(vals)
        ax.plot(i, mn, '^', color='#E74C3C', markersize=9, zorder=5,
                label='Mean' if i==1 else '')
        ax.plot(i, md, 'o', color=PALETTE['primary'], markersize=7, zorder=5,
                label='Median' if i==1 else '')

    # BIS threshold
    if isinstance(bis, tuple):
        ax.axhline(bis[0], color='green', ls='--', lw=1.5, alpha=0.8, label=f'BIS min={bis[0]}')
        ax.axhline(bis[1], color='green', ls='--', lw=1.5, alpha=0.8, label=f'BIS max={bis[1]}')
        ax.axhspan(bis[0], bis[1], alpha=0.07, color='green')
    else:
        ax.axhline(bis, color='#E74C3C', ls='-', lw=2, alpha=0.85,
                   label=f'BIS limit={bis}')

    ax.set_title(f'{pname} ({unit})', fontsize=11, fontweight='bold',
                 color=PALETTE['primary'], pad=4)
    ax.set_ylabel(unit, fontsize=9)
    ax.tick_params(labelsize=9)

    # legend only on first
    if pname == 'pH':
        handles = [
            Line2D([0],[0], marker='^', color='w', markerfacecolor='#E74C3C',
                   markersize=9, label='Mean'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor=PALETTE['primary'],
                   markersize=7, label='Median'),
            Line2D([0],[0], color='#E74C3C', lw=2, label='BIS Limit'),
            mpatches.Patch(facecolor=PALETTE['secondary'], alpha=0.55, label='Groundwater'),
            mpatches.Patch(facecolor=PALETTE['accent'],    alpha=0.55, label='Surface Water'),
        ]
        ax.legend(handles=handles, fontsize=8, loc='upper right', framealpha=0.9)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(f"{OUT}/Fig4_Boxplots_BIS_Thresholds.png", dpi=180, bbox_inches='tight')
plt.close()
print("  Saved Fig4_Boxplots_BIS_Thresholds.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — K-MEANS CLUSTERING SCATTER (k=2) — GW & SW
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 5: K-Means Clustering scatter...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Scatter Plot — K-Means Clustering (k=2) Applied to WQI Feature Space\n'
             'Dimensionality reduced via PCA (PC1 vs PC2)',
             fontsize=13, fontweight='bold', y=1.02, color=PALETTE['primary'])

cluster_colors = ['#2E86AB', '#E84855']
cluster_markers = ['o', 's']
wqi_class_colors = {'Excellent':'#2ECC71','Good':'#F1C40F','Poor':'#E67E22',
                    'Very Poor':'#E74C3C','Unsuitable':'#8E44AD',
                    'Medium':'#3498DB'}  # SW uses Medium

for ax, X_raw, y_labels, class_names_arr, title, label_col in [
    (axes[0], X_gw, y_gw,  gw_classes, 'Groundwater', 'wqi_class'),
    (axes[1], X_sw, y_sw,  sw_classes,  'Surface Water','WQI_Class'),
]:
    # Impute & scale
    imp_raw = KNNImputer(n_neighbors=5)
    X_imp   = imp_raw.fit_transform(X_raw)
    sc_raw  = StandardScaler()
    X_sc    = sc_raw.fit_transform(X_imp)

    # PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    # KMeans k=2
    km2 = KMeans(n_clusters=2, random_state=42, n_init=15)
    clusters = km2.fit_predict(X_sc)

    centroids_pca = pca.transform(km2.cluster_centers_)

    # Plot cluster regions (convex hull approximation via scatter)
    for ci, (col, mrk) in enumerate(zip(cluster_colors, cluster_markers)):
        mask = clusters == ci
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=col, marker=mrk, s=70, alpha=0.65,
                   edgecolors='white', linewidths=0.8,
                   label=f'Cluster {ci+1} (n={mask.sum()})', zorder=3)

    # Centroids
    for ci in range(2):
        ax.scatter(centroids_pca[ci,0], centroids_pca[ci,1],
                   c=cluster_colors[ci], marker='*', s=320, zorder=5,
                   edgecolors='black', linewidths=1.2,
                   label=f'Centroid {ci+1}')

    # Overlay true WQI class as text labels (small)
    y_arr = np.array(y_labels)
    for i, (xp, yp, lbl) in enumerate(zip(X_pca[:,0], X_pca[:,1], y_arr)):
        cls_name = class_names_arr[lbl] if lbl < len(class_names_arr) else str(lbl)
        short = cls_name[:3]
        ax.text(xp, yp+0.08, short, fontsize=6.5, ha='center',
                color='#333', alpha=0.85)

    # Explained variance
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}% variance)', fontsize=10)
    ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}% variance)', fontsize=10)
    ax.set_title(f'{title} — K-Means (k=2)', fontsize=12,
                 fontweight='bold', color=PALETTE['primary'], pad=8)

    # Inertia annotation
    ax.text(0.02, 0.98, f'Inertia: {km2.inertia_:.1f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CCC', alpha=0.9))

    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig(f"{OUT}/Fig5_KMeans_Clustering.png", dpi=180, bbox_inches='tight')
plt.close()
print("  Saved Fig5_KMeans_Clustering.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES 6–10 — CLASSIFIER METRICS COMPARISON (one per metric)
# ═══════════════════════════════════════════════════════════════════════════
print("Figs 6–10: Classifier metric comparisons...")

metrics_list = [
    ('Accuracy',  'Fig6_Accuracy_Comparison.png',  [0,1],  True),
    ('Precision', 'Fig7_Precision_Comparison.png', [0,1],  True),
    ('Recall',    'Fig8_Recall_Comparison.png',    [0,1],  True),
    ('F1 Score',  'Fig9_F1_Comparison.png',        [0,1],  True),
    ('MCC',       'Fig10_MCC_Comparison.png',      [-0.3,1.05], False),
]

model_order = ['KNN','Random Forest','SVM','Gradient Boosting',
               'Decision Tree','XGBoost','K-Means']

for metric, fname, ylims, add_perfect in metrics_list:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)
    fig.suptitle(f'{metric} — All Classifiers + K-Means\nGroundwater (left) vs Surface Water (right)',
                 fontsize=14, fontweight='bold', y=1.01, color=PALETTE['primary'])

    for ax, df_res, water in [(axes[0], gw_df,'Groundwater'),
                               (axes[1], sw_df,'Surface Water')]:
        df_plot = df_res.set_index('Model').reindex(
            [m for m in model_order if m in df_res['Model'].values]
        ).reset_index()

        bars_x = np.arange(len(df_plot))
        bar_cols = [MODEL_COLORS.get(m,'#888888') for m in df_plot['Model']]
        vals = df_plot[metric].values

        bars = ax.bar(bars_x, vals, color=bar_cols, edgecolor='white',
                      linewidth=1.5, alpha=0.88, width=0.65, zorder=3)

        # Value labels on bars
        for bar, val in zip(bars, vals):
            ypos = bar.get_height() + 0.01
            if metric=='MCC' and val < 0: ypos = val - 0.04
            ax.text(bar.get_x()+bar.get_width()/2, ypos,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=9.5, fontweight='bold', color='#333')

        # Highlight best
        best_idx = int(np.argmax(vals))
        bars[best_idx].set_edgecolor(PALETTE['primary'])
        bars[best_idx].set_linewidth(3)

        # K-Means bar — hatched to distinguish
        km_idx = list(df_plot['Model']).index('K-Means') if 'K-Means' in list(df_plot['Model']) else None
        if km_idx is not None:
            bars[km_idx].set_hatch('///')
            bars[km_idx].set_alpha(0.75)

        if add_perfect:
            ax.axhline(1.0, color='green', ls='--', lw=1.5, alpha=0.7, label='Perfect=1.0')
        if metric == 'MCC':
            ax.axhline(0, color='grey', ls='-', lw=1, alpha=0.5)

        ax.set_xticks(bars_x)
        ax.set_xticklabels(df_plot['Model'], rotation=30, ha='right', fontsize=10)
        ax.set_ylim(ylims)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{water}', fontsize=12, fontweight='bold',
                     color=PALETTE['secondary'], pad=6)
        ax.tick_params(labelsize=9)

        # Legend patches
        patches = [mpatches.Patch(color=MODEL_COLORS.get(m,'#888'), label=m,
                                  alpha=0.88)
                   for m in df_plot['Model']]
        km_patch = mpatches.Patch(facecolor=MODEL_COLORS['K-Means'],
                                  hatch='///', alpha=0.75, label='K-Means (clustering)')
        patches[-1] = km_patch
        ax.legend(handles=patches, fontsize=8.5, loc='lower right',
                  framealpha=0.9, title='Models')

        # Best label
        best_name = df_plot.loc[best_idx,'Model']
        ax.text(0.02, 0.96, f'Best: {best_name} ({vals[best_idx]:.3f})',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='#E8F4FD', ec='#2E86AB'))

    plt.tight_layout()
    plt.savefig(f"{OUT}/{fname}", dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 11 — MODEL COMPLEXITY RADAR CHART
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 11: Model complexity radar chart...")

radar_metrics = ['GW Acc','GW F1','GW MCC','SW Acc','SW F1','SW MCC',
                 'Interpretability','CV Stability']

interp_scores = {
    'KNN':               0.55, 'Random Forest':     0.60,
    'SVM':               0.45, 'Gradient Boosting': 0.55,
    'Decision Tree':     0.95, 'XGBoost':           0.50,
    'K-Means':           0.70,
}
cv_stability = {   # normalised from 5-fold CV std (lower std = higher stability)
    'KNN':               0.87, 'Random Forest':     0.98,
    'SVM':               0.91, 'Gradient Boosting': 0.98,
    'Decision Tree':     0.92, 'XGBoost':           0.87,
    'K-Means':           0.75,
}

def normalize_mcc(v):
    return max(0, (v + 0.21) / 1.21)  # shift so -0.21 → 0, 1.0 → 1.0

def get_radar_values(model):
    gw_row = gw_df[gw_df['Model']==model]
    sw_row = sw_df[sw_df['Model']==model]
    if gw_row.empty or sw_row.empty: return None
    return [
        float(gw_row['Accuracy'].iloc[0]),
        float(gw_row['F1 Score'].iloc[0]),
        normalize_mcc(float(gw_row['MCC'].iloc[0])),
        float(sw_row['Accuracy'].iloc[0]),
        float(sw_row['F1 Score'].iloc[0]),
        normalize_mcc(float(sw_row['MCC'].iloc[0])),
        interp_scores.get(model, 0.5),
        cv_stability.get(model, 0.8),
    ]

N = len(radar_metrics)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

model_plot_list = [m for m in model_order if m in list(gw_df['Model'])]
nmod = len(model_plot_list)

# ── Use GridSpec so we fully control layout without subplot numbering issues ─
# Row 0: one wide overlay radar (spans all 4 columns)
# Rows 1+: individual radars, 4 per row
COLS = 4
ind_rows = (nmod + COLS - 1) // COLS
fig = plt.figure(figsize=(18, 5 + ind_rows * 4.5))
fig.suptitle('Model Complexity & Performance Radar Chart\n'
             'GW Accuracy · GW F1 · GW MCC · SW Accuracy · SW F1 · SW MCC · '
             'Interpretability · CV Stability',
             fontsize=13, fontweight='bold', y=0.99, color=PALETTE['primary'])

gs = gridspec.GridSpec(1 + ind_rows, COLS,
                       figure=fig,
                       hspace=0.55, wspace=0.45)

# ── Overlay — spans entire first row ─────────────────────────────────────────
ax_main = fig.add_subplot(gs[0, :], polar=True)
ax_main.set_facecolor('#F0F4F8')

for model in model_plot_list:
    vals = get_radar_values(model)
    if not vals:
        continue
    vals_c = vals + vals[:1]
    col = MODEL_COLORS.get(model, '#888')
    ax_main.plot(angles, vals_c, 'o-', lw=2.2, color=col, label=model, alpha=0.88)
    ax_main.fill(angles, vals_c, alpha=0.07, color=col)

ax_main.set_xticks(angles[:-1])
ax_main.set_xticklabels(radar_metrics, size=9.5, fontweight='bold')
ax_main.set_ylim(0, 1.05)
ax_main.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_main.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], size=7, color='grey')
ax_main.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.45)
ax_main.set_title('All Models — Overlay', fontsize=12, fontweight='bold',
                  color=PALETTE['primary'], pad=22)
ax_main.legend(loc='upper right', bbox_to_anchor=(1.52, 1.18),
               fontsize=9.5, framealpha=0.92, title='Models', title_fontsize=9)

# ── Individual panels ─────────────────────────────────────────────────────────
for idx, model in enumerate(model_plot_list):
    vals = get_radar_values(model)
    if not vals:
        continue
    row_i = 1 + idx // COLS
    col_i = idx % COLS
    ax = fig.add_subplot(gs[row_i, col_i], polar=True)
    ax.set_facecolor('#F0F4F8')
    vals_c = vals + vals[:1]
    col = MODEL_COLORS.get(model, '#888')
    ax.plot(angles, vals_c, 'o-', lw=2, color=col, alpha=0.92)
    ax.fill(angles, vals_c, alpha=0.22, color=col)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics, size=7)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(['0.5', '1.0'], size=6.5, color='grey')
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.4)
    ax.set_title(model, fontsize=10, fontweight='bold', color=col, pad=15)

plt.savefig(f"{OUT}/Fig11_Radar_Model_Complexity.png", dpi=180, bbox_inches='tight')
plt.close()
print("  Saved Fig11_Radar_Model_Complexity.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 12 — CONFUSION MATRICES (all classifiers, GW + SW)
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 12: Confusion matrices...")

def plot_confusion_grid(models_dict, X_tr, y_tr, X_te, y_te,
                        class_names_arr, title_prefix, savepath):
    # include K-Means even though it's not a pre-trained sklearn model in models_dict
    model_names = [m for m in model_order if m in models_dict or m == 'K-Means']
    n = len(model_names)
    cols_cm = 4; rows_cm = (n + cols_cm - 1) // cols_cm

    fig, axes = plt.subplots(rows_cm, cols_cm,
                              figsize=(cols_cm*4.5, rows_cm*4.2))
    fig.suptitle(f'Confusion Matrices — {title_prefix}\n'
                 f'WQI Classification (Test Set, Augmented + SMOTE Pipeline)',
                 fontsize=13, fontweight='bold', y=1.01, color=PALETTE['primary'])
    axes_flat = axes.flat if rows_cm > 1 else [axes] if cols_cm == 1 else axes.flat

    present_labels = sorted(np.unique(y_te))
    present_names  = [class_names_arr[i] for i in present_labels
                      if i < len(class_names_arr)]

    cmaps_per = {
        'KNN':'Blues','Random Forest':'Greens','SVM':'Reds',
        'Gradient Boosting':'Oranges','Decision Tree':'Purples',
        'XGBoost':'YlOrBr','K-Means':'PuBuGn',
    }

    for ax, mname in zip(axes_flat, model_names):
        if mname == 'K-Means':
            km_c = KMeans(n_clusters=len(set(y_tr.values if hasattr(y_tr,'values') else y_tr)),
                          random_state=42, n_init=10)
            km_c.fit(X_tr)
            tr_cl = km_c.predict(X_tr)
            y_arr = y_tr.values if hasattr(y_tr,'values') else np.array(y_tr)
            cluster_lbl = {}
            for ci in np.unique(tr_cl):
                idx = np.where(tr_cl==ci)[0]
                cluster_lbl[ci] = int(pd.Series(y_arr[idx]).mode()[0])
            te_cl = km_c.predict(X_te)
            y_pred = np.array([cluster_lbl[c] for c in te_cl])
        else:
            m = models_dict[mname]
            y_pred = m.predict(X_te)

        y_te_arr = np.array(y_te)
        cm = confusion_matrix(y_te_arr, y_pred, labels=present_labels)
        acc = accuracy_score(y_te_arr, y_pred)

        # Normalise for colour only
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)

        cmap = cmaps_per.get(mname, 'Blues')
        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)

        for ri in range(len(present_labels)):
            for ci in range(len(present_labels)):
                txt_col = 'white' if cm_norm[ri,ci] > 0.55 else 'black'
                ax.text(ci, ri, f'{cm[ri,ci]}\n({cm_norm[ri,ci]*100:.0f}%)',
                        ha='center', va='center', fontsize=8.5,
                        fontweight='bold', color=txt_col)

        col = MODEL_COLORS.get(mname,'#888')
        ax.set_title(f'{mname}\nAcc={acc:.3f}',
                     fontsize=10, fontweight='bold', color=col, pad=6)
        ax.set_xticks(range(len(present_labels)))
        ax.set_yticks(range(len(present_labels)))
        ax.set_xticklabels(present_names, rotation=30, ha='right', fontsize=8.5)
        ax.set_yticklabels(present_names, fontsize=8.5)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('Actual', fontsize=9)

    # Hide unused axes
    for ax in list(axes_flat)[len(model_names):]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(savepath, dpi=180, bbox_inches='tight')
    plt.close()

plot_confusion_grid(gw_models, X_gw_sm, y_gw_sm, X_gw_te, y_gw_te,
                    gw_classes, 'Groundwater WQI',
                    f"{OUT}/Fig12a_CM_Groundwater.png")
print("  Saved Fig12a_CM_Groundwater.png")

plot_confusion_grid(sw_models, X_sw_sm, y_sw_sm, X_sw_te, y_sw_te,
                    sw_classes, 'Surface Water WQI',
                    f"{OUT}/Fig12b_CM_SurfaceWater.png")
print("  Saved Fig12b_CM_SurfaceWater.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 13 — COMBINED METRICS DASHBOARD (all 5 metrics in one figure)
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 13: Combined metrics dashboard...")

metrics_dash = ['Accuracy','Precision','Recall','F1 Score','MCC']
fig, axes = plt.subplots(2, 5, figsize=(24, 10))
fig.suptitle('Comprehensive Classifier Performance Dashboard\n'
             'Groundwater (top row) · Surface Water (bottom row)',
             fontsize=14, fontweight='bold', y=1.01, color=PALETTE['primary'])

for row_idx, (df_res, water) in enumerate([(gw_df,'Groundwater'),(sw_df,'Surface Water')]):
    for col_idx, metric in enumerate(metrics_dash):
        ax = axes[row_idx, col_idx]
        df_plot = df_res.set_index('Model').reindex(
            [m for m in model_order if m in df_res['Model'].values]
        ).reset_index()

        vals   = df_plot[metric].values
        bcolors = [MODEL_COLORS.get(m,'#888') for m in df_plot['Model']]
        bars_x  = np.arange(len(df_plot))

        bars = ax.bar(bars_x, vals, color=bcolors, edgecolor='white',
                      lw=1.2, alpha=0.85, width=0.7, zorder=3)

        # KMeans hatch
        km_idx = list(df_plot['Model']).index('K-Means') \
                 if 'K-Means' in list(df_plot['Model']) else None
        if km_idx is not None:
            bars[km_idx].set_hatch('///')
            bars[km_idx].set_alpha(0.72)

        best_idx = int(np.argmax(vals))
        bars[best_idx].set_edgecolor('#1A3A5C'); bars[best_idx].set_linewidth(2.8)

        for bar, val in zip(bars, vals):
            yp = bar.get_height() + (0.01 if val >= 0 else -0.04)
            ax.text(bar.get_x()+bar.get_width()/2, yp, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold')

        if metric != 'MCC':
            ax.axhline(1.0, color='green', ls='--', lw=1.2, alpha=0.6)
            ax.set_ylim(0, 1.15)
        else:
            ax.axhline(0, color='grey', ls='-', lw=0.8, alpha=0.5)
            ax.set_ylim(-0.35, 1.15)

        ax.set_xticks(bars_x)
        ax.set_xticklabels([m.replace(' ','\n') for m in df_plot['Model']],
                            fontsize=7.5, rotation=0)
        ax.set_title(f'{metric}\n({water})',
                     fontsize=9.5, fontweight='bold',
                     color=PALETTE['primary'] if row_idx==0 else PALETTE['secondary'])
        ax.tick_params(labelsize=8)
        ax.set_ylabel(metric, fontsize=8.5)

# Single legend at bottom
handles_leg = [mpatches.Patch(color=MODEL_COLORS.get(m,'#888'), alpha=0.85, label=m)
               for m in model_order if m != 'K-Means']
handles_leg.append(mpatches.Patch(facecolor=MODEL_COLORS['K-Means'],
                                   hatch='///', alpha=0.72, label='K-Means'))
fig.legend(handles=handles_leg, loc='lower center', ncol=7,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig(f"{OUT}/Fig13_Combined_Metrics_Dashboard.png", dpi=180, bbox_inches='tight')
plt.close()
print("  Saved Fig13_Combined_Metrics_Dashboard.png")

print("\n" + "="*60)
print("ALL PLOTS GENERATED SUCCESSFULLY")
print(f"Output directory: {OUT}")
import glob
files = sorted(glob.glob(f"{OUT}/*.png"))
for f in files:
    size_kb = os.path.getsize(f)//1024
    print(f"  {os.path.basename(f):50s}  {size_kb:>5} KB")
print("="*60)
