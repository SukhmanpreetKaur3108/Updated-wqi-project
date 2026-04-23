"""
Interactive EDA + iPlots — Groundwater & Surface Water WQI
Generates HTML files openable in any browser (no server needed).
Run: python interactive_eda.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── sklearn pipeline ──────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, r2_score,
                              mean_squared_error, mean_absolute_error)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE

# ── plotly ────────────────────────────────────────────────────────────────────
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

BASE = r"D:\My Projects\Py-DS-ML-Bootcamp-master"
OUT  = r"D:\My Projects\Py-DS-ML-Bootcamp-master\research paper\interactive_plots"
import os; os.makedirs(OUT, exist_ok=True)

# ── Colour theme ──────────────────────────────────────────────────────────────
WQI_COLORS = {
    'Excellent': '#2ECC71', 'Good': '#F1C40F', 'Medium': '#3498DB',
    'Poor': '#E67E22', 'Very Poor': '#E74C3C', 'Unsuitable': '#8E44AD',
}
MODEL_COLORS = {
    'KNN':'#3498DB', 'Random Forest':'#2ECC71', 'SVM':'#E74C3C',
    'Gradient Boosting':'#F39C12', 'Decision Tree':'#9B59B6',
    'XGBoost':'#1ABC9C', 'K-Means':'#E84855',
}
PLOT_TEMPLATE = 'plotly_white'

# ════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
gw = pd.read_csv(f"{BASE}/groundwater_wqi.csv")
sw = pd.read_csv(f"{BASE}/surface_wqi.csv")

# normalise sw columns
sw.columns = [c.strip() for c in sw.columns]

GW_PARAMS = ['ph','turbidity','conductivity','chloride_(ppm)','sulphates(ppm)',
             'iron_(ppm)','cod(ppm)','bod(ppm)','do(ppm)',
             'ammonia(ppm)','nitrate(ppm)','total_bacterial_count_(cfu/ml)']
SW_PARAMS  = ['pH','Turbidity','Conductivity','Chloride (ppm)','Sulphates(ppm)',
              'Iron (ppm)','COD(ppm)','BOD(ppm)','DO(ppm)',
              'Ammonia(ppm)','Nitrate(ppm)','Total Bacterial Count (cfu/ml)']
NICE_NAMES = ['pH','Turbidity','Conductivity','Chloride','Sulphates','Iron',
              'COD','BOD','DO','Ammonia','Nitrate','TBC']
BIS_LIMITS = {'pH':(6.5,8.5),'Turbidity':5,'Chloride':250,'Sulphates':200,
              'Nitrate':45,'Iron':0.3,'Conductivity':2000,'BOD':3,'DO':6}

# ─── pipeline helpers ─────────────────────────────────────────────────────────
def prep_gw_cls():
    df = gw.copy().drop(columns=[c for c in gw.columns if gw[c].isna().all()],
                        errors='ignore')
    df.drop(columns=['iron_(ppm)'], errors='ignore', inplace=True)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['wqi_class'])
    X = df.select_dtypes('number').drop(columns=['wqi','label']).astype(float)
    return X, df['label'], le.classes_

def prep_sw_cls():
    df = sw.copy()
    for c in ['WQI','WQI_Pred','Odour','Lead(ppm)','Pesticide (µg/l)','Phosphates(ppm)']:
        if c in df.columns: df.drop(columns=[c], inplace=True)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['WQI_Class'])
    X = df.select_dtypes('number').drop(columns=['label']).astype(float)
    return X, df['label'], le.classes_

def prep_gw_reg():
    df = gw.copy().drop(columns=[c for c in gw.columns if gw[c].isna().all()],
                        errors='ignore')
    X = df.select_dtypes('number').drop(columns=['wqi']).astype(float)
    return X, df['wqi']

def prep_sw_reg():
    df = sw.copy()
    for c in ['WQI_Pred','Odour','Lead(ppm)','Pesticide (µg/l)','Phosphates(ppm)','WQI_Class']:
        if c in df.columns: df.drop(columns=[c], inplace=True)
    X = df.select_dtypes('number').astype(float)
    if 'WQI' in X.columns: X = X.drop(columns=['WQI'])
    return X, df['WQI']

def augment_pipeline_cls(X, y, k_smote=4):
    X = X.copy().astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                           random_state=42, stratify=y)
    rng = np.random.RandomState(42); stds = Xtr.std()
    copies = [Xtr.copy()]
    for _ in range(3):
        Xn = Xtr.copy()
        for col in Xtr.columns:
            s = stds[col]
            if s==0 or np.isnan(s): continue
            noise = rng.normal(0, 0.01*s, len(Xtr))
            mask = Xn[col].notna()
            Xn.loc[mask, col] += noise[mask]
        copies.append(Xn)
    Xa = pd.concat(copies, ignore_index=True)
    ya = pd.concat([ytr]*4, ignore_index=True)
    imp = KNNImputer(n_neighbors=5); sc = StandardScaler()
    Xa_sc = sc.fit_transform(imp.fit_transform(Xa))
    Xte_sc = sc.transform(imp.transform(Xte))
    k = max(1, min(k_smote, pd.Series(ya).value_counts().min()-1))
    sm = SMOTE(random_state=42, k_neighbors=k)
    Xsm, ysm = sm.fit_resample(Xa_sc, ya)
    return Xsm, ysm, Xte_sc, yte, sc, imp

def augment_pipeline_reg(X, y):
    X = X.copy().astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rng = np.random.RandomState(42); stds = Xtr.std()
    copies_X, copies_y = [Xtr.copy()], [ytr.copy()]
    for _ in range(3):
        Xn = Xtr.copy()
        for col in Xtr.columns:
            s = stds[col]
            if s==0 or np.isnan(s): continue
            noise = rng.normal(0, 0.01*s, len(Xtr))
            mask = Xn[col].notna()
            Xn.loc[mask, col] += noise[mask]
        yn = ytr + rng.normal(0, 0.005*ytr.std(), len(ytr))
        copies_X.append(Xn); copies_y.append(yn)
    Xa = pd.concat(copies_X, ignore_index=True)
    ya = pd.concat(copies_y, ignore_index=True)
    imp = KNNImputer(n_neighbors=5); sc = StandardScaler()
    Xa_sc = sc.fit_transform(imp.fit_transform(Xa))
    Xte_sc = sc.transform(imp.transform(Xte))
    return Xa_sc, ya, Xte_sc, yte

def save(fig, fname):
    path = f"{OUT}/{fname}"
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"  Saved {fname}")

print("Loading and preparing data...")
X_gwc, y_gwc, gwc_cls = prep_gw_cls()
X_swc, y_swc, swc_cls = prep_sw_cls()
X_gwr, y_gwr          = prep_gw_reg()
X_swr, y_swr          = prep_sw_reg()

Xgwc_sm, ygwc_sm, Xgwc_te, ygwc_te, _, _ = augment_pipeline_cls(X_gwc, y_gwc)
Xswc_sm, yswc_sm, Xswc_te, yswc_te, _, _ = augment_pipeline_cls(X_swc, y_swc)
Xgwr_tr, ygwr_tr, Xgwr_te, ygwr_te       = augment_pipeline_reg(X_gwr, y_gwr)
Xswr_tr, yswr_tr, Xswr_te, yswr_te       = augment_pipeline_reg(X_swr, y_swr)
print("Done.\n")


# ════════════════════════════════════════════════════════════════════════════
# EDA-1  WQI Distribution — GW & SW (Interactive histogram + violin)
# ════════════════════════════════════════════════════════════════════════════
print("EDA-1: WQI distributions...")
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=['Groundwater WQI Distribution',
                                    'Surface Water WQI Distribution'])

gw_wqi_class = gw['wqi_class'].fillna('Unknown')
for cls in gw['wqi_class'].dropna().unique():
    sub = gw[gw['wqi_class']==cls]['wqi']
    fig.add_trace(go.Violin(x=[cls]*len(sub), y=sub, name=cls,
                            fillcolor=WQI_COLORS.get(cls,'#888'),
                            line_color='white', opacity=0.8,
                            box_visible=True, meanline_visible=True,
                            points='all', pointpos=0,
                            marker=dict(size=5, opacity=0.6)),
                  row=1, col=1)

for cls in sw['WQI_Class'].dropna().unique():
    sub = sw[sw['WQI_Class']==cls]['WQI']
    fig.add_trace(go.Violin(x=[cls]*len(sub), y=sub, name=cls,
                            fillcolor=WQI_COLORS.get(cls,'#888'),
                            line_color='white', opacity=0.8,
                            box_visible=True, meanline_visible=True,
                            points='all', pointpos=0,
                            marker=dict(size=5, opacity=0.6),
                            showlegend=False),
                  row=1, col=2)

fig.update_layout(title='WQI Distribution by Class — Groundwater & Surface Water',
                  template=PLOT_TEMPLATE, height=520,
                  font=dict(family='Arial', size=12))
fig.update_yaxes(title_text='WQI Value', row=1, col=1)
fig.update_yaxes(title_text='WQI Value', row=1, col=2)
save(fig, 'EDA1_WQI_Distribution.html')


# ════════════════════════════════════════════════════════════════════════════
# EDA-2  Parameter Histograms with KDE — interactive (GW vs SW)
# ════════════════════════════════════════════════════════════════════════════
print("EDA-2: Parameter histograms...")
fig = make_subplots(rows=4, cols=3,
                    subplot_titles=NICE_NAMES,
                    vertical_spacing=0.09, horizontal_spacing=0.07)

for i, (gcol, scol, name) in enumerate(zip(GW_PARAMS, SW_PARAMS, NICE_NAMES)):
    r, c = divmod(i, 3)
    gv = gw[gcol].dropna() if gcol in gw.columns else pd.Series([], dtype=float)
    sv = sw[scol].dropna() if scol in sw.columns else pd.Series([], dtype=float)

    if len(gv):
        fig.add_trace(go.Histogram(x=gv, name='GW '+name, opacity=0.55,
                                   marker_color='#2E86AB', nbinsx=15,
                                   histnorm='probability density',
                                   legendgroup='GW', showlegend=(i==0)),
                      row=r+1, col=c+1)
    if len(sv):
        fig.add_trace(go.Histogram(x=sv, name='SW '+name, opacity=0.55,
                                   marker_color='#E84855', nbinsx=15,
                                   histnorm='probability density',
                                   legendgroup='SW', showlegend=(i==0)),
                      row=r+1, col=c+1)

    # BIS limit vertical line
    bis = BIS_LIMITS.get(name)
    if bis:
        lim = bis[1] if isinstance(bis, tuple) else bis
        fig.add_vline(x=lim, line_dash='dash', line_color='green',
                      opacity=0.8, row=r+1, col=c+1)

fig.update_layout(title='Parameter Distributions — Groundwater (blue) vs Surface Water (red)<br>'
                        '<sup>Dashed green = BIS permissible limit</sup>',
                  template=PLOT_TEMPLATE, height=1100, barmode='overlay',
                  font=dict(family='Arial', size=11))
save(fig, 'EDA2_Parameter_Histograms.html')


# ════════════════════════════════════════════════════════════════════════════
# EDA-3  Interactive Correlation Heatmap — GW & SW
# ════════════════════════════════════════════════════════════════════════════
print("EDA-3: Correlation heatmaps...")

def corr_heatmap(df, cols, names, title):
    valid = [c for c in cols if c in df.columns]
    labels = [names[cols.index(c)] for c in valid]
    corr = df[valid].corr(method='pearson').round(3)
    corr.columns = labels; corr.index = labels
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate='%{text}', textfont_size=9,
        hoverongaps=False,
        colorbar=dict(title='Pearson r', thickness=15)))
    fig.update_layout(title=title, template=PLOT_TEMPLATE,
                      height=620, width=700,
                      font=dict(family='Arial', size=11),
                      xaxis_tickangle=-40)
    return fig

fig_gw_corr = corr_heatmap(gw, GW_PARAMS+['wqi'], NICE_NAMES+['WQI'],
                            'Pearson Correlation — Groundwater Parameters')
fig_sw_corr = corr_heatmap(sw, SW_PARAMS+['WQI'], NICE_NAMES+['WQI'],
                            'Pearson Correlation — Surface Water Parameters')
save(fig_gw_corr, 'EDA3a_Correlation_GW.html')
save(fig_sw_corr, 'EDA3b_Correlation_SW.html')


# ════════════════════════════════════════════════════════════════════════════
# EDA-4  Interactive Box Plots with BIS limits
# ════════════════════════════════════════════════════════════════════════════
print("EDA-4: Box plots...")

def box_with_bis(df_gw, df_sw, gcols, scols, names, title):
    fig = make_subplots(rows=2, cols=6, subplot_titles=names,
                        vertical_spacing=0.18, horizontal_spacing=0.05)
    for i, (gn, sn, nm) in enumerate(zip(gcols, scols, names)):
        r, c = divmod(i, 6)
        gv = df_gw[gn].dropna() if gn in df_gw.columns else pd.Series([], dtype=float)
        sv = df_sw[sn].dropna() if sn in df_sw.columns else pd.Series([], dtype=float)
        show_leg = (i == 0)
        if len(gv):
            fig.add_trace(go.Box(y=gv, name='Groundwater', marker_color='#2E86AB',
                                 boxmean='sd', legendgroup='GW',
                                 showlegend=show_leg), row=r+1, col=c+1)
        if len(sv):
            fig.add_trace(go.Box(y=sv, name='Surface Water', marker_color='#E84855',
                                 boxmean='sd', legendgroup='SW',
                                 showlegend=show_leg), row=r+1, col=c+1)
        bis = BIS_LIMITS.get(nm)
        if bis:
            lim = bis[1] if isinstance(bis, tuple) else bis
            fig.add_hline(y=lim, line_dash='dot', line_color='green',
                          opacity=0.9, row=r+1, col=c+1)
    fig.update_layout(title=title+'<br><sup>Dotted green = BIS permissible limit | '
                      'Box shows IQR | ✕ = mean | whiskers = ±1 std</sup>',
                      template=PLOT_TEMPLATE, height=700, boxmode='group',
                      font=dict(family='Arial', size=11))
    return fig

fig_box = box_with_bis(gw, sw, GW_PARAMS[:12], SW_PARAMS[:12], NICE_NAMES[:12],
                       'Water Quality Parameters — Box Plots GW vs SW')
save(fig_box, 'EDA4_BoxPlots_BIS.html')


# ════════════════════════════════════════════════════════════════════════════
# EDA-5  Temporal Trend — WQI by Year (GW & SW)
# ════════════════════════════════════════════════════════════════════════════
print("EDA-5: Temporal trends...")

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=['Groundwater WQI Trend (2016–2018)',
                                    'Surface Water WQI Trend (2016–2018)'])
for col_idx, (df, ycol, wcol, ttl) in enumerate([
        (gw, 'year', 'wqi', 'GW'),
        (sw, 'Year', 'WQI', 'SW')]):
    if ycol not in df.columns: continue
    grp = df.groupby(ycol)[wcol].agg(['mean','median','std']).reset_index()
    fig.add_trace(go.Scatter(x=grp[ycol], y=grp['mean'], mode='lines+markers',
                              name=f'{ttl} Mean', line=dict(width=3),
                              marker=dict(size=10),
                              error_y=dict(type='data', array=grp['std'],
                                           visible=True, thickness=1.5)),
                  row=1, col=col_idx+1)
    fig.add_trace(go.Scatter(x=grp[ycol], y=grp['median'], mode='lines+markers',
                              name=f'{ttl} Median', line=dict(dash='dash', width=2),
                              marker=dict(size=8, symbol='diamond')),
                  row=1, col=col_idx+1)

fig.update_layout(title='Temporal WQI Trend — Mean ± Std & Median (2016–2018)',
                  template=PLOT_TEMPLATE, height=450,
                  font=dict(family='Arial', size=12))
fig.update_xaxes(tickvals=[2016,2017,2018])
fig.update_yaxes(title_text='WQI', row=1, col=1)
save(fig, 'EDA5_Temporal_Trend.html')


# ════════════════════════════════════════════════════════════════════════════
# EDA-6  PCA + K-Means Scatter (k=2, interactive, GW & SW)
# ════════════════════════════════════════════════════════════════════════════
print("EDA-6: K-Means PCA scatter...")

def kmeans_pca_plot(X_raw, y_raw, class_names, title):
    X_f = X_raw.astype(float)
    imp = KNNImputer(n_neighbors=5)
    Xsc = StandardScaler().fit_transform(imp.fit_transform(X_f))
    pca = PCA(n_components=3, random_state=42)
    Xp  = pca.fit_transform(Xsc)
    km  = KMeans(n_clusters=2, random_state=42, n_init=15)
    cl  = km.fit_predict(Xsc)
    ev  = pca.explained_variance_ratio_

    # Hover text
    y_arr = np.array(y_raw)
    labels = [class_names[i] if i < len(class_names) else str(i) for i in y_arr]
    df_plot = pd.DataFrame({'PC1':Xp[:,0], 'PC2':Xp[:,1], 'PC3':Xp[:,2],
                             'Cluster': [f'Cluster {c+1}' for c in cl],
                             'WQI Class': labels})

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f'PC1 vs PC2  ({ev[0]*100:.1f}% + {ev[1]*100:.1f}%)',
                                        f'PC2 vs PC3  ({ev[1]*100:.1f}% + {ev[2]*100:.1f}%)'],
                        specs=[[{'type':'scatter'},{'type':'scatter'}]])

    for cname, grp in df_plot.groupby('Cluster'):
        col = '#2E86AB' if '1' in cname else '#E84855'
        for r, xc, yc in [(1,'PC1','PC2'),(1,'PC2','PC3')]:
            fig.add_trace(go.Scatter(
                x=grp[xc], y=grp[yc], mode='markers',
                name=cname, marker=dict(color=col, size=9, opacity=0.75,
                                         line=dict(width=1, color='white')),
                text=grp['WQI Class'],
                hovertemplate=f'<b>{cname}</b><br>WQI Class: %{{text}}<br>'
                              f'{xc}: %{{x:.2f}}<br>{yc}: %{{y:.2f}}<extra></extra>',
                legendgroup=cname, showlegend=(r==1 and xc=='PC1')),
                row=1, col=1 if xc=='PC1' else 2)

    # Centroids projected
    cent_pca = pca.transform(km.cluster_centers_)
    for ci, cp in enumerate(cent_pca):
        fig.add_trace(go.Scatter(x=[cp[0]], y=[cp[1]], mode='markers',
                                  name=f'Centroid {ci+1}',
                                  marker=dict(symbol='star', size=18,
                                              color='#FFD700',
                                              line=dict(width=2,color='black')),
                                  hovertemplate=f'Centroid {ci+1}<extra></extra>'),
                      row=1, col=1)

    fig.update_layout(title=title, template=PLOT_TEMPLATE,
                      height=500, font=dict(family='Arial', size=12))
    fig.update_xaxes(title_text='PC1', row=1, col=1)
    fig.update_yaxes(title_text='PC2', row=1, col=1)
    fig.update_xaxes(title_text='PC2', row=1, col=2)
    fig.update_yaxes(title_text='PC3', row=1, col=2)
    return fig

fig_gw_km = kmeans_pca_plot(X_gwc, y_gwc, gwc_cls,
                             'K-Means Clustering (k=2) — Groundwater (PCA)')
fig_sw_km = kmeans_pca_plot(X_swc, y_swc, swc_cls,
                             'K-Means Clustering (k=2) — Surface Water (PCA)')
save(fig_gw_km, 'EDA6a_KMeans_GW.html')
save(fig_sw_km, 'EDA6b_KMeans_SW.html')


# ════════════════════════════════════════════════════════════════════════════
# TRAIN CLASSIFIERS (GW + SW)
# ════════════════════════════════════════════════════════════════════════════
print("\nTraining classifiers...")
clf_models = {
    'KNN':               KNeighborsClassifier(n_neighbors=3),
    'Random Forest':     RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':               SVC(kernel='linear', C=1, random_state=42, probability=True),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                                    max_depth=3, random_state=42),
    'Decision Tree':     DecisionTreeClassifier(random_state=42),
    'XGBoost':           XGBClassifier(eval_metric='mlogloss', verbosity=0, random_state=42),
}
reg_models = {
    'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                                    max_depth=3, random_state=42),
    'Decision Tree':     DecisionTreeRegressor(random_state=42),
    'XGBoost':           XGBRegressor(verbosity=0, random_state=42),
    'SVR':               SVR(kernel='rbf', C=100, gamma='scale'),
}

import copy as _copy

gw_clf, sw_clf = {}, {}
for name, m in clf_models.items():
    mg = _copy.deepcopy(m); mg.fit(Xgwc_sm, ygwc_sm); gw_clf[name] = mg
    ms = _copy.deepcopy(m); ms.fit(Xswc_sm, yswc_sm); sw_clf[name] = ms
    print(f"  Clf {name} trained.")

gw_reg, sw_reg = {}, {}
for name, m in reg_models.items():
    mg = _copy.deepcopy(m); mg.fit(Xgwr_tr, ygwr_tr); gw_reg[name] = mg
    ms = _copy.deepcopy(m); ms.fit(Xswr_tr, yswr_tr); sw_reg[name] = ms
    print(f"  Reg {name} trained.")


# ════════════════════════════════════════════════════════════════════════════
# CLS-1  Interactive Confusion Matrices — GW & SW (all models in one figure)
# ════════════════════════════════════════════════════════════════════════════
print("\nCLS-1: Interactive confusion matrices...")

def interactive_cm(clf_dict, X_te, y_te, class_names, title):
    model_names = list(clf_dict.keys())
    n = len(model_names)
    cols = 3; rows = (n + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=model_names,
                        horizontal_spacing=0.08,
                        vertical_spacing=0.14)
    y_te_arr = np.array(y_te)
    present  = sorted(np.unique(y_te_arr))
    names_p  = [class_names[i] for i in present if i < len(class_names)]

    for idx, (mname, model) in enumerate(clf_dict.items()):
        r, c = divmod(idx, cols)
        y_pred = model.predict(X_te)
        cm = confusion_matrix(y_te_arr, y_pred, labels=present)
        acc = (y_te_arr == y_pred).mean()

        cm_norm  = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        text_arr = [[f'{cm[i,j]}<br>({cm_norm[i,j]*100:.0f}%)'
                     for j in range(len(present))] for i in range(len(present))]

        col_sc = MODEL_COLORS.get(mname,'#3498DB')
        # Custom colorscale per model
        cs = [[0,'#FFFFFF'],[1, col_sc]]
        fig.add_trace(go.Heatmap(z=cm_norm,
                                  x=names_p, y=names_p,
                                  text=text_arr, texttemplate='%{text}',
                                  textfont_size=11,
                                  colorscale=cs, showscale=False,
                                  zmin=0, zmax=1,
                                  hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>'
                                                'Count: %{text}<extra></extra>'),
                      row=r+1, col=c+1)
        # Update subplot title with acc
        fig.layout.annotations[idx].text = f'<b>{mname}</b><br>Acc={acc:.3f}'
        fig.layout.annotations[idx].font.color = col_sc

    fig.update_layout(title=title, template=PLOT_TEMPLATE,
                      height=350*rows, font=dict(family='Arial', size=11))
    fig.update_xaxes(title_text='Predicted')
    fig.update_yaxes(title_text='Actual')
    return fig

fig_gw_cm = interactive_cm(gw_clf, Xgwc_te, ygwc_te, gwc_cls,
                            'Confusion Matrices — Groundwater WQI Classification')
fig_sw_cm = interactive_cm(sw_clf, Xswc_te, yswc_te, swc_cls,
                            'Confusion Matrices — Surface Water WQI Classification')
save(fig_gw_cm, 'CLS1a_ConfusionMatrix_GW.html')
save(fig_sw_cm, 'CLS1b_ConfusionMatrix_SW.html')


# ════════════════════════════════════════════════════════════════════════════
# CLS-2  Classifier Metrics Comparison — Interactive grouped bar (GW & SW)
# ════════════════════════════════════════════════════════════════════════════
print("CLS-2: Classifier metrics bars...")

from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

def clf_metrics_df(clf_dict, X_te, y_te):
    rows = []
    y_te_arr = np.array(y_te)
    for name, m in clf_dict.items():
        yp = m.predict(X_te)
        rows.append({'Model': name,
                     'Accuracy':  round((y_te_arr==yp).mean(), 4),
                     'Precision': round(precision_score(y_te_arr,yp,average='weighted',zero_division=0),4),
                     'Recall':    round(recall_score(y_te_arr,yp,average='weighted',zero_division=0),4),
                     'F1 Score':  round(f1_score(y_te_arr,yp,average='weighted',zero_division=0),4),
                     'MCC':       round(matthews_corrcoef(y_te_arr,yp),4)})
    return pd.DataFrame(rows)

# Add K-Means as pseudo-classifier
def kmeans_clf(X_tr, y_tr, X_te, y_te, k=4):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_tr)
    cl = km.predict(X_tr)
    y_arr = y_tr if isinstance(y_tr, np.ndarray) else np.array(y_tr)
    lbl_map = {}
    for ci in np.unique(cl):
        idx = np.where(cl==ci)[0]
        lbl_map[ci] = int(pd.Series(y_arr[idx]).mode()[0])
    y_pred = np.array([lbl_map[c] for c in km.predict(X_te)])
    y_te_arr = np.array(y_te)
    return {'Model':'K-Means',
            'Accuracy':  round((y_te_arr==y_pred).mean(),4),
            'Precision': round(precision_score(y_te_arr,y_pred,average='weighted',zero_division=0),4),
            'Recall':    round(recall_score(y_te_arr,y_pred,average='weighted',zero_division=0),4),
            'F1 Score':  round(f1_score(y_te_arr,y_pred,average='weighted',zero_division=0),4),
            'MCC':       round(matthews_corrcoef(y_te_arr,y_pred),4)}

gw_clf_df = clf_metrics_df(gw_clf, Xgwc_te, ygwc_te)
sw_clf_df = clf_metrics_df(sw_clf, Xswc_te, yswc_te)
gw_km_row = kmeans_clf(Xgwc_sm, ygwc_sm, Xgwc_te, ygwc_te)
sw_km_row = kmeans_clf(Xswc_sm, yswc_sm, Xswc_te, yswc_te)
gw_clf_df = pd.concat([gw_clf_df, pd.DataFrame([gw_km_row])], ignore_index=True)
sw_clf_df = pd.concat([sw_clf_df, pd.DataFrame([sw_km_row])], ignore_index=True)

metrics_to_plot = ['Accuracy','Precision','Recall','F1 Score','MCC']

def clf_bar_fig(df_gw, df_sw, metric):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f'Groundwater — {metric}',
                                        f'Surface Water — {metric}'])
    for col_i, (df, wtype) in enumerate([(df_gw,'GW'),(df_sw,'SW')]):
        colors = [MODEL_COLORS.get(m,'#888') for m in df['Model']]
        fig.add_trace(go.Bar(x=df['Model'], y=df[metric],
                              marker_color=colors, name=wtype,
                              text=df[metric].round(3),
                              textposition='outside',
                              textfont_size=11,
                              hovertemplate='%{x}: %{y:.4f}<extra></extra>'),
                      row=1, col=col_i+1)
        if metric != 'MCC':
            fig.add_hline(y=1.0, line_dash='dot', line_color='green',
                          opacity=0.5, row=1, col=col_i+1)

    fig.update_layout(title=f'{metric} — All Classifiers + K-Means (GW vs SW)',
                      template=PLOT_TEMPLATE, height=480, showlegend=False,
                      font=dict(family='Arial', size=12))
    if metric == 'MCC':
        fig.add_hline(y=0, line_dash='dash', line_color='grey', opacity=0.6)
    return fig

for metric in metrics_to_plot:
    fig_m = clf_bar_fig(gw_clf_df, sw_clf_df, metric)
    fname = f'CLS2_{metric.replace(" ","_")}_Comparison.html'
    save(fig_m, fname)


# ════════════════════════════════════════════════════════════════════════════
# CLS-3  Radar Chart — All classifiers (interactive)
# ════════════════════════════════════════════════════════════════════════════
print("CLS-3: Interactive radar chart...")

interp  = {'KNN':0.55,'Random Forest':0.60,'SVM':0.45,
           'Gradient Boosting':0.55,'Decision Tree':0.95,
           'XGBoost':0.50,'K-Means':0.70}
cv_stab = {'KNN':0.87,'Random Forest':0.98,'SVM':0.91,
           'Gradient Boosting':0.98,'Decision Tree':0.92,
           'XGBoost':0.87,'K-Means':0.75}
radar_cats = ['GW Acc','GW F1','GW MCC*','SW Acc','SW F1','SW MCC*',
              'Interpretability','CV Stability']

def norm_mcc(v): return max(0, (v+0.21)/1.21)

fig_radar = go.Figure()
for _, row in gw_clf_df.iterrows():
    model = row['Model']
    sw_row = sw_clf_df[sw_clf_df['Model']==model]
    if sw_row.empty: continue
    sw_row = sw_row.iloc[0]
    vals = [row['Accuracy'], row['F1 Score'], norm_mcc(row['MCC']),
            sw_row['Accuracy'], sw_row['F1 Score'], norm_mcc(sw_row['MCC']),
            interp.get(model,0.5), cv_stab.get(model,0.8)]
    vals_c = vals + [vals[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_c, theta=radar_cats+[radar_cats[0]],
        fill='toself', opacity=0.25, name=model,
        line=dict(color=MODEL_COLORS.get(model,'#888'), width=2.5),
        marker=dict(size=8),
        hovertemplate='%{theta}: %{r:.3f}<extra>'+model+'</extra>'))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0,1.05],
                               tickvals=[0.25,0.5,0.75,1.0],
                               tickfont_size=9),
               angularaxis=dict(tickfont_size=11)),
    title='Model Complexity Radar — GW & SW Performance + Interpretability + CV Stability<br>'
          '<sup>MCC* normalised to 0–1 scale</sup>',
    template=PLOT_TEMPLATE, height=580,
    legend=dict(orientation='v', x=1.1, y=0.5),
    font=dict(family='Arial', size=12))
save(fig_radar, 'CLS3_Radar_Chart.html')


# ════════════════════════════════════════════════════════════════════════════
# REG-1  Actual vs Predicted — all regressors (GW & SW, interactive)
# ════════════════════════════════════════════════════════════════════════════
print("REG-1: Actual vs Predicted...")

def actual_vs_pred(reg_dict, X_te, y_te, title):
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=list(reg_dict.keys()),
                        horizontal_spacing=0.08, vertical_spacing=0.15)
    y_te_arr = np.array(y_te)
    for idx, (name, m) in enumerate(reg_dict.items()):
        r, c = divmod(idx, 3)
        yp = m.predict(X_te)
        r2  = r2_score(y_te_arr, yp)
        rmse= np.sqrt(mean_squared_error(y_te_arr, yp))
        col = MODEL_COLORS.get(name,'#3498DB')

        fig.add_trace(go.Scatter(x=y_te_arr, y=yp, mode='markers',
                                  name=name,
                                  marker=dict(color=col, size=9, opacity=0.8,
                                              line=dict(width=1,color='white')),
                                  hovertemplate='Actual: %{x:.1f}<br>Predicted: %{y:.1f}'
                                                f'<extra>{name}</extra>'),
                      row=r+1, col=c+1)
        # Perfect line
        lims = [min(y_te_arr.min(), yp.min()), max(y_te_arr.max(), yp.max())]
        fig.add_trace(go.Scatter(x=lims, y=lims, mode='lines',
                                  line=dict(color='red', dash='dash', width=1.5),
                                  showlegend=False, name='Perfect'),
                      row=r+1, col=c+1)
        fig.layout.annotations[idx].text = (f'<b>{name}</b><br>'
                                             f'R²={r2:.4f}  RMSE={rmse:.2f}')
        fig.layout.annotations[idx].font.color = col

    fig.update_layout(title=title, template=PLOT_TEMPLATE,
                      height=650, showlegend=False,
                      font=dict(family='Arial', size=11))
    fig.update_xaxes(title_text='Actual WQI')
    fig.update_yaxes(title_text='Predicted WQI')
    return fig

fig_gwr_avp = actual_vs_pred(gw_reg, Xgwr_te, ygwr_te,
                              'Actual vs Predicted WQI — Groundwater Regressors')
fig_swr_avp = actual_vs_pred(sw_reg, Xswr_te, yswr_te,
                              'Actual vs Predicted WQI — Surface Water Regressors')
save(fig_gwr_avp, 'REG1a_ActualVsPred_GW.html')
save(fig_swr_avp, 'REG1b_ActualVsPred_SW.html')


# ════════════════════════════════════════════════════════════════════════════
# REG-2  Residual Plots — all regressors (interactive)
# ════════════════════════════════════════════════════════════════════════════
print("REG-2: Residual plots...")

def residual_plots(reg_dict, X_te, y_te, title):
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=list(reg_dict.keys()),
                        horizontal_spacing=0.08, vertical_spacing=0.15)
    y_arr = np.array(y_te)
    for idx, (name, m) in enumerate(reg_dict.items()):
        r, c = divmod(idx, 3)
        yp  = m.predict(X_te)
        res = y_arr - yp
        col = MODEL_COLORS.get(name,'#3498DB')
        fig.add_trace(go.Scatter(x=yp, y=res, mode='markers',
                                  name=name,
                                  marker=dict(color=col, size=8, opacity=0.75,
                                              line=dict(width=1,color='white')),
                                  hovertemplate='Predicted: %{x:.1f}<br>Residual: %{y:.1f}'
                                                f'<extra>{name}</extra>'),
                      row=r+1, col=c+1)
        fig.add_hline(y=0, line_dash='dash', line_color='red',
                      opacity=0.7, row=r+1, col=c+1)
        mae  = mean_absolute_error(y_arr, yp)
        fig.layout.annotations[idx].text = f'<b>{name}</b>  MAE={mae:.2f}'
        fig.layout.annotations[idx].font.color = col

    fig.update_layout(title=title, template=PLOT_TEMPLATE,
                      height=650, showlegend=False,
                      font=dict(family='Arial', size=11))
    fig.update_xaxes(title_text='Predicted WQI')
    fig.update_yaxes(title_text='Residual (Actual − Predicted)')
    return fig

fig_gwr_res = residual_plots(gw_reg, Xgwr_te, ygwr_te,
                              'Residual Plots — Groundwater Regressors')
fig_swr_res = residual_plots(sw_reg, Xswr_te, yswr_te,
                              'Residual Plots — Surface Water Regressors')
save(fig_gwr_res, 'REG2a_Residuals_GW.html')
save(fig_swr_res, 'REG2b_Residuals_SW.html')


# ════════════════════════════════════════════════════════════════════════════
# REG-3  Regression Metrics Comparison — interactive bar (GW & SW)
# ════════════════════════════════════════════════════════════════════════════
print("REG-3: Regression metrics bars...")

def reg_metrics_df(reg_dict, X_te, y_te):
    rows = []
    y_arr = np.array(y_te)
    for name, m in reg_dict.items():
        yp = m.predict(X_te)
        rows.append({'Model': name,
                     'R²':   round(r2_score(y_arr,yp),4),
                     'RMSE': round(np.sqrt(mean_squared_error(y_arr,yp)),3),
                     'MAE':  round(mean_absolute_error(y_arr,yp),3),
                     'NSE':  round(1 - np.sum((y_arr-yp)**2)/
                                      np.sum((y_arr-np.mean(y_arr))**2), 4)})
    return pd.DataFrame(rows)

gw_reg_df = reg_metrics_df(gw_reg, Xgwr_te, ygwr_te)
sw_reg_df = reg_metrics_df(sw_reg, Xswr_te, yswr_te)

fig_reg = make_subplots(rows=2, cols=2,
                         subplot_titles=['R² (higher=better)','RMSE (lower=better)',
                                         'MAE (lower=better)','NSE (higher=better)'])
for row_i, (metric, higher_better) in enumerate(
        [('R²',True),('RMSE',False),('MAE',False),('NSE',True)]):
    r, c = divmod(row_i, 2)
    for df_src, wtype, offset in [(gw_reg_df,'GW',0),(sw_reg_df,'SW',0.3)]:
        colors = [MODEL_COLORS.get(m,'#888') for m in df_src['Model']]
        x = list(range(len(df_src)))
        fig_reg.add_trace(
            go.Bar(x=[xi+offset for xi in x], y=df_src[metric],
                   name=wtype, marker_color=colors,
                   text=df_src[metric].round(3), textposition='outside',
                   textfont_size=10, width=0.28,
                   legendgroup=wtype, showlegend=(row_i==0),
                   hovertemplate='%{text}<extra>'+wtype+'</extra>'),
            row=r+1, col=c+1)
    xtick_labels = list(gw_reg_df['Model'])
    fig_reg.update_xaxes(tickvals=[i+0.15 for i in range(len(xtick_labels))],
                          ticktext=xtick_labels, tickangle=-35, row=r+1, col=c+1)

fig_reg.update_layout(title='Regression Model Performance — Groundwater vs Surface Water',
                       template=PLOT_TEMPLATE, height=750, barmode='group',
                       font=dict(family='Arial', size=11))
save(fig_reg, 'REG3_Regression_Metrics.html')


# ════════════════════════════════════════════════════════════════════════════
# REG-4  Feature Importance — Random Forest (GW & SW Regression)
# ════════════════════════════════════════════════════════════════════════════
print("REG-4: Feature importance...")

fig_fi = make_subplots(rows=1, cols=2,
                        subplot_titles=['GW Feature Importance (RF Regressor)',
                                        'SW Feature Importance (RF Regressor)'])
for col_i, (reg_dict, X_raw, wtype) in enumerate([
        (gw_reg, X_gwr, 'GW'), (sw_reg, X_swr, 'SW')]):
    if 'Random Forest' not in reg_dict: continue
    m = reg_dict['Random Forest']
    fi = pd.Series(m.feature_importances_,
                   index=X_raw.columns).sort_values(ascending=True)
    fig_fi.add_trace(go.Bar(x=fi.values, y=fi.index, orientation='h',
                             name=wtype,
                             marker=dict(color=fi.values,
                                         colorscale='Blues', showscale=False),
                             hovertemplate='%{y}: %{x:.4f}<extra></extra>'),
                     row=1, col=col_i+1)

fig_fi.update_layout(title='Feature Importance — Random Forest Regressor (GW & SW)',
                      template=PLOT_TEMPLATE, height=600, showlegend=False,
                      font=dict(family='Arial', size=11))
fig_fi.update_xaxes(title_text='Importance Score')
save(fig_fi, 'REG4_Feature_Importance.html')


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
import glob
files = sorted(glob.glob(f"{OUT}/*.html"))
print(f"\n{'='*60}")
print(f"ALL INTERACTIVE PLOTS GENERATED — {len(files)} files")
print(f"Open any .html file in your browser (double-click or drag)")
print(f"Output: {OUT}")
for f in files:
    kb = os.path.getsize(f)//1024
    print(f"  {os.path.basename(f):45s}  {kb:>4} KB")
print('='*60)
