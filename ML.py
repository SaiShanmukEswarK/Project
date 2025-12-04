#!/usr/bin/env python
# coding: utf-8

# Imports
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from pyspark.sql import SparkSession, functions as F
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# PART 1: Spark aggregation to state-year (weighted rates/means)

BASE = "file:///s/chopin/b/grad/C837217249/project"
COMPLETE_DIR = f"{BASE}/complete"

spark = SparkSession.builder.appName("Project").getOrCreate()

micro = spark.read.parquet(f"{COMPLETE_DIR}/pums_*.parquet")
print("Total micro rows (all years):", micro.count())

weight_col = 'PWGTP'
group_cols = ['STATE_NAME', 'YEAR']
wt_sum = F.sum(weight_col)

features = [
    wt_sum.alias('state_year_population'),
    F.first('TOTAL_HEALTH_SPENDING').alias('TOTAL_HEALTH_SPENDING'),

    # Demographics
    (F.sum(F.when(F.col('AGEP') >= 65, F.col(weight_col)).otherwise(0)) / wt_sum).alias('AGEP_65plus_share'),
    (F.sum(F.when(F.col('AGEP') <= 17, F.col(weight_col)).otherwise(0)) / wt_sum).alias('AGEP_0_17_share'),
    (F.sum(F.when(F.col('SEX') == 2, F.col(weight_col)).otherwise(0)) / wt_sum).alias('SEX_female_share'),
    (F.sum(F.when(F.col('MAR').isin(1, 2, 3), F.col(weight_col)).otherwise(0)) / wt_sum).alias('MAR_ever_married_share'),

    # Health insurance coverage
    (F.sum(F.when(F.col('HICOV') == 2, F.col(weight_col)).otherwise(0)) / wt_sum).alias('HICOV_uninsured_share'),
    (F.sum(F.when(F.col('HINS1') == 1, F.col(weight_col)).otherwise(0)) / wt_sum).alias('HINS1_employer_share'),
    (F.sum(F.when(F.col('HINS2') == 1, F.col(weight_col)).otherwise(0)) / wt_sum).alias('HINS2_direct_share'),
    (F.sum(F.when(F.col('HINS3') == 1, F.col(weight_col)).otherwise(0)) / wt_sum).alias('HINS3_medicare_share'),
    (F.sum(F.when(F.col('HINS4') == 1, F.col(weight_col)).otherwise(0)) / wt_sum).alias('HINS4_medicaid_share'),
    (F.sum(F.when(F.col('HINS5') == 1, F.col(weight_col)).otherwise(0)) / wt_sum).alias('HINS5_tricare_share'),
    (F.sum(F.when(F.col('HINS6') == 1, F.col(weight_col)).otherwise(0)) / wt_sum).alias('HINS6_va_share'),

    # Economic & income
    (F.sum(F.when(F.col('POVPIP') < 138, F.col(weight_col)).otherwise(0)) / wt_sum).alias('POVPIP_lt138_share'),
    (F.sum(F.col('PINCP') * F.col(weight_col)) / wt_sum).alias('PINCP_mean'),
    (F.sum(F.col('WAGP') * F.col(weight_col)) / wt_sum).alias('WAGP_mean'),
    (F.sum(F.col('SEMP') * F.col(weight_col)) / F.sum(F.when(F.col('SEMP') > 0, F.col(weight_col)).otherwise(None))).alias('SEMP_mean_if_positive'),
    (F.sum(F.col('SSP') * F.col(weight_col)) / wt_sum).alias('SSP_per_capita'),
    (F.sum(F.when(F.col('RETP') > 0, F.col(weight_col)).otherwise(0)) / wt_sum).alias('RETP_positive_share'),

    # Employment & labor
    (F.sum(F.when(F.col('ESR').isin(3, 6), F.col(weight_col)).otherwise(0)) / wt_sum).alias('ESR_unemp_or_nilf_share'),
    (F.sum(F.when(F.col('ESR').isin(1, 2, 3, 4, 5), F.col(weight_col)).otherwise(0)) / wt_sum).alias('ESR_in_labor_force_share'),

    # Education & social
    (F.sum(F.when(F.col('SCHL') >= 21, F.col(weight_col)).otherwise(0)) / wt_sum).alias('SCHL_bach_plus_share'),
    (F.sum(F.when(F.col('DOUT') == 1, F.col(weight_col)).otherwise(0)) / wt_sum).alias('DOUT_diff_share'),
    (F.sum(F.when(F.col('DPHY') == 1, F.col(weight_col)).otherwise(0)) / wt_sum).alias('DPHY_diff_share'),
    (F.sum(F.when(F.col('DIS') == 1, F.col(weight_col)).otherwise(0)) / wt_sum).alias('DIS_any_share'),
    (
        1000.0
        * F.sum(
            F.when(
                (F.col('SEX') == 2) & (F.col('AGEP').between(15, 50)) & (F.col('FER') == 1),
                F.col(weight_col),
            ).otherwise(0)
        )
        / F.sum(F.when((F.col('SEX') == 2) & (F.col('AGEP').between(15, 50)), F.col(weight_col)).otherwise(None))
    ).alias('FER_births_per_1000_women_15_50'),

    # Access & lifestyle
    (F.sum(F.when(F.col('BROADBND') == 1, F.col(weight_col)).otherwise(0)) / wt_sum).alias('BROADBND_yes_share'),
    (F.sum(F.col('VEH') * F.col(weight_col)) / wt_sum).alias('VEH_mean_per_person'),
    (F.sum(F.when(F.col('MIL').isin(2, 3), F.col(weight_col)).otherwise(0)) / wt_sum).alias('MIL_veteran_share'),
    (F.sum(F.when(F.col('MIL') == 1, F.col(weight_col)).otherwise(0)) / wt_sum).alias('MIL_active_duty_share'),
]

print("Aggregating microdata to state-year panel ...")
df_state_year = (
    micro
    .groupBy(*group_cols)
    .agg(*features)
    .toPandas()
)
print('Final panel shape:', df_state_year.shape)
print(df_state_year.head())

# PART 2: Pooled RF model (per-capita target, shuffled split 70/20/10)

leakage_counts = df_state_year.groupby(['STATE_NAME', 'YEAR'])['TOTAL_HEALTH_SPENDING'].nunique()
print('Target nunique per state-year:')
print(leakage_counts.value_counts().sort_index())
print('Summary:')
print(leakage_counts.describe())

target = 'TOTAL_HEALTH_SPENDING'
categorical_cols = []
drop_from_X = {
    target, 'STATE_NAME', 'ST', 'state_year_population',
    'SSP_per_capita', 'RETP_positive_share',
    'DOUT_diff_share', 'DPHY_diff_share',
}
numeric_cols = [c for c in df_state_year.columns if c not in drop_from_X]

X_full = df_state_year[categorical_cols + numeric_cols]
y_full = df_state_year[target]

X_tmp, X_test, y_tmp, y_test = train_test_split(
    X_full, y_full, test_size=0.10, random_state=42, shuffle=True
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.20, random_state=42, shuffle=True
)
print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

preprocess = ColumnTransformer([
    ('num', 'passthrough', numeric_cols),
])

def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def eval_split(name, y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} -> R^2: {r2:,.3f} | MAE: {mae:,.2f} | RMSE: {rmse:,.2f}")
    return r2, mae, rmse

from itertools import product
param_grid = {
    'n_estimators': [300, 500],
    'max_depth': [10, 15],
    'min_samples_leaf': [1, 2],
}
results = []
best_rmse = None
best_pipe = None
best_params = None
for n_est, depth, leaf in product(param_grid['n_estimators'], param_grid['max_depth'], param_grid['min_samples_leaf']):
    rf = RandomForestRegressor(
        n_estimators=n_est,
        max_depth=depth,
        min_samples_leaf=leaf,
        n_jobs=-1,
        random_state=42,
    )
    pipe_candidate = Pipeline([('pre', preprocess), ('rf', rf)])
    pipe_candidate.fit(X_train, y_train)
    val_pred = pipe_candidate.predict(X_val)
    r2, mae, rmse = eval_split(f'Val (n={n_est}, depth={depth}, leaf={leaf})', y_val, val_pred)
    results.append((rmse, n_est, depth, leaf, r2, mae))
    if best_rmse is None or rmse < best_rmse:
        best_rmse = rmse
        best_pipe = pipe_candidate
        best_params = {'n_estimators': n_est, 'max_depth': depth, 'min_samples_leaf': leaf}

print('Best params (val RMSE):', best_params)
print(f"Best val RMSE: {best_rmse:,.2f}")

X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
best_pipe.fit(X_trainval, y_trainval)
y_pred = best_pipe.predict(X_test)
test_r2, test_mae, test_rmse = eval_split('Test (best model)', y_test, y_pred)

pipe = best_pipe

# 5-fold cross-validation on pooled RF pipeline (full data)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'r2': 'r2',
    'mae': 'neg_mean_absolute_error',
    'rmse': 'neg_root_mean_squared_error',
}
cv_res = cross_validate(
    pipe, X_full, y_full, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False
)
mean_r2 = cv_res['test_r2'].mean()
std_r2 = cv_res['test_r2'].std()
mean_mae = (-cv_res['test_mae']).mean()
mean_rmse = (-cv_res['test_rmse']).mean()
print(f"5-fold CV -> R^2: {mean_r2:.3f} +/- {std_r2:.3f} | MAE: {mean_mae:,.2f} | RMSE: {mean_rmse:,.2f}")

# Residual analysis (val/test) + save plots
if 'y_pred' not in globals() or 'y_test' not in globals() or 'val_pred' not in globals():
    print('Need y_pred/y_test and val_pred/y_val from the tuned model; run the RF cell first.')
else:
    val_resid = y_val - val_pred
    test_resid = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(val_resid, bins=20, alpha=0.7, color='steelblue')
    axes[0].set_title('Val residuals')
    axes[0].set_xlabel('Residual')
    axes[1].hist(test_resid, bins=20, alpha=0.7, color='indianred')
    axes[1].set_title('Test residuals')
    axes[1].set_xlabel('Residual')
    plt.tight_layout()
    plt.savefig("residual_histograms.png")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(val_pred, val_resid, alpha=0.6, color='steelblue')
    axes[0].axhline(0, color='k', linestyle='--')
    axes[0].set_title('Val: residual vs. fitted')
    axes[0].set_xlabel('Fitted')
    axes[0].set_ylabel('Residual')

    axes[1].scatter(y_pred, test_resid, alpha=0.6, color='indianred')
    axes[1].axhline(0, color='k', linestyle='--')
    axes[1].set_title('Test: residual vs. fitted')
    axes[1].set_xlabel('Fitted')
    axes[1].set_ylabel('Residual')
    plt.tight_layout()
    plt.savefig("residual_vs_fitted.png")
    plt.close()

# PART 2b: Random Forest feature importances

if hasattr(pipe, 'named_steps') and 'rf' in pipe.named_steps:
    rf_model = pipe.named_steps['rf']
    importances = rf_model.feature_importances_
    fi = pd.DataFrame({
        'feature': numeric_cols,
        'importance': importances,
    }).sort_values('importance', ascending=False)
    print('Top 15 feature importances:')
    print(fi.head(15).to_string(index=False))
else:
    print('RandomForestRegressor not found in pipeline; skipping importances')

# PART 2c: Top 5 feature importances per year (separate RF per year)

years = sorted(df_state_year['YEAR'].unique())
print('Per-year RF importances (top 5):')

for yr in years:
    df_y = df_state_year[df_state_year['YEAR'] == yr].copy()
    if df_y.shape[0] < 10:
        print(f'Year {yr}: skipped (too few rows: {df_y.shape[0]})')
        continue

    X_y = df_y[numeric_cols]
    y_y = df_y[target]

    rf_y = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        n_jobs=-1,
        random_state=42,
    )
    rf_y.fit(X_y, y_y)

    fi = pd.DataFrame({
        'feature': numeric_cols,
        'importance': rf_y.feature_importances_,
    }).sort_values('importance', ascending=False).head(5)
    print(f'Year {yr} (rows={len(df_y)}):')
    print(fi.to_string(index=False))

# XGBoost baseline model
X = df_state_year[numeric_cols]
y = df_state_year[target]
X_train_x, X_test_x, y_train_x, y_test_x = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

xgb = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    objective='reg:squarederror',
)

xgb.fit(X_train_x, y_train_x)

for name, yt, yp in [('Train', y_train_x, xgb.predict(X_train_x)), ('Test', y_test_x, xgb.predict(X_test_x))]:
    rmse = root_mean_squared_error(yt, yp)
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    print(f"{name} -> RMSE: {rmse:,.2f}  MAE: {mae:,.2f}  R^2: {r2:.3f}")

xgb_baseline_model = xgb
y_pred_xgb = xgb.predict(X_test_x)

sns.set_theme(style="whitegrid", context="talk")

if 'xgb_baseline_model' in globals():
    rmse = root_mean_squared_error(y_test_x, y_pred_xgb)
    mae = mean_absolute_error(y_test_x, y_pred_xgb)
    r2 = r2_score(y_test_x, y_pred_xgb)
    plt.figure(figsize=(7, 7))
    sns.scatterplot(
        x=y_test_x,
        y=y_pred_xgb,
        hue=X_test_x['YEAR'] if 'YEAR' in X_test_x else None,
        palette='Blues',
        alpha=0.7,
    )
    lo = min(y_test_x.min(), y_pred_xgb.min())
    hi = max(y_test_x.max(), y_pred_xgb.max())
    plt.plot([lo, hi], [lo, hi], color='red', linestyle='--', label='y = x')
    plt.xlabel('Actual ($)')
    plt.ylabel('Predicted ($)')
    plt.title(f'Baseline XGB: R^2={r2:.3f}, RMSE={rmse:,.1f}, MAE={mae:,.1f}')
    plt.legend(title='YEAR', bbox_to_anchor=(1.02, 0.5), loc='center left')
    plt.tight_layout()
    plt.savefig("xgb_actual_vs_pred.png")
    plt.close()
else:
    print('XGB baseline not available in this environment')

top_n = 5

target_col = 'TOTAL_HEALTH_SPENDING'
if 'target' in globals():
    target_col = target

top_feat_by_year = {}
if 'df_state_year' not in globals() or 'numeric_cols' not in globals():
    print('Run the prep cell first to define df_state_year and numeric_cols.')
elif target_col not in df_state_year.columns:
    print(f"Target column '{target_col}' not found in df_state_year.")
else:
    for yr in sorted(df_state_year['YEAR'].unique()):
        df_y = df_state_year[df_state_year['YEAR'] == yr]
        if df_y.shape[0] < 10:
            continue
        X_y = df_y[numeric_cols]
        y_y = df_y[target_col]
        rf_y = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            n_jobs=-1,
            random_state=42,
        )
        rf_y.fit(X_y, y_y)
        s = pd.Series(rf_y.feature_importances_, index=numeric_cols).nlargest(top_n)
        top_feat_by_year[yr] = ', '.join([f"{feat} ({imp:.3f})" for feat, imp in s.items()])
    if top_feat_by_year:
        print(f'Computed top {top_n} features for {len(top_feat_by_year)} years.')
    else:
        print('No per-year importances computed (too few rows or data issues).')

metric_col = 'TOTAL_HEALTH_SPENDING'
color_range = None

target_top_n = globals().get('top_n', 5)
STATE_TO_ABBR = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
    'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
    'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR',
    'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA',
    'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
}

if 'df_state_year' not in globals() or 'numeric_cols' not in globals():
    print('df_state_year or numeric_cols missing; run the prep cell first.')
elif 'YEAR' not in df_state_year.columns:
    print("Column 'YEAR' missing in df_state_year; cannot animate.")
elif metric_col not in df_state_year.columns:
    print(f"Column {metric_col} not in df_state_year; choose another metric_col.")
else:
    df_plot = df_state_year.copy()
    if 'ST' not in df_plot.columns and 'STATE_NAME' in df_plot.columns:
        df_plot['ST'] = df_plot['STATE_NAME'].map(STATE_TO_ABBR)
    if 'ST' not in df_plot.columns:
        print("Need 'ST' or 'STATE_NAME' to map to state codes before plotting.")
    else:
        df_plot = df_plot.dropna(subset=['ST', metric_col, 'YEAR'])
        if df_plot.empty:
            print('No data after cleaning for ST/metric/YEAR.')
        else:
            if 'top_feat_by_year' in globals():
                df_plot['top_features'] = df_plot['YEAR'].map(top_feat_by_year).fillna('n/a')
            else:
                df_plot['top_features'] = 'n/a (run precompute cell)'

            if color_range is None:
                color_range = (df_plot[metric_col].min(), df_plot[metric_col].max())
            fig = px.choropleth(
                df_plot,
                locations='ST',
                locationmode='USA-states',
                color=metric_col,
                hover_name='STATE_NAME' if 'STATE_NAME' in df_plot.columns else 'ST',
                hover_data={'top_features': True},
                scope='usa',
                color_continuous_scale='Blues',
                animation_frame='YEAR',
                labels={metric_col: metric_col.replace('_', ' '), 'top_features': f'Top {target_top_n} features'},
                range_color=color_range,
                height=550,
            )
            fig.update_layout(title=f"{metric_col.replace('_', ' ')} by state (animated) — hover for top {target_top_n} features")
            fig.update_layout(transition={'duration': 0})
            years_sorted = sorted(df_plot['YEAR'].unique())
            fig.update_layout(
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': True,
                    'x': 1.05,
                    'y': 1.15,
                    'buttons': [
                        {
                            'label': 'Play',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 800, 'redraw': True},
                                'transition': {'duration': 0},
                                'fromcurrent': True,
                            }]
                        },
                        {
                            'label': 'Pause',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate',
                            }]
                        }
                    ]
                }],
                sliders=[{
                    'pad': {'t': 50},
                    'x': 0.1,
                    'len': 0.8,
                    'currentvalue': {'prefix': 'Year: '},
                    'steps': [
                        {
                            'label': str(yr),
                            'method': 'animate',
                            'args': [[str(yr)], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': True}, 'transition': {'duration': 0}}]
                        }
                        for yr in years_sorted
                    ],
                }]
            )

            out_path = 'choropleth_anim.html'
            fig.write_html(out_path, include_plotlyjs='cdn', full_html=True)
            print(f'Wrote animated choropleth to {out_path}')

spark.stop()
