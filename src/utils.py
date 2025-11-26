import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import mahalanobis
from scipy.stats import mstats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def compute_ar1(series):
    series = series.dropna()
    if len(series) < 3:
        return np.nan
    y = series.values
    y_t = y[1:]
    y_t1 = y[:-1]
    if np.std(y_t1) == 0:
        return np.nan
    return np.corrcoef(y_t, y_t1)[0, 1]


def compute_variance(series):
    series = series.dropna()
    if len(series) < 2:
        return np.nan
    return series.var()


def compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1-1)*group1.std()**2 + (n2-1)*group2.std()**2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else np.nan


def winsorize_data(data, limits=(0.05, 0.05)):
    return mstats.winsorize(data.dropna(), limits=limits)


def compute_mahalanobis_distance(data, threshold=3.5):
    if len(data) <= data.shape[1]:
        return np.zeros(len(data), dtype=bool)
    
    mean = data.mean()
    cov = data.cov()
    
    if np.linalg.det(cov) < 1e-10:
        return np.zeros(len(data), dtype=bool)
    
    inv_cov = np.linalg.pinv(cov)
    distances = data.apply(lambda x: mahalanobis(x, mean, inv_cov), axis=1)
    chi2_threshold = np.sqrt(stats.chi2.ppf(0.999, df=data.shape[1]))
    
    return distances > chi2_threshold


def compute_roc_metrics(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    youden_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[youden_idx]
    
    y_pred = (y_score >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'auc': auc,
        'optimal_threshold': optimal_threshold,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        'fpr': fpr,
        'tpr': tpr
    }


def standardize_variables(df, columns):
    scaler = StandardScaler()
    df_scaled = df.copy()
    for col in columns:
        if col in df.columns:
            df_scaled[f'{col}_z'] = scaler.fit_transform(df[[col]])
    return df_scaled


def compute_trajectory_slope(y_values):
    if len(y_values) < 2:
        return np.nan
    x = np.arange(len(y_values))
    slope, _, _, _, _ = stats.linregress(x, y_values)
    return slope


def classify_by_tertiles(values, labels=('Decline', 'Stable', 'Improved')):
    q1, q3 = values.quantile([0.33, 0.67])
    groups = pd.Series(index=values.index, dtype='object')
    groups[values <= q1] = labels[0]
    groups[values >= q3] = labels[2]
    groups[(values > q1) & (values < q3)] = labels[1]
    return groups


def plot_group_comparison(data, metric, groups, colors=None, ax=None):
    if colors is None:
        colors = {'Decline': '#e74c3c', 'Stable': '#3498db', 'Improved': '#2ecc71'}
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    group_list = ['Decline', 'Stable', 'Improved']
    means = [data[data['trajectory_group'] == g][metric].mean() for g in group_list]
    sems = [data[data['trajectory_group'] == g][metric].sem() for g in group_list]
    
    bars = ax.bar(group_list, means, yerr=sems, 
                  color=[colors[g] for g in group_list], capsize=5)
    ax.set_ylabel(f'Mean {metric}')
    ax.set_title(f'{metric} by Trajectory Group')
    
    return ax


def plot_roc_curve(fpr, tpr, auc, ax=None, label='Model'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return ax


def format_pvalue(p):
    if p < 1e-300:
        return "< 10⁻³⁰⁰"
    elif p < 1e-100:
        exp = int(np.floor(np.log10(p)))
        return f"< 10^{exp}"
    elif p < 0.001:
        return f"{p:.2e}"
    elif p < 0.05:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"


def generate_summary_statistics(df, numeric_cols, group_col=None):
    summary = {}
    
    if group_col and group_col in df.columns:
        for group in df[group_col].unique():
            group_df = df[df[group_col] == group]
            summary[group] = {}
            for col in numeric_cols:
                if col in group_df.columns:
                    summary[group][col] = {
                        'mean': group_df[col].mean(),
                        'sd': group_df[col].std(),
                        'median': group_df[col].median(),
                        'iqr': group_df[col].quantile(0.75) - group_df[col].quantile(0.25),
                        'n': group_df[col].notna().sum()
                    }
    else:
        for col in numeric_cols:
            if col in df.columns:
                summary[col] = {
                    'mean': df[col].mean(),
                    'sd': df[col].std(),
                    'median': df[col].median(),
                    'iqr': df[col].quantile(0.75) - df[col].quantile(0.25),
                    'n': df[col].notna().sum()
                }
    
    return summary
