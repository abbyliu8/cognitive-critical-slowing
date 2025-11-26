import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import warnings
warnings.filterwarnings('ignore')


def load_cohort_data(filepath):
    return pd.read_csv(filepath, low_memory=False)


def compute_ar1_coefficient(series):
    series = series.dropna()
    if len(series) < 3:
        return np.nan
    
    y = series.values
    y_t = y[1:]
    y_t1 = y[:-1]
    
    if np.std(y_t1) == 0:
        return np.nan
    
    correlation = np.corrcoef(y_t, y_t1)[0, 1]
    return correlation


def compute_variance(series):
    series = series.dropna()
    if len(series) < 2:
        return np.nan
    return series.var()


def compute_csd_indicators(df, outcome_col, id_col='ID', wave_col='wave'):
    results = []
    
    for pid in df[id_col].unique():
        person_data = df[df[id_col] == pid].sort_values(wave_col)
        outcome_series = person_data[outcome_col]
        
        ar1 = compute_ar1_coefficient(outcome_series)
        variance = compute_variance(outcome_series)
        
        group = person_data['trajectory_group'].iloc[0] if 'trajectory_group' in person_data.columns else None
        
        results.append({
            'ID': pid,
            'ar1': ar1,
            'variance': variance,
            'trajectory_group': group,
            'n_observations': outcome_series.notna().sum()
        })
    
    return pd.DataFrame(results)


def compute_cci(csd_df):
    valid_df = csd_df.dropna(subset=['ar1', 'variance'])
    
    scaler = StandardScaler()
    ar1_z = scaler.fit_transform(valid_df[['ar1']])
    variance_z = scaler.fit_transform(valid_df[['variance']])
    
    cci = (ar1_z.flatten() + variance_z.flatten()) / 2
    
    valid_df = valid_df.copy()
    valid_df['cci'] = cci
    valid_df['ar1_z'] = ar1_z.flatten()
    valid_df['variance_z'] = variance_z.flatten()
    
    return valid_df


def group_comparison_statistics(csd_df, metric, group_col='trajectory_group'):
    groups = csd_df[group_col].unique()
    groups = [g for g in groups if pd.notna(g)]
    
    group_data = {g: csd_df[csd_df[group_col] == g][metric].dropna() for g in groups}
    
    f_stat, p_value = stats.f_oneway(*[group_data[g] for g in groups if len(group_data[g]) > 0])
    
    results = {
        'f_statistic': f_stat,
        'p_value': p_value,
        'group_means': {g: group_data[g].mean() for g in groups},
        'group_sds': {g: group_data[g].std() for g in groups},
        'group_ns': {g: len(group_data[g]) for g in groups}
    }
    
    if 'Decline' in groups and 'Stable' in groups:
        d1 = group_data['Decline']
        d2 = group_data['Stable']
        pooled_std = np.sqrt(((len(d1) - 1) * d1.std()**2 + (len(d2) - 1) * d2.std()**2) / (len(d1) + len(d2) - 2))
        cohens_d = (d1.mean() - d2.mean()) / pooled_std if pooled_std > 0 else np.nan
        results['cohens_d'] = cohens_d
        
        t_stat, t_pvalue = stats.ttest_ind(d1, d2)
        results['t_statistic'] = t_stat
        results['t_pvalue'] = t_pvalue
    
    return results


def evaluate_prediction_performance(csd_df, predictor='cci', target_group='Decline', group_col='trajectory_group'):
    valid_df = csd_df.dropna(subset=[predictor, group_col])
    
    y_true = (valid_df[group_col] == target_group).astype(int)
    y_score = valid_df[predictor]
    
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    youden_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[youden_idx]
    
    y_pred = (y_score >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    results = {
        'auc': auc,
        'optimal_threshold': optimal_threshold,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }
    
    return results


def create_visualization(csd_df, performance, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    groups = ['Decline', 'Stable', 'Improved']
    colors = {'Decline': '#e74c3c', 'Stable': '#3498db', 'Improved': '#2ecc71'}
    
    ax = axes[0, 0]
    for group in groups:
        data = csd_df[csd_df['trajectory_group'] == group]['ar1'].dropna()
        ax.hist(data, bins=30, alpha=0.6, label=group, color=colors[group])
    ax.set_xlabel('AR(1) Coefficient')
    ax.set_ylabel('Frequency')
    ax.set_title('AR(1) Distribution by Group')
    ax.legend()
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    ax = axes[0, 1]
    for group in groups:
        data = csd_df[csd_df['trajectory_group'] == group]['variance'].dropna()
        ax.hist(data, bins=30, alpha=0.6, label=group, color=colors[group])
    ax.set_xlabel('Variance')
    ax.set_ylabel('Frequency')
    ax.set_title('Variance Distribution by Group')
    ax.legend()
    
    ax = axes[0, 2]
    for group in groups:
        data = csd_df[csd_df['trajectory_group'] == group]['cci'].dropna()
        ax.hist(data, bins=30, alpha=0.6, label=group, color=colors[group])
    ax.set_xlabel('CCI')
    ax.set_ylabel('Frequency')
    ax.set_title('CCI Distribution by Group')
    ax.legend()
    
    ax = axes[1, 0]
    group_means = [csd_df[csd_df['trajectory_group'] == g]['ar1'].mean() for g in groups]
    group_sems = [csd_df[csd_df['trajectory_group'] == g]['ar1'].sem() for g in groups]
    bars = ax.bar(groups, group_means, yerr=group_sems, color=[colors[g] for g in groups], capsize=5)
    ax.set_ylabel('Mean AR(1)')
    ax.set_title('AR(1) by Trajectory Group')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax = axes[1, 1]
    ax.plot(performance['fpr'], performance['tpr'], 'b-', linewidth=2, 
            label=f'AUC = {performance["auc"]:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve for Decline Prediction')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    ax = axes[1, 2]
    metrics = ['Sensitivity', 'Specificity', 'PPV', 'NPV']
    values = [performance['sensitivity'], performance['specificity'], 
              performance['ppv'], performance['npv']]
    bars = ax.bar(metrics, values, color=['#e74c3c', '#3498db', '#9b59b6', '#2ecc71'])
    ax.set_ylabel('Value')
    ax.set_title('Prediction Performance Metrics')
    ax.set_ylim([0, 1])
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()


def generate_analysis_report(csd_df, ar1_stats, var_stats, cci_stats, performance, output_path=None):
    report = []
    report.append("=" * 70)
    report.append("CRITICAL SLOWING DOWN ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")
    report.append("[SAMPLE]")
    report.append(f"Total participants: {len(csd_df):,}")
    for group in ['Decline', 'Stable', 'Improved']:
        n = (csd_df['trajectory_group'] == group).sum()
        pct = n / len(csd_df) * 100
        report.append(f"  {group}: {n:,} ({pct:.1f}%)")
    report.append("")
    report.append("[AR(1) AUTOREGRESSIVE COEFFICIENT]")
    for group, mean in ar1_stats['group_means'].items():
        sd = ar1_stats['group_sds'][group]
        report.append(f"  {group}: {mean:.3f} ± {sd:.3f}")
    report.append(f"F-statistic: {ar1_stats['f_statistic']:.2f}")
    report.append(f"p-value: {ar1_stats['p_value']:.2e}")
    if 'cohens_d' in ar1_stats:
        report.append(f"Cohen's d (Decline vs Stable): {ar1_stats['cohens_d']:.3f}")
    report.append("")
    report.append("[VARIANCE]")
    for group, mean in var_stats['group_means'].items():
        sd = var_stats['group_sds'][group]
        report.append(f"  {group}: {mean:.3f} ± {sd:.3f}")
    report.append(f"F-statistic: {var_stats['f_statistic']:.2f}")
    report.append(f"p-value: {var_stats['p_value']:.2e}")
    report.append("")
    report.append("[COMPREHENSIVE CRITICALITY INDEX (CCI)]")
    for group, mean in cci_stats['group_means'].items():
        sd = cci_stats['group_sds'][group]
        report.append(f"  {group}: {mean:.3f} ± {sd:.3f}")
    report.append(f"F-statistic: {cci_stats['f_statistic']:.2f}")
    report.append(f"p-value: {cci_stats['p_value']:.2e}")
    report.append("")
    report.append("[PREDICTION PERFORMANCE]")
    report.append(f"AUC: {performance['auc']:.3f}")
    report.append(f"Optimal threshold: {performance['optimal_threshold']:.3f}")
    report.append(f"Sensitivity: {performance['sensitivity']:.3f}")
    report.append(f"Specificity: {performance['specificity']:.3f}")
    report.append(f"PPV: {performance['ppv']:.3f}")
    report.append(f"NPV: {performance['npv']:.3f}")
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
    
    return report_text


def main(args):
    import os
    os.makedirs(args.output, exist_ok=True)
    
    print("Loading cohort data...")
    df = load_cohort_data(args.input)
    
    outcome_col = args.outcome if args.outcome else 'memory'
    if outcome_col not in df.columns:
        outcome_col = 'total_cognition'
    
    print("Computing critical slowing down indicators...")
    csd_df = compute_csd_indicators(df, outcome_col)
    
    print("Computing Comprehensive Criticality Index...")
    csd_df = compute_cci(csd_df)
    
    print("Computing group comparison statistics...")
    ar1_stats = group_comparison_statistics(csd_df, 'ar1')
    var_stats = group_comparison_statistics(csd_df, 'variance')
    cci_stats = group_comparison_statistics(csd_df, 'cci')
    
    print("Evaluating prediction performance...")
    performance = evaluate_prediction_performance(csd_df, predictor='cci')
    
    print("Generating visualization...")
    fig_path = f"{args.output}/csd_analysis.png"
    create_visualization(csd_df, performance, fig_path)
    
    report_path = f"{args.output}/csd_analysis_report.txt"
    report = generate_analysis_report(csd_df, ar1_stats, var_stats, cci_stats, performance, report_path)
    print(report)
    
    results_path = f"{args.output}/csd_results.csv"
    csd_df.to_csv(results_path, index=False)
    
    print(f"Analysis complete. Results saved to {args.output}/")
    return csd_df, performance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Critical Slowing Down Analysis")
    parser.add_argument("--input", type=str, required=True, help="Input cohort CSV")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--outcome", type=str, default="memory", help="Primary outcome variable")
    
    args = parser.parse_args()
    main(args)
