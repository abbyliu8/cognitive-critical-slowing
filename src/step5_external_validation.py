import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mstats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
import warnings
import os
warnings.filterwarnings('ignore')


def load_elsa_data(filepath):
    return pd.read_csv(filepath, low_memory=False)


def construct_elsa_cohort(df, memory_col='memory', id_col='idauniq', wave_col='wave',
                          min_age=50, min_wave=4, max_wave=9):
    n_waves = max_wave - min_wave + 1
    
    df_filtered = df[(df[wave_col] >= min_wave) & (df[wave_col] <= max_wave)].copy()
    
    wave_memory_counts = df_filtered.groupby(id_col).apply(
        lambda x: x[x[memory_col].notna()][wave_col].nunique()
    )
    complete_ids = wave_memory_counts[wave_memory_counts == n_waves].index
    
    cohort = df_filtered[df_filtered[id_col].isin(complete_ids)].copy()
    
    baseline = cohort[cohort[wave_col] == min_wave]
    age_eligible = baseline[baseline['age'] >= min_age][id_col].unique()
    cohort = cohort[cohort[id_col].isin(age_eligible)]
    
    return cohort


def compute_trajectory_slope(df, outcome_col, id_col='idauniq', wave_col='wave'):
    slopes = {}
    for pid in df[id_col].unique():
        person_data = df[df[id_col] == pid].sort_values(wave_col)
        y = person_data[outcome_col].dropna().values
        x = np.arange(len(y))
        if len(y) >= 2:
            slope, _, _, _, _ = stats.linregress(x, y)
            slopes[pid] = slope
    return pd.Series(slopes)


def classify_trajectories(slopes):
    q1, q3 = slopes.quantile([0.33, 0.67])
    groups = pd.Series(index=slopes.index, dtype='object')
    groups[slopes <= q1] = 'Decline'
    groups[slopes >= q3] = 'Improved'
    groups[(slopes > q1) & (slopes < q3)] = 'Stable'
    return groups


def compute_ar1_coefficient(series):
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


def compute_csd_indicators(df, outcome_col, groups, id_col='idauniq', wave_col='wave'):
    results = []
    for pid in df[id_col].unique():
        person_data = df[df[id_col] == pid].sort_values(wave_col)
        outcome_series = person_data[outcome_col]
        ar1 = compute_ar1_coefficient(outcome_series)
        variance = compute_variance(outcome_series)
        group = groups.get(pid, None)
        results.append({
            'ID': pid,
            'ar1': ar1,
            'variance': variance,
            'trajectory_group': group,
            'n_observations': outcome_series.notna().sum()
        })
    return pd.DataFrame(results)


def winsorize_statistics(data, limits=(0.05, 0.05)):
    clean_data = data.dropna()
    if len(clean_data) < 10:
        return clean_data.mean(), clean_data.std()
    winsorized = mstats.winsorize(clean_data, limits=limits)
    return float(np.mean(winsorized)), float(np.std(winsorized))


def compute_cci(csd_df):
    valid_df = csd_df.dropna(subset=['ar1', 'variance']).copy()
    scaler = StandardScaler()
    valid_df['ar1_z'] = scaler.fit_transform(valid_df[['ar1']])
    scaler2 = StandardScaler()
    valid_df['variance_z'] = scaler2.fit_transform(valid_df[['variance']])
    valid_df['cci'] = (valid_df['ar1_z'] + valid_df['variance_z']) / 2
    return valid_df


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (group1.mean() - group2.mean()) / pooled_std


def compute_statistics(csd_df, metric, use_winsorize=False):
    groups = ['Decline', 'Stable', 'Improved']
    group_data = {g: csd_df[csd_df['trajectory_group'] == g][metric].dropna() for g in groups}
    
    valid_groups = [group_data[g] for g in groups if len(group_data[g]) > 0]
    if len(valid_groups) < 2:
        return None
    
    f_stat, p_value = stats.f_oneway(*valid_groups)
    
    if use_winsorize:
        group_means = {}
        group_sds = {}
        for g in groups:
            mean, sd = winsorize_statistics(group_data[g])
            group_means[g] = mean
            group_sds[g] = sd
    else:
        group_means = {g: group_data[g].mean() for g in groups}
        group_sds = {g: group_data[g].std() for g in groups}
    
    d1, d2 = group_data['Decline'], group_data['Stable']
    t_stat, t_pvalue = stats.ttest_ind(d1, d2)
    effect_size = cohens_d(d1, d2)
    
    results = {
        'f_statistic': f_stat,
        'p_value': p_value,
        'group_means': group_means,
        'group_sds': group_sds,
        'group_n': {g: len(group_data[g]) for g in groups},
        't_statistic': t_stat,
        't_pvalue': t_pvalue,
        'cohens_d': effect_size
    }
    
    return results


def evaluate_prediction(csd_df, predictor='cci'):
    valid_df = csd_df.dropna(subset=[predictor, 'trajectory_group'])
    y_true = (valid_df['trajectory_group'] == 'Decline').astype(int)
    y_score = valid_df[predictor]
    
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    youden_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[youden_idx]
    
    y_pred = (y_score >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return {
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'optimal_threshold': optimal_threshold,
        'fpr': fpr,
        'tpr': tpr
    }


def create_validation_figure(csd_df, performance, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    groups = ['Decline', 'Stable', 'Improved']
    colors = {'Decline': '#e74c3c', 'Stable': '#3498db', 'Improved': '#2ecc71'}
    
    ax = axes[0, 0]
    for group in groups:
        data = csd_df[csd_df['trajectory_group'] == group]['ar1'].dropna()
        ax.hist(data, bins=30, alpha=0.6, label=group, color=colors[group])
    ax.set_xlabel('AR(1) Coefficient')
    ax.set_ylabel('Frequency')
    ax.set_title('AR(1) Distribution by Trajectory Group')
    ax.legend()
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    ax = axes[0, 1]
    for group in groups:
        data = csd_df[csd_df['trajectory_group'] == group]['variance'].dropna()
        data_clipped = data[data < data.quantile(0.95)]
        ax.hist(data_clipped, bins=30, alpha=0.6, label=group, color=colors[group])
    ax.set_xlabel('Variance')
    ax.set_ylabel('Frequency')
    ax.set_title('Variance Distribution by Trajectory Group')
    ax.legend()
    
    ax = axes[1, 0]
    group_means = [csd_df[csd_df['trajectory_group'] == g]['ar1'].mean() for g in groups]
    group_sems = [csd_df[csd_df['trajectory_group'] == g]['ar1'].sem() for g in groups]
    ax.bar(groups, group_means, yerr=group_sems, color=[colors[g] for g in groups], capsize=5)
    ax.set_ylabel('Mean AR(1) Coefficient')
    ax.set_title('AR(1) by Trajectory Group')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax = axes[1, 1]
    ax.plot(performance['fpr'], performance['tpr'], 'b-', linewidth=2,
            label=f'AUC = {performance["auc"]:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve: CCI Predicting Cognitive Decline')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()


def generate_validation_report(csd_df, ar1_stats, var_stats, performance, output_path=None):
    report = []
    report.append("=" * 70)
    report.append("ELSA EXTERNAL VALIDATION REPORT")
    report.append("=" * 70)
    report.append("")
    
    report.append("[SAMPLE CHARACTERISTICS]")
    report.append(f"Total participants: {len(csd_df):,}")
    report.append(f"Observation waves: 6 (Waves 4-9, 2008-2018)")
    for group in ['Decline', 'Stable', 'Improved']:
        n = (csd_df['trajectory_group'] == group).sum()
        pct = n / len(csd_df) * 100
        report.append(f"  {group}: {n:,} ({pct:.1f}%)")
    report.append("")
    
    report.append("[AR(1) AUTOREGRESSIVE COEFFICIENT]")
    for group in ['Decline', 'Stable', 'Improved']:
        mean = ar1_stats['group_means'][group]
        sd = ar1_stats['group_sds'][group]
        n = ar1_stats['group_n'][group]
        report.append(f"  {group}: {mean:.3f} +/- {sd:.3f} (n={n})")
    report.append(f"")
    report.append(f"Decline vs Stable:")
    report.append(f"  t-statistic: {ar1_stats['t_statistic']:.3f}")
    report.append(f"  p-value: {ar1_stats['t_pvalue']:.2e}")
    report.append(f"  Cohen's d: {ar1_stats['cohens_d']:.3f}")
    report.append("")
    
    report.append("[VARIANCE (Winsorized at 5th/95th percentile)]")
    for group in ['Decline', 'Stable', 'Improved']:
        mean = var_stats['group_means'][group]
        sd = var_stats['group_sds'][group]
        report.append(f"  {group}: {mean:.3f} +/- {sd:.3f}")
    report.append(f"")
    report.append(f"Decline vs Stable:")
    report.append(f"  Cohen's d: {var_stats['cohens_d']:.3f}")
    report.append("")
    
    report.append("[CCI PREDICTION PERFORMANCE]")
    report.append(f"  AUC: {performance['auc']:.3f}")
    report.append(f"  Sensitivity: {performance['sensitivity']:.3f}")
    report.append(f"  Specificity: {performance['specificity']:.3f}")
    report.append(f"  PPV: {performance['ppv']:.3f}")
    report.append(f"  NPV: {performance['npv']:.3f}")
    report.append(f"  Optimal threshold: {performance['optimal_threshold']:.3f}")
    report.append("")
    
    report.append("[CROSS-COHORT COMPARISON]")
    report.append("                      CHARLS          ELSA")
    report.append(f"  Sample size:        3,487           {len(csd_df):,}")
    report.append(f"  AR(1) Decline:      0.054           {ar1_stats['group_means']['Decline']:.3f}")
    report.append(f"  AR(1) Stable:       -0.514          {ar1_stats['group_means']['Stable']:.3f}")
    report.append(f"  Cohen's d:          1.001           {ar1_stats['cohens_d']:.3f}")
    report.append(f"  AUC:                0.752           {performance['auc']:.3f}")
    report.append("")
    
    report.append("[VALIDATION CONCLUSION]")
    ar1_validated = ar1_stats['group_means']['Decline'] > ar1_stats['group_means']['Stable']
    var_validated = var_stats['group_means']['Decline'] > var_stats['group_means']['Stable']
    auc_validated = performance['auc'] >= 0.70
    
    report.append(f"  AR(1) elevated in Decline group: {'CONFIRMED' if ar1_validated else 'NOT CONFIRMED'}")
    report.append(f"  Variance elevated in Decline group: {'CONFIRMED' if var_validated else 'NOT CONFIRMED'}")
    report.append(f"  CCI predictive validity (AUC >= 0.70): {'CONFIRMED' if auc_validated else 'NOT CONFIRMED'}")
    
    if ar1_validated and var_validated and auc_validated:
        report.append("")
        report.append("  EXTERNAL VALIDATION: SUCCESSFUL")
        report.append("  Critical slowing down theory replicated in independent cohort.")
    
    report.append("")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
    return report_text


def main(args):
    os.makedirs(args.output, exist_ok=True)
    
    print("Loading ELSA data...")
    df = load_elsa_data(args.input)
    
    print("Constructing cohort (Waves 4-9, age >= 50, complete memory data)...")
    cohort = construct_elsa_cohort(df, min_wave=4, max_wave=9)
    print(f"  Cohort size: {cohort['idauniq'].nunique():,} participants")
    
    print("Computing trajectory slopes...")
    slopes = compute_trajectory_slope(cohort, 'memory', id_col='idauniq')
    
    print("Classifying trajectories (tertile-based)...")
    groups = classify_trajectories(slopes)
    
    print("Computing critical slowing down indicators...")
    csd_df = compute_csd_indicators(cohort, 'memory', groups, id_col='idauniq')
    csd_df = compute_cci(csd_df)
    print(f"  Valid CSD observations: {len(csd_df.dropna(subset=['ar1', 'variance'])):,}")
    
    print("Computing group statistics...")
    ar1_stats = compute_statistics(csd_df, 'ar1', use_winsorize=False)
    var_stats = compute_statistics(csd_df, 'variance', use_winsorize=True)
    
    print("Evaluating CCI prediction performance...")
    performance = evaluate_prediction(csd_df, predictor='cci')
    
    print("Generating visualization...")
    fig_path = os.path.join(args.output, "elsa_validation.png")
    create_validation_figure(csd_df, performance, fig_path)
    
    print("Generating validation report...")
    report_path = os.path.join(args.output, "elsa_validation_report.txt")
    report = generate_validation_report(csd_df, ar1_stats, var_stats, performance, report_path)
    print("")
    print(report)
    
    results_path = os.path.join(args.output, "elsa_csd_results.csv")
    csd_df.to_csv(results_path, index=False)
    
    print(f"\nExternal validation complete. Results saved to {args.output}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ELSA External Validation for Critical Slowing Down")
    parser.add_argument("--input", type=str, required=True, help="Path to ELSA merged CSV file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for results")
    
    args = parser.parse_args()
    main(args)
