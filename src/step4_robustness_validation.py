import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
import warnings
import os
warnings.filterwarnings('ignore')


def load_cohort_data(filepath):
    return pd.read_csv(filepath, low_memory=False)


def create_cohort_by_waves(df, outcome_col, min_waves, id_col='ID', wave_col='wave'):
    wave_counts = df.groupby(id_col).apply(lambda x: x[outcome_col].notna().sum())
    eligible_ids = wave_counts[wave_counts >= min_waves].index
    cohort = df[df[id_col].isin(eligible_ids)].copy()
    return cohort


def compute_trajectory_slope(df, outcome_col, id_col='ID', wave_col='wave'):
    slopes = {}
    for pid in df[id_col].unique():
        person_data = df[df[id_col] == pid].sort_values(wave_col)
        mask = person_data[outcome_col].notna()
        x = person_data.loc[mask, wave_col].values.astype(float)
        y = person_data.loc[mask, outcome_col].values.astype(float)
        if len(y) >= 2 and np.std(x) > 0:
            result = stats.linregress(x, y)
            slopes[pid] = result.slope
    return pd.Series(slopes)


def classify_trajectories(slopes):
    q1, q3 = slopes.quantile([0.33, 0.67])
    groups = pd.Series(index=slopes.index, dtype='object')
    groups[slopes <= q1] = 'Decline'
    groups[slopes >= q3] = 'Improved'
    groups[(slopes > q1) & (slopes < q3)] = 'Stable'
    return groups


def compute_ar1_ols(series, detrend=True):
    series = series.dropna()
    if len(series) < 3:
        return np.nan
    y = series.values.astype(float)
    
    if detrend:
        x = np.arange(len(y))
        trend = stats.linregress(x, y)
        y = y - (trend.slope * x + trend.intercept)
    
    y_t = y[1:]
    y_t1 = y[:-1]
    
    if np.std(y_t1) == 0:
        return np.nan
    
    result = stats.linregress(y_t1, y_t)
    return result.slope


def compute_variance_detrended(series, detrend=True):
    series = series.dropna()
    if len(series) < 2:
        return np.nan
    y = series.values.astype(float)
    
    if detrend:
        x = np.arange(len(y))
        trend = stats.linregress(x, y)
        y = y - (trend.slope * x + trend.intercept)
    
    return np.var(y, ddof=1)


def compute_csd_indicators(df, outcome_col, groups, id_col='ID', wave_col='wave', detrend=True):
    results = []
    for pid in df[id_col].unique():
        person_data = df[df[id_col] == pid].sort_values(wave_col)
        outcome_series = person_data[outcome_col]
        ar1 = compute_ar1_ols(outcome_series, detrend=detrend)
        variance = compute_variance_detrended(outcome_series, detrend=detrend)
        group = groups.get(pid, None)
        results.append({
            'ID': pid,
            'ar1': ar1,
            'variance': variance,
            'trajectory_group': group,
            'n_observations': outcome_series.notna().sum()
        })
    return pd.DataFrame(results)


def compute_cci(csd_df):
    valid_df = csd_df.dropna(subset=['ar1', 'variance']).copy()
    if len(valid_df) < 2:
        return valid_df
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    valid_df['ar1_z'] = scaler1.fit_transform(valid_df[['ar1']])
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


def compute_statistics(csd_df, metric):
    groups_list = ['Decline', 'Stable', 'Improved']
    group_data = {g: csd_df[csd_df['trajectory_group'] == g][metric].dropna() for g in groups_list}
    
    valid_groups = [group_data[g] for g in groups_list if len(group_data[g]) > 1]
    if len(valid_groups) < 2:
        return None
    
    h_stat, kw_pvalue = stats.kruskal(*valid_groups)
    
    d1, d2 = group_data['Decline'], group_data['Stable']
    if len(d1) < 2 or len(d2) < 2:
        return None
    
    t_stat, t_pvalue = stats.ttest_ind(d1, d2, equal_var=False)
    u_stat, mw_pvalue = stats.mannwhitneyu(d1, d2, alternative='two-sided')
    effect_size = cohens_d(d1, d2)
    
    results = {
        'kruskal_h': h_stat,
        'kruskal_p': kw_pvalue,
        'group_means': {g: group_data[g].mean() for g in groups_list},
        'group_sds': {g: group_data[g].std() for g in groups_list},
        'group_medians': {g: group_data[g].median() for g in groups_list},
        'group_n': {g: len(group_data[g]) for g in groups_list},
        'welch_t': t_stat,
        'welch_p': t_pvalue,
        'mannwhitney_u': u_stat,
        'mannwhitney_p': mw_pvalue,
        'cohens_d': effect_size
    }
    
    return results


def evaluate_prediction(csd_df, predictor='cci', threshold=None):
    valid_df = csd_df.dropna(subset=[predictor, 'trajectory_group'])
    y_true = (valid_df['trajectory_group'] == 'Decline').astype(int)
    y_score = valid_df[predictor]
    
    if y_true.nunique() < 2:
        return None
    
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    if threshold is None:
        youden_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[youden_idx]
    else:
        optimal_threshold = threshold
    
    y_pred = (y_score >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    
    return {
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'threshold': optimal_threshold,
        'fpr': fpr,
        'tpr': tpr
    }


def analyze_cohort(df, outcome_col, groups, id_col='ID', detrend=True):
    csd_df = compute_csd_indicators(df, outcome_col, groups, id_col=id_col, detrend=detrend)
    csd_df = compute_cci(csd_df)
    
    ar1_stats = compute_statistics(csd_df, 'ar1')
    var_stats = compute_statistics(csd_df, 'variance')
    
    results = {
        'n': len(csd_df.dropna(subset=['ar1', 'variance'])),
        'ar1_stats': ar1_stats,
        'var_stats': var_stats,
        'csd_df': csd_df
    }
    
    return results


def create_comparison_figure(silver, gold, eligible_n, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    cohorts = ['Silver\n(>=3 waves)', 'Gold\n(>=4 waves)']
    colors = ['#3498db', '#f39c12']
    
    ax = axes[0]
    sizes = [silver['n'], gold['n']]
    bars = ax.bar(cohorts, sizes, color=colors)
    ax.set_ylabel('Sample Size')
    ax.set_title('A. Sample Size Comparison')
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'n = {size:,}', ha='center', va='bottom', fontsize=10)
        pct = size / eligible_n * 100 if eligible_n > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{pct:.1f}%\nof silver', ha='center', va='center', color='white', fontsize=9)
    
    ax = axes[1]
    d_silver = silver['ar1_stats']['cohens_d'] if silver['ar1_stats'] else np.nan
    d_gold = gold['ar1_stats']['cohens_d'] if gold['ar1_stats'] else np.nan
    effect_sizes = [d_silver, d_gold]
    
    valid_es = [es for es in effect_sizes if not np.isnan(es)]
    if len(valid_es) > 0:
        bars = ax.bar(cohorts, [es if not np.isnan(es) else 0 for es in effect_sizes], color=colors)
        ax.set_ylabel("Cohen's d (Decline vs Stable)")
        ax.set_title('B. Effect Size Comparison')
        ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small (0.2)')
        ax.axhline(y=0.5, color='gray', linestyle='-.', alpha=0.5, label='Medium (0.5)')
        ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='Large (0.8)')
        ax.legend(loc='upper right', fontsize=8)
        for bar, es in zip(bars, effect_sizes):
            if not np.isnan(es):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'd = {es:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax = axes[2]
    auc_silver = silver['performance']['auc'] if silver.get('performance') else np.nan
    auc_gold = gold['performance']['auc'] if gold.get('performance') else np.nan
    aucs = [auc_silver, auc_gold]
    
    valid_aucs = [a for a in aucs if not np.isnan(a)]
    if len(valid_aucs) > 0:
        bars = ax.bar(cohorts, [a if not np.isnan(a) else 0.5 for a in aucs], color=colors)
        ax.set_ylabel('AUC (Decline Prediction)')
        ax.set_title('C. Predictive Performance')
        ax.set_ylim([0.4, 0.9])
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance (0.5)')
        ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Acceptable (0.7)')
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent (0.8)')
        ax.legend(loc='upper right', fontsize=8)
        for bar, auc in zip(bars, aucs):
            if not np.isnan(auc):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'AUC = {auc:.3f}', ha='center', va='bottom', fontsize=10)
    
    delta_d = d_gold - d_silver if not (np.isnan(d_gold) or np.isnan(d_silver)) else np.nan
    delta_auc = auc_gold - auc_silver if not (np.isnan(auc_gold) or np.isnan(auc_silver)) else np.nan
    
    summary_text = 'Robustness Assessment: '
    if not np.isnan(delta_d):
        summary_text += f'Effect size d = {delta_d:+.3f}, '
    if not np.isnan(delta_auc):
        summary_text += f'AUC = {delta_auc:+.3f}'
    
    fig.text(0.5, -0.02, summary_text, ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()


def generate_report(silver, gold, output_path=None):
    report = []
    report.append("=" * 70)
    report.append("ROBUSTNESS VALIDATION REPORT")
    report.append("Silver (>=3 waves) vs Gold (>=4 waves) Cohort Comparison")
    report.append("=" * 70)
    report.append("")
    report.append("[METHODOLOGICAL NOTES]")
    report.append("  - AR(1) estimated via OLS regression (phi parameter)")
    report.append("  - Detrended before AR(1) and variance calculation")
    report.append("  - Trajectory slope computed using actual wave values")
    report.append("  - Statistical tests: Welch t-test, Mann-Whitney U, Kruskal-Wallis")
    report.append("  - Threshold defined on Silver, applied unchanged to Gold")
    report.append("")
    
    report.append("[SAMPLE SIZE]")
    report.append(f"Silver cohort: {silver['n']:,}")
    report.append(f"Gold cohort: {gold['n']:,}")
    report.append(f"Retention: {gold['n']/silver['n']*100:.1f}%" if silver['n'] > 0 else "N/A")
    report.append("")
    
    report.append("[AR(1) AUTOREGRESSIVE COEFFICIENT - OLS phi]")
    report.append("")
    if silver['ar1_stats']:
        report.append("Silver cohort:")
        for g in ['Decline', 'Stable', 'Improved']:
            m = silver['ar1_stats']['group_means'].get(g, np.nan)
            s = silver['ar1_stats']['group_sds'].get(g, np.nan)
            n = silver['ar1_stats']['group_n'].get(g, 0)
            med = silver['ar1_stats']['group_medians'].get(g, np.nan)
            report.append(f"  {g}: {m:.3f} +/- {s:.3f} (median={med:.3f}, n={n})")
        report.append(f"  Kruskal-Wallis H = {silver['ar1_stats']['kruskal_h']:.2f}, p = {silver['ar1_stats']['kruskal_p']:.2e}")
        report.append(f"  Welch t = {silver['ar1_stats']['welch_t']:.2f}, p = {silver['ar1_stats']['welch_p']:.2e}")
        report.append(f"  Mann-Whitney p = {silver['ar1_stats']['mannwhitney_p']:.2e}")
        report.append(f"  Cohen's d: {silver['ar1_stats']['cohens_d']:.3f}")
    report.append("")
    
    if gold['ar1_stats']:
        report.append("Gold cohort:")
        for g in ['Decline', 'Stable', 'Improved']:
            m = gold['ar1_stats']['group_means'].get(g, np.nan)
            s = gold['ar1_stats']['group_sds'].get(g, np.nan)
            n = gold['ar1_stats']['group_n'].get(g, 0)
            med = gold['ar1_stats']['group_medians'].get(g, np.nan)
            report.append(f"  {g}: {m:.3f} +/- {s:.3f} (median={med:.3f}, n={n})")
        report.append(f"  Kruskal-Wallis H = {gold['ar1_stats']['kruskal_h']:.2f}, p = {gold['ar1_stats']['kruskal_p']:.2e}")
        report.append(f"  Welch t = {gold['ar1_stats']['welch_t']:.2f}, p = {gold['ar1_stats']['welch_p']:.2e}")
        report.append(f"  Mann-Whitney p = {gold['ar1_stats']['mannwhitney_p']:.2e}")
        report.append(f"  Cohen's d: {gold['ar1_stats']['cohens_d']:.3f}")
    report.append("")
    
    report.append("[VARIANCE (Detrended)]")
    report.append("")
    if silver['var_stats']:
        report.append("Silver cohort:")
        for g in ['Decline', 'Stable', 'Improved']:
            m = silver['var_stats']['group_means'].get(g, np.nan)
            s = silver['var_stats']['group_sds'].get(g, np.nan)
            report.append(f"  {g}: {m:.3f} +/- {s:.3f}")
        report.append(f"  Kruskal-Wallis H = {silver['var_stats']['kruskal_h']:.2f}, p = {silver['var_stats']['kruskal_p']:.2e}")
    report.append("")
    
    if gold['var_stats']:
        report.append("Gold cohort:")
        for g in ['Decline', 'Stable', 'Improved']:
            m = gold['var_stats']['group_means'].get(g, np.nan)
            s = gold['var_stats']['group_sds'].get(g, np.nan)
            report.append(f"  {g}: {m:.3f} +/- {s:.3f}")
        report.append(f"  Kruskal-Wallis H = {gold['var_stats']['kruskal_h']:.2f}, p = {gold['var_stats']['kruskal_p']:.2e}")
    report.append("")
    
    report.append("[CCI PREDICTION PERFORMANCE]")
    report.append("Note: Threshold defined on Silver, applied to Gold (no re-optimization)")
    report.append("")
    
    if silver.get('performance') and gold.get('performance'):
        report.append("                    Silver      Gold (same threshold)")
        report.append(f"  AUC:              {silver['performance']['auc']:.3f}       {gold['performance']['auc']:.3f}")
        report.append(f"  Sensitivity:      {silver['performance']['sensitivity']:.3f}       {gold['performance']['sensitivity']:.3f}")
        report.append(f"  Specificity:      {silver['performance']['specificity']:.3f}       {gold['performance']['specificity']:.3f}")
        report.append(f"  PPV:              {silver['performance']['ppv']:.3f}       {gold['performance']['ppv']:.3f}")
        report.append(f"  NPV:              {silver['performance']['npv']:.3f}       {gold['performance']['npv']:.3f}")
        report.append(f"  Threshold:        {silver['performance']['threshold']:.3f}       {gold['performance']['threshold']:.3f}")
    report.append("")
    
    report.append("[ROBUSTNESS ASSESSMENT]")
    
    ar1_robust = False
    var_robust = False
    auc_robust = False
    
    if silver['ar1_stats'] and gold['ar1_stats']:
        ar1_robust = (silver['ar1_stats']['group_means']['Decline'] > silver['ar1_stats']['group_means']['Stable'] and
                      gold['ar1_stats']['group_means']['Decline'] > gold['ar1_stats']['group_means']['Stable'])
        delta_d = gold['ar1_stats']['cohens_d'] - silver['ar1_stats']['cohens_d']
        report.append(f"  AR(1) direction consistent: {'CONFIRMED' if ar1_robust else 'NOT CONFIRMED'}")
        report.append(f"  Effect size change: {delta_d:+.3f}")
    
    if silver['var_stats'] and gold['var_stats']:
        var_robust = (silver['var_stats']['group_means']['Decline'] > silver['var_stats']['group_means']['Stable'] and
                      gold['var_stats']['group_means']['Decline'] > gold['var_stats']['group_means']['Stable'])
        report.append(f"  Variance direction consistent: {'CONFIRMED' if var_robust else 'NOT CONFIRMED'}")
    
    if silver.get('performance') and gold.get('performance'):
        delta_auc = gold['performance']['auc'] - silver['performance']['auc']
        auc_robust = abs(delta_auc) < 0.05
        report.append(f"  AUC stable (|delta| < 0.05): {'CONFIRMED' if auc_robust else 'NOT CONFIRMED'}")
        report.append(f"  AUC change: {delta_auc:+.3f}")
    
    report.append("")
    
    if ar1_robust and var_robust and auc_robust:
        report.append("  ROBUSTNESS VALIDATION: SUCCESSFUL")
    elif ar1_robust and var_robust:
        report.append("  ROBUSTNESS VALIDATION: PARTIAL (direction consistent, AUC shifted)")
    else:
        report.append("  ROBUSTNESS VALIDATION: REQUIRES REVIEW")
    
    report.append("")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
    return report_text


def main(args):
    os.makedirs(args.output, exist_ok=True)
    
    print("Loading data...")
    df = load_cohort_data(args.input)
    
    outcome_col = args.outcome
    id_col = 'ID'
    wave_col = 'wave'
    
    if outcome_col not in df.columns:
        if 'memory' in df.columns:
            outcome_col = 'memory'
        elif 'total_cognition' in df.columns:
            outcome_col = 'total_cognition'
        else:
            raise ValueError(f"Outcome column not found: {args.outcome}")
    
    print(f"Outcome variable: {outcome_col}")
    print(f"Detrend: {args.detrend}")
    
    print("Creating silver cohort (>=3 waves)...")
    silver_df = create_cohort_by_waves(df, outcome_col, min_waves=3, id_col=id_col)
    eligible_n = silver_df[id_col].nunique()
    print(f"  Silver cohort: {eligible_n:,} participants")
    
    print("Computing trajectory slopes (silver)...")
    silver_slopes = compute_trajectory_slope(silver_df, outcome_col, id_col=id_col, wave_col=wave_col)
    silver_groups = classify_trajectories(silver_slopes)
    
    print("Analyzing silver cohort...")
    silver_results = analyze_cohort(silver_df, outcome_col, silver_groups, id_col=id_col, detrend=args.detrend)
    
    print("Evaluating prediction (silver - threshold selection)...")
    silver_perf = evaluate_prediction(silver_results['csd_df'], predictor='cci', threshold=None)
    silver_results['performance'] = silver_perf
    
    print("Creating gold cohort (>=4 waves)...")
    gold_df = create_cohort_by_waves(df, outcome_col, min_waves=4, id_col=id_col)
    gold_n = gold_df[id_col].nunique()
    print(f"  Gold cohort: {gold_n:,} participants")
    
    print("Computing trajectory slopes (gold)...")
    gold_slopes = compute_trajectory_slope(gold_df, outcome_col, id_col=id_col, wave_col=wave_col)
    gold_groups = classify_trajectories(gold_slopes)
    
    print("Analyzing gold cohort...")
    gold_results = analyze_cohort(gold_df, outcome_col, gold_groups, id_col=id_col, detrend=args.detrend)
    
    print("Evaluating prediction (gold - using silver threshold)...")
    silver_threshold = silver_perf['threshold'] if silver_perf else None
    gold_perf = evaluate_prediction(gold_results['csd_df'], predictor='cci', threshold=silver_threshold)
    gold_results['performance'] = gold_perf
    
    print("Generating comparison figure...")
    fig_path = os.path.join(args.output, "robustness_comparison.png")
    create_comparison_figure(silver_results, gold_results, eligible_n, fig_path)
    
    print("Generating report...")
    report_path = os.path.join(args.output, "robustness_report.txt")
    report = generate_report(silver_results, gold_results, report_path)
    print("")
    print(report)
    
    silver_results['csd_df'].to_csv(os.path.join(args.output, "silver_csd_results.csv"), index=False)
    gold_results['csd_df'].to_csv(os.path.join(args.output, "gold_csd_results.csv"), index=False)
    
    print(f"\nRobustness validation complete. Results saved to {args.output}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustness Validation: Silver vs Gold Cohort")
    parser.add_argument("--input", type=str, required=True, help="Input cohort CSV file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--outcome", type=str, default="memory", help="Outcome variable name")
    parser.add_argument("--detrend", action="store_true", default=True, help="Detrend before AR(1)/variance")
    parser.add_argument("--no-detrend", dest="detrend", action="store_false", help="Do not detrend")
    
    args = parser.parse_args()
    main(args)
