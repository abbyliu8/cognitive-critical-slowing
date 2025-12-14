import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
import os


def load_data(filepath):
    return pd.read_csv(filepath, low_memory=False)


def compute_slope(y, x):
    if len(y) < 2:
        return np.nan
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if np.unique(x).size < 2:
        return np.nan
    result = stats.linregress(x, y)
    return result.slope


def classify_trajectories(slopes, q_lower=0.33, q_upper=0.67):
    valid_slopes = slopes.dropna()
    if len(valid_slopes) < 10:
        return pd.Series(index=slopes.index, dtype='object'), {'q33': np.nan, 'q67': np.nan}
    
    q33 = valid_slopes.quantile(q_lower)
    q67 = valid_slopes.quantile(q_upper)
    
    groups = pd.Series(index=slopes.index, dtype='object')
    groups[slopes <= q33] = 'Decline'
    groups[(slopes > q33) & (slopes <= q67)] = 'Stable'
    groups[slopes > q67] = 'Improved'
    
    return groups, {'q33': q33, 'q67': q67}


def create_min_wave_cohort(df, outcome_col, min_waves, id_col='ID', wave_col='wave'):
    wave_counts = df.groupby(id_col).apply(
        lambda g: g[[wave_col, outcome_col]].dropna()[wave_col].nunique()
    )
    eligible_ids = wave_counts[wave_counts >= min_waves].index
    cohort_df = df[df[id_col].isin(eligible_ids)].copy()
    return cohort_df, wave_counts


def compute_ar1_ols(y, x, detrend=True):
    if len(y) < 3:
        return np.nan
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if detrend and np.unique(x).size < 2:
        return np.nan
    if detrend:
        trend = stats.linregress(x, y)
        y = y - (trend.slope * x + trend.intercept)
    y_t = y[1:]
    y_t1 = y[:-1]
    if np.std(y_t1) == 0:
        return np.nan
    result = stats.linregress(y_t1, y_t)
    return result.slope


def compute_variance_detrended(y, x, detrend=True):
    if len(y) < 2:
        return np.nan
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if detrend and np.unique(x).size < 2:
        return np.nan
    if detrend:
        trend = stats.linregress(x, y)
        y = y - (trend.slope * x + trend.intercept)
    return np.var(y, ddof=1)


def compute_csd_indicators(df, outcome_col, groups, id_col='ID', wave_col='wave'):
    results = []
    for pid in df[id_col].unique():
        person_data = df[df[id_col] == pid].sort_values(wave_col)
        mask = person_data[outcome_col].notna()
        y = person_data.loc[mask, outcome_col].values
        x = person_data.loc[mask, wave_col].values.astype(float)
        
        ar1 = compute_ar1_ols(y, x, detrend=True)
        variance = compute_variance_detrended(y, x, detrend=True)
        group = groups.get(pid, np.nan)
        
        results.append({
            'participant_id': pid,
            'ar1': ar1,
            'variance': variance,
            'trajectory_group': group,
            'n_observations': len(y)
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


def group_statistics(csd_df, metric):
    groups = ['Decline', 'Stable', 'Improved']
    group_data = {g: csd_df[csd_df['trajectory_group'] == g][metric].dropna() for g in groups}
    
    valid_groups = [group_data[g] for g in groups if len(group_data[g]) > 1]
    if len(valid_groups) < 2:
        return None
    
    h_stat, kw_p = stats.kruskal(*valid_groups)
    
    results = {
        'kruskal_h': h_stat,
        'kruskal_p': kw_p,
        'group_means': {g: group_data[g].mean() for g in groups if len(group_data[g]) > 0},
        'group_sds': {g: group_data[g].std() for g in groups if len(group_data[g]) > 0},
        'group_ns': {g: len(group_data[g]) for g in groups}
    }
    
    if len(group_data['Decline']) > 1 and len(group_data['Stable']) > 1:
        d1, d2 = group_data['Decline'], group_data['Stable']
        t_stat, welch_p = stats.ttest_ind(d1, d2, equal_var=False)
        u_stat, mw_p = stats.mannwhitneyu(d1, d2, alternative='two-sided')
        results['welch_t'] = t_stat
        results['welch_p'] = welch_p
        results['mannwhitney_u'] = u_stat
        results['mannwhitney_p'] = mw_p
        results['cohens_d'] = cohens_d(d1, d2)
    
    return results


def evaluate_prediction(csd_df, predictor='cci', target_group='Decline'):
    valid_df = csd_df.dropna(subset=[predictor, 'trajectory_group'])
    y_true = (valid_df['trajectory_group'] == target_group).astype(int)
    y_score = valid_df[predictor]
    
    if y_true.nunique() < 2:
        return None
    if y_true.sum() < 2 or (len(y_true) - y_true.sum()) < 2:
        return None
    
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    y_pred = (y_score >= 0.0).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'auc': auc,
        'threshold_used': 0.0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        'fpr': fpr, 'tpr': tpr,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def analyze_cohort(df, outcome_col, groups, id_col='ID', wave_col='wave'):
    csd_df = compute_csd_indicators(df, outcome_col, groups, id_col, wave_col)
    csd_df = compute_cci(csd_df)
    
    ar1_stats = group_statistics(csd_df, 'ar1')
    var_stats = group_statistics(csd_df, 'variance')
    performance = evaluate_prediction(csd_df, predictor='cci')
    
    return {
        'csd_df': csd_df,
        'ar1_stats': ar1_stats,
        'var_stats': var_stats,
        'performance': performance
    }


def create_comparison_figure(silver_results, gold_results, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    silver_n = len(silver_results['csd_df'])
    gold_n = len(gold_results['csd_df'])
    pct_gold = gold_n / silver_n * 100 if silver_n > 0 else np.nan
    
    ax = axes[0]
    bars = ax.bar(['Silver\n(>=3 waves)', 'Gold\n(>=4 waves)'], [silver_n, gold_n],
                  color=['#3498db', '#f39c12'])
    ax.set_ylabel('Sample Size (valid CSD)')
    ax.set_title('A. Sample Size Comparison')
    ax.text(bars[0].get_x() + bars[0].get_width()/2, bars[0].get_height() + 50,
            f'n = {silver_n:,}', ha='center', va='bottom')
    ax.text(bars[1].get_x() + bars[1].get_width()/2, bars[1].get_height() + 50,
            f'n = {gold_n:,}\n({pct_gold:.1f}%)', ha='center', va='bottom')
    
    silver_d = silver_results['ar1_stats']['cohens_d'] if silver_results['ar1_stats'] else np.nan
    gold_d = gold_results['ar1_stats']['cohens_d'] if gold_results['ar1_stats'] else np.nan
    
    ax = axes[1]
    x_pos = [0, 1]
    heights = [silver_d if not np.isnan(silver_d) else 0, gold_d if not np.isnan(gold_d) else 0]
    bars = ax.bar(x_pos, heights, color=['#3498db', '#f39c12'], width=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Silver\n(>=3 waves)', 'Gold\n(>=4 waves)'])
    ax.set_ylabel("Cohen's d (Decline vs Stable)")
    ax.set_title("B. Effect Size Comparison")
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small (0.2)')
    ax.axhline(y=0.5, color='gray', linestyle='-.', alpha=0.5, label='Medium (0.5)')
    ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='Large (0.8)')
    ax.legend(loc='upper right', fontsize=8)
    for i, (bar, val) in enumerate(zip(bars, [silver_d, gold_d])):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'd = {val:.3f}', ha='center', va='bottom')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, 0.1, 'N/A', ha='center', va='bottom')
    
    silver_auc = silver_results['performance']['auc'] if silver_results.get('performance') else np.nan
    gold_auc = gold_results['performance']['auc'] if gold_results.get('performance') else np.nan
    
    ax = axes[2]
    x_pos = [0, 1]
    heights = [silver_auc if not np.isnan(silver_auc) else 0, gold_auc if not np.isnan(gold_auc) else 0]
    bars = ax.bar(x_pos, heights, color=['#3498db', '#f39c12'], width=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Silver\n(>=3 waves)', 'Gold\n(>=4 waves)'])
    ax.set_ylabel('AUC (Decline discrimination)')
    ax.set_title('C. Discriminative Performance')
    ax.set_ylim([0.4, 1.0])
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance (0.5)')
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Acceptable (0.7)')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent (0.8)')
    ax.legend(loc='lower right', fontsize=8)
    for i, (bar, val) in enumerate(zip(bars, [silver_auc, gold_auc])):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'AUC = {val:.3f}', ha='center', va='bottom')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, 0.55, 'N/A', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()


def generate_report(silver_results, gold_results, silver_thresholds, gold_thresholds, output_path):
    report = []
    report.append("=" * 70)
    report.append("ROBUSTNESS VALIDATION REPORT: Silver vs Gold Cohort")
    report.append("=" * 70)
    report.append("")
    
    silver_n = len(silver_results['csd_df'])
    gold_n = len(gold_results['csd_df'])
    
    report.append("[COHORT SIZES]")
    report.append(f"  Silver (>=3 waves, valid CSD): {silver_n:,}")
    report.append(f"  Gold (>=4 waves, valid CSD): {gold_n:,}")
    report.append(f"  Retention: {gold_n/silver_n*100:.1f}%")
    report.append("")
    
    report.append("[TRAJECTORY THRESHOLDS]")
    report.append(f"  Silver: q33={silver_thresholds['q33']:.4f}, q67={silver_thresholds['q67']:.4f}")
    report.append(f"  Gold: applied from Silver (threshold freeze)")
    report.append("")
    
    report.append("[AR(1) COMPARISON]")
    for name, res in [('Silver', silver_results), ('Gold', gold_results)]:
        if res['ar1_stats']:
            s = res['ar1_stats']
            report.append(f"  {name}:")
            report.append(f"    Decline: {s['group_means'].get('Decline', np.nan):.3f} +/- {s['group_sds'].get('Decline', np.nan):.3f}")
            report.append(f"    Stable:  {s['group_means'].get('Stable', np.nan):.3f} +/- {s['group_sds'].get('Stable', np.nan):.3f}")
            if 'cohens_d' in s:
                report.append(f"    Cohen's d: {s['cohens_d']:.3f}")
    report.append("")
    
    report.append("[DISCRIMINATION COMPARISON]")
    for name, res in [('Silver', silver_results), ('Gold', gold_results)]:
        if res.get('performance'):
            p = res['performance']
            report.append(f"  {name}:")
            report.append(f"    AUC: {p['auc']:.3f}")
            report.append(f"    Sensitivity: {p['sensitivity']:.3f}")
            report.append(f"    Specificity: {p['specificity']:.3f}")
            report.append(f"    NPV: {p['npv']:.3f}")
        else:
            report.append(f"  {name}: AUC not available")
    report.append("")
    
    report.append("[ROBUSTNESS ASSESSMENT]")
    if silver_results['ar1_stats'] and gold_results['ar1_stats']:
        d_silver = silver_results['ar1_stats'].get('cohens_d', np.nan)
        d_gold = gold_results['ar1_stats'].get('cohens_d', np.nan)
        if not np.isnan(d_silver) and not np.isnan(d_gold):
            delta_d = d_gold - d_silver
            report.append(f"  Effect size change: {delta_d:+.3f}")
            if delta_d >= 0:
                report.append("  Interpretation: Effect maintained or strengthened in gold cohort")
            else:
                report.append("  Interpretation: Effect attenuated in gold cohort")
    
    if silver_results.get('performance') and gold_results.get('performance'):
        auc_silver = silver_results['performance']['auc']
        auc_gold = gold_results['performance']['auc']
        delta_auc = auc_gold - auc_silver
        report.append(f"  AUC change: {delta_auc:+.3f}")
    
    report.append("")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    with open(output_path, 'w') as f:
        f.write(report_text)
    return report_text


def main(args):
    os.makedirs(args.output, exist_ok=True)
    
    print("Loading data...")
    df = load_data(args.input)
    
    outcome_col = args.outcome
    if outcome_col not in df.columns:
        if 'memory' in df.columns:
            outcome_col = 'memory'
        elif 'total_cognition' in df.columns:
            outcome_col = 'total_cognition'
        else:
            raise ValueError(f"Outcome column not found: {args.outcome}")
    
    id_col = args.id_col
    wave_col = args.wave_col
    
    print(f"Outcome: {outcome_col}")
    print(f"ID column: {id_col}")
    print(f"Wave column: {wave_col}")
    
    print("\nCreating Silver cohort (>=3 waves)...")
    silver_df, silver_wave_counts = create_min_wave_cohort(df, outcome_col, min_waves=3, 
                                                            id_col=id_col, wave_col=wave_col)
    silver_n_total = silver_df[id_col].nunique()
    print(f"  Silver eligible: {silver_n_total:,}")
    
    print("Computing slopes for Silver cohort...")
    silver_slopes = silver_df.groupby(id_col).apply(
        lambda g: compute_slope(
            g.sort_values(wave_col)[outcome_col].dropna().values,
            g.sort_values(wave_col).loc[g[outcome_col].notna(), wave_col].values.astype(float)
        )
    )
    
    print("Classifying trajectories (Silver)...")
    silver_groups, silver_thresholds = classify_trajectories(silver_slopes)
    if silver_groups.isna().all():
        raise ValueError("Silver trajectory classification failed (insufficient valid slopes).")
    print(f"  Thresholds: q33={silver_thresholds['q33']:.4f}, q67={silver_thresholds['q67']:.4f}")
    
    print("\nCreating Gold cohort (>=4 waves)...")
    gold_df, gold_wave_counts = create_min_wave_cohort(df, outcome_col, min_waves=4,
                                                        id_col=id_col, wave_col=wave_col)
    gold_n_total = gold_df[id_col].nunique()
    print(f"  Gold eligible: {gold_n_total:,}")
    
    print("Computing slopes for Gold cohort...")
    gold_slopes = gold_df.groupby(id_col).apply(
        lambda g: compute_slope(
            g.sort_values(wave_col)[outcome_col].dropna().values,
            g.sort_values(wave_col).loc[g[outcome_col].notna(), wave_col].values.astype(float)
        )
    )
    
    print("Applying Silver thresholds to Gold (threshold freeze)...")
    gold_groups = pd.Series(index=gold_slopes.index, dtype='object')
    gold_groups[gold_slopes <= silver_thresholds['q33']] = 'Decline'
    gold_groups[(gold_slopes > silver_thresholds['q33']) & (gold_slopes <= silver_thresholds['q67'])] = 'Stable'
    gold_groups[gold_slopes > silver_thresholds['q67']] = 'Improved'
    
    if gold_groups.isna().all():
        raise ValueError("Gold trajectory classification failed (insufficient valid slopes).")
    
    print("\nAnalyzing Silver cohort...")
    silver_results = analyze_cohort(silver_df, outcome_col, silver_groups, id_col, wave_col)
    silver_results['thresholds'] = silver_thresholds
    if silver_results['performance'] is None:
        print("[WARNING] Silver AUC not computed (insufficient class counts).")
    
    print("Analyzing Gold cohort...")
    gold_results = analyze_cohort(gold_df, outcome_col, gold_groups, id_col, wave_col)
    gold_results['thresholds'] = silver_thresholds
    if gold_results['performance'] is None:
        print("[WARNING] Gold AUC not computed (insufficient class counts).")
    
    print("\nGenerating comparison figure...")
    fig_path = os.path.join(args.output, "robustness_comparison.png")
    create_comparison_figure(silver_results, gold_results, fig_path)
    
    print("Generating report...")
    report_path = os.path.join(args.output, "robustness_report.txt")
    report = generate_report(silver_results, gold_results, silver_thresholds, 
                              silver_thresholds, report_path)
    print("")
    print(report)
    
    silver_results['csd_df'].to_csv(os.path.join(args.output, "silver_csd_results.csv"), index=False)
    gold_results['csd_df'].to_csv(os.path.join(args.output, "gold_csd_results.csv"), index=False)
    
    print(f"\nAnalysis complete. Results saved to {args.output}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--outcome", type=str, default="memory")
    parser.add_argument("--id-col", type=str, default="ID", dest="id_col")
    parser.add_argument("--wave-col", type=str, default="wave", dest="wave_col")
    
    args = parser.parse_args()
    main(args)
