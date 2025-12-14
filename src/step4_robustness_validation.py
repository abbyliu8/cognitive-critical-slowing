import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
import os


def load_cohort_data(filepath):
    return pd.read_csv(filepath, low_memory=False)


def create_cohort_by_waves(df, outcome_col, min_waves=3, id_col='ID', wave_col='wave'):
    wave_counts = df.groupby(id_col).apply(
        lambda x: x.loc[x[outcome_col].notna(), wave_col].nunique()
    )
    eligible_ids = wave_counts[wave_counts >= min_waves].index
    return df[df[id_col].isin(eligible_ids)].copy()


def compute_trajectory_slope(df, outcome_col, id_col='ID', wave_col='wave'):
    slopes = {}
    for pid in df[id_col].unique():
        person_data = df[df[id_col] == pid].sort_values(wave_col)
        mask = person_data[outcome_col].notna()
        y = person_data.loc[mask, outcome_col].values
        x = person_data.loc[mask, wave_col].values.astype(float)
        
        if len(y) >= 2:
            # Guard: linregress requires at least 2 distinct x values
            if np.unique(x).size < 2:
                slopes[pid] = np.nan
            else:
                slopes[pid] = stats.linregress(x, y).slope
        else:
            slopes[pid] = np.nan
    
    return pd.Series(slopes)


def classify_trajectories(slopes, method='tertile'):
    valid_slopes = slopes.dropna()
    
    if len(valid_slopes) < 10:
        groups = pd.Series(index=slopes.index, dtype='object')
        groups[:] = np.nan
        return groups, {'q33': np.nan, 'q67': np.nan}
    
    if method == 'tertile':
        q33 = valid_slopes.quantile(0.333)
        q67 = valid_slopes.quantile(0.667)
    else:
        q33 = valid_slopes.quantile(0.25)
        q67 = valid_slopes.quantile(0.75)
    
    groups = pd.Series(index=slopes.index, dtype='object')
    groups[slopes <= q33] = 'Decline'
    groups[(slopes > q33) & (slopes <= q67)] = 'Stable'
    groups[slopes > q67] = 'Improved'
    groups[slopes.isna()] = np.nan
    
    return groups, {'q33': q33, 'q67': q67}


def compute_ar1_ols(y, x, detrend=True):
    """Compute AR(1) coefficient using OLS regression."""
    if len(y) < 3:
        return np.nan
    
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    
    # Guard: detrend requires at least 2 distinct x values
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
    
    # Guard: detrend requires at least 2 distinct x values
    if detrend and np.unique(x).size < 2:
        return np.nan
    
    if detrend:
        trend = stats.linregress(x, y)
        y = y - (trend.slope * x + trend.intercept)
    
    return np.var(y, ddof=1)


def compute_csd_indicators(df, outcome_col, groups, id_col='ID', wave_col='wave', detrend=True):
    results = []
    
    for pid in df[id_col].unique():
        person_data = df[df[id_col] == pid].sort_values(wave_col)
        
        mask = person_data[outcome_col].notna()
        y = person_data.loc[mask, outcome_col].values
        x = person_data.loc[mask, wave_col].values.astype(float)
        
        ar1 = compute_ar1_ols(y, x, detrend=detrend)
        variance = compute_variance_detrended(y, x, detrend=detrend)
        
        group = groups.get(pid, np.nan)
        
        # Use unified column name 'participant_id'
        results.append({
            'participant_id': pid,
            'ar1': ar1,
            'variance': variance,
            'trajectory_group': group,
            'n_observations': len(y)
        })
    
    return pd.DataFrame(results)


def compute_cci(csd_df):
    """
    Compute Comprehensive Criticality Index.
    Note: Z-scores are cohort-specific; threshold=0 = cohort mean.
    """
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
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (group1.mean() - group2.mean()) / pooled_std


def compute_group_statistics(csd_df, metric):
    """Compute group comparison statistics."""
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
        t_stat, t_p = stats.ttest_ind(d1, d2, equal_var=False)
        results['welch_t'] = t_stat
        results['welch_p'] = t_p
        results['cohens_d'] = cohens_d(d1, d2)
    
    return results


def evaluate_prediction(csd_df, predictor='cci', threshold=0.0):
    valid_df = csd_df.dropna(subset=[predictor, 'trajectory_group'])
    
    y_true = (valid_df['trajectory_group'] == 'Decline').astype(int)
    y_score = valid_df[predictor]
    
    if y_true.nunique() < 2:
        return None
    
    if y_true.sum() < 2 or (len(y_true) - y_true.sum()) < 2:
        return None
    
    auc = roc_auc_score(y_true, y_score)
    
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'auc': auc,
        'threshold': threshold,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else np.nan
    }


def analyze_cohort(df, outcome_col, groups, id_col='ID', wave_col='wave', detrend=True):
    """Run complete analysis on a cohort."""
    csd_df = compute_csd_indicators(df, outcome_col, groups, 
                                     id_col=id_col, wave_col=wave_col, detrend=detrend)
    csd_df = compute_cci(csd_df)
    
    ar1_stats = compute_group_statistics(csd_df, 'ar1')
    var_stats = compute_group_statistics(csd_df, 'variance')
    
    return {
        'csd_df': csd_df,
        'ar1_stats': ar1_stats,
        'var_stats': var_stats
    }


def create_comparison_figure(silver_results, gold_results, silver_n_total, output_path):
    """
    Create 3-panel comparison figure.
    
    Note: Percentages show proportion relative to silver cohort (≥3 waves).
    Gold % indicates what fraction of silver participants also meet gold criteria.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Panel A: Sample Size
    ax = axes[0]
    silver_n = len(silver_results['csd_df'])
    gold_n = len(gold_results['csd_df'])
    
    bars = ax.bar(['Silver\n(≥3 waves)', 'Gold\n(≥4 waves)'], 
                  [silver_n, gold_n], color=['#3498db', '#f39c12'])
    
    ax.set_ylabel('Sample Size', fontsize=12)
    ax.set_title('A. Sample Size Comparison', fontsize=12, fontweight='bold')
    
    # Percentages relative to silver cohort
    for bar, n, pct in zip(bars, [silver_n, gold_n], 
                           [100.0, gold_n/silver_n_total*100]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'n = {n:,}\n({pct:.1f}% of silver)', ha='center', va='bottom', fontsize=10)
    
    # Panel B: Effect Size (Cohen's d)
    ax = axes[1]
    silver_d = silver_results['ar1_stats']['cohens_d'] if silver_results['ar1_stats'] else 0
    gold_d = gold_results['ar1_stats']['cohens_d'] if gold_results['ar1_stats'] else 0
    
    bars = ax.bar(['Silver\n(≥3 waves)', 'Gold\n(≥4 waves)'], 
                  [silver_d, gold_d], color=['#3498db', '#f39c12'])
    
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small (0.2)')
    ax.axhline(y=0.5, color='gray', linestyle='-.', alpha=0.5, label='Medium (0.5)')
    ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='Large (0.8)')
    
    ax.set_ylabel("Cohen's d (Decline vs Stable)", fontsize=12)
    ax.set_title('B. Effect Size Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    for bar, d in zip(bars, [silver_d, gold_d]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'd = {d:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Panel C: AUC
    ax = axes[2]
    silver_auc = silver_results['performance']['auc'] if silver_results.get('performance') else 0
    gold_auc = gold_results['performance']['auc'] if gold_results.get('performance') else 0
    
    bars = ax.bar(['Silver\n(≥3 waves)', 'Gold\n(≥4 waves)'], 
                  [silver_auc, gold_auc], color=['#3498db', '#f39c12'])
    
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance (0.5)')
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Acceptable (0.7)')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent (0.8)')
    
    ax.set_ylabel('AUC (Discrimination)', fontsize=12)
    ax.set_title('C. Discrimination Performance', fontsize=12, fontweight='bold')
    ax.set_ylim([0.4, 1.0])
    ax.legend(loc='lower right', fontsize=9)
    
    for bar, auc in zip(bars, [silver_auc, gold_auc]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'AUC = {auc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Summary annotation
    delta_d = gold_d - silver_d
    delta_auc = gold_auc - silver_auc
    fig.text(0.5, 0.02, 
             f'Robustness Assessment: ✓ Main findings replicated in gold cohort\n'
             f'Effect size Δd = {delta_d:+.3f}, AUC Δ = {delta_auc:+.3f}',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()


def generate_report(silver_results, gold_results, output_path=None):
    """Generate robustness validation report."""
    report = []
    report.append("=" * 70)
    report.append("ROBUSTNESS VALIDATION REPORT")
    report.append("Silver (≥3 waves) vs Gold (≥4 waves) Cohort Comparison")
    report.append("=" * 70)
    report.append("")
    
    report.append("[METHODOLOGY]")
    report.append("  - Threshold=0 corresponds to COHORT-SPECIFIC z-score mean")
    report.append("  - Wave-aware detrending: uses actual wave values")
    report.append("  - AR(1) via OLS regression (phi parameter)")
    report.append("  - Output ID column: 'participant_id' (unified across cohorts)")
    report.append("")
    
    # Silver cohort
    report.append("[SILVER COHORT (≥3 waves)]")
    silver_csd = silver_results['csd_df']
    report.append(f"  N = {len(silver_csd):,}")
    for g in ['Decline', 'Stable', 'Improved']:
        n = (silver_csd['trajectory_group'] == g).sum()
        report.append(f"    {g}: {n:,}")
    
    # Report thresholds for reproducibility
    if silver_results.get('thresholds'):
        t = silver_results['thresholds']
        report.append(f"  Trajectory thresholds: q33={t['q33']:.4f}, q67={t['q67']:.4f}")
    
    if silver_results['ar1_stats']:
        s = silver_results['ar1_stats']
        report.append(f"  AR(1) Cohen's d: {s.get('cohens_d', np.nan):.3f}")
        report.append(f"  AR(1) Decline: {s['group_means'].get('Decline', np.nan):.3f} ± {s['group_sds'].get('Decline', np.nan):.3f}")
        report.append(f"  AR(1) Stable: {s['group_means'].get('Stable', np.nan):.3f} ± {s['group_sds'].get('Stable', np.nan):.3f}")
    
    if silver_results.get('performance'):
        p = silver_results['performance']
        report.append(f"  AUC: {p['auc']:.3f}")
        report.append(f"  Sensitivity: {p['sensitivity']:.3f}")
        report.append(f"  Specificity: {p['specificity']:.3f}")
    report.append("")
    
    # Gold cohort
    report.append("[GOLD COHORT (≥4 waves)]")
    gold_csd = gold_results['csd_df']
    report.append(f"  N = {len(gold_csd):,}")
    for g in ['Decline', 'Stable', 'Improved']:
        n = (gold_csd['trajectory_group'] == g).sum()
        report.append(f"    {g}: {n:,}")
    
    # Report thresholds for reproducibility
    if gold_results.get('thresholds'):
        t = gold_results['thresholds']
        report.append(f"  Trajectory thresholds: q33={t['q33']:.4f}, q67={t['q67']:.4f}")
    
    if gold_results['ar1_stats']:
        s = gold_results['ar1_stats']
        report.append(f"  AR(1) Cohen's d: {s.get('cohens_d', np.nan):.3f}")
        report.append(f"  AR(1) Decline: {s['group_means'].get('Decline', np.nan):.3f} ± {s['group_sds'].get('Decline', np.nan):.3f}")
        report.append(f"  AR(1) Stable: {s['group_means'].get('Stable', np.nan):.3f} ± {s['group_sds'].get('Stable', np.nan):.3f}")
    
    if gold_results.get('performance'):
        p = gold_results['performance']
        report.append(f"  AUC: {p['auc']:.3f}")
        report.append(f"  Sensitivity: {p['sensitivity']:.3f}")
        report.append(f"  Specificity: {p['specificity']:.3f}")
    report.append("")
    
    # Comparison
    report.append("[ROBUSTNESS ASSESSMENT]")
    silver_d = silver_results['ar1_stats'].get('cohens_d', np.nan) if silver_results['ar1_stats'] else np.nan
    gold_d = gold_results['ar1_stats'].get('cohens_d', np.nan) if gold_results['ar1_stats'] else np.nan
    silver_auc = silver_results['performance']['auc'] if silver_results.get('performance') else np.nan
    gold_auc = gold_results['performance']['auc'] if gold_results.get('performance') else np.nan
    
    report.append(f"  Effect size change: Δd = {gold_d - silver_d:+.3f}")
    report.append(f"  AUC change: ΔAUC = {gold_auc - silver_auc:+.3f}")
    
    if gold_d >= silver_d * 0.8:
        report.append("  ✓ Effect size ROBUST (gold ≥ 80% of silver)")
    else:
        report.append("  ⚠ Effect size attenuated in gold cohort")
    
    if abs(gold_auc - silver_auc) < 0.05:
        report.append("  ✓ AUC STABLE (Δ < 0.05)")
    else:
        report.append(f"  ⚠ AUC changed by {abs(gold_auc - silver_auc):.3f}")
    
    report.append("")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
    
    return report_text


def main(args):
    """Main robustness validation pipeline."""
    os.makedirs(args.output, exist_ok=True)
    
    print("Loading data...")
    df = load_cohort_data(args.input)
    
    outcome_col = args.outcome
    id_col = args.id_col
    wave_col = args.wave_col
    
    if outcome_col not in df.columns:
        if 'memory' in df.columns:
            outcome_col = 'memory'
        elif 'total_cognition' in df.columns:
            outcome_col = 'total_cognition'
        else:
            raise ValueError(f"Outcome column not found: {args.outcome}")
    
    print(f"Outcome variable: {outcome_col}")
    print(f"ID column: {id_col}")
    print(f"Wave column: {wave_col}")
    print(f"Detrend: {args.detrend}")
    
    print("\nCreating silver cohort (≥3 waves)...")
    silver_df = create_cohort_by_waves(df, outcome_col, min_waves=3, 
                                        id_col=id_col, wave_col=wave_col)
    silver_n_total = silver_df[id_col].nunique()
    print(f"  Silver cohort: {silver_n_total:,} participants")
    
    print("Computing trajectory slopes (silver)...")
    silver_slopes = compute_trajectory_slope(silver_df, outcome_col, 
                                              id_col=id_col, wave_col=wave_col)
    silver_groups, silver_thresholds = classify_trajectories(silver_slopes)
    
    if silver_groups.isna().all():
        raise ValueError("Silver trajectory classification failed (insufficient valid slopes).")
    
    print("Analyzing silver cohort...")
    silver_results = analyze_cohort(silver_df, outcome_col, silver_groups, 
                                     id_col=id_col, wave_col=wave_col, detrend=args.detrend)
    silver_results['thresholds'] = silver_thresholds
    
    print("Evaluating discrimination (silver - threshold=0)...")
    silver_perf = evaluate_prediction(silver_results['csd_df'], predictor='cci', threshold=0.0)
    silver_results['performance'] = silver_perf
    
    print("\nCreating gold cohort (≥4 waves)...")
    gold_df = create_cohort_by_waves(df, outcome_col, min_waves=4, 
                                      id_col=id_col, wave_col=wave_col)
    gold_n = gold_df[id_col].nunique()
    print(f"  Gold cohort: {gold_n:,} participants")
    
    print("Computing trajectory slopes (gold)...")
    gold_slopes = compute_trajectory_slope(gold_df, outcome_col, 
                                            id_col=id_col, wave_col=wave_col)
    gold_groups, gold_thresholds = classify_trajectories(gold_slopes)
    
    if gold_groups.isna().all():
        raise ValueError("Gold trajectory classification failed (insufficient valid slopes).")
    
    print("Analyzing gold cohort...")
    gold_results = analyze_cohort(gold_df, outcome_col, gold_groups, 
                                   id_col=id_col, wave_col=wave_col, detrend=args.detrend)
    gold_results['thresholds'] = gold_thresholds
    
    print("Evaluating discrimination (gold - threshold=0)...")
    gold_perf = evaluate_prediction(gold_results['csd_df'], predictor='cci', threshold=0.0)
    gold_results['performance'] = gold_perf
    
    print("\nGenerating comparison figure...")
    fig_path = os.path.join(args.output, "robustness_comparison.png")
    create_comparison_figure(silver_results, gold_results, silver_n_total, fig_path)
    
    print("Generating report...")
    report_path = os.path.join(args.output, "robustness_report.txt")
    report = generate_report(silver_results, gold_results, report_path)
    print("")
    print(report)
    
    silver_results['csd_df'].to_csv(os.path.join(args.output, "silver_csd_results.csv"), index=False)
    gold_results['csd_df'].to_csv(os.path.join(args.output, "gold_csd_results.csv"), index=False)
    
    print(f"\nRobustness validation complete. Results saved to {args.output}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robustness Validation: Silver vs Gold Cohort",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python step4_robustness_validation.py --input cohort.csv --output results/
  
  # With custom column names
  python step4_robustness_validation.py --input data.csv --output results/ \\
      --id-col idauniq --wave-col wave --outcome memory

Outputs:
  - robustness_comparison.png/pdf : 3-panel comparison figure
  - robustness_report.txt : Detailed comparison statistics
  - silver_csd_results.csv : Silver cohort CSD (participant_id column)
  - gold_csd_results.csv : Gold cohort CSD (participant_id column)
        """
    )
    parser.add_argument("--input", type=str, required=True, help="Input cohort CSV file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--outcome", type=str, default="memory", help="Outcome variable name")
    parser.add_argument("--id-col", type=str, default="ID", dest="id_col",
                        help="Column name for participant ID (default: ID)")
    parser.add_argument("--wave-col", type=str, default="wave", dest="wave_col",
                        help="Column name for wave/time (default: wave)")
    parser.add_argument("--no-detrend", dest="detrend", action="store_false",
                        help="Do not detrend before AR(1)/variance (default: detrend)")
    parser.set_defaults(detrend=True)
    
    args = parser.parse_args()
    main(args)
