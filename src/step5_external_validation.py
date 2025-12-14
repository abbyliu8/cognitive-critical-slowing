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


def create_min_wave_cohort(df, outcome_col, min_waves=6, id_col='idauniq', wave_col='wave'):
    wave_counts = df.groupby(id_col).apply(
        lambda x: x.loc[x[outcome_col].notna(), wave_col].nunique()
    )
    eligible_ids = wave_counts[wave_counts >= min_waves].index
    cohort_df = df[df[id_col].isin(eligible_ids)].copy()
    return cohort_df, wave_counts


def compute_trajectory_slope(df, outcome_col, id_col='idauniq', wave_col='wave'):
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
    """Classify into Decline/Stable/Improved based on slope tertiles."""
    valid_slopes = slopes.dropna()
    
    if len(valid_slopes) < 10:
        groups = pd.Series(index=slopes.index, dtype='object')
        groups[:] = np.nan
        return groups, {'q33': np.nan, 'q67': np.nan}
    
    q33 = valid_slopes.quantile(0.333)
    q67 = valid_slopes.quantile(0.667)
    
    groups = pd.Series(index=slopes.index, dtype='object')
    groups[slopes <= q33] = 'Decline'
    groups[(slopes > q33) & (slopes <= q67)] = 'Stable'
    groups[slopes > q67] = 'Improved'
    groups[slopes.isna()] = np.nan
    
    return groups, {'q33': q33, 'q67': q67}


def compute_ar1_ols(y, x, detrend=True):
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


def compute_csd_indicators(df, outcome_col, groups, id_col='idauniq', wave_col='wave', detrend=True):
    results = []
    
    for pid in df[id_col].unique():
        person_data = df[df[id_col] == pid].sort_values(wave_col)
        
        mask = person_data[outcome_col].notna()
        y = person_data.loc[mask, outcome_col].values
        x = person_data.loc[mask, wave_col].values.astype(float)
        
        ar1 = compute_ar1_ols(y, x, detrend=detrend)
        variance = compute_variance_detrended(y, x, detrend=detrend)
        
        group = groups.get(pid, np.nan)
        
        # Use unified column name 'participant_id' for cross-cohort compatibility
        results.append({
            'participant_id': pid,
            'ar1': ar1,
            'variance': variance,
            'trajectory_group': group,
            'n_observations': len(y),
            'waves': str(sorted(x.astype(int).tolist()))
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
    """Compute comprehensive group statistics."""
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
        'group_medians': {g: group_data[g].median() for g in groups if len(group_data[g]) > 0},
        'group_ns': {g: len(group_data[g]) for g in groups}
    }
    
    if len(group_data['Decline']) > 1 and len(group_data['Stable']) > 1:
        d1, d2 = group_data['Decline'], group_data['Stable']
        t_stat, t_p = stats.ttest_ind(d1, d2, equal_var=False)
        u_stat, mw_p = stats.mannwhitneyu(d1, d2, alternative='two-sided')
        
        results['welch_t'] = t_stat
        results['welch_p'] = t_p
        results['mannwhitney_u'] = u_stat
        results['mannwhitney_p'] = mw_p
        results['cohens_d'] = cohens_d(d1, d2)
    
    return results


def evaluate_association(csd_df, predictor='cci'):
    """Evaluate association strength using AUC with fixed threshold."""
    valid_df = csd_df.dropna(subset=[predictor, 'trajectory_group'])
    
    y_true = (valid_df['trajectory_group'] == 'Decline').astype(int)
    y_score = valid_df[predictor]
    
    if y_true.nunique() < 2:
        return None
    
    if y_true.sum() < 2 or (len(y_true) - y_true.sum()) < 2:
        return None
    
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    y_pred = (y_score >= 0).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'auc': auc,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        'threshold_used': 0.0,
        'fpr': fpr, 'tpr': tpr,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def create_visualization(csd_df, ar1_stats, var_stats, association, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    groups = ['Decline', 'Stable', 'Improved']
    colors = {'Decline': '#e74c3c', 'Stable': '#3498db', 'Improved': '#2ecc71'}
    
    # Panel A: AR(1) distribution
    ax = axes[0, 0]
    for group in groups:
        data = csd_df[csd_df['trajectory_group'] == group]['ar1'].dropna()
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.6, label=f'{group} (n={len(data)})', color=colors[group])
    ax.set_xlabel('AR(1) Coefficient (OLS, detrended)')
    ax.set_ylabel('Frequency')
    ax.set_title('A. AR(1) Distribution - ELSA')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.legend()
    
    # Panel B: Variance distribution  
    ax = axes[0, 1]
    for group in groups:
        data = csd_df[csd_df['trajectory_group'] == group]['variance'].dropna()
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.6, label=group, color=colors[group])
    ax.set_xlabel('Variance (detrended)')
    ax.set_ylabel('Frequency')
    ax.set_title('B. Variance Distribution - ELSA')
    ax.legend()
    
    # Panel C: CCI distribution - axvline BEFORE legend
    ax = axes[0, 2]
    for group in groups:
        data = csd_df[csd_df['trajectory_group'] == group]['cci'].dropna()
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.6, label=group, color=colors[group])
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Threshold=0')
    ax.set_xlabel('Comprehensive Criticality Index')
    ax.set_ylabel('Frequency')
    ax.set_title('C. CCI Distribution - ELSA')
    ax.legend()
    
    # Panel D: AR(1) bar plot
    ax = axes[1, 0]
    if ar1_stats:
        means = [ar1_stats['group_means'].get(g, 0) for g in groups]
        sds = [ar1_stats['group_sds'].get(g, 0) for g in groups]
        ns = [ar1_stats['group_ns'].get(g, 0) for g in groups]
        sems = [s / np.sqrt(n) if n > 1 else 0 for s, n in zip(sds, ns)]
        
        ax.bar(groups, means, yerr=sems, color=[colors[g] for g in groups], capsize=5, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        if 'cohens_d' in ar1_stats:
            ax.text(0.95, 0.95, f"Cohen's d = {ar1_stats['cohens_d']:.3f}",
                    transform=ax.transAxes, ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_ylabel('Mean AR(1) +/- SEM')
    ax.set_title('D. AR(1) by Group - ELSA')
    
    # Panel E: ROC curve
    ax = axes[1, 1]
    if association:
        ax.plot(association['fpr'], association['tpr'], 'b-', linewidth=2,
                label=f'ELSA AUC = {association["auc"]:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('E. ROC Curve - ELSA (association)')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Panel F: Cross-cohort comparison table
    ax = axes[1, 2]
    ax.axis('off')
    
    # Summary statistics table
    table_data = [
        ['Metric', 'ELSA'],
        ['N', f"{len(csd_df):,}"],
        ['AR(1) d', f"{ar1_stats['cohens_d']:.3f}" if ar1_stats and 'cohens_d' in ar1_stats else 'N/A'],
        ['AUC', f"{association['auc']:.3f}" if association else 'N/A'],
        ['Sensitivity', f"{association['sensitivity']:.3f}" if association else 'N/A'],
        ['Specificity', f"{association['specificity']:.3f}" if association else 'N/A']
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.4, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    for i in range(len(table_data)):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white', fontweight='bold')
            elif i % 2 == 0:
                cell.set_facecolor('#D6DCE5')
    
    ax.set_title('F. ELSA Validation Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()


def generate_report(csd_df, ar1_stats, var_stats, association, thresholds, output_path=None):
    """Generate ELSA validation report."""
    report = []
    report.append("=" * 70)
    report.append("ELSA EXTERNAL VALIDATION REPORT")
    report.append("Critical Slowing Down Analysis - English Longitudinal Study of Ageing")
    report.append("=" * 70)
    report.append("")
    
    report.append("[METHODOLOGY]")
    report.append("  - Wave-aware detrending: uses actual wave values (not sequential indices)")
    report.append("  - AR(1) via OLS regression (phi parameter)")
    report.append("  - Trajectory classification: tertile-based on memory slope")
    report.append("  - CCI: z-score normalized within cohort")
    report.append("  - Threshold=0 corresponds to cohort-specific mean (relative threshold)")
    report.append("")
    
    report.append("[SAMPLE]")
    report.append(f"  Total with valid CSD: {len(csd_df):,}")
    for g in ['Decline', 'Stable', 'Improved']:
        n = (csd_df['trajectory_group'] == g).sum()
        pct = n / len(csd_df) * 100
        report.append(f"    {g}: {n:,} ({pct:.1f}%)")
    report.append("")
    
    report.append("[TRAJECTORY THRESHOLDS]")
    report.append(f"  Decline: slope <= {thresholds['q33']:.4f}")
    report.append(f"  Improved: slope > {thresholds['q67']:.4f}")
    report.append("")
    
    report.append("[AR(1) AUTOREGRESSIVE COEFFICIENT]")
    if ar1_stats:
        for g in ['Decline', 'Stable', 'Improved']:
            if g in ar1_stats['group_means']:
                m = ar1_stats['group_means'][g]
                s = ar1_stats['group_sds'][g]
                med = ar1_stats['group_medians'][g]
                n = ar1_stats['group_ns'][g]
                report.append(f"    {g}: {m:.3f} +/- {s:.3f}, median {med:.3f} (n={n})")
        
        report.append(f"  Kruskal-Wallis H = {ar1_stats['kruskal_h']:.2f}, p = {ar1_stats['kruskal_p']:.2e}")
        
        if 'cohens_d' in ar1_stats:
            report.append(f"  Decline vs Stable:")
            report.append(f"    Welch t = {ar1_stats['welch_t']:.2f}, p = {ar1_stats['welch_p']:.2e}")
            report.append(f"    Mann-Whitney p = {ar1_stats['mannwhitney_p']:.2e}")
            report.append(f"    Cohen's d = {ar1_stats['cohens_d']:.3f}")
    report.append("")
    
    report.append("[VARIANCE]")
    if var_stats:
        for g in ['Decline', 'Stable', 'Improved']:
            if g in var_stats['group_means']:
                m = var_stats['group_means'][g]
                s = var_stats['group_sds'][g]
                report.append(f"    {g}: {m:.3f} +/- {s:.3f}")
        
        if 'cohens_d' in var_stats:
            report.append(f"  Cohen's d (Decline vs Stable) = {var_stats['cohens_d']:.3f}")
    report.append("")
    
    report.append("[ASSOCIATION STRENGTH]")
    if association:
        report.append(f"  AUC: {association['auc']:.3f}")
        report.append(f"  Threshold used: {association['threshold_used']:.1f} (cohort-specific mean of z-scored CCI)")
        report.append(f"    Sensitivity: {association['sensitivity']:.3f}")
        report.append(f"    Specificity: {association['specificity']:.3f}")
        report.append(f"    PPV: {association['ppv']:.3f}")
        report.append(f"    NPV: {association['npv']:.3f}")
    report.append("")
    
    report.append("[THEORETICAL INTERPRETATION]")
    if ar1_stats and 'cohens_d' in ar1_stats:
        d = ar1_stats['cohens_d']
        effect = "Large" if d > 0.8 else "Medium" if d > 0.5 else "Small" if d > 0.2 else "Negligible"
        
        ar1_d = ar1_stats['group_means'].get('Decline', 0)
        ar1_s = ar1_stats['group_means'].get('Stable', 0)
        
        if ar1_d > ar1_s:
            report.append(f"  [OK] AR(1): Decline > Stable (d={d:.2f}, {effect} effect)")
            report.append("    Consistent with critical slowing down theory")
        else:
            report.append(f"  [!] AR(1): Decline <= Stable (unexpected)")
    report.append("")
    
    report.append("[NOTE ON THRESHOLD INTERPRETATION]")
    report.append("  Threshold=0 is a RELATIVE threshold (cohort-specific z-score mean).")
    report.append("  It is NOT an absolute clinical cutoff that generalizes across cohorts.")
    report.append("  For cross-cohort prediction, consider:")
    report.append("    - Using fixed scaler from training cohort, or")
    report.append("    - Percentile-based thresholds (e.g., top 25%)")
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
    df = load_data(args.input)
    
    outcome_col = args.outcome
    id_col = args.id_col
    wave_col = args.wave_col
    
    if outcome_col not in df.columns:
        if 'memory' in df.columns:
            outcome_col = 'memory'
        else:
            raise ValueError(f"Outcome column not found: {args.outcome}")
    
    print(f"Outcome variable: {outcome_col}")
    print(f"ID column: {id_col}")
    print(f"Wave column: {wave_col}")
    print(f"Detrend: {args.detrend}")
    
    # Check available waves
    waves = sorted(df[wave_col].unique())
    print(f"Available waves: {waves}")
    
    print(f"\nCreating cohort (≥{args.min_waves} distinct non-missing outcome waves)...")
    cohort_df = create_min_wave_cohort(df, outcome_col, min_waves=args.min_waves,
                                             id_col=id_col, wave_col=wave_col)
    n_cohort = cohort_df[id_col].nunique()
    print(f"  Cohort size: {n_cohort:,} participants")
    
    print("Computing trajectory slopes (wave-aware)...")
    slopes = compute_trajectory_slope(cohort_df, outcome_col, id_col=id_col, wave_col=wave_col)
    groups, thresholds = classify_trajectories(slopes)
    
    if groups.isna().all():
        raise ValueError(
            "Trajectory classification failed: insufficient valid slopes. "
            "Consider lowering --min-waves or checking outcome missingness."
        )
    
    print("Trajectory groups:")
    for g in ['Decline', 'Stable', 'Improved']:
        n = (groups == g).sum()
        print(f"  {g}: {n:,}")
    
    print("\nComputing CSD indicators (wave-aware)...")
    csd_df = compute_csd_indicators(cohort_df, outcome_col, groups,
                                     id_col=id_col, wave_col=wave_col, detrend=args.detrend)
    csd_df = compute_cci(csd_df)
    
    print("Computing group statistics...")
    ar1_stats = compute_group_statistics(csd_df, 'ar1')
    var_stats = compute_group_statistics(csd_df, 'variance')
    
    print("Evaluating association...")
    association = evaluate_association(csd_df, predictor='cci')
    if association is None:
        print("[WARNING] AUC not computed (insufficient class counts).")
    
    print("\nGenerating visualization...")
    fig_path = os.path.join(args.output, "elsa_validation.png")
    create_visualization(csd_df, ar1_stats, var_stats, association, fig_path)
    
    print("Generating report...")
    report_path = os.path.join(args.output, "elsa_validation_report.txt")
    report = generate_report(csd_df, ar1_stats, var_stats, association, thresholds, report_path)
    print("")
    print(report)
    
    # Save results
    csd_df.to_csv(os.path.join(args.output, "elsa_csd_results.csv"), index=False)
    
    # Summary
    summary = {
        'cohort': 'ELSA',
        'n_total': len(csd_df),
        'n_decline': (csd_df['trajectory_group'] == 'Decline').sum(),
        'n_stable': (csd_df['trajectory_group'] == 'Stable').sum(),
        'ar1_cohens_d': ar1_stats['cohens_d'] if ar1_stats and 'cohens_d' in ar1_stats else np.nan,
        'auc': association['auc'] if association else np.nan
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.output, "elsa_summary.csv"), index=False)
    
    print(f"\nELSA validation complete. Results saved to {args.output}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ELSA External Validation for Critical Slowing Down",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python step5_external_validation.py --input elsa_data.csv --output results/
  
  # With custom parameters
  python step5_external_validation.py --input elsa.csv --output results/ \\
      --id-col idauniq --wave-col wave --outcome memory --min-waves 6

Outputs:
  - elsa_validation.png/pdf : 6-panel visualization
  - elsa_validation_report.txt : Detailed statistics
  - elsa_csd_results.csv : Individual-level CSD indicators (participant_id column)
  - elsa_summary.csv : Key metrics summary
        """
    )
    parser.add_argument("--input", type=str, required=True, help="Input ELSA CSV file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--outcome", type=str, default="memory", help="Outcome variable name")
    parser.add_argument("--id-col", type=str, default="idauniq", dest="id_col",
                        help="Column name for participant ID (default: idauniq)")
    parser.add_argument("--wave-col", type=str, default="wave", dest="wave_col",
                        help="Column name for wave (default: wave)")
    parser.add_argument("--min-waves", type=int, default=6, dest="min_waves",
                        help="Minimum waves required (default: 6)")
    parser.add_argument("--no-detrend", dest="detrend", action="store_false",
                        help="Do not detrend before CSD calculation")
    parser.set_defaults(detrend=True)
    
    args = parser.parse_args()
    main(args)
