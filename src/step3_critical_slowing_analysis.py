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


def validate_input_data(df, id_col='ID', wave_col='wave', outcome_col='memory'):
    required_cols = [id_col, wave_col, outcome_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    if 'trajectory_group' not in df.columns:
        raise ValueError("trajectory_group column not found. Run step2 first.")
    
    group_per_id = df.groupby(id_col)['trajectory_group'].nunique()
    inconsistent = group_per_id[group_per_id > 1]
    if len(inconsistent) > 0:
        raise ValueError(f"{len(inconsistent)} IDs have inconsistent trajectory_group.")
    
    if df['trajectory_group'].isna().any():
        n_missing = df[df['trajectory_group'].isna()][id_col].nunique()
        raise ValueError(f"trajectory_group missing for {n_missing} participants.")
    
    id_groups = df.sort_values([id_col, wave_col]).groupby(id_col)['trajectory_group'].first()
    counts = id_groups.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError(f"Insufficient group sizes: {counts.to_dict()}")
    
    print(f"Validated: {df[id_col].nunique():,} participants")
    print(f"  Groups: {counts.to_dict()}")


def compute_ar1_ols(y, x, detrend=True):
    """
    Compute AR(1) coefficient using OLS regression (phi parameter).
    
    Parameters
    ----------
    y : array-like
        Outcome values (already sorted by time)
    x : array-like
        Actual wave/time values (same length as y)
    detrend : bool
        If True, remove linear trend before AR(1) estimation
    
    Returns
    -------
    float
        AR(1) phi coefficient, or NaN if insufficient data
    
    Notes
    -----
    Uses OLS regression: y_t = alpha + phi * y_{t-1} + epsilon
    This estimates the phi parameter directly, unlike correlation which
    conflates trend effects when the mean drifts over time.
    """
    if len(y) < 3:
        return np.nan
    
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    
    # Guard: detrend requires at least 2 distinct x values
    if detrend and np.unique(x).size < 2:
        return np.nan
    
    if detrend:
        # Detrend using actual wave values (not sequential indices)
        trend = stats.linregress(x, y)
        y = y - (trend.slope * x + trend.intercept)
    
    y_t = y[1:]
    y_t1 = y[:-1]
    
    if np.std(y_t1) == 0:
        return np.nan
    
    # OLS: y_t = alpha + phi * y_{t-1} + epsilon
    result = stats.linregress(y_t1, y_t)
    return result.slope


def compute_variance_detrended(y, x, detrend=True):
    """
    Compute variance, optionally after detrending.
    
    Parameters
    ----------
    y : array-like
        Outcome values
    x : array-like
        Actual wave/time values
    detrend : bool
        If True, compute variance of residuals after removing linear trend
    
    Returns
    -------
    float
        Variance (ddof=1), or NaN if insufficient data
    """
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


def compute_csd_indicators(df, outcome_col, id_col='ID', wave_col='wave', detrend=True):
    """
    Compute critical slowing down indicators for each participant.
    Uses actual wave values for detrending (not sequential indices).
    
    Parameters
    ----------
    df : DataFrame
        Longitudinal cohort data with trajectory_group column
    outcome_col : str
        Name of the outcome variable (e.g., 'memory')
    id_col : str
        Name of the participant ID column
    wave_col : str
        Name of the wave/time column
    detrend : bool
        Whether to detrend before computing AR(1) and variance
    
    Returns
    -------
    DataFrame
        One row per participant with columns:
        - participant_id: unified ID column name for cross-cohort compatibility
        - ar1, variance, trajectory_group, n_observations, wave_min, wave_max
    """
    results = []
    
    for pid in df[id_col].unique():
        person_data = df[df[id_col] == pid].sort_values(wave_col)
        
        # Get non-missing outcome and corresponding waves
        mask = person_data[outcome_col].notna()
        y = person_data.loc[mask, outcome_col].values
        x = person_data.loc[mask, wave_col].values.astype(float)
        
        ar1 = compute_ar1_ols(y, x, detrend=detrend)
        variance = compute_variance_detrended(y, x, detrend=detrend)
        
        # trajectory_group is guaranteed unique per ID (validated earlier)
        group = person_data['trajectory_group'].iloc[0]
        
        # Use unified column name 'participant_id' for cross-cohort compatibility
        results.append({
            'participant_id': pid,
            'ar1': ar1,
            'variance': variance,
            'trajectory_group': group,
            'n_observations': len(y),
            'wave_min': x.min() if len(x) > 0 else np.nan,
            'wave_max': x.max() if len(x) > 0 else np.nan
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


def group_comparison_statistics(csd_df, metric, group_col='trajectory_group'):
    """
    Compute group comparison statistics using both parametric and non-parametric tests.
    
    Returns dictionary with:
    - Kruskal-Wallis H (3-group non-parametric)
    - ANOVA F (3-group parametric, for reference)
    - Welch t-test (Decline vs Stable, unequal variance)
    - Mann-Whitney U (Decline vs Stable, non-parametric)
    - Cohen's d (effect size)
    - Group means, SDs, medians, IQRs
    """
    groups = ['Decline', 'Stable', 'Improved']
    group_data = {g: csd_df[csd_df[group_col] == g][metric].dropna() for g in groups}
    
    valid_groups = [group_data[g] for g in groups if len(group_data[g]) > 1]
    if len(valid_groups) < 2:
        return None
    
    # Non-parametric: Kruskal-Wallis
    h_stat, kw_pvalue = stats.kruskal(*valid_groups)
    
    # Parametric: ANOVA (for reference)
    f_stat, f_pvalue = stats.f_oneway(*valid_groups)
    
    results = {
        'kruskal_h': h_stat,
        'kruskal_p': kw_pvalue,
        'f_statistic': f_stat,
        'f_pvalue': f_pvalue,
        'group_means': {g: group_data[g].mean() for g in groups if len(group_data[g]) > 0},
        'group_sds': {g: group_data[g].std() for g in groups if len(group_data[g]) > 0},
        'group_medians': {g: group_data[g].median() for g in groups if len(group_data[g]) > 0},
        'group_iqrs': {g: group_data[g].quantile(0.75) - group_data[g].quantile(0.25) 
                       for g in groups if len(group_data[g]) > 0},
        'group_ns': {g: len(group_data[g]) for g in groups}
    }
    
    # Pairwise: Decline vs Stable
    if len(group_data['Decline']) > 1 and len(group_data['Stable']) > 1:
        d1, d2 = group_data['Decline'], group_data['Stable']
        
        t_stat, t_pvalue = stats.ttest_ind(d1, d2, equal_var=False)  # Welch
        u_stat, mw_pvalue = stats.mannwhitneyu(d1, d2, alternative='two-sided')
        
        results['welch_t'] = t_stat
        results['welch_p'] = t_pvalue
        results['mannwhitney_u'] = u_stat
        results['mannwhitney_p'] = mw_pvalue
        results['cohens_d'] = cohens_d(d1, d2)
    
    return results


def evaluate_association_strength(csd_df, predictor='cci', target_group='Decline'):
    """
    Evaluate association between CCI and trajectory group using AUC.
    
    Note: This is association/discrimination, not prospective prediction.
    AUC quantifies how well CCI discriminates between groups defined on the same series.
    """
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


def create_visualization(csd_df, association, output_path):
    """Create 6-panel visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    groups = ['Decline', 'Stable', 'Improved']
    colors = {'Decline': '#e74c3c', 'Stable': '#3498db', 'Improved': '#2ecc71'}
    
    # Panel A: AR(1) distribution
    ax = axes[0, 0]
    for group in groups:
        data = csd_df[csd_df['trajectory_group'] == group]['ar1'].dropna()
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.6, label=f'{group} (n={len(data)})', color=colors[group])
    ax.set_xlabel('AR(1) Coefficient (OLS phi, detrended)')
    ax.set_ylabel('Frequency')
    ax.set_title('A. AR(1) Distribution by Group')
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
    ax.set_title('B. Variance Distribution by Group')
    ax.legend()
    
    # Panel C: CCI distribution - threshold line BEFORE legend
    ax = axes[0, 2]
    for group in groups:
        data = csd_df[csd_df['trajectory_group'] == group]['cci'].dropna()
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.6, label=group, color=colors[group])
    ax.set_xlabel('Comprehensive Criticality Index')
    ax.set_ylabel('Frequency')
    ax.set_title('C. CCI Distribution by Group')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Threshold=0')
    ax.legend()
    
    # Panel D: AR(1) bar plot
    ax = axes[1, 0]
    group_means = []
    group_sems = []
    for g in groups:
        data = csd_df[csd_df['trajectory_group'] == g]['ar1'].dropna()
        group_means.append(data.mean() if len(data) > 0 else 0)
        group_sems.append(data.sem() if len(data) > 1 else 0)
    
    ax.bar(groups, group_means, yerr=group_sems, 
           color=[colors[g] for g in groups], capsize=5, alpha=0.8)
    ax.set_ylabel('Mean AR(1) ± SEM')
    ax.set_title('D. AR(1) by Trajectory Group')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Panel E: ROC curve
    ax = axes[1, 1]
    if association:
        ax.plot(association['fpr'], association['tpr'], 'b-', linewidth=2, 
                label=f'AUC = {association["auc"]:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('E. ROC Curve (CCI vs Decline)')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.text(0.55, 0.15, 'Association strength\n(not prospective prediction)', 
            fontsize=9, style='italic', alpha=0.7)
    
    # Panel F: Variance bar plot
    ax = axes[1, 2]
    group_means = []
    group_sems = []
    for g in groups:
        data = csd_df[csd_df['trajectory_group'] == g]['variance'].dropna()
        group_means.append(data.mean() if len(data) > 0 else 0)
        group_sems.append(data.sem() if len(data) > 1 else 0)
    
    ax.bar(groups, group_means, yerr=group_sems, 
           color=[colors[g] for g in groups], capsize=5, alpha=0.8)
    ax.set_ylabel('Mean Variance ± SEM')
    ax.set_title('F. Variance by Trajectory Group')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()


def generate_analysis_report(csd_df, ar1_stats, var_stats, cci_stats, association, detrend, output_path=None):
    """Generate comprehensive analysis report."""
    report = []
    report.append("=" * 70)
    report.append("CRITICAL SLOWING DOWN ANALYSIS REPORT")
    report.append("Association between CSD indicators and cognitive trajectory")
    report.append("=" * 70)
    report.append("")
    
    report.append("[METHODOLOGICAL NOTES]")
    report.append("  - AR(1) estimated via OLS regression: y_t = alpha + phi*y_{t-1} + e")
    report.append(f"  - Detrending before CSD calculation: {'Yes (using actual wave values)' if detrend else 'No'}")
    report.append("  - Statistical tests: Kruskal-Wallis (3-group), Welch t-test (pairwise)")
    report.append("  - AUC represents ASSOCIATION strength, NOT prospective prediction")
    report.append("  - For prospective prediction, use prefix-window design (step3b)")
    report.append("")
    
    report.append("[SAMPLE]")
    n_total = len(csd_df)
    report.append(f"Total participants with valid CSD: {n_total:,}")
    for group in ['Decline', 'Stable', 'Improved']:
        n = (csd_df['trajectory_group'] == group).sum()
        pct = n / n_total * 100 if n_total > 0 else 0
        report.append(f"  {group}: {n:,} ({pct:.1f}%)")
    report.append("")
    
    report.append("[AR(1) AUTOREGRESSIVE COEFFICIENT]")
    if ar1_stats:
        report.append("  Group statistics:")
        for group in ['Decline', 'Stable', 'Improved']:
            if group in ar1_stats['group_means']:
                m = ar1_stats['group_means'][group]
                s = ar1_stats['group_sds'][group]
                med = ar1_stats['group_medians'][group]
                iqr = ar1_stats['group_iqrs'][group]
                n = ar1_stats['group_ns'][group]
                report.append(f"    {group}: {m:.3f} +/- {s:.3f}, median {med:.3f} [IQR {iqr:.3f}] (n={n})")
        report.append(f"  Kruskal-Wallis H = {ar1_stats['kruskal_h']:.2f}, p = {ar1_stats['kruskal_p']:.2e}")
        if 'welch_t' in ar1_stats:
            report.append(f"  Decline vs Stable:")
            report.append(f"    Welch t = {ar1_stats['welch_t']:.2f}, p = {ar1_stats['welch_p']:.2e}")
            report.append(f"    Mann-Whitney p = {ar1_stats['mannwhitney_p']:.2e}")
            report.append(f"    Cohen's d = {ar1_stats['cohens_d']:.3f}")
    report.append("")
    
    report.append("[VARIANCE]")
    if var_stats:
        for group in ['Decline', 'Stable', 'Improved']:
            if group in var_stats['group_means']:
                m = var_stats['group_means'][group]
                s = var_stats['group_sds'][group]
                med = var_stats['group_medians'][group]
                iqr = var_stats['group_iqrs'][group]
                report.append(f"    {group}: {m:.3f} +/- {s:.3f}, median {med:.3f} [IQR {iqr:.3f}]")
        report.append(f"  Kruskal-Wallis H = {var_stats['kruskal_h']:.2f}, p = {var_stats['kruskal_p']:.2e}")
        if 'cohens_d' in var_stats:
            report.append(f"  Cohen's d (Decline vs Stable) = {var_stats['cohens_d']:.3f}")
    report.append("")
    
    report.append("[COMPREHENSIVE CRITICALITY INDEX (CCI)]")
    report.append("  CCI = (z_AR1 + z_Variance) / 2")
    if cci_stats:
        for group in ['Decline', 'Stable', 'Improved']:
            if group in cci_stats['group_means']:
                m = cci_stats['group_means'][group]
                s = cci_stats['group_sds'][group]
                report.append(f"    {group}: {m:.3f} +/- {s:.3f}")
        report.append(f"  Kruskal-Wallis H = {cci_stats['kruskal_h']:.2f}, p = {cci_stats['kruskal_p']:.2e}")
    report.append("")
    
    report.append("[ASSOCIATION STRENGTH]")
    report.append("  NOTE: AUC quantifies association, NOT prospective prediction accuracy.")
    report.append("        CCI and trajectory_group are computed from the SAME time series.")
    report.append("  IMPORTANT: Threshold=0 corresponds to the COHORT-SPECIFIC mean of")
    report.append("             the standardized CCI, not an absolute clinical threshold.")
    report.append("             For cross-cohort comparison, use fixed scaler or percentile cutoffs.")
    if association:
        report.append(f"  AUC: {association['auc']:.3f}")
        report.append(f"  At fixed threshold = 0 (cohort-specific z-score mean):")
        report.append(f"    Sensitivity: {association['sensitivity']:.3f}")
        report.append(f"    Specificity: {association['specificity']:.3f}")
    report.append("")
    
    report.append("[THEORETICAL INTERPRETATION]")
    if ar1_stats and 'cohens_d' in ar1_stats:
        d = ar1_stats['cohens_d']
        effect_label = "Large" if d > 0.8 else "Medium" if d > 0.5 else "Small" if d > 0.2 else "Negligible"
        
        ar1_decline = ar1_stats['group_means'].get('Decline', 0)
        ar1_stable = ar1_stats['group_means'].get('Stable', 0)
        
        if ar1_decline > ar1_stable:
            report.append(f"  AR(1): Decline > Stable (d={d:.2f}, {effect_label} effect)")
            report.append("  Interpretation: Decline group shows HIGHER autocorrelation,")
            report.append("                  consistent with critical slowing down theory.")
            report.append("                  (Reduced resilience = slower recovery from perturbations)")
        else:
            report.append(f"  AR(1): Decline <= Stable (unexpected direction)")
    
    report.append("")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
    
    return report_text


def main(args):
    os.makedirs(args.output, exist_ok=True)
    
    print("Loading cohort data...")
    df = load_cohort_data(args.input)
    
    outcome_col = args.outcome
    if outcome_col not in df.columns:
        if 'memory' in df.columns:
            outcome_col = 'memory'
        elif 'total_cognition' in df.columns:
            outcome_col = 'total_cognition'
        else:
            raise ValueError(f"Outcome column not found: {args.outcome}")
    
    print(f"Outcome variable: {outcome_col}")
    print(f"ID column: {args.id_col}")
    print(f"Wave column: {args.wave_col}")
    print(f"Detrend: {args.detrend}")
    
    print("\nValidating input data...")
    validate_input_data(df, id_col=args.id_col, wave_col=args.wave_col, outcome_col=outcome_col)
    
    print("\nComputing critical slowing down indicators (wave-aware)...")
    csd_df = compute_csd_indicators(df, outcome_col, id_col=args.id_col, 
                                     wave_col=args.wave_col, detrend=args.detrend)
    
    print("Computing Comprehensive Criticality Index...")
    csd_df = compute_cci(csd_df)
    
    # Check group sizes after CSD filtering (may differ from input due to NaN drops)
    final_counts = csd_df['trajectory_group'].value_counts()
    if len(final_counts) < 2 or final_counts.min() < 2:
        print(f"[WARNING] Group sizes after CSD filtering are small: {final_counts.to_dict()}")
    
    print("Computing group comparison statistics...")
    ar1_stats = group_comparison_statistics(csd_df, 'ar1')
    var_stats = group_comparison_statistics(csd_df, 'variance')
    cci_stats = group_comparison_statistics(csd_df, 'cci')
    
    if ar1_stats is None or var_stats is None:
        print("[WARNING] Some statistics not computed (insufficient group sizes).")
    
    print("Evaluating association strength...")
    association = evaluate_association_strength(csd_df, predictor='cci')
    if association is None:
        print("[WARNING] AUC not computed (insufficient class counts).")
    
    print("Generating visualization...")
    fig_path = os.path.join(args.output, "csd_analysis.png")
    create_visualization(csd_df, association, fig_path)
    
    print("Generating report...")
    report_path = os.path.join(args.output, "csd_analysis_report.txt")
    report = generate_analysis_report(csd_df, ar1_stats, var_stats, cci_stats, 
                                       association, args.detrend, report_path)
    print("")
    print(report)
    
    # Save results
    results_path = os.path.join(args.output, "csd_results.csv")
    csd_df.to_csv(results_path, index=False)
    
    # Save summary
    summary = {
        'n_total': len(csd_df),
        'n_decline': (csd_df['trajectory_group'] == 'Decline').sum(),
        'n_stable': (csd_df['trajectory_group'] == 'Stable').sum(),
        'ar1_cohens_d': ar1_stats['cohens_d'] if ar1_stats and 'cohens_d' in ar1_stats else np.nan,
        'var_cohens_d': var_stats['cohens_d'] if var_stats and 'cohens_d' in var_stats else np.nan,
        'auc': association['auc'] if association else np.nan
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.output, "csd_summary.csv"), index=False)
    
    print(f"\nAnalysis complete. Results saved to {args.output}/")
    return csd_df, association


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Critical Slowing Down Analysis (Wave-Aware)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        
Outputs:
  - csd_analysis.png/pdf  : 6-panel visualization
  - csd_analysis_report.txt : Comprehensive statistics report
  - csd_results.csv : Individual-level AR(1), variance, CCI (participant_id column)
  - csd_summary.csv : Key metrics (n, Cohen's d, AUC)
        """
    )
    parser.add_argument("--input", type=str, required=True, 
                        help="Input cohort CSV with trajectory_group column")
    parser.add_argument("--output", type=str, required=True, 
                        help="Output directory")
    parser.add_argument("--outcome", type=str, default="memory", 
                        help="Primary outcome variable name (default: memory)")
    parser.add_argument("--id-col", type=str, default="ID", dest="id_col",
                        help="Column name for participant ID (default: ID)")
    parser.add_argument("--wave-col", type=str, default="wave", dest="wave_col",
                        help="Column name for wave/time (default: wave)")
    parser.add_argument("--no-detrend", dest="detrend", action="store_false",
                        help="Do not detrend before CSD calculation (default: detrend)")
    parser.set_defaults(detrend=True)
    
    args = parser.parse_args()
    main(args)
