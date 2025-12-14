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


def compute_csd_indicators(df, outcome_col, id_col='ID', wave_col='wave', detrend=True):
    results = []
    
    for pid in df[id_col].unique():
        person_data = df[df[id_col] == pid].sort_values(wave_col)
        
        mask = person_data[outcome_col].notna()
        y = person_data.loc[mask, outcome_col].values
        x = person_data.loc[mask, wave_col].values.astype(float)
        
        ar1 = compute_ar1_ols(y, x, detrend=detrend)
        variance = compute_variance_detrended(y, x, detrend=detrend)
        
        group = person_data.sort_values(wave_col)['trajectory_group'].iloc[0]
        
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


def group_comparison_statistics(csd_df, metric):
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


def evaluate_association_strength(csd_df, predictor='cci', target_group='Decline'):
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
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    groups = ['Decline', 'Stable', 'Improved']
    colors = {'Decline': '#e74c3c', 'Stable': '#3498db', 'Improved': '#2ecc71'}
    
    ax = axes[0, 0]
    for group in groups:
        data = csd_df[csd_df['trajectory_group'] == group]['ar1'].dropna()
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.5, label=f'{group} (n={len(data)})', color=colors[group])
    ax.set_xlabel('AR(1) Coefficient')
    ax.set_ylabel('Frequency')
    ax.set_title('A. AR(1) Distribution by Trajectory Group')
    ax.legend()
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    ax = axes[0, 1]
    for group in groups:
        data = csd_df[csd_df['trajectory_group'] == group]['variance'].dropna()
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.5, label=f'{group} (n={len(data)})', color=colors[group])
    ax.set_xlabel('Variance')
    ax.set_ylabel('Frequency')
    ax.set_title('B. Variance Distribution by Trajectory Group')
    ax.legend()
    
    ax = axes[0, 2]
    for group in groups:
        data = csd_df[csd_df['trajectory_group'] == group]['cci'].dropna()
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.5, label=f'{group} (n={len(data)})', color=colors[group])
    ax.set_xlabel('CCI')
    ax.set_ylabel('Frequency')
    ax.set_title('C. CCI Distribution by Trajectory Group')
    ax.legend()
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    ax = axes[1, 0]
    box_data = [csd_df[csd_df['trajectory_group'] == g]['ar1'].dropna() for g in groups]
    bp = ax.boxplot(box_data, labels=groups, patch_artist=True)
    for patch, group in zip(bp['boxes'], groups):
        patch.set_facecolor(colors[group])
        patch.set_alpha(0.5)
    ax.set_ylabel('AR(1) Coefficient')
    ax.set_title('D. AR(1) by Group (Box Plot)')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax = axes[1, 1]
    if association:
        ax.plot(association['fpr'], association['tpr'], 'b-', linewidth=2,
                label=f"AUC = {association['auc']:.3f}")
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('E. ROC Curve: CCI vs Decline')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    else:
        ax.text(0.5, 0.5, 'AUC not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('E. ROC Curve: CCI vs Decline')
    
    ax = axes[1, 2]
    if association:
        metrics = ['Sensitivity', 'Specificity', 'PPV', 'NPV']
        values = [association['sensitivity'], association['specificity'],
                  association['ppv'], association['npv']]
        bars = ax.bar(metrics, values, color=['#3498db', '#2ecc71', '#f39c12', '#9b59b6'])
        ax.set_ylim([0, 1])
        ax.set_ylabel('Value')
        ax.set_title(f"F. Performance at threshold=0")
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom')
    else:
        ax.text(0.5, 0.5, 'Metrics not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('F. Performance Metrics')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()


def generate_analysis_report(csd_df, ar1_stats, var_stats, cci_stats, association, detrend, output_path=None):
    report = []
    report.append("=" * 70)
    report.append("CRITICAL SLOWING DOWN ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")
    
    report.append("[SAMPLE]")
    report.append(f"  Total with valid CSD: {len(csd_df):,}")
    for g in ['Decline', 'Stable', 'Improved']:
        n = (csd_df['trajectory_group'] == g).sum()
        report.append(f"    {g}: {n:,}")
    report.append("")
    
    report.append("[METHODOLOGY]")
    report.append(f"  Detrending: {'Yes (wave-aware)' if detrend else 'No'}")
    report.append("  AR(1): OLS regression (phi parameter)")
    report.append("  Tests: Kruskal-Wallis, Welch t-test, Mann-Whitney U")
    report.append("  Threshold: 0 (cohort-specific z-score mean)")
    report.append("")
    
    if ar1_stats:
        report.append("[AR(1) RESULTS]")
        for g in ['Decline', 'Stable', 'Improved']:
            m = ar1_stats['group_means'].get(g, np.nan)
            s = ar1_stats['group_sds'].get(g, np.nan)
            n = ar1_stats['group_ns'].get(g, 0)
            report.append(f"  {g}: {m:.3f} +/- {s:.3f} (n={n})")
        report.append(f"  Kruskal-Wallis H = {ar1_stats['kruskal_h']:.2f}, p = {ar1_stats['kruskal_p']:.2e}")
        if 'cohens_d' in ar1_stats:
            report.append(f"  Cohen's d (Decline vs Stable) = {ar1_stats['cohens_d']:.3f}")
        report.append("")
    
    if var_stats:
        report.append("[VARIANCE RESULTS]")
        for g in ['Decline', 'Stable', 'Improved']:
            m = var_stats['group_means'].get(g, np.nan)
            s = var_stats['group_sds'].get(g, np.nan)
            n = var_stats['group_ns'].get(g, 0)
            report.append(f"  {g}: {m:.3f} +/- {s:.3f} (n={n})")
        report.append(f"  Kruskal-Wallis H = {var_stats['kruskal_h']:.2f}, p = {var_stats['kruskal_p']:.2e}")
        report.append("")
    
    if association:
        report.append("[CCI DISCRIMINATION]")
        report.append(f"  AUC: {association['auc']:.3f}")
        report.append(f"  Threshold: {association['threshold_used']:.1f}")
        report.append(f"  Sensitivity: {association['sensitivity']:.3f}")
        report.append(f"  Specificity: {association['specificity']:.3f}")
        report.append(f"  PPV: {association['ppv']:.3f}")
        report.append(f"  NPV: {association['npv']:.3f}")
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
    
    n_input = df[args.id_col].nunique()
    
    print("\nComputing CSD indicators...")
    csd_df = compute_csd_indicators(df, outcome_col, id_col=args.id_col,
                                     wave_col=args.wave_col, detrend=args.detrend)
    
    print("Computing Comprehensive Criticality Index...")
    csd_df = compute_cci(csd_df)
    
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
    
    results_path = os.path.join(args.output, "csd_results.csv")
    csd_df.to_csv(results_path, index=False)
    
    summary = {
        'n_input': n_input,
        'n_valid_csd': len(csd_df),
        'n_decline': (csd_df['trajectory_group'] == 'Decline').sum(),
        'n_stable': (csd_df['trajectory_group'] == 'Stable').sum(),
        'ar1_cohens_d': ar1_stats['cohens_d'] if ar1_stats and 'cohens_d' in ar1_stats else np.nan,
        'auc': association['auc'] if association else np.nan,
        'sensitivity': association['sensitivity'] if association else np.nan,
        'specificity': association['specificity'] if association else np.nan,
        'npv': association['npv'] if association else np.nan
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(args.output, "csd_summary.csv"), index=False)
    
    print(f"\nAnalysis complete. Results saved to {args.output}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--outcome", type=str, default="memory")
    parser.add_argument("--id-col", type=str, default="ID", dest="id_col")
    parser.add_argument("--wave-col", type=str, default="wave", dest="wave_col")
    parser.add_argument("--no-detrend", dest="detrend", action="store_false")
    parser.set_defaults(detrend=True)
    
    args = parser.parse_args()
    main(args)
