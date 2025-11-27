import pandas as pd
import numpy as np
from scipy import stats
import argparse
import os
import warnings
warnings.filterwarnings('ignore')


def load_cleaned_data(filepath):
    """Load cleaned dataset from Step 1"""
    return pd.read_csv(filepath, low_memory=False)


def assess_wave_coverage(df, outcome_col, id_col='ID', wave_col='wave'):
    """
    Assess number of waves with valid outcome data per participant
    
    Parameters
    ----------
    df : DataFrame
        Input dataset
    outcome_col : str
        Name of outcome variable to assess
    id_col : str
        Participant ID column
    wave_col : str
        Wave indicator column
    
    Returns
    -------
    Series
        Count of valid outcome observations per participant
    """
    outcome_counts = df.groupby(id_col).apply(
        lambda x: x[outcome_col].notna().sum()
    )
    return outcome_counts


def construct_cohort(df, outcome_col, min_waves=3, id_col='ID', wave_col='wave'):
    """
    Construct analysis cohort based on minimum wave requirement
    
    Parameters
    ----------
    df : DataFrame
        Input dataset
    outcome_col : str
        Primary outcome variable
    min_waves : int
        Minimum number of waves with valid outcome data required
    id_col : str
        Participant ID column
    wave_col : str
        Wave indicator column
    
    Returns
    -------
    tuple
        (cohort DataFrame, wave counts Series)
    """
    wave_counts = assess_wave_coverage(df, outcome_col, id_col, wave_col)
    eligible_ids = wave_counts[wave_counts >= min_waves].index
    cohort = df[df[id_col].isin(eligible_ids)].copy()
    return cohort, wave_counts


def compute_trajectory_slope(df, outcome_col, id_col='ID', wave_col='wave'):
    """
    Compute linear trajectory slope for each participant
    
    Uses actual wave values as x-axis to correctly handle
    non-consecutive waves (e.g., waves 1,2,4 when wave 3 is missing)
    
    Parameters
    ----------
    df : DataFrame
        Input dataset
    outcome_col : str
        Outcome variable for trajectory
    id_col : str
        Participant ID column
    wave_col : str
        Wave indicator column
    
    Returns
    -------
    Series
        Trajectory slopes indexed by participant ID
    """
    slopes = {}
    
    for pid in df[id_col].unique():
        person_data = df[df[id_col] == pid].sort_values(wave_col)
        valid_data = person_data[[wave_col, outcome_col]].dropna()
        
        if len(valid_data) >= 2:
            x = valid_data[wave_col].values
            y = valid_data[outcome_col].values
            slope, _, _, _, _ = stats.linregress(x, y)
            slopes[pid] = slope
    
    return pd.Series(slopes)


def classify_trajectories(slopes, method='tertile'):
    """
    Classify participants into trajectory groups
    
    Parameters
    ----------
    slopes : Series
        Trajectory slopes indexed by participant ID
    method : str
        Classification method: 'tertile' or 'median_split'
    
    Returns
    -------
    tuple
        (group assignments Series, threshold dict)
    """
    if method == 'tertile':
        q1, q3 = slopes.quantile([0.33, 0.67])
        
        groups = pd.Series(index=slopes.index, dtype='object')
        groups[slopes <= q1] = 'Decline'
        groups[slopes >= q3] = 'Improved'
        groups[(slopes > q1) & (slopes < q3)] = 'Stable'
        
        return groups, {'decline_threshold': q1, 'improve_threshold': q3}
    
    elif method == 'median_split':
        median = slopes.median()
        groups = pd.Series(index=slopes.index, dtype='object')
        groups[slopes < median] = 'Decline'
        groups[slopes >= median] = 'Stable'
        return groups, {'threshold': median}
    
    return None, None


def get_baseline_characteristics(df, id_col='ID', wave_col='wave'):
    """
    Extract baseline demographic characteristics
    
    Uses each participant's earliest available observation as baseline,
    not restricted to wave 1 (handles participants entering at later waves)
    
    Parameters
    ----------
    df : DataFrame
        Cohort dataset
    id_col : str
        Participant ID column
    wave_col : str
        Wave indicator column
    
    Returns
    -------
    dict
        Baseline characteristics summary
    """
    df_sorted = df.sort_values([id_col, wave_col])
    baseline = df_sorted.groupby(id_col).first().reset_index()
    
    characteristics = {
        'n': len(baseline),
        'age_mean': baseline['age'].mean() if 'age' in baseline.columns else np.nan,
        'age_sd': baseline['age'].std() if 'age' in baseline.columns else np.nan,
        'female_n': (baseline['gender'] == 2).sum() if 'gender' in baseline.columns else np.nan,
        'female_pct': ((baseline['gender'] == 2).sum() / len(baseline) * 100) if 'gender' in baseline.columns else np.nan
    }
    
    return characteristics


def generate_cohort_report(df, groups, characteristics, thresholds, output_path=None):
    """
    Generate cohort construction report
    
    Parameters
    ----------
    df : DataFrame
        Cohort dataset
    groups : Series
        Trajectory group assignments
    characteristics : dict
        Baseline characteristics
    thresholds : dict
        Classification thresholds
    output_path : str, optional
        Path to save report
    
    Returns
    -------
    str
        Report text
    """
    report = []
    report.append("=" * 60)
    report.append("COHORT CONSTRUCTION REPORT")
    report.append("=" * 60)
    report.append("")
    report.append("[COHORT CHARACTERISTICS]")
    report.append(f"Sample size: {characteristics['n']:,}")
    report.append(f"Total observations: {len(df):,}")
    report.append(f"Mean observations per person: {len(df) / characteristics['n']:.2f}")
    report.append("")
    report.append("[DEMOGRAPHICS]")
    report.append(f"Age: {characteristics['age_mean']:.1f} ± {characteristics['age_sd']:.1f} years")
    report.append(f"Female: {characteristics['female_n']:,.0f} ({characteristics['female_pct']:.1f}%)")
    report.append("")
    report.append("[TRAJECTORY GROUPS]")
    group_counts = groups.value_counts()
    for group in ['Decline', 'Stable', 'Improved']:
        if group in group_counts.index:
            count = group_counts[group]
            pct = count / len(groups) * 100
            report.append(f"  {group}: {count:,} ({pct:.1f}%)")
    report.append("")
    report.append("[CLASSIFICATION THRESHOLDS]")
    for key, value in thresholds.items():
        report.append(f"  {key}: {value:.3f}")
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
    
    return report_text


def main(args):
    """Main execution function"""
    
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print("Loading cleaned data...")
    df = load_cleaned_data(args.input)
    
    outcome_col = args.outcome if args.outcome else 'memory'
    if outcome_col not in df.columns:
        outcome_col = 'total_cognition'
    
    print(f"Constructing cohort with >= {args.min_waves} waves of {outcome_col}...")
    cohort, wave_counts = construct_cohort(df, outcome_col, min_waves=args.min_waves)
    
    print("Computing trajectory slopes...")
    slopes = compute_trajectory_slope(cohort, outcome_col)
    
    print("Classifying trajectories...")
    groups, thresholds = classify_trajectories(slopes, method=args.method)
    
    characteristics = get_baseline_characteristics(cohort)
    
    report = generate_cohort_report(cohort, groups, characteristics, thresholds, args.report)
    print(report)
    
    cohort['trajectory_group'] = cohort['ID'].map(groups)
    cohort['trajectory_slope'] = cohort['ID'].map(slopes)
    
    print(f"\nSaving cohort data to {args.output}...")
    cohort.to_csv(args.output, index=False)
    
    groups_df = pd.DataFrame({
        'ID': groups.index,
        'trajectory_group': groups.values,
        'trajectory_slope': slopes.values
    })
    groups_output = args.output.replace('.csv', '_groups.csv')
    groups_df.to_csv(groups_output, index=False)
    
    print("Cohort construction complete.")
    return cohort, groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cohort Construction Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input cleaned CSV from Step 1")
    parser.add_argument("--output", type=str, required=True, help="Output cohort CSV")
    parser.add_argument("--report", type=str, default=None, help="Report output path")
    parser.add_argument("--outcome", type=str, default="memory", help="Primary outcome variable")
    parser.add_argument("--min_waves", type=int, default=3, help="Minimum waves required")
    parser.add_argument("--method", type=str, default="tertile", choices=["tertile", "median_split"])
    
    args = parser.parse_args()
    main(args)
