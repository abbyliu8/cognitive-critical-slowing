
import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import mahalanobis
import argparse
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    try:
        df = pd.read_csv(filepath, low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, low_memory=False, encoding='gbk')
        except:
            df = pd.read_csv(filepath, low_memory=False, encoding='latin1')
    
    df = df.drop_duplicates(['ID', 'wave'], keep='first')
    return df


def filter_by_age(df, min_age=45, age_col='age', wave_col='wave', id_col='ID'):
    df_sorted = df.sort_values([id_col, wave_col])
    first_obs = df_sorted.groupby(id_col).first().reset_index()
    eligible_ids = first_obs[first_obs[age_col] >= min_age][id_col].unique()
    df_filtered = df[df[id_col].isin(eligible_ids)].copy()
    return df_filtered


def validate_gender(df, gender_col='gender', id_col='ID'):
    gender_check = df.groupby(id_col)[gender_col].nunique()
    consistent_ids = gender_check[gender_check == 1].index
    df_valid = df[df[id_col].isin(consistent_ids)].copy()
    df_valid = df_valid[df_valid[gender_col].notna()]
    final_ids = df_valid[id_col].unique()
    df_valid = df_valid[df_valid[id_col].isin(final_ids)]
    return df_valid


def detect_multivariate_outliers(df, vars_list, wave_col='wave'):
    outlier_indices = []
    
    for wave in df[wave_col].unique():
        wave_data = df[df[wave_col] == wave][vars_list].dropna()
        
        if len(wave_data) > len(vars_list) + 1:
            try:
                mean = wave_data.mean()
                cov = wave_data.cov()
                
                if np.linalg.det(cov) > 1e-10:
                    inv_cov = np.linalg.pinv(cov)
                    distances = wave_data.apply(
                        lambda x: mahalanobis(x, mean, inv_cov), axis=1
                    )
                    chi2_threshold = np.sqrt(stats.chi2.ppf(0.999, df=len(vars_list)))
                    outlier_mask = distances > chi2_threshold
                    outlier_indices.extend(wave_data[outlier_mask].index.tolist())
            except Exception:
                continue
    
    return outlier_indices


def check_temporal_consistency(df, outcome_col, id_col='ID', wave_col='wave', sd_multiplier=3):
    df_wide = df.pivot_table(index=id_col, columns=wave_col, values=outcome_col, aggfunc='first')
    
    changes = []
    for i in range(len(df_wide.columns) - 1):
        col1, col2 = df_wide.columns[i], df_wide.columns[i + 1]
        delta = df_wide[col2] - df_wide[col1]
        changes.append(delta)
    
    all_changes = pd.concat(changes)
    extreme_threshold = all_changes.std() * sd_multiplier
    extreme_ids = all_changes[abs(all_changes) > extreme_threshold].dropna().index.unique()
    
    return extreme_ids


def compute_coverage_rates(df, var_list):
    coverage = {}
    for var in var_list:
        if var in df.columns:
            coverage[var] = (df[var].notna().sum() / len(df)) * 100
    return coverage


def generate_report(df_original, df_cleaned, coverage_rates, output_path=None):
    n_original = df_original['ID'].nunique()
    n_cleaned = df_cleaned['ID'].nunique()
    
    df_sorted = df_cleaned.sort_values(['ID', 'wave'])
    baseline = df_sorted.groupby('ID').first()
    
    if 'gender' in baseline.columns:
        female_pct = (baseline['gender'] == 0).mean() * 100
        if female_pct < 30:
            female_pct = (baseline['gender'] == 2).mean() * 100
    else:
        female_pct = np.nan
    
    mean_age = baseline['age'].mean() if 'age' in baseline.columns else np.nan
    
    report = []
    report.append("CHARLS Data Cleaning Report - Step 1")
    report.append("=" * 60)
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    report.append("[Cleaning Steps]")
    report.append("1. Age filter: Baseline age >= 45")
    report.append("2. Gender validation: Consistent and non-missing")
    report.append("3. Multivariate outlier detection: Mahalanobis distance")
    report.append("4. Temporal consistency check: +/- 3 SD")
    report.append("")
    report.append("[Results]")
    report.append(f"Original: {n_original:,} individuals")
    report.append(f"Final: {n_cleaned:,} individuals")
    report.append(f"Observations: {len(df_cleaned):,}")
    report.append(f"Female: {female_pct:.1f}%")
    report.append(f"Mean age: {mean_age:.1f} years")
    report.append("")
    report.append("[Variable Coverage]")
    for var, rate in coverage_rates.items():
        report.append(f"  {var}: {rate:.1f}%")
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    return report_text


def main(args):
    print("Loading data...")
    df = load_data(args.input)
    df_original = df.copy()
    print(f"Loaded {df['ID'].nunique():,} individuals, {len(df):,} observations")
    
    print("Filtering by age...")
    df = filter_by_age(df, min_age=args.min_age)
    print(f"After age filter: {df['ID'].nunique():,} individuals")
    
    print("Validating gender...")
    df = validate_gender(df)
    print(f"After gender validation: {df['ID'].nunique():,} individuals")
    
    print("Detecting multivariate outliers...")
    core_vars = ['total_cognition', 'memeory', 'cesd10', 'age']
    available_vars = [v for v in core_vars if v in df.columns]
    
    if len(available_vars) >= 2:
        outlier_indices = detect_multivariate_outliers(df, available_vars)
        df = df[~df.index.isin(outlier_indices)]
        print(f"Removed {len(outlier_indices)} outlier observations")
    
    print("Checking temporal consistency...")
    outcome_col = 'memeory' if 'memeory' in df.columns else 'total_cognition'
    if outcome_col in df.columns:
        extreme_ids = check_temporal_consistency(df, outcome_col)
        df = df[~df['ID'].isin(extreme_ids)]
        print(f"Removed {len(extreme_ids)} individuals with extreme changes")
    
    all_vars = [
        'total_cognition', 'memeory', 'executive', 'cesd10', 'sleep',
        'hibpe', 'diabe', 'bl_crp', 'bl_hbalc', 'bl_tg'
    ]
    coverage = compute_coverage_rates(df, all_vars)
    
    report = generate_report(df_original, df, coverage, args.report)
    print("\n" + report)
    
    print(f"\nSaving cleaned data to {args.output}...")
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    
    print("\nStep 1 complete.")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument("--min_age", type=int, default=45)
    
    args = parser.parse_args()
    main(args)
