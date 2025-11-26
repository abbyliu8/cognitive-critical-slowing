import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import mahalanobis
import argparse
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    return df


def filter_by_age(df, min_age=45, age_col='age', wave_col='wave', id_col='ID'):
    baseline = df[df[wave_col] == df[wave_col].min()]
    eligible_ids = baseline[baseline[age_col] >= min_age][id_col].unique()
    df_filtered = df[df[id_col].isin(eligible_ids)].copy()
    return df_filtered


def validate_gender(df, gender_col='gender', id_col='ID', wave_col='wave'):
    gender_check = df.groupby(id_col)[gender_col].nunique()
    consistent_ids = gender_check[gender_check == 1].index
    df_valid = df[df[id_col].isin(consistent_ids)].copy()
    df_valid = df_valid[df_valid[gender_col].notna()]
    return df_valid


def detect_multivariate_outliers(df, vars_list, threshold=3.5, wave_col='wave'):
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
    report = []
    report.append("=" * 60)
    report.append("CHARLS DATA CLEANING REPORT")
    report.append("=" * 60)
    report.append("")
    report.append("[CLEANING STEPS]")
    report.append("1. Age filter: Baseline age >= 45")
    report.append("2. Gender validation: Consistent and non-missing")
    report.append("3. Multivariate outlier detection: Mahalanobis distance")
    report.append("4. Temporal consistency check: ±3 SD")
    report.append("")
    report.append("[SAMPLE SUMMARY]")
    report.append(f"Original sample: {df_original['ID'].nunique():,} individuals")
    report.append(f"Final sample: {df_cleaned['ID'].nunique():,} individuals")
    report.append(f"Total observations: {len(df_cleaned):,}")
    report.append(f"Retention rate: {df_cleaned['ID'].nunique() / df_original['ID'].nunique() * 100:.1f}%")
    report.append("")
    report.append("[VARIABLE COVERAGE RATES]")
    for var, rate in coverage_rates.items():
        report.append(f"  {var}: {rate:.1f}%")
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
    
    return report_text


def main(args):
    print("Loading data...")
    df = load_data(args.input)
    df_original = df.copy()
    
    print("Filtering by age...")
    df = filter_by_age(df, min_age=args.min_age)
    
    print("Validating gender...")
    df = validate_gender(df)
    
    print("Detecting multivariate outliers...")
    core_vars = ['total_cognition', 'memory', 'cesd10', 'age']
    available_vars = [v for v in core_vars if v in df.columns]
    
    if len(available_vars) >= 2:
        outlier_indices = detect_multivariate_outliers(df, available_vars)
        df = df[~df.index.isin(outlier_indices)]
    
    print("Checking temporal consistency...")
    outcome_col = 'memory' if 'memory' in df.columns else 'total_cognition'
    if outcome_col in df.columns:
        extreme_ids = check_temporal_consistency(df, outcome_col)
        df = df[~df['ID'].isin(extreme_ids)]
    
    all_vars = [
        'total_cognition', 'memory', 'executive', 'cesd10', 'sleep',
        'hibpe', 'diabe', 'bl_crp', 'bl_hbalc', 'bl_tg'
    ]
    coverage = compute_coverage_rates(df, all_vars)
    
    report = generate_report(df_original, df, coverage, args.report)
    print(report)
    
    print(f"Saving cleaned data to {args.output}...")
    df.to_csv(args.output, index=False)
    
    print("Data cleaning complete.")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHARLS Data Cleaning Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
    parser.add_argument("--report", type=str, default=None, help="Report output path")
    parser.add_argument("--min_age", type=int, default=45, help="Minimum baseline age")
    
    args = parser.parse_args()
    main(args)
