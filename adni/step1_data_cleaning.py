import pandas as pd
import numpy as np
from scipy import stats
import argparse
import os


def load_adni_data(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    return df


def standardize_column_names(df):
    if 'RID' in df.columns:
        df = df.rename(columns={'RID': 'participant_id'})
    elif 'PTID' in df.columns:
        df = df.rename(columns={'PTID': 'participant_id'})
    
    rename_map = {
        'VISCODE': 'visit_code',
        'EXAMDATE': 'exam_date',
        'AGE': 'age',
        'PTGENDER': 'gender',
        'PTEDUCAT': 'education',
        'PTETHCAT': 'ethnicity',
        'PTRACCAT': 'race',
        'PTMARRY': 'marital_status',
        'DX': 'diagnosis',
        'DX_bl': 'diagnosis_baseline',
        'MMSE': 'mmse',
        'CDRSB': 'cdr_sb',
        'ADAS11': 'adas11',
        'ADAS13': 'adas13',
        'ADASQ4': 'adas_delayed_recall',
        'RAVLT_immediate': 'ravlt_immediate',
        'RAVLT_learning': 'ravlt_learning',
        'RAVLT_forgetting': 'ravlt_forgetting',
        'RAVLT_perc_forgetting': 'ravlt_perc_forgetting',
        'LDELTOTAL': 'logical_memory_delayed',
        'LIMMTOTAL': 'logical_memory_immediate',
        'DIGITSCOR': 'digit_span',
        'TRABSCOR': 'trails_b',
        'FAQ': 'faq',
        'Hippocampus': 'hippocampus_vol',
        'WholeBrain': 'wholebrain_vol',
        'Entorhinal': 'entorhinal_vol',
        'Fusiform': 'fusiform_vol',
        'MidTemp': 'midtemp_vol',
        'ICV': 'icv',
        'APOE4': 'apoe4'
    }
    
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def convert_visit_to_months(df, visit_col='visit_code'):
    if visit_col not in df.columns:
        return df, []
    
    month_mapping = {
        'bl': 0, 'sc': 0, 'm03': 3, 'm06': 6, 'm12': 12, 'm18': 18,
        'm24': 24, 'm30': 30, 'm36': 36, 'm42': 42, 'm48': 48,
        'm54': 54, 'm60': 60, 'm66': 66, 'm72': 72, 'm78': 78,
        'm84': 84, 'm90': 90, 'm96': 96, 'm102': 102, 'm108': 108,
        'm114': 114, 'm120': 120, 'm126': 126, 'm132': 132, 'm144': 144,
        'm156': 156, 'm168': 168, 'm180': 180
    }
    
    original_na = df[visit_col].isna()
    s = df[visit_col].astype(str).str.strip().str.lower()
    df['months'] = s.map(month_mapping)
    df.loc[original_na, 'months'] = np.nan
    
    unmapped_mask = df['months'].isna() & ~original_na
    unmapped = sorted(set(s[unmapped_mask]))
    if len(unmapped) > 0:
        print(f"  [WARNING] Unmapped visit codes ({len(unmapped)}): {unmapped[:10]}")
    
    return df, unmapped


def standardize_gender(df, gender_col='gender'):
    if gender_col not in df.columns:
        return df
    
    s_raw = df[gender_col]
    na_mask = s_raw.isna()
    s = s_raw.astype(str).str.strip().str.lower()
    s[na_mask] = np.nan
    
    unique_vals = s.dropna().unique()
    print(f"  Gender unique values: {unique_vals[:10]}")
    
    gender_map = {'male': 0, 'female': 1, 'm': 0, 'f': 1}
    df['gender_coded'] = s.map(gender_map)
    
    unmapped = s[df['gender_coded'].isna() & s.notna()].unique()
    if len(unmapped) > 0:
        print(f"  [WARNING] Unmapped gender values: {unmapped[:10]}")
    
    return df


def filter_age(df, min_age=50, age_col='age', id_col='participant_id'):
    if age_col not in df.columns:
        print(f"  [WARNING] Age column '{age_col}' not found. Skipping age filter.")
        return df
    
    df_time = df.dropna(subset=['months']).sort_values([id_col, 'months'])
    baseline_age = df_time.groupby(id_col)[age_col].first()
    valid_ids = baseline_age[baseline_age >= min_age].index
    
    n_before = df[id_col].nunique()
    df_filtered = df[df[id_col].isin(valid_ids)].copy()
    n_after = df_filtered[id_col].nunique()
    
    n_no_time = df[id_col].nunique() - df_time[id_col].nunique()
    if n_no_time > 0:
        print(f"  [WARNING] {n_no_time} participants excluded due to no valid time data")
    
    print(f"  Age filter (>={min_age}): {n_before:,} -> {n_after:,} ({n_after/n_before*100:.1f}%)")
    return df_filtered


def create_memory_composite(df):
    memory_cols = ['ravlt_immediate', 'ravlt_learning', 'logical_memory_delayed',
                   'logical_memory_immediate', 'adas_delayed_recall']
    
    available_cols = [c for c in memory_cols if c in df.columns]
    
    if len(available_cols) == 0:
        print("  [WARNING] No memory columns found for composite score.")
        return df
    
    df = df.copy()
    
    for col in available_cols:
        col_data = df[col].copy()
        col_data = col_data.where(col_data >= 0)
        
        if col == 'adas_delayed_recall':
            df[f'{col}_z'] = -stats.zscore(col_data, nan_policy='omit')
        else:
            df[f'{col}_z'] = stats.zscore(col_data, nan_policy='omit')
    
    z_cols = [f'{c}_z' for c in available_cols]
    df['memory_composite'] = df[z_cols].mean(axis=1, skipna=True)
    
    print(f"  Memory composite: {len(available_cols)} variables ({', '.join(available_cols)})")
    return df


def compute_normalized_volumes(df):
    df = df.copy()
    
    if 'hippocampus_vol' in df.columns and 'icv' in df.columns:
        df['hippocampus_norm'] = df['hippocampus_vol'] / df['icv'] * 1000
        print("  Normalized hippocampus: hippocampus_vol / ICV * 1000")
    
    if 'entorhinal_vol' in df.columns and 'icv' in df.columns:
        df['entorhinal_norm'] = df['entorhinal_vol'] / df['icv'] * 1000
        print("  Normalized entorhinal: entorhinal_vol / ICV * 1000")
    
    return df


def validate_longitudinal_structure(df, id_col='participant_id', time_col='months'):
    df_valid = df.dropna(subset=[time_col])
    wave_counts = df_valid.groupby(id_col)[time_col].nunique()
    
    print(f"\n[LONGITUDINAL STRUCTURE]")
    print(f"  Total participants: {df[id_col].nunique():,}")
    print(f"  With valid time data: {df_valid[id_col].nunique():,}")
    print(f"  Total observations: {len(df_valid):,}")
    print(f"  Mean timepoints/participant: {wave_counts.mean():.2f}")
    
    for n in [2, 3, 4, 6, 8]:
        count = (wave_counts >= n).sum()
        pct = count / len(wave_counts) * 100
        print(f"  >= {n} timepoints: {count:,} ({pct:.1f}%)")
    
    return wave_counts


def compute_variable_coverage(df):
    var_groups = {
        'Core': ['age', 'gender', 'education', 'diagnosis'],
        'Cognitive': ['mmse', 'cdr_sb', 'adas13', 'ravlt_immediate',
                      'logical_memory_delayed', 'memory_composite'],
        'Neuroimaging': ['hippocampus_vol', 'hippocampus_norm',
                         'entorhinal_vol', 'wholebrain_vol', 'icv'],
        'Genetic': ['apoe4']
    }
    
    print(f"\n[VARIABLE COVERAGE]")
    
    coverage_dict = {}
    for group_name, var_list in var_groups.items():
        print(f"  {group_name}:")
        for var in var_list:
            if var in df.columns:
                n_valid = df[var].notna().sum()
                pct = n_valid / len(df) * 100
                print(f"    {var}: {pct:.1f}%")
                coverage_dict[var] = pct
            else:
                print(f"    {var}: not found")
                coverage_dict[var] = 0.0
    
    return coverage_dict


def compute_diagnosis_distribution(df, id_col='participant_id'):
    if 'diagnosis' not in df.columns:
        return None
    
    df_time = df.dropna(subset=['months']).sort_values([id_col, 'months'])
    baseline_dx = df_time.groupby(id_col)['diagnosis'].first()
    dx_counts = baseline_dx.value_counts()
    
    print(f"\n[DIAGNOSIS DISTRIBUTION - Baseline]")
    for dx, count in dx_counts.items():
        pct = count / len(baseline_dx) * 100
        print(f"  {dx}: {count:,} ({pct:.1f}%)")
    
    missing = baseline_dx.isna().sum()
    if missing > 0:
        print(f"  Missing: {missing:,} ({missing/len(baseline_dx)*100:.1f}%)")
    
    return dx_counts


def generate_cleaning_report(df, output_path, id_col='participant_id', time_col='months',
                             wave_counts=None, coverage_dict=None, unmapped_codes=None):
    lines = []
    lines.append("=" * 70)
    lines.append("ADNI DATA CLEANING REPORT - Step 1")
    lines.append("=" * 70)
    lines.append("")
    
    lines.append("[SAMPLE OVERVIEW]")
    lines.append(f"  Participants: {df[id_col].nunique():,}")
    lines.append(f"  Observations: {len(df):,}")
    
    if 'gender_coded' in df.columns:
        g = df.groupby(id_col)['gender_coded'].first()
        n_valid = g.notna().sum()
        lines.append(f"  Gender coded: {n_valid:,} / {len(g):,} ({n_valid/len(g)*100:.1f}%)")
        if n_valid > 0:
            female_pct = g.mean() * 100
            lines.append(f"  Female: {female_pct:.1f}%")
    
    df_time_base = df.dropna(subset=[time_col]).sort_values([id_col, time_col])
    
    if 'age' in df.columns and len(df_time_base) > 0:
        baseline_age = df_time_base.groupby(id_col)['age'].first()
        lines.append(f"  Age (baseline): {baseline_age.mean():.1f} +/- {baseline_age.std():.1f}")
    
    if 'education' in df.columns and len(df_time_base) > 0:
        baseline_edu = df_time_base.groupby(id_col)['education'].first()
        lines.append(f"  Education: {baseline_edu.mean():.1f} +/- {baseline_edu.std():.1f} years")
    
    lines.append("")
    
    if 'diagnosis' in df.columns:
        lines.append("[DIAGNOSIS DISTRIBUTION - Baseline]")
        df_time = df.dropna(subset=[time_col]).sort_values([id_col, time_col])
        baseline_dx = df_time.groupby(id_col)['diagnosis'].first()
        dx_counts = baseline_dx.value_counts()
        for dx, count in dx_counts.items():
            pct = count / len(baseline_dx) * 100
            lines.append(f"  {dx}: {count:,} ({pct:.1f}%)")
        missing = baseline_dx.isna().sum()
        if missing > 0:
            lines.append(f"  Missing: {missing:,} ({missing/len(baseline_dx)*100:.1f}%)")
        lines.append("")
    
    if wave_counts is not None:
        lines.append("[LONGITUDINAL COVERAGE]")
        lines.append(f"  Mean timepoints/participant: {wave_counts.mean():.2f}")
        for n in [2, 3, 4, 6, 8]:
            count = (wave_counts >= n).sum()
            pct = count / len(wave_counts) * 100
            lines.append(f"  >= {n} timepoints: {count:,} ({pct:.1f}%)")
        lines.append("")
    
    if coverage_dict is not None:
        lines.append("[KEY VARIABLE COVERAGE]")
        key_vars = ['memory_composite', 'mmse', 'adas13', 'hippocampus_norm', 'apoe4']
        for var in key_vars:
            if var in coverage_dict:
                lines.append(f"  {var}: {coverage_dict[var]:.1f}%")
        lines.append("")
    
    lines.append("[TIME VARIABLE]")
    lines.append("  Using 'months' (months since baseline) derived from VISCODE")
    if 'months' in df.columns:
        valid_months = df['months'].notna().sum()
        pct_valid = valid_months / len(df) * 100
        lines.append(f"  Valid time mapping: {valid_months:,} / {len(df):,} ({pct_valid:.1f}%)")
    if unmapped_codes is not None and len(unmapped_codes) > 0:
        lines.append(f"  Unmapped visit codes ({len(unmapped_codes)}): {unmapped_codes[:10]}")
    lines.append("")
    
    lines.append("[NOTE]")
    lines.append("  Diagnostic status (DX) is NOT used to filter participants.")
    lines.append("  DX is retained for descriptive characterization and sensitivity analyses.")
    lines.append("  DX missingness is visit-level; baseline DX missingness reported above.")
    lines.append("")
    
    lines.append("=" * 70)
    
    report_text = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    return report_text


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ADNI Step 1: Data Cleaning")
    print("=" * 60)
    
    print("\n[1] Loading data...")
    df = load_adni_data(args.input)
    print(f"  Raw data: {len(df):,} rows, {len(df.columns)} columns")
    
    print("\n[2] Standardizing column names...")
    df = standardize_column_names(df)
    
    id_col = 'participant_id'
    if id_col not in df.columns:
        raise ValueError(f"participant_id column not found after standardization. "
                         f"Available columns: {df.columns.tolist()[:20]}")
    
    print(f"  Unique participants: {df[id_col].nunique():,}")
    
    print("\n[3] Converting visit codes to months...")
    df, unmapped_codes = convert_visit_to_months(df)
    time_col = 'months'
    
    valid_time = df[time_col].notna().sum()
    print(f"  Valid time observations: {valid_time:,} / {len(df):,} ({valid_time/len(df)*100:.1f}%)")
    
    print("\n[4] Standardizing gender...")
    df = standardize_gender(df)
    
    print("\n[5] Applying age filter...")
    n_before = df[id_col].nunique()
    df = filter_age(df, min_age=args.min_age, id_col=id_col)
    n_after = df[id_col].nunique()
    
    print("\n[6] Creating derived variables...")
    df = create_memory_composite(df)
    df = compute_normalized_volumes(df)
    
    print("\n[7] Validating longitudinal structure...")
    wave_counts = validate_longitudinal_structure(df, id_col=id_col, time_col=time_col)
    
    print("\n[8] Computing variable coverage...")
    coverage_dict = compute_variable_coverage(df)
    
    compute_diagnosis_distribution(df, id_col=id_col)
    
    print("\n[9] Saving outputs...")
    
    output_file = os.path.join(args.output_dir, "ADNI_cleaned_step1.csv")
    df.to_csv(output_file, index=False)
    print(f"  Cleaned data: {output_file}")
    
    baseline_df = df.dropna(subset=[time_col]).sort_values([id_col, time_col]).groupby(id_col).first().reset_index()
    baseline_file = os.path.join(args.output_dir, "ADNI_baseline_step1.csv")
    baseline_df.to_csv(baseline_file, index=False)
    print(f"  Baseline data: {baseline_file}")
    
    report_file = os.path.join(args.output_dir, "ADNI_cleaning_report_step1.txt")
    report = generate_cleaning_report(
        df, report_file, id_col=id_col, time_col=time_col,
        wave_counts=wave_counts, coverage_dict=coverage_dict,
        unmapped_codes=unmapped_codes
    )
    print(f"  Report: {report_file}")
    
    print("\n" + "=" * 60)
    print("CLEANING REPORT")
    print("=" * 60)
    print(report)
    
    print("\n" + "=" * 60)
    print("Step 1 Complete")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADNI Step 1: Data Cleaning")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to ADNIMERGE.csv or similar")
    parser.add_argument("--output-dir", type=str, required=True, dest="output_dir",
                        help="Output directory for cleaned data")
    parser.add_argument("--min-age", type=int, default=50, dest="min_age",
                        help="Minimum baseline age (default: 50)")
    
    args = parser.parse_args()
    main(args)
