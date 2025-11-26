# Statistical Analysis Plan

## Study Objectives

### Primary Objective
To test whether individuals exhibiting cognitive decline display elevated critical slowing down signatures (AR(1) autoregressive coefficient and variance) compared to cognitively stable individuals.

### Secondary Objectives
1. To develop and validate a Comprehensive Criticality Index (CCI) for predicting rapid cognitive decline
2. To assess robustness of findings across cohorts of varying data quality
3. To externally validate findings in an independent population

## Study Population

### CHARLS Cohort

**Inclusion Criteria**:
- Age ≥ 45 years at baseline (Wave 1, 2011)
- Memory assessment available at ≥ 3 waves (Silver cohort) or ≥ 4 waves (Gold cohort)
- Consistent gender coding across waves

**Exclusion Criteria**:
- Missing baseline age or gender
- Multivariate outliers (Mahalanobis distance > χ²₀.₉₉₉ threshold)
- Extreme trajectory changes (> 3 SD year-over-year change)

### ELSA Cohort (External Validation)

**Inclusion Criteria**:
- Age ≥ 50 years at Wave 4 (2008)
- Memory assessment available at all 6 waves (Waves 4-9)

**Exclusion Criteria**:
- Missing age or sex information

## Outcome Definitions

### Primary Outcome: Trajectory Group

Memory trajectory slope computed via linear regression of memory scores against wave number. Groups defined by tertile cutpoints:
- **Decline**: Slope ≤ 33rd percentile
- **Stable**: 33rd percentile < Slope < 67th percentile
- **Improved**: Slope ≥ 67th percentile

### Critical Slowing Down Indicators

**AR(1) Coefficient**: Pearson correlation between Y(t) and Y(t-1)

**Variance**: Sample variance of memory scores across waves

**CCI**: Mean of z-standardized AR(1) and variance

## Statistical Analyses

### Descriptive Statistics

Continuous variables: Mean ± SD, Median (IQR)
Categorical variables: N (%)
Group comparisons: ANOVA F-test, independent t-test

### Primary Analysis

**Hypothesis**: AR(1)_Decline > AR(1)_Stable

**Test**: One-way ANOVA with post-hoc pairwise comparisons
**Effect Size**: Cohen's d for Decline vs Stable contrast
**Significance Level**: α = 0.05 (two-sided)

### Prediction Performance

**Metric**: Area Under ROC Curve (AUC)
**Primary Comparison**: CCI as predictor of Decline group membership
**Threshold Optimization**: Youden index (maximize sensitivity + specificity - 1)
**Additional Metrics**: Sensitivity, Specificity, PPV, NPV

### Robustness Analyses

1. **Cohort Stringency**: Compare Silver (≥3 waves) vs Gold (≥4 waves)
2. **External Validation**: Replicate in ELSA cohort
3. **Outlier Sensitivity**: Winsorized vs raw statistics for variance

## Sample Size Considerations

With N = 3,487 (Silver cohort) and anticipated effect size d = 1.0, statistical power exceeds 99% for detecting AR(1) differences between Decline and Stable groups at α = 0.05.

## Missing Data

Complete case analysis is employed. Participants lacking sufficient waves are excluded rather than imputed, as AR(1) estimation requires observed longitudinal sequences.

## Software

All analyses conducted in Python 3.8+ using:
- NumPy 1.21+
- Pandas 1.3+
- SciPy 1.7+
- Scikit-learn 1.0+
- Matplotlib 3.4+

## Results Reporting

Results reported following STROBE guidelines for observational studies. Effect sizes reported with 95% confidence intervals where applicable. P-values < 0.001 reported as exact values in scientific notation.

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-01 | Initial protocol |
| 1.1 | 2025-03-01 | Added ELSA validation |
| 1.2 | 2025-06-01 | Added Winsorization for ELSA variance |
