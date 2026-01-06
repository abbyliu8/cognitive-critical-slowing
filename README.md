# Cognitive decline follows critical phase transition dynamics across populations and neuropathological scales.<img width="468" height="55" alt="image" src="https://github.com/user-attachments/assets/7df5d56a-2fc3-4e75-afa0-732602eecc5e" />


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the analysis code for detecting critical slowing down signatures in cognitive aging trajectories. The study applies dynamical systems theory to identify early warning signals that precede rapid cognitive decline in older adults.

**Key Finding**: Across 10,559 individuals from three independent cohorts, individuals approaching cognitive decline exhibit elevated autoregressive coefficients (AR(1) ≈ 0.05–0.58) compared to cognitively stable individuals (AR(1) ≈ −0.18 to −0.51), with large effect sizes (Cohen's d = 1.00–1.25) and consistent predictive performance (AUC = 0.75–0.84). Dynamical signatures emerge 2–4 years before clinical deterioration and are independent of hippocampal atrophy.

## Theoretical Framework

Complex adaptive systems approaching critical transitions exhibit characteristic early warning signals including increased temporal autocorrelation and elevated variance. This study tests whether cognitive systems nearing functional decline display analogous critical slowing down phenomena.

**Core Hypothesis**: Cognitive decline represents a phase transition where the cognitive system loses self-organized criticality, manifesting as detectable dynamical signatures years before clinical deterioration.

## Data Sources

- **Discovery Cohort**: China Health and Retirement Longitudinal Study (CHARLS), 2011–2020, 5 waves
- **External Validation**: English Longitudinal Study of Ageing (ELSA), 2008–2018, 6 waves
- **Mechanistic Validation**: Alzheimer's Disease Neuroimaging Initiative (ADNI), 2005–2023

| Cohort | N | Waves | Primary Outcome | AUC | Cohen's d |
|--------|---|-------|-----------------|-----|-----------|
| CHARLS | 3,487 | ≥3 | Episodic Memory | 0.752 | 1.001 |
| ELSA | 5,298 | 6 | Episodic Memory | 0.761 | 1.155 |
| ADNI | 1,774 | ≥4 | ADAS-Cog13 | 0.843 | 1.250 |
| **Total** | **10,559** | — | — | — | — |

## Repository Structure

```
cognitive-critical-slowing/
├── src/
│   ├── charls/
│   │   ├── step1_data_cleaning.py
│   │   ├── step2_cohort_construction.py
│   │   ├── step3_critical_slowing_analysis.py
│   │   └── step4_robustness_validation.py
│   │   └── step5_biomarker_validation.py
│   ├── adni/
│   │   ├── ADNI_cleaning_pipeline.py
│   │   ├── ADNI_critical_slowing.py
│   └── utils.py
├── docs/
│   ├── methods_supplement.md
│   └── statistical_analysis_plan.md
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/abbyliu8/cognitive-critical-slowing.git
cd cognitive-critical-slowing
pip install -r requirements.txt
```

## Usage

Execute analysis pipeline sequentially:

```bash
# Discovery cohort (CHARLS)
python src/step1_data_cleaning.py --input data/CHARLS_raw.csv --output data/CHARLS_cleaned.csv
python src/step2_cohort_construction.py --input data/CHARLS_cleaned.csv --output data/CHARLS_cohort.csv
python src/step3_critical_slowing_analysis.py --input data/CHARLS_cohort.csv --output results/

# Robustness validation
python src/step4_robustness_validation.py --input data/CHARLS_cohort.csv --output results/

# External validation (ELSA)
python src/step5_external_validation_elsa.py --input data/ELSA_raw.csv --output results/

# Mechanistic validation (ADNI)
python src/step6_mechanistic_validation_adni.py --input data/ADNIMERGE.csv --output results/
```

## Key Results

### AR(1) Autoregressive Coefficients

| Group | CHARLS | ELSA | ADNI | Interpretation |
|-------|--------|------|------|----------------|
| Decline | 0.054 | 0.140 | 0.584 | Loss of resilience |
| Stable | −0.514 | −0.311 | −0.181 | Maintained elasticity |
| Δ (Decline − Stable) | 0.568 | 0.451 | 0.765 | Consistent pattern |
| p-value | < 10⁻¹⁶⁸ | < 10⁻²⁵⁵ | < 10⁻¹⁸ | Highly significant |

### Structural Independence (ADNI)

AR(1) correlation with baseline hippocampal volume: r = −0.057 (P = 0.025)

After adjusting for hippocampal atrophy, age, education, and APOE ε4:
- AR(1) → Decline: OR = 1.826, 95% CI 1.556–2.144, P = 1.78 × 10⁻¹³

### Biological Validation

| Biomarker | Correlation with Variance | P-value |
|-----------|---------------------------|---------|
| CSF pTau₁₈₁ (n=398) | ρ = 0.327 | < 10⁻²⁸ |
| Braak Stage (n=72) | ρ = 0.365 | < 0.001 |

### Comprehensive Criticality Index (CCI)

| Metric | CHARLS | ELSA | ADNI |
|--------|--------|------|------|
| AUC | 0.752 | 0.761 | 0.843 |
| Sensitivity | 74.2% | 77.2% | 77.3% |
| Specificity | 68.3% | 71.2% | 82.6% |
| NPV | 89.1% | 89.0% | 88.0% |

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.21.0
- Pandas ≥ 1.3.0
- SciPy ≥ 1.7.0
- Scikit-learn ≥ 1.0.0
- Statsmodels ≥ 0.13.0
- Matplotlib ≥ 3.4.0
- Seaborn ≥ 0.11.0

## Citation

If you use this code, please cite:

```bibtex
@article{liu2025critical,
  title={Critical Slowing Down as an Early Warning Signal for Cognitive Decline: 
         A Three-Cohort Validation Study with Neurobiological Mechanism},
  author={Liu, Abby and colleagues},
  journal={Nature Aging (in preparation)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration inquiries, please open an issue or contact [uctqhhl@ucl.ac.uk].
