# Critical Slowing Down as an Early Warning Signal for Cognitive Decline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the analysis code for detecting critical slowing down signatures in cognitive aging trajectories. The study applies dynamical systems theory to identify early warning signals that precede rapid cognitive decline in older adults.

**Key Finding**: Individuals exhibiting cognitive decline show significantly elevated autoregressive coefficients (AR(1) ≈ 0.05-0.14) compared to cognitively stable individuals (AR(1) ≈ -0.31 to -0.51), with large effect sizes (Cohen's d = 1.00-1.16) and predictive performance (AUC = 0.75-0.76).

## Theoretical Framework

Complex adaptive systems approaching critical transitions exhibit characteristic early warning signals including increased temporal autocorrelation and elevated variance. This study tests whether cognitive systems nearing functional decline display analogous critical slowing down phenomena.

**Core Hypothesis**: Cognitive decline represents a phase transition where the cognitive system loses self-organized criticality, manifesting as detectable dynamical signatures years before clinical deterioration.

## Data Sources

- **Primary Cohort**: China Health and Retirement Longitudinal Study (CHARLS), 2011-2020, 5 waves
- **External Validation**: English Longitudinal Study of Ageing (ELSA), 2008-2018, 6 waves

| Cohort | N | Waves | Primary Outcome | AUC |
|--------|---|-------|-----------------|-----|
| CHARLS Silver | 3,487 | ≥3 | Memory | 0.752 |
| CHARLS Gold | 2,833 | ≥4 | Memory | 0.755 |
| ELSA | 5,298 | 6 | Memory | 0.761 |

## Repository Structure

```
cognitive-critical-slowing/
├── src/
│   ├── step1_data_cleaning.py
│   ├── step2_cohort_construction.py
│   ├── step3_critical_slowing_analysis.py
│   ├── step4_robustness_validation.py
│   ├── step5_external_validation.py
│   └── utils.py
├── docs/
│   ├── methods_supplement.md
│   └── statistical_analysis_plan.md
├
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
python src/step1_data_cleaning.py --input data/CHARLS_raw.csv --output data/CHARLS_cleaned.csv
python src/step2_cohort_construction.py --input data/CHARLS_cleaned.csv --output data/CHARLS_cohort.csv
python src/step3_critical_slowing_analysis.py --input data/CHARLS_cohort.csv --output results/
python src/step4_robustness_validation.py --input data/CHARLS_cohort.csv --output results/
python src/step5_external_validation.py --input data/ELSA_raw.csv --output results/
```

## Key Results

### AR(1) Autoregressive Coefficients

| Group | CHARLS | ELSA | Interpretation |
|-------|--------|------|----------------|
| Decline | 0.054 ± 0.700 | 0.140 ± 0.410 | Loss of resilience |
| Stable | -0.514 ± 0.393 | -0.311 ± 0.303 | Maintained elasticity |
| Difference | Δ = 0.568 | Δ = 0.451 | Consistent pattern |
| p-value | < 10⁻¹⁶⁸ | < 10⁻²⁵⁵ | Highly significant |
| Effect Size | d = 1.001 | d = 1.155 | Large effect |

### Comprehensive Criticality Index (CCI)

The CCI integrates AR(1) and variance into a composite early warning score:

- **Sensitivity**: 74.2%
- **Specificity**: 68.3%
- **NPV**: 89.1% (suitable for clinical screening)

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
         Evidence from Two Longitudinal Cohort Studies},
  author={Liu, Abby and colleagues},
  journal={In preparation},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## Contact

For questions or collaboration inquiries, please open an issue or contact [uctqhhl@ucl.ac.uk].
