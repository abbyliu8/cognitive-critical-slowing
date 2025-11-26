# Supplementary Methods

## Theoretical Framework

### Critical Slowing Down in Complex Systems

Complex adaptive systems approaching critical transitions exhibit characteristic dynamical signatures termed "critical slowing down." As a system loses resilience and approaches a tipping point, its recovery rate from perturbations decreases, leading to measurable changes in temporal dynamics. Two primary indicators characterize this phenomenon: elevated temporal autocorrelation and increased variance.

The autoregressive coefficient of order 1 (AR(1)) quantifies the degree to which the current state depends on the immediately preceding state. In a resilient system, perturbations dissipate quickly, yielding low autocorrelation. Conversely, as the system approaches criticality, perturbations persist longer, manifesting as elevated AR(1) values approaching unity.

Variance similarly increases near critical transitions because the system's restoring forces weaken, allowing fluctuations to accumulate. These dynamical signatures have been validated across diverse domains including climate systems, ecological networks, and financial markets.

### Application to Cognitive Systems

The cognitive system constitutes a complex adaptive network integrating neural, psychological, and physiological subsystems. Under the critical slowing down framework, cognitive decline represents not merely the accumulation of pathological insults but rather a phase transition wherein the system loses its capacity for self-organized criticality.

This theoretical reframing yields a testable prediction: individuals destined for cognitive decline should exhibit elevated AR(1) coefficients and variance in their cognitive trajectories during the pre-decline phase, before manifest deterioration becomes apparent through conventional assessments.

## Statistical Methods

### Sample Selection

The primary analysis cohort (Silver standard) included participants with cognitive assessments at three or more waves. This threshold balances statistical power for AR(1) estimation (minimum three observations required) against sample retention. The robustness cohort (Gold standard) restricted to four or more waves to assess whether findings strengthen with additional temporal resolution.

### Trajectory Classification

Cognitive trajectories were characterized using ordinary least squares regression of memory scores against time. The resulting slopes were trichotomized at the 33rd and 67th percentiles to define Decline, Stable, and Improved groups. This approach ensures balanced group sizes while capturing the tails of the trajectory distribution where critical slowing down effects should be most pronounced.

### AR(1) Estimation

For each participant, the AR(1) coefficient was computed as the Pearson correlation between consecutive observations:

$$\rho = \text{cor}(Y_t, Y_{t-1})$$

where $Y_t$ denotes the cognitive score at time $t$. This simplified estimator assumes equally spaced observations and stationary dynamics within individuals. Although hierarchical Bayesian approaches could yield more precise estimates by borrowing strength across participants, the current implementation prioritizes transparency and reproducibility.

### Variance Estimation

Individual-level variance was computed as the sample variance of cognitive scores across all available waves:

$$\sigma^2 = \frac{1}{n-1}\sum_{t=1}^{n}(Y_t - \bar{Y})^2$$

For the ELSA validation cohort, extreme values (> 3 × IQR above Q3) were Winsorized at the 5th and 95th percentiles to ensure robust statistics. This preprocessing step was not required for CHARLS data, which exhibited well-behaved variance distributions.

### Comprehensive Criticality Index (CCI)

The CCI integrates AR(1) and variance into a single composite score. Both indicators were first standardized to z-scores using the full sample distribution, then averaged:

$$\text{CCI} = \frac{z_{\rho} + z_{\sigma^2}}{2}$$

This equal weighting reflects the theoretical expectation that both indicators contribute comparably to early warning signals. Alternative weighting schemes (e.g., principal component loadings) yielded qualitatively similar results.

### Prediction Performance

Discriminative performance was assessed using the area under the receiver operating characteristic curve (AUC). Sensitivity and specificity were evaluated at the Youden-optimal threshold maximizing (sensitivity + specificity - 1). Negative predictive value (NPV) was prioritized for clinical interpretation given the intended screening application.

### Effect Size Estimation

Between-group differences were quantified using Cohen's d:

$$d = \frac{\bar{X}_1 - \bar{X}_2}{s_p}$$

where $s_p$ denotes the pooled standard deviation. Effect sizes were interpreted following conventional benchmarks: 0.2 (small), 0.5 (medium), 0.8 (large).

## Robustness Analyses

### Cohort Stringency

Analyses were repeated in the Gold cohort (≥4 waves) to assess sensitivity to data completeness requirements. Stronger effect sizes in this higher-quality subset would support the validity of AR(1) estimation and argue against bias from selective attrition.

### External Validation

The ELSA cohort provided independent replication in a Western population with distinct cultural and healthcare contexts. Consistent findings across CHARLS and ELSA would demonstrate cross-population generalizability of critical slowing down signatures.

### Outlier Handling

Variance estimates proved sensitive to extreme values in ELSA. Sensitivity analyses comparing raw statistics, Winsorized statistics, and trimmed means confirmed that the primary conclusions remained robust across analytical approaches.

## Limitations

Several limitations warrant acknowledgment. First, the observational design precludes causal inference regarding whether critical slowing down signatures mechanistically precede decline or merely co-occur with it. Second, the biennial assessment intervals may inadequately capture rapid dynamics. Third, restricting to participants with three or more waves introduces potential survivor bias, although the stronger effects observed in the Gold cohort argue against this concern. Fourth, the use of memory as the sole cognitive indicator limits generalizability to other cognitive domains.
